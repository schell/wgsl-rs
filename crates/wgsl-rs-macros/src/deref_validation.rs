//! Compile-time validation of deref expressions (wgsl-rs#153).
//!
//! Module variables are value references in WGSL, not pointers. A `*` in
//! front of a linkage accessor (`*get!(VAR)`) is a Rust-side guard
//! artifact and is elided at IR conversion — see `ir_convert`. A deref of
//! a *local*, however, is only valid WGSL when the local holds a real
//! pointer (`let p = &x`, `ptr!`). Dereferencing a local that holds a
//! module variable *value* — `let u = get!(U); *u` — compiles on the CPU
//! (the guard derefs to `T`) but renders `*u` in WGSL, which wgpu rejects
//! with "the operand of the `*` operator must be a pointer". Those derefs
//! are compile errors here, with a note pointing at `load!`.
//!
//! Locals that cannot be proven to hold values (function parameters,
//! calls, field accesses, names shadowed across scopes with differing
//! shapes) fail open — wgpu validation is the backstop. These checks are
//! a footgun net, not a type system.

use std::collections::HashMap;

use crate::{
    parse::{Block, ElseBody, Error, Expr, Item, Stmt, UnOp},
    parse_visitor::{ParseVisitorMut, walk_expr, walk_fn},
};

/// What a let-bound local holds, as far as its initializer can prove.
#[derive(Clone, PartialEq, Eq)]
enum LocalShape {
    /// Holds a module variable value (`get!(VAR)`, `*get!(VAR)`): a WGSL
    /// value, not a pointer. Carries the module variable's name so the
    /// error can suggest the concrete `load!` spelling.
    Value(String),
    /// Holds a WGSL pointer (`&expr`): deref is valid.
    Pointer,
    /// Shape is not provable from the initializer alone; fail open.
    Unknown,
}

impl LocalShape {
    /// The shape's kind, ignoring the `Value` payload.
    fn kind(&self) -> ShapeKind {
        match self {
            LocalShape::Value(_) => ShapeKind::Value,
            LocalShape::Pointer => ShapeKind::Pointer,
            LocalShape::Unknown => ShapeKind::Unknown,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShapeKind {
    Value,
    Pointer,
    Unknown,
}

/// Validate deref expressions in `items`.
pub(crate) fn validate_items(items: &mut [Item]) -> Result<(), Error> {
    let mut visitor = DerefVisitor {
        locals: HashMap::new(),
    };
    for item in items {
        visitor.visit_item(item)?;
    }
    Ok(())
}

/// Visitor that errors on derefs of locals that provably hold module
/// variable values.
struct DerefVisitor {
    /// Locals of the function currently being visited, mapped to the shape
    /// of their bindings. Parameters are seeded as `Unknown`, and a name
    /// bound multiple times with differing shapes is `Unknown` (fail
    /// open), since the scopes are flattened.
    locals: HashMap<String, LocalShape>,
}

impl ParseVisitorMut for DerefVisitor {
    fn visit_fn(&mut self, f: &mut crate::parse::ItemFn) -> Result<(), Error> {
        self.locals.clear();
        // Seed parameters as `Unknown`: a parameter may be a real WGSL
        // pointer (`ptr!`), and a branch-local `let` shadowing a parameter
        // must not poison derefs of the parameter outside that branch.
        // With the parameter in the map, a colliding `let` downgrades the
        // name to `Unknown` (fail open) instead of `Value`.
        for arg in &f.inputs {
            note_shape(&mut self.locals, arg.ident.to_string(), LocalShape::Unknown);
        }
        collect_locals(&f.block, &mut self.locals);
        walk_fn(self, f)
    }

    fn visit_expr(&mut self, e: &mut Expr) -> Result<(), Error> {
        if let Expr::Unary {
            op: UnOp::Deref(_),
            expr,
        } = e
            && let Expr::Ident(ident) = expr.peel_parens()
            && let Some(LocalShape::Value(var)) = self.locals.get(&ident.to_string())
        {
            return Err(Error::unsupported(
                ident.span(),
                format!(
                    "'*{ident}' dereferences a local that holds a module variable value. In WGSL, \
                     module variables are values, not pointers — bind the value with load! (let \
                     {ident} = load!({var});) or deref the accessor directly (*get!({var}))"
                ),
            ));
        }
        walk_expr(self, e)
    }
}

/// Collect every `let` binding in `block` into `locals`, flattening
/// nested scopes.
fn collect_locals(block: &Block, locals: &mut HashMap<String, LocalShape>) {
    for stmt in &block.stmt {
        collect_stmt(stmt, locals);
    }
}

/// Recurse into the statements that carry bindings or nested blocks.
fn collect_stmt(stmt: &Stmt, locals: &mut HashMap<String, LocalShape>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                note_shape(locals, local.ident.to_string(), init_shape(&init.expr));
            }
        }
        Stmt::If(i) => {
            collect_locals(&i.then_block, locals);
            collect_else(&i.else_branch, locals);
        }
        Stmt::While { body, .. } | Stmt::Loop { body, .. } => collect_locals(body, locals),
        Stmt::Block(b) => collect_locals(b, locals),
        Stmt::For(f) => collect_locals(&f.body, locals),
        Stmt::Switch(s) => {
            for arm in &s.arms {
                collect_locals(&arm.body, locals);
            }
        }
        _ => {}
    }
}

/// Recurse into the else branch of an if chain.
fn collect_else(
    else_branch: &Option<crate::parse::ElseBranch>,
    locals: &mut HashMap<String, LocalShape>,
) {
    if let Some(branch) = else_branch {
        match &branch.body {
            ElseBody::Block(b) => collect_locals(b, locals),
            ElseBody::If(i) => {
                collect_locals(&i.then_block, locals);
                collect_else(&i.else_branch, locals);
            }
        }
    }
}

/// Record a binding's shape. Re-binding a name with a differing shape
/// makes it `Unknown` — the scopes are flattened, so a deref cannot be
/// attributed to a specific binding; fail open rather than risk a false
/// error on a valid pointer use.
fn note_shape(locals: &mut HashMap<String, LocalShape>, name: String, shape: LocalShape) {
    locals
        .entry(name)
        .and_modify(|existing| {
            if existing.kind() != shape.kind() {
                *existing = LocalShape::Unknown;
            }
        })
        .or_insert(shape);
}

/// The shape a binding takes from its initializer.
fn init_shape(expr: &Expr) -> LocalShape {
    match expr.peel_parens() {
        // `&expr` is always a WGSL pointer.
        Expr::Reference { .. } => LocalShape::Pointer,
        // A linkage accessor yields a guard on the CPU; in WGSL the bare
        // variable reference is a value, not a pointer.
        Expr::LinkageAccess { ident, .. } => LocalShape::Value(ident.to_string()),
        Expr::Unary {
            op: UnOp::Deref(_),
            expr,
        } => match expr.peel_parens() {
            // `*get!(VAR)` copies the value out — a value, not a pointer.
            Expr::LinkageAccess { ident, .. } => LocalShape::Value(ident.to_string()),
            // `*&expr` takes the referenced value's shape, which is not
            // provable here; fail open.
            _ => LocalShape::Unknown,
        },
        _ => LocalShape::Unknown,
    }
}
