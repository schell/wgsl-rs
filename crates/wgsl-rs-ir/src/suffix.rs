//! Type-directed integer literal suffix insertion.
//!
//! Rust infers the type of an unsuffixed integer literal from context —
//! `0` in a `u32` function is `u32`. WGSL instead defaults unsuffixed
//! integer literals to `i32` wherever the surrounding context does not
//! force a concrete type (most visibly inside polymorphic builtins like
//! `select`), so Rust that compiles cleanly can render to WGSL that naga
//! rejects (wgsl-rs#145).
//!
//! This pass propagates expected types from anchor points down to leaf
//! literals and writes Rust-style suffixes (`u32` / `i32`) onto
//! empty-suffix integer literals, the exact forms [`crate::render`]
//! translates to WGSL `u` / `i`. Anchors:
//!
//! * function return types (`return` statements and trailing expressions),
//! * typed `let`/`var` initializers and `const` items,
//! * assignment and compound-assignment targets (identifiers, fields, array
//!   elements, swizzles),
//! * array literals and struct constructor fields (struct constructors
//!   self-anchor through the struct registry even without an outer annotation),
//! * casts,
//! * binary and comparison operands (a bare literal adopts the provable scalar
//!   type of the other operand),
//! * `for` bounds when the loop variable's type is explicit,
//! * `switch` case selectors from the selector's type.
//!
//! Like [`crate::deshadow`], this is a whole-module IR mutation intended
//! to run on freshly built and substituted IR, after deshadowing and
//! immediately before rendering (see the DEVLOG entry for 2026-09-21).
//! Call-argument propagation is part of this pass: user function
//! signatures (seeded across sources via [`fn_signatures`] /
//! [`suffix_module_with_imports`]), same-type builtin groups (`select` /
//! `min` / `max` / `clamp`), and vector constructors.
//!
//! Deliberate exclusions:
//!
//! * Type positions ([`Type::Array`] lengths, [`Expr::ZeroValueArray`] lengths)
//!   are never suffixed — WGSL wants abstract integers there.
//! * Array-index expressions stay bare: WGSL coerces the abstract integer in
//!   `a[0]` to either index type, so there is nothing to fix.
//! * Float expectations never touch integer literals: the IR cannot contain an
//!   integer literal in an `f32` context, because Rust would have rejected the
//!   source.

use std::collections::HashMap;

use crate::{
    mangle::mangle,
    types::{
        BinOp, Block, CaseSelector, ElseBranch, Expr, FnPath, ForLoop, ImplItem, Item, ItemFn, Lit,
        Local, Module, ReturnType, ScalarType, Stmt, StmtIf, StmtSwitch, Type, UnOp,
    },
};

/// Insert type-directed suffixes onto unsuffixed integer literals across
/// every item in `module`.
///
/// Only literals with an empty suffix are written, so the pass is
/// idempotent and safe to run more than once.
pub fn suffix_module(module: &mut Module) {
    SuffixPass::default().walk_module(module);
}

/// Like [`suffix_module`], but for a bare slice of items — e.g. an
/// instantiated cross-source template that is rendered without a
/// [`Module`] wrapper.
pub fn suffix_items(items: &mut [Item]) {
    SuffixPass::default().walk_items(items);
}

/// Like [`suffix_module`], but seeds the user-function signature registry
/// with `imports` — signatures harvested from other sources via
/// [`fn_signatures`] — so calls to imported functions anchor their
/// arguments too. The module's own signatures shadow imported ones,
/// matching Rust name resolution.
pub fn suffix_module_with_imports(module: &mut Module, imports: &HashMap<String, Vec<Type>>) {
    SuffixPass {
        fn_sigs: imports.clone(),
        ..SuffixPass::default()
    }
    .walk_module(module);
}

/// Like [`suffix_items`], but seeds the user-function signature registry
/// with `imports` — signatures harvested from other sources via
/// [`fn_signatures_in_items`] — so calls to imported functions anchor
/// their arguments too. Used for cross-source template instantiation,
/// where the template's items reference functions defined in other
/// chunks of the assembled translation unit.
pub fn suffix_items_with_imports(items: &mut [Item], imports: &HashMap<String, Vec<Type>>) {
    SuffixPass {
        fn_sigs: imports.clone(),
        ..SuffixPass::default()
    }
    .walk_items(items);
}

/// Harvest the module's user-function signatures — callee name → declared
/// parameter types — for seeding [`suffix_module_with_imports`] /
/// [`suffix_items_with_imports`] on a source that imports or instantiates
/// this module. Free functions key by their name; impl methods key by
/// their mangled render name.
pub fn fn_signatures(module: &Module) -> HashMap<String, Vec<Type>> {
    fn_signatures_in_items(&module.items)
}

/// Like [`fn_signatures`], but for a bare slice of items — e.g. an
/// instantiated cross-source template rendered without a [`Module`]
/// wrapper. Note that signatures are taken from the items as-is, so
/// call any renaming (such as template instance mangling) before
/// harvesting.
pub fn fn_signatures_in_items(items: &[Item]) -> HashMap<String, Vec<Type>> {
    let mut sigs = HashMap::new();
    for item in items {
        match item {
            Item::Fn(f) => {
                sigs.insert(
                    f.name.to_string(),
                    f.inputs.iter().map(|a| a.ty.clone()).collect(),
                );
            }
            Item::Impl(imp) => {
                for impl_item in &imp.items {
                    if let ImplItem::Fn(f) = impl_item {
                        sigs.insert(
                            mangle(&[imp.self_ty.as_str(), &f.name]),
                            f.inputs.iter().map(|a| a.ty.clone()).collect(),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    sigs
}

/// A collected struct definition: type parameter names plus fields as
/// `(name, type)` pairs.
struct StructDef {
    type_params: Vec<String>,
    fields: Vec<(String, Type)>,
}

/// Scope-tracking state for the suffix pass.
///
/// Structurally mirrors the deshadow pass: one instance walks a whole
/// module, pushing a scope frame per block so that the innermost
/// declaration of a name wins when resolving assignment targets.
#[derive(Default)]
struct SuffixPass {
    /// Stack of scope frames mapping local names to their declared types.
    /// The bottom frame holds the current function's parameters.
    scopes: Vec<HashMap<String, Type>>,
    /// The current function's declared return type, if any.
    return_ty: Option<Type>,
    /// Struct definitions by name, collected before the walk so struct
    /// constructor fields and field assignments can resolve types.
    structs: HashMap<String, StructDef>,
    /// User function signatures: callee name → declared parameter
    /// types. Free functions key by their name; impl methods key by
    /// their mangled render name (`mangle(&[self_ty, method])`).
    fn_sigs: HashMap<String, Vec<Type>>,
    /// Module-level `const` item types by name, consulted as the
    /// fallback of scope lookup.
    consts: HashMap<String, Type>,
}

impl SuffixPass {
    /// Walk every item in the module.
    fn walk_module(&mut self, module: &mut Module) {
        self.walk_items(&mut module.items);
    }

    /// First collect struct definitions, then walk each top-level item
    /// that can anchor a type expectation: `const` items via their
    /// declared type, and functions (free or impl methods) via their
    /// signatures.
    fn walk_items(&mut self, items: &mut [Item]) {
        for item in items.iter() {
            match item {
                Item::Struct(s) => {
                    self.structs.insert(
                        s.name.clone(),
                        StructDef {
                            type_params: s.type_params.clone(),
                            fields: s
                                .fields
                                .iter()
                                .map(|f| (f.name.clone(), f.ty.clone()))
                                .collect(),
                        },
                    );
                }
                Item::Fn(f) => {
                    self.fn_sigs.insert(
                        f.name.to_string(),
                        f.inputs.iter().map(|a| a.ty.clone()).collect(),
                    );
                }
                Item::Const(c) => {
                    self.consts.insert(c.name.clone(), c.ty.clone());
                }
                Item::Impl(imp) => {
                    for impl_item in &imp.items {
                        if let ImplItem::Fn(f) = impl_item {
                            self.fn_sigs.insert(
                                mangle(&[imp.self_ty.as_str(), &f.name]),
                                f.inputs.iter().map(|a| a.ty.clone()).collect(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        for item in items {
            match item {
                Item::Const(c) => self.expect(&mut c.expr, Some(&c.ty)),
                Item::Fn(f) => self.walk_fn(f),
                Item::Impl(imp) => {
                    for impl_item in &mut imp.items {
                        match impl_item {
                            ImplItem::Fn(f) => self.walk_fn(f),
                            // Associated consts anchor from their
                            // declared type, like module-level consts.
                            ImplItem::Const(c) => self.expect(&mut c.expr, Some(&c.ty)),
                            // Associated type aliases carry no
                            // expression.
                            ImplItem::Type(_) => {}
                        }
                    }
                }
                // Structs were collected above; uniforms, storage,
                // samplers, textures, and enums carry no expression to
                // suffix.
                _ => {}
            }
        }
    }

    /// Walk one function body with its parameters and return type as the
    /// seed context.
    fn walk_fn(&mut self, f: &mut ItemFn) {
        self.scopes.clear();
        self.return_ty = match &f.return_type {
            ReturnType::Type { ty, .. } => Some(ty.clone()),
            ReturnType::Default => None,
        };
        let params: HashMap<String, Type> = f
            .inputs
            .iter()
            .map(|arg| (arg.name.clone(), arg.ty.clone()))
            .collect();
        self.scopes.push(params);
        self.walk_block(&mut f.block);
        self.scopes.clear();
        self.return_ty = None;
    }

    /// Walk a block's statements in a fresh scope frame.
    fn walk_block(&mut self, block: &mut Block) {
        self.scopes.push(HashMap::new());
        for stmt in &mut block.stmts {
            self.walk_stmt(stmt);
        }
        self.scopes.pop();
    }

    /// Apply the anchors that produce a type expectation for a statement.
    fn walk_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Local(local) => self.walk_local(local),
            Stmt::Const(c) => {
                self.expect(&mut c.expr, Some(&c.ty));
                // Register the const's declared type in scope after
                // walking its initializer, like a typed local — later
                // expressions anchor from it
                // (`const C: u32 = 1; select(C, 0, cond)`).
                let name = c.name.clone();
                let ty = c.ty.clone();
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(name, ty);
                }
            }
            Stmt::Assignment { lhs, rhs } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
                self.walk_assignment_rhs(lhs, rhs);
            }
            Stmt::Return(Some(expr)) => {
                let ret = self.return_ty.clone();
                self.expect(expr, ret.as_ref());
            }
            Stmt::Expr { expr, has_semi } => {
                // A trailing expression is an implicit `return expr;`.
                let expected = if *has_semi {
                    None
                } else {
                    self.return_ty.clone()
                };
                self.expect(expr, expected.as_ref());
            }
            Stmt::If(i) => self.walk_if(i),
            Stmt::While { condition, body } => {
                self.expect(condition, None);
                self.walk_block(body);
            }
            Stmt::Loop { body } => self.walk_block(body),
            Stmt::For(f) => self.walk_for(f),
            Stmt::Switch(s) => self.walk_switch(s),
            Stmt::Block(b) => self.walk_block(b),
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
            } => {
                // The renderer emits a u32 loop counter, compares it
                // against `size`, and adds the offsets to it — all
                // three flow through u32 arithmetic.
                let u32_ty = Type::Scalar(ScalarType::U32);
                self.expect(src_offset, Some(&u32_ty));
                self.expect(dest_offset, Some(&u32_ty));
                self.expect(size, Some(&u32_ty));
                self.expect(src, None);
                self.expect(dest, None);
            }
            // Control flow and extension macros carry no type
            // expectation in this pass.
            Stmt::Return(None)
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Discard
            | Stmt::Macro { .. } => {}
        }
    }

    /// Suffix a local's initializer and register the local's type in the
    /// current scope.
    ///
    /// Typed locals propagate their declared type; un-annotated locals
    /// still get walked so self-anchored expressions (struct
    /// constructors) inside them are suffixed. Rust infers `let x = 0;`
    /// as `i32`, matching WGSL's default, so bare literals stay bare.
    fn walk_local(&mut self, local: &mut Local) {
        match (&local.ty, &mut local.init) {
            (Some(ty), Some(init)) => self.expect(init, Some(ty)),
            (None, Some(init)) => self.expect(init, None),
            _ => {}
        }
        if let Some(ty) = &local.ty
            && let Some(scope) = self.scopes.last_mut()
        {
            scope.insert(local.name.clone(), ty.clone());
        }
    }

    /// Suffix the RHS of an assignment from the type of the assignment
    /// target, when that is provable (identifier, field, array element,
    /// or swizzle rooted at a typed base).
    fn walk_assignment_rhs(&mut self, lhs: &Expr, rhs: &mut Expr) {
        let target_ty = self.expr_ty(lhs);
        self.expect(rhs, target_ty.as_ref());
    }

    /// Walk an `if` / `else if` / `else` chain.
    fn walk_if(&mut self, i: &mut StmtIf) {
        self.expect(&mut i.condition, None);
        self.walk_block(&mut i.then_block);
        if let Some(else_branch) = &mut i.else_branch {
            match else_branch {
                ElseBranch::Block(b) => self.walk_block(b),
                ElseBranch::If(inner) => self.walk_if(inner),
            }
        }
    }

    /// Walk a `for` loop: bounds inherit the loop variable's type, and
    /// the loop variable is visible in the body.
    ///
    /// Parsed Rust loops never annotate the loop variable (`for i: u32
    /// in …` is not valid Rust — the IR field stays `None`), so its
    /// type is inferred from a provable range bound, mirroring Rust's
    /// own inference. Fully-bare ranges (`for i in 0..8`) infer `i32`
    /// in both Rust and WGSL and stay bare.
    fn walk_for(&mut self, f: &mut ForLoop) {
        let var_ty = f
            .var_ty
            .clone()
            .or_else(|| self.expr_ty(&f.from).or_else(|| self.expr_ty(&f.to)));
        if let Some(ty) = &var_ty {
            self.expect(&mut f.from, Some(ty));
            self.expect(&mut f.to, Some(ty));
        }
        let mut seeded = false;
        if let Some(ty) = var_ty {
            let mut frame = HashMap::new();
            frame.insert(f.var.clone(), ty);
            self.scopes.push(frame);
            seeded = true;
        }
        self.walk_block(&mut f.body);
        if seeded {
            self.scopes.pop();
        }
    }

    /// Walk a `switch`: the selector subtree is walked first so nested
    /// call-site and binary anchoring applies (`match x + select(0, 1,
    /// c)` suffixes the select through the binary anchor), then case
    /// selectors inherit the selector's derived type, and arm bodies
    /// are walked for their own anchors.
    fn walk_switch(&mut self, s: &mut StmtSwitch) {
        self.expect(&mut s.selector, None);
        let selector_ty = self.expr_ty(&s.selector);
        for arm in &mut s.arms {
            if let Some(ty) = &selector_ty {
                for sel in &mut arm.selectors {
                    match sel {
                        CaseSelector::Literal(l) => suffix_lit(l, ty),
                        CaseSelector::Expr(e) => self.expect(e, Some(ty)),
                        CaseSelector::Default => {}
                    }
                }
            }
            self.walk_block(&mut arm.body);
        }
    }

    /// Look up the declared type of `name`, innermost scope first.
    /// Module-level consts are the fallback after every scope misses;
    /// locals, parameters, and function-body consts shadow them,
    /// matching Rust name resolution.
    fn lookup(&self, name: &str) -> Option<Type> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name).cloned())
            .or_else(|| self.consts.get(name).cloned())
    }

    /// The concrete type of `expr`, when provable without a full type
    /// checker: typed identifiers from scope, suffixed literals, casts,
    /// and field / element accesses rooted at a typed base.
    fn expr_ty(&self, expr: &Expr) -> Option<Type> {
        match expr {
            Expr::Ident(name) => self.lookup(name),
            Expr::Lit(Lit::Int { suffix, .. }) => match suffix.as_str() {
                "u32" | "usize" => Some(Type::Scalar(ScalarType::U32)),
                "i32" | "isize" => Some(Type::Scalar(ScalarType::I32)),
                _ => None,
            },
            Expr::Paren(inner) => self.expr_ty(inner),
            Expr::Cast { ty, .. } => Some((**ty).clone()),
            Expr::FieldAccess { base, field } => {
                let Type::Struct { name, type_args } = self.expr_ty(base)? else {
                    return None;
                };
                self.field_ty(&name, &type_args, field)
            }
            Expr::ArrayIndexing { lhs, .. } => match self.expr_ty(lhs)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem } => Some((*elem).clone()),
                _ => None,
            },
            Expr::Swizzle { lhs, .. } => match self.expr_ty(lhs)? {
                Type::Vector {
                    scalar_ty: Some(scalar),
                    ..
                } => Some(Type::Scalar(scalar)),
                // A swizzle on a matrix selects a column (a vector),
                // not a scalar — no scalar expectation is derivable.
                _ => None,
            },
            // Arithmetic and shift operators yield their operands'
            // type; comparisons and logical operators yield `bool`.
            Expr::Binary { lhs, op, rhs } => {
                if is_arithmetic(op) {
                    self.expr_ty(lhs).or_else(|| self.expr_ty(rhs))
                } else {
                    Some(Type::Scalar(ScalarType::Bool))
                }
            }
            // `!`, bitwise complement, and negation preserve the
            // operand type; a deref yields the pointee, which the
            // pointer expression's type does not carry.
            Expr::Unary { op, expr } => match op {
                UnOp::Not | UnOp::Complement | UnOp::Neg => self.expr_ty(expr),
                UnOp::Deref => None,
            },
            // Same-type builtin value groups yield their first value
            // argument's type.
            Expr::FnCall { path, params, .. } => {
                let FnPath::Ident(name) = path else {
                    return None;
                };
                if builtin_value_args(name.as_str()).is_some() {
                    params.first().and_then(|p| self.expr_ty(p))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Resolve the declared type of `field` on struct `name`,
    /// substituting the struct's type parameters positionally from
    /// `type_args`.
    fn field_ty(&self, name: &str, type_args: &[Type], field: &str) -> Option<Type> {
        let def = self.structs.get(name)?;
        let mut ty = def
            .fields
            .iter()
            .find(|(field_name, _)| field_name == field)?
            .1
            .clone();
        substitute_params(&mut ty, &def.type_params, type_args);
        Some(ty)
    }

    /// Propagate expectations into a call's arguments.
    ///
    /// User function signatures are consulted first — they shadow the
    /// builtin rules, mirroring how Rust resolves a local `fn select`
    /// over a glob-imported builtin, and how WGSL resolves user
    /// declarations over predeclared ones. Falls back to builtin rules:
    /// vector constructors carry their element type in the name, and
    /// same-type builtins (`select`, `min`/`max`, `clamp`) propagate one
    /// shared value type across their value arguments.
    fn walk_fn_call(
        &mut self,
        path: &FnPath,
        type_args: &[Type],
        params: &mut [Expr],
        expected: Option<&Type>,
    ) {
        let callee = match path {
            FnPath::Ident(name) => name.clone(),
            FnPath::TypeMethod { ty, method } => mangle(&[ty, method]),
        };
        if let Some(param_tys) = self.fn_sigs.get(&callee).cloned() {
            for (i, arg) in params.iter_mut().enumerate() {
                self.expect(arg, param_tys.get(i));
            }
            return;
        }
        if let Some(elem_ty) = vec_ctor_elem(&callee, type_args) {
            for arg in params {
                self.expect(arg, Some(&elem_ty));
            }
            return;
        }
        if let Some(value_indices) = builtin_value_args(&callee) {
            // The shared value type: an outer expectation first, else a
            // provable value argument (mirrors the binary-operand
            // anchor — `select(x, 0, cond)` with `x: u32`).
            let group_ty = expected.cloned().or_else(|| {
                value_indices
                    .iter()
                    .find_map(|&i| params.get(i).and_then(|a| self.expr_ty(a)))
            });
            for (i, arg) in params.iter_mut().enumerate() {
                if value_indices.contains(&i) {
                    self.expect(arg, group_ty.as_ref());
                } else {
                    // `select`'s condition is not a value argument.
                    self.expect(arg, None);
                }
            }
            return;
        }
        for arg in params {
            self.expect(arg, None);
        }
    }

    /// Propagate `expected` into `expr`, suffixing bare integer literals
    /// when the expectation resolves to a concrete scalar integer type.
    ///
    /// `expected` is `None` for contexts with no outer annotation; the
    /// walk still descends so self-anchored sub-expressions (struct
    /// constructors, casts, anchored binary operands) get suffixed.
    fn expect(&mut self, expr: &mut Expr, expected: Option<&Type>) {
        match expr {
            Expr::Lit(lit) => {
                if let Some(ty) = expected {
                    suffix_lit(lit, ty);
                }
            }
            Expr::Paren(inner) => self.expect(inner, expected),
            Expr::Array { elems } => {
                let elem_ty = expected.and_then(|ty| match ty {
                    Type::Array { elem, .. } | Type::RuntimeArray { elem } => {
                        Some((**elem).clone())
                    }
                    _ => None,
                });
                for e in elems {
                    self.expect(e, elem_ty.as_ref());
                }
            }
            Expr::Struct {
                name,
                type_args: ctor_args,
                fields,
            } => {
                // The expectation's type arguments come from an explicit
                // annotation and take precedence; otherwise the
                // constructor's own arguments carry the instantiation.
                let args: Vec<Type> = match expected {
                    Some(Type::Struct {
                        name: exp_name,
                        type_args,
                    }) if exp_name.as_str() == name.as_str() => type_args.clone(),
                    _ => ctor_args.clone(),
                };
                for field in fields {
                    let field_ty = self.field_ty(name, &args, &field.member);
                    self.expect(&mut field.expr, field_ty.as_ref());
                }
            }
            Expr::Cast { lhs, ty } => self.expect(lhs, Some(ty)),
            Expr::Binary { lhs, op, rhs } => {
                // Arithmetic and shift operators pass an outer
                // expectation to their operands (the operands share the
                // result type); comparisons and logical operators yield
                // `bool`, so their operands are walked bare. Either
                // way the operands are walked so nested call-site and
                // binary anchoring applies
                // (`let y = select(x, 0, c) + 1;`).
                let operand_expected = if is_arithmetic(op) { expected } else { None };
                self.expect(lhs, operand_expected);
                self.expect(rhs, operand_expected);
                // Anchor: a bare literal adopts the provable scalar type
                // of the other operand.
                if is_arithmetic(op) || is_comparison(op) {
                    let lhs_ty = self.expr_ty(lhs);
                    let rhs_ty = self.expr_ty(rhs);
                    if let Some(ty) = lhs_ty {
                        self.expect(rhs, Some(&ty));
                    }
                    if let Some(ty) = rhs_ty {
                        self.expect(lhs, Some(&ty));
                    }
                }
            }
            Expr::FnCall {
                path,
                type_args,
                params,
            } => self.walk_fn_call(path, type_args, params, expected),
            Expr::Unary { op, expr } => {
                // `!` (logical not), bitwise complement, and negation
                // preserve the operand type, so the expectation flows
                // through (`fn f() -> u32 { !0 }` renders `~0u`). A
                // deref's operand is the pointer — its type is not the
                // deref result — so it is walked bare.
                match op {
                    UnOp::Not | UnOp::Complement | UnOp::Neg => self.expect(expr, expected),
                    UnOp::Deref => self.expect(expr, None),
                }
            }
            Expr::ArrayIndexing { lhs, index } => {
                // An outer scalar expectation pins the indexed base's
                // element type (`[select(0, 1, c)][0]` in a `u32`
                // function): carry it into the base as a sequence
                // expectation.
                let seq_expected = expected.map(|ty| Type::RuntimeArray {
                    elem: Box::new(ty.clone()),
                });
                self.expect(lhs, seq_expected.as_ref());
                // Index expressions stay bare (WGSL coerces the
                // abstract integer in `a[0]` to either index type).
                self.expect(index, None);
            }
            Expr::Swizzle { lhs, params, .. } => {
                self.expect(lhs, None);
                if let Some(params) = params {
                    for p in params {
                        self.expect(p, None);
                    }
                }
            }
            Expr::FieldAccess { base, .. } => self.expect(base, None),
            Expr::Reference(inner) => self.expect(inner, None),
            // Identifiers, type paths, and zero-value array lengths
            // (type positions) carry nothing to suffix.
            Expr::Ident(_) | Expr::TypePath { .. } | Expr::ZeroValueArray { .. } => {}
        }
    }
}

/// Suffix a single bare integer literal from `expected` — the leaf of
/// every expectation path. Literals that already carry a suffix are
/// left alone, as are non-integer expectations a literal cannot have in
/// valid Rust.
fn suffix_lit(lit: &mut Lit, expected: &Type) {
    let Lit::Int { suffix, .. } = lit else {
        return;
    };
    if !suffix.is_empty() {
        return;
    }
    match concrete_scalar(expected) {
        Some(ScalarType::U32) => *suffix = "u32".to_string(),
        Some(ScalarType::I32) => *suffix = "i32".to_string(),
        // A `bool` / `f32` expectation cannot apply to an integer
        // literal in valid Rust; leave it bare.
        Some(ScalarType::F32 | ScalarType::Bool) | None => {}
    }
}

/// The scalar element expectation carried by a vector constructor: the
/// WGSL predeclared `vecN<f|i|u>` shorthand encodes it in the name, and
/// an explicit `vecN<T>` type argument carries it in `type_args[0]`.
///
/// Abstract constructors (`vec3(...)`) return `None` — WGSL's
/// abstract-int defaults match Rust's `i32`, so there is nothing to
/// fix. Bool shorthands (`vec2b`) cannot host integer literals in
/// valid Rust.
fn vec_ctor_elem(name: &str, type_args: &[Type]) -> Option<Type> {
    if let [ty] = type_args {
        return Some(ty.clone());
    }
    let rest = name.strip_prefix("vec")?;
    let suffix = rest.strip_prefix(|c: char| matches!(c, '2' | '3' | '4'))?;
    Some(Type::Scalar(match suffix {
        "i" => ScalarType::I32,
        "u" => ScalarType::U32,
        // `vec3f` is float-typed: an integer literal cannot appear
        // there in valid Rust, and the pass leaves floats alone.
        "f" => ScalarType::F32,
        _ => return None,
    }))
}

/// Argument indices of the builtins whose value arguments share one
/// type `T` — `select(f, t, cond)`'s condition is not a value argument.
///
/// Returns `None` for every other name. There is no general signature
/// table for WGSL builtins (`builtin_lookup.rs` translates names only),
/// so this list stays deliberately small: only builtins that appear in
/// wgsl-rs with polymorphic value parameters where a bare literal is
/// otherwise ambiguous.
fn builtin_value_args(name: &str) -> Option<&'static [usize]> {
    Some(match name {
        "select" => &[0, 1],
        "min" | "max" => &[0, 1],
        "clamp" => &[0, 1, 2],
        _ => return None,
    })
}

/// Resolve `ty` to a concrete scalar type, unwrapping single-element
/// wrappers (`atomic<T>`, pointers). Returns `None` for anything the
/// pass cannot prove concrete — type parameters, abstract vectors,
/// structs, and arrays all need more context than a bare [`Type`]
/// carries.
fn concrete_scalar(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Scalar(scalar) => Some(*scalar),
        Type::Atomic { elem } | Type::Ptr { elem, .. } => concrete_scalar(elem),
        _ => None,
    }
}

/// Whether `op` produces a result of its operands' type (so an outer
/// expectation flows into the operands).
fn is_arithmetic(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Shl
            | BinOp::Shr
    )
}

/// Whether `op` compares two same-typed operands (so one operand's
/// type can anchor the other's literals).
fn is_comparison(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

/// Replace [`Type::TypeParam`] references in `ty` with the corresponding
/// entry of `args` (positional). Unmatched parameters are left in place;
/// the pass no-ops on them downstream.
fn substitute_params(ty: &mut Type, params: &[String], args: &[Type]) {
    match ty {
        Type::TypeParam { name } => {
            if let Some(pos) = params.iter().position(|p| p == name)
                && let Some(arg) = args.get(pos)
            {
                *ty = arg.clone();
            }
        }
        Type::Array { elem, .. }
        | Type::RuntimeArray { elem }
        | Type::Atomic { elem }
        | Type::Ptr { elem, .. }
        | Type::Phantom { elem }
        | Type::AssocType { ty: elem, .. } => substitute_params(elem, params, args),
        Type::Struct { type_args, .. } => {
            for arg in type_args {
                substitute_params(arg, params, args);
            }
        }
        _ => {}
    }
}
