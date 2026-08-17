//! Deshadow pass: rename same-scope variable redeclarations so the
//! rendered WGSL is valid.
//!
//! Rust allows shadowing within the same scope (`let x = 1.0; let x =
//! 2.0;`), but WGSL rejects two declarations with the same name and the
//! same end-of-scope. This pass walks each function body with a scope
//! stack and renames any shadowed binding (and all subsequent references
//! to it) to a unique mangled name like `x_1`.
//!
//! WGSL does allow shadowing in nested blocks (different end-of-scope),
//! so this pass only renames when a name is redeclared within the same
//! block scope.

use std::collections::HashSet;

use crate::{Block, ElseBranch, Expr, ForLoop, Item, ItemFn, Module, Stmt, StmtIf, StmtSwitch};

/// Rename same-scope shadowed locals in every function in the module so
/// the rendered WGSL is valid.
pub fn deshadow_module(module: &mut Module) {
    for item in &mut module.items {
        if let Item::Fn(f) = item {
            deshadow_fn(f);
        }
    }
}

/// Like [`deshadow_module`] but operates on a bare slice of items.
pub fn deshadow_items(items: &mut [Item]) {
    for item in items {
        if let Item::Fn(f) = item {
            deshadow_fn(f);
        }
    }
}

/// Deshadow a single function. Function parameters and the function
/// body share the same scope (same end-of-scope in WGSL), so locals
/// in the top-level body that shadow a parameter must be renamed.
fn deshadow_fn(f: &mut ItemFn) {
    let mut ctx = DeshadowCtx::new();
    let mut base = HashSet::new();
    for arg in &f.inputs {
        base.insert(arg.name.clone());
    }
    ctx.scopes.push(base);

    // Process the function body statements directly in the param scope
    // (don't push a new scope — params and body share the same
    // end-of-scope in WGSL).
    let mut renames: Vec<(String, String)> = Vec::new();
    for stmt in &mut f.block.stmts {
        for (from, to) in &renames {
            rename_in_expr_stmt(stmt, from, to);
        }
        collect_decls_and_recurse(stmt, &mut ctx, &mut renames);
    }

    ctx.scopes.pop();
}

/// Scope-tracking context for the deshadow pass.
struct DeshadowCtx {
    /// Stack of scopes, each containing the set of names declared in
    /// that scope.
    scopes: Vec<HashSet<String>>,
    /// Monotonic counter for generating unique names.
    counter: usize,
}

impl DeshadowCtx {
    fn new() -> Self {
        DeshadowCtx {
            scopes: Vec::new(),
            counter: 0,
        }
    }

    /// Generate a unique name for a shadowed variable. Checks all scopes
    /// on the stack to avoid collisions with any in-scope name.
    fn unique_name(&mut self, original: &str) -> String {
        loop {
            self.counter += 1;
            let candidate = format!("{original}_{}", self.counter);
            if !self.scopes.iter().any(|s| s.contains(&candidate)) {
                return candidate;
            }
        }
    }

    /// Check if a name is already declared in the current (innermost)
    /// scope.
    fn current_scope_contains(&self, name: &str) -> bool {
        self.scopes.last().is_some_and(|s| s.contains(name))
    }

    /// Insert a name into the current (innermost) scope.
    fn insert_into_current_scope(&mut self, name: String) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name);
        }
    }
}

/// Deshadow a block. Pushes a new scope, processes statements in order,
/// and applies any pending renames to subsequent statements.
fn deshadow_block(block: &mut Block, ctx: &mut DeshadowCtx) {
    ctx.scopes.push(HashSet::new());

    // Renames accumulated as we walk: (original_name, new_name).
    // Each subsequent statement (and its sub-expressions) must have
    // these renames applied so references resolve to the new binding.
    let mut renames: Vec<(String, String)> = Vec::new();

    for stmt in &mut block.stmts {
        // Apply all pending renames to this statement first, so that
        // references to previously-shadowed names point to the new
        // binding.
        for (from, to) in &renames {
            rename_in_expr_stmt(stmt, from, to);
        }

        // Now process the statement for new shadowing declarations.
        collect_decls_and_recurse(stmt, ctx, &mut renames);
    }

    ctx.scopes.pop();
}

/// Process a statement: collect any declarations (checking for
/// shadowing), and recurse into nested scopes.
fn collect_decls_and_recurse(
    stmt: &mut Stmt,
    ctx: &mut DeshadowCtx,
    renames: &mut Vec<(String, String)>,
) {
    match stmt {
        Stmt::Local(l) => {
            if ctx.current_scope_contains(&l.name) {
                // Shadowed! Generate a unique name.
                let new_name = ctx.unique_name(&l.name);
                let old_name = std::mem::replace(&mut l.name, new_name.clone());
                // Do NOT rename l.init — in Rust, the initializer of a
                // shadowing binding refers to the *outer* variable (which
                // keeps its original name).
                ctx.insert_into_current_scope(new_name.clone());
                renames.push((old_name, new_name));
            } else {
                ctx.insert_into_current_scope(l.name.clone());
            }
        }
        Stmt::Const(c) => {
            if ctx.current_scope_contains(&c.name) {
                let new_name = ctx.unique_name(&c.name);
                let old_name = std::mem::replace(&mut c.name, new_name.clone());
                // const init refers to outer scope, don't rename it.
                ctx.insert_into_current_scope(new_name.clone());
                renames.push((old_name, new_name));
            } else {
                ctx.insert_into_current_scope(c.name.clone());
            }
        }
        Stmt::Block(b) => {
            deshadow_block(b, ctx);
        }
        Stmt::While { body, .. } => {
            deshadow_block(body, ctx);
        }
        Stmt::Loop { body } => {
            deshadow_block(body, ctx);
        }
        Stmt::For(f) => {
            deshadow_for(f, ctx);
        }
        Stmt::If(i) => {
            deshadow_if(i, ctx);
        }
        Stmt::Switch(s) => {
            deshadow_switch(s, ctx);
        }
        // Other statements don't introduce bindings or nested scopes.
        _ => {}
    }
}

/// Deshadow a for loop. The loop variable is scoped to the loop body.
fn deshadow_for(f: &mut ForLoop, ctx: &mut DeshadowCtx) {
    // The for loop introduces its loop variable in a scope that wraps
    // the body. We push a scope, register the loop var, then deshadow
    // the body within it.
    ctx.scopes.push(HashSet::new());

    if ctx.current_scope_contains(&f.var) {
        let new_name = ctx.unique_name(&f.var);
        let old_name = std::mem::replace(&mut f.var, new_name.clone());
        ctx.insert_into_current_scope(new_name.clone());
        // Rename references in the body (and from/to expressions).
        rename_in_expr(&mut f.from, &old_name, &new_name);
        rename_in_expr(&mut f.to, &old_name, &new_name);
        rename_in_block(&mut f.body, &old_name, &new_name);
    } else {
        ctx.insert_into_current_scope(f.var.clone());
    }

    deshadow_block(&mut f.body, ctx);
    ctx.scopes.pop();
}

/// Deshadow an if statement. Each branch (then, else block, else-if)
/// gets its own nested scope.
fn deshadow_if(i: &mut StmtIf, ctx: &mut DeshadowCtx) {
    deshadow_block(&mut i.then_block, ctx);
    if let Some(else_branch) = &mut i.else_branch {
        match else_branch {
            ElseBranch::Block(b) => deshadow_block(b, ctx),
            ElseBranch::If(nested) => deshadow_if(nested, ctx),
        }
    }
}

/// Deshadow a switch statement. Each arm body gets its own nested scope.
fn deshadow_switch(s: &mut StmtSwitch, ctx: &mut DeshadowCtx) {
    for arm in &mut s.arms {
        deshadow_block(&mut arm.body, ctx);
    }
}

// ===== Rename helpers =====

/// Rename all `Expr::Ident(from)` to `to` within a statement, recursing
/// into all sub-expressions and nested blocks. This is a flat rename —
/// it does not track scopes, so it should only be used when the caller
/// has already established that `from` is the shadowed name visible in
/// all positions being renamed.
fn rename_in_expr_stmt(stmt: &mut Stmt, from: &str, to: &str) {
    match stmt {
        Stmt::Local(l) => {
            // Rename init (references to the shadowed binding), but NOT
            // l.name itself (that may have already been renamed).
            if let Some(init) = &mut l.init {
                rename_in_expr(init, from, to);
            }
        }
        Stmt::Const(c) => {
            rename_in_expr(&mut c.expr, from, to);
        }
        Stmt::Assignment { lhs, rhs } => {
            rename_in_expr(lhs, from, to);
            rename_in_expr(rhs, from, to);
        }
        Stmt::CompoundAssignment { lhs, rhs, .. } => {
            rename_in_expr(lhs, from, to);
            rename_in_expr(rhs, from, to);
        }
        Stmt::While { condition, body } => {
            rename_in_expr(condition, from, to);
            rename_in_block(body, from, to);
        }
        Stmt::Loop { body } => {
            rename_in_block(body, from, to);
        }
        Stmt::Expr { expr, .. } => {
            rename_in_expr(expr, from, to);
        }
        Stmt::If(i) => {
            rename_in_if(i, from, to);
        }
        Stmt::Return(Some(e)) => {
            rename_in_expr(e, from, to);
        }
        Stmt::For(f) => {
            rename_in_expr(&mut f.from, from, to);
            rename_in_expr(&mut f.to, from, to);
            // Don't rename f.var (it's a declaration name, not a ref)
            rename_in_block(&mut f.body, from, to);
        }
        Stmt::Switch(s) => {
            rename_in_expr(&mut s.selector, from, to);
            for arm in &mut s.arms {
                for sel in &mut arm.selectors {
                    if let crate::CaseSelector::Expr(e) = sel {
                        rename_in_expr(e, from, to);
                    }
                }
                rename_in_block(&mut arm.body, from, to);
            }
        }
        Stmt::Block(b) => {
            rename_in_block(b, from, to);
        }
        Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            rename_in_expr(src, from, to);
            rename_in_expr(src_offset, from, to);
            rename_in_expr(dest, from, to);
            rename_in_expr(dest_offset, from, to);
            rename_in_expr(size, from, to);
        }
        // No expressions to rename.
        Stmt::Return(None) | Stmt::Break | Stmt::Continue | Stmt::Discard | Stmt::Macro { .. } => {}
    }
}

/// Rename all `Expr::Ident(from)` to `to` within a block.
fn rename_in_block(block: &mut Block, from: &str, to: &str) {
    for stmt in &mut block.stmts {
        rename_in_expr_stmt(stmt, from, to);
    }
}

/// Rename all `Expr::Ident(from)` to `to` within an if statement.
fn rename_in_if(i: &mut StmtIf, from: &str, to: &str) {
    rename_in_expr(&mut i.condition, from, to);
    rename_in_block(&mut i.then_block, from, to);
    if let Some(else_branch) = &mut i.else_branch {
        match else_branch {
            ElseBranch::Block(b) => rename_in_block(b, from, to),
            ElseBranch::If(nested) => rename_in_if(nested, from, to),
        }
    }
}

/// Rename all `Expr::Ident(from)` to `to` within an expression tree.
fn rename_in_expr(e: &mut Expr, from: &str, to: &str) {
    match e {
        Expr::Ident(name) => {
            if name == from {
                *name = to.to_string();
            }
        }
        Expr::Lit(_) | Expr::TypePath { .. } => {}
        Expr::Array { elems } => {
            for x in elems {
                rename_in_expr(x, from, to);
            }
        }
        Expr::Paren(inner) | Expr::Reference(inner) => {
            rename_in_expr(inner, from, to);
        }
        Expr::Binary { lhs, rhs, .. } => {
            rename_in_expr(lhs, from, to);
            rename_in_expr(rhs, from, to);
        }
        Expr::Unary { expr, .. } => {
            rename_in_expr(expr, from, to);
        }
        Expr::ArrayIndexing { lhs, index } => {
            rename_in_expr(lhs, from, to);
            rename_in_expr(index, from, to);
        }
        Expr::Swizzle { lhs, params, .. } => {
            rename_in_expr(lhs, from, to);
            if let Some(args) = params {
                for a in args {
                    rename_in_expr(a, from, to);
                }
            }
        }
        Expr::Cast { lhs, .. } => {
            rename_in_expr(lhs, from, to);
        }
        Expr::FnCall { params, .. } => {
            for p in params {
                rename_in_expr(p, from, to);
            }
        }
        Expr::Struct { fields, .. } => {
            for f in fields {
                rename_in_expr(&mut f.expr, from, to);
            }
        }
        Expr::FieldAccess { base, .. } => {
            rename_in_expr(base, from, to);
        }
        Expr::ZeroValueArray { len, .. } => {
            rename_in_expr(len, from, to);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{FnArg, FnAttrs, ItemFn, Local, ReturnType, render_module};
    use std::borrow::Cow;

    /// Build a minimal `ItemFn` with the given body statements.
    fn fn_with_stmts(stmts: Vec<Stmt>) -> ItemFn {
        ItemFn {
            type_params: Vec::new(),
            const_params: Vec::new(),
            fn_attrs: FnAttrs::None,
            name: Cow::Borrowed("test_fn"),
            inputs: Vec::new(),
            return_type: ReturnType::Default,
            block: Block { stmts },
            attrs: Vec::new(),
        }
    }

    /// Build a `Module` containing a single function and render it to
    /// WGSL, after running the deshadow pass.
    fn render_after_deshadow(f: ItemFn) -> String {
        let mut module = Module {
            name: "test",
            items: vec![Item::Fn(f)],
            attrs: Vec::new(),
        };
        deshadow_module(&mut module);
        render_module(&module)
    }

    fn let_stmt(name: &str, init: Expr) -> Stmt {
        Stmt::Local(Local {
            mutable: false,
            name: name.to_string(),
            ty: None,
            init: Some(init),
        })
    }

    fn ident(name: &str) -> Expr {
        Expr::Ident(name.to_string())
    }

    fn lit_f32(val: &str) -> Expr {
        Expr::Lit(crate::Lit::Float {
            text: val.to_string(),
        })
    }

    #[test]
    fn basic_shadowing() {
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("x", lit_f32("2.0")),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(
            wgsl.contains("let x = 1.0;"),
            "first x unchanged, got: {wgsl}"
        );
        assert!(
            wgsl.contains("let x_1 = 2.0;"),
            "second x renamed to x_1, got: {wgsl}"
        );
    }

    #[test]
    fn shadow_with_reference_in_init() {
        // let x = 1.0; let x = x + 1.0;
        // The init of the second x refers to the FIRST x (outer binding).
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt(
                "x",
                Expr::Binary {
                    lhs: Box::new(ident("x")),
                    op: crate::BinOp::Add,
                    rhs: Box::new(lit_f32("1.0")),
                },
            ),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(
            wgsl.contains("let x = 1.0;"),
            "first x unchanged, got: {wgsl}"
        );
        // Second binding renamed, but its init still references the old x.
        assert!(
            wgsl.contains("let x_1 = x + 1.0;"),
            "second x renamed to x_1 with init referencing old x, got: {wgsl}"
        );
    }

    #[test]
    fn shadow_then_use() {
        // let x = 1.0; let x = 2.0; let y = x;
        // After deshadow: let x = 1.0; let x_1 = 2.0; let y = x_1;
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("x", lit_f32("2.0")),
            let_stmt("y", ident("x")),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(
            wgsl.contains("let y = x_1;"),
            "y should reference x_1 (the shadowed binding), got: {wgsl}"
        );
    }

    #[test]
    fn triple_shadow() {
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("x", lit_f32("2.0")),
            let_stmt("x", lit_f32("3.0")),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(wgsl.contains("let x = 1.0;"), "got: {wgsl}");
        assert!(wgsl.contains("let x_1 = 2.0;"), "got: {wgsl}");
        assert!(wgsl.contains("let x_2 = 3.0;"), "got: {wgsl}");
    }

    #[test]
    fn nested_block_shadow_is_allowed() {
        // let x = 1.0; { let x = 2.0; } — valid in WGSL (different
        // end-of-scope), should NOT be renamed.
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            Stmt::Block(Block {
                stmts: vec![let_stmt("x", lit_f32("2.0"))],
            }),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(
            wgsl.contains("let x = 1.0;"),
            "outer x unchanged, got: {wgsl}"
        );
        assert!(
            wgsl.contains("let x = 2.0;"),
            "inner block x should NOT be renamed (different end-of-scope), got: {wgsl}"
        );
    }

    #[test]
    fn shadow_in_nested_block_same_scope() {
        // { let x = 1.0; let x = 2.0; } — same scope within the block,
        // should be renamed.
        let f = fn_with_stmts(vec![Stmt::Block(Block {
            stmts: vec![let_stmt("x", lit_f32("1.0")), let_stmt("x", lit_f32("2.0"))],
        })]);
        let wgsl = render_after_deshadow(f);
        assert!(wgsl.contains("let x = 1.0;"), "got: {wgsl}");
        assert!(wgsl.contains("let x_1 = 2.0;"), "got: {wgsl}");
    }

    #[test]
    fn fn_param_shadowed_by_local() {
        let mut f = fn_with_stmts(vec![let_stmt("x", lit_f32("1.0"))]);
        f.inputs = vec![FnArg {
            inter_stage_io: Vec::new(),
            name: "x".to_string(),
            ty: crate::Type::Scalar(crate::ScalarType::F32),
            attrs: Vec::new(),
        }];
        let wgsl = render_after_deshadow(f);
        // The local `x` shadows the param `x` in the same scope, so it
        // should be renamed.
        assert!(
            wgsl.contains("let x_1 = 1.0;"),
            "local x should be renamed to avoid shadowing param, got: {wgsl}"
        );
    }

    #[test]
    fn no_shadowing_no_rename() {
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("y", lit_f32("2.0")),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(wgsl.contains("let x = 1.0;"), "got: {wgsl}");
        assert!(wgsl.contains("let y = 2.0;"), "got: {wgsl}");
    }

    #[test]
    fn shadow_inside_if_body() {
        // let x = 1.0; if x > 0.0 { let x = 2.0; let x = 3.0; }
        // The two inner x's are in the same if-body scope, so the second
        // should be renamed. The outer x and the first inner x have
        // different end-of-scope, so they're fine.
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            Stmt::If(StmtIf {
                condition: Expr::Binary {
                    lhs: Box::new(ident("x")),
                    op: crate::BinOp::Gt,
                    rhs: Box::new(lit_f32("0.0")),
                },
                then_block: Block {
                    stmts: vec![let_stmt("x", lit_f32("2.0")), let_stmt("x", lit_f32("3.0"))],
                },
                else_branch: None,
            }),
        ]);
        let wgsl = render_after_deshadow(f);
        // Outer x unchanged.
        assert!(wgsl.contains("let x = 1.0;"), "got: {wgsl}");
        // First inner x is in a nested scope — fine, no rename.
        assert!(
            wgsl.contains("let x = 2.0;"),
            "first inner x unchanged, got: {wgsl}"
        );
        // Second inner x shadows the first inner x — rename.
        assert!(
            wgsl.contains("let x_1 = 3.0;"),
            "second inner x renamed, got: {wgsl}"
        );
    }

    #[test]
    fn shadow_then_assign() {
        // let x = 1.0; let x = 2.0; x = 3.0;
        // After deshadow: let x = 1.0; let x_1 = 2.0; x_1 = 3.0;
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("x", lit_f32("2.0")),
            Stmt::Assignment {
                lhs: ident("x"),
                rhs: lit_f32("3.0"),
            },
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(
            wgsl.contains("x_1 = 3.0;"),
            "assignment should target x_1, got: {wgsl}"
        );
    }

    #[test]
    fn shadow_with_existing_suffixed_name() {
        // If x_1 already exists, the generated name should skip to x_2.
        let f = fn_with_stmts(vec![
            let_stmt("x", lit_f32("1.0")),
            let_stmt("x_1", lit_f32("0.5")),
            let_stmt("x", lit_f32("2.0")),
        ]);
        let wgsl = render_after_deshadow(f);
        assert!(wgsl.contains("let x = 1.0;"), "got: {wgsl}");
        assert!(
            wgsl.contains("let x_1 = 0.5;"),
            "existing x_1 unchanged, got: {wgsl}"
        );
        assert!(
            wgsl.contains("let x_2 = 2.0;"),
            "shadowed x should skip to x_2 (x_1 already taken), got: {wgsl}"
        );
    }
}
