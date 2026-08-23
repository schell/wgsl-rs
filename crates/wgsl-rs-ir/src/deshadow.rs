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
//!
//! # Rename propagation
//!
//! Renames are propagated lazily via [`DeshadowCtx::resolve_rename`], a
//! map from original name to mangled name. When resolving an expression,
//! `rename_expr` consults this map.

use crate::{
    Block, CaseSelector, ElseBranch, Expr, FnArg, ForLoop, ImplItem, Item, ItemFn, Module, Stmt,
    StmtIf, StmtSwitch, deshadow::ctx::DeshadowCtx,
};

mod ctx;

/// The deshadow pass. Owns the scope-tracking context and walks the IR
/// tree, renaming same-scope shadowed bindings.
#[derive(Default)]
struct DeshadowPass {
    ctx: DeshadowCtx,
}

/// Pub wrappers preserving the old free-function API.
pub fn deshadow_module(module: &mut Module) {
    DeshadowPass::default().deshadow_module(module);
}

pub fn deshadow_items(items: &mut [Item]) {
    DeshadowPass::default().deshadow_items(items);
}

impl DeshadowPass {
    /// Rename same-scope shadowed locals in every function in the module so
    /// the rendered WGSL is valid.
    fn deshadow_module(&mut self, module: &mut Module) {
        for item in &mut module.items {
            self.deshadow_item(item);
        }
    }

    /// Like [`deshadow_module`](Self::deshadow_module) but operates on a bare
    /// slice of items.
    fn deshadow_items(&mut self, items: &mut [Item]) {
        for item in items {
            self.deshadow_item(item);
        }
    }

    /// Deshadow a single top-level item. Only functions (free or inside
    /// `impl` blocks) contain bindings that can shadow; other items are
    /// no-ops.
    fn deshadow_item(&mut self, item: &mut Item) {
        match item {
            Item::Fn(item_fn) => {
                self.deshadow_item_fn(item_fn);
            }
            Item::Impl(item_impl) => {
                for impl_item in &mut item_impl.items {
                    if let ImplItem::Fn(f) = impl_item {
                        self.deshadow_item_fn(f);
                    }
                }
            }
            Item::Const(_)
            | Item::Uniform(_)
            | Item::Storage(_)
            | Item::Workgroup(_)
            | Item::Sampler(_)
            | Item::Texture(_)
            | Item::Struct(_)
            | Item::Enum(_) => {}
        }
    }

    /// Deshadow a single function. Function parameters and the function
    /// body share the same scope (same end-of-scope in WGSL), so locals
    /// in the top-level body that shadow a parameter must be renamed.
    fn deshadow_item_fn(&mut self, f: &mut ItemFn) {
        let ItemFn {
            type_params: _,
            const_params: _,
            fn_attrs: _,
            name: _,
            inputs,
            return_type: _,
            block,
            attrs: _,
        } = f;
        self.deshadow_block(block, Some(inputs));
    }

    /// Deshadow a block. Pushes a new scope, optionally seeds it with
    /// function parameter names (params and body share the same
    /// end-of-scope in WGSL), processes statements in order, then pops
    /// the scope. Renames are scoped automatically via the push/pop —
    /// no manual save/restore needed.
    fn deshadow_block(&mut self, b: &mut Block, fn_inputs: Option<&[FnArg]>) {
        self.ctx.push_scope();

        if let Some(fn_args) = fn_inputs {
            for arg in fn_args {
                // Params are the first declarations in this scope; they
                // can't shadow each other (WGSL forbids duplicate
                // params), but a later local can shadow them.
                self.ctx.declare_name(&arg.name);
            }
        }

        let Block { stmts } = b;
        for stmt in stmts.iter_mut() {
            self.deshadow_stmt(stmt);
        }

        self.ctx.pop_scope();
    }

    /// Deshadow a statement: resolve references via the scope stack,
    /// then handle any declarations (checking for same-scope shadowing)
    /// and recurse into nested scopes.
    fn deshadow_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Local(l) => {
                // Resolve the initializer in the *enclosing* scope (Rust
                // semantics: the init of a shadowing binding refers to
                // the outer variable).
                if let Some(init) = &mut l.init {
                    self.rename_expr(init);
                }
                if let Some(new_name) = self.ctx.declare_name(&l.name) {
                    l.name = new_name;
                }
            }
            Stmt::Const(c) => {
                self.rename_expr(&mut c.expr);
                if let Some(new_name) = self.ctx.declare_name(&c.name) {
                    c.name = new_name;
                }
            }
            Stmt::Assignment { lhs, rhs } => {
                self.rename_expr(lhs);
                self.rename_expr(rhs);
            }
            Stmt::CompoundAssignment { lhs, rhs, .. } => {
                self.rename_expr(lhs);
                self.rename_expr(rhs);
            }
            Stmt::While { condition, body } => {
                self.rename_expr(condition);
                self.deshadow_block(body, None);
            }
            Stmt::Loop { body } => {
                self.deshadow_block(body, None);
            }
            Stmt::Expr { expr, .. } => {
                self.rename_expr(expr);
            }
            Stmt::If(i) => {
                self.deshadow_if(i);
            }
            Stmt::Break | Stmt::Continue | Stmt::Discard => {}
            Stmt::Return(e) => {
                if let Some(e) = e {
                    self.rename_expr(e);
                }
            }
            Stmt::For(f) => {
                self.deshadow_for(f);
            }
            Stmt::Switch(s) => {
                self.deshadow_switch(s);
            }
            Stmt::Block(b) => {
                self.deshadow_block(b, None);
            }
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
            } => {
                self.rename_expr(src);
                self.rename_expr(src_offset);
                self.rename_expr(dest);
                self.rename_expr(dest_offset);
                self.rename_expr(size);
            }
            Stmt::Macro { .. } => {}
        }
    }

    /// Deshadow an if statement. The condition is resolved in the
    /// enclosing scope; each branch (then, else block, else-if) gets its
    /// own nested scope.
    fn deshadow_if(&mut self, i: &mut StmtIf) {
        self.rename_expr(&mut i.condition);
        self.deshadow_block(&mut i.then_block, None);
        if let Some(else_branch) = &mut i.else_branch {
            match else_branch {
                ElseBranch::Block(b) => self.deshadow_block(b, None),
                ElseBranch::If(nested) => self.deshadow_if(nested),
            }
        }
    }

    /// Deshadow a for loop. The `from`/`to` range expressions are
    /// resolved in the enclosing scope (Rust evaluates the range before
    /// entering the loop). The loop variable is declared in a scope
    /// wrapping the body; the body gets its own nested scope. Renames
    /// are scoped automatically via the push/pop.
    fn deshadow_for(&mut self, f: &mut ForLoop) {
        // Resolve range in the enclosing scope.
        self.rename_expr(&mut f.from);
        self.rename_expr(&mut f.to);

        // The loop variable lives in a scope wrapping the body. Per the
        // WGSL spec, `for` desugars to `{ initializer; loop { body } }`,
        // so the loop var is always in its own scope and can never
        // same-scope-collide with an enclosing binding.
        self.ctx.push_scope();
        if let Some(new_name) = self.ctx.declare_name(&f.var) {
            f.var = new_name;
        }

        self.deshadow_block(&mut f.body, None);

        self.ctx.pop_scope();
    }

    /// Deshadow a switch statement. The selector and any expression
    /// case selectors are resolved in the enclosing scope; each arm
    /// body gets its own nested scope.
    fn deshadow_switch(&mut self, s: &mut StmtSwitch) {
        self.rename_expr(&mut s.selector);
        for arm in &mut s.arms {
            for sel in &mut arm.selectors {
                if let CaseSelector::Expr(e) = sel {
                    self.rename_expr(e);
                }
            }
            self.deshadow_block(&mut arm.body, None);
        }
    }

    /// Rename all `Expr::Ident` nodes within an expression tree by
    /// consulting the scope stack ([`DeshadowCtx::resolve_rename`]).
    /// Recurses into all sub-expressions.
    fn rename_expr(&mut self, e: &mut Expr) {
        match e {
            Expr::Ident(name) => {
                if let Some(to) = self.ctx.resolve_rename(name) {
                    *name = to.to_string();
                }
            }
            Expr::Lit(_) | Expr::TypePath { .. } => {}
            Expr::Array { elems } => {
                for x in elems {
                    self.rename_expr(x);
                }
            }
            Expr::Paren(inner) | Expr::Reference(inner) => {
                self.rename_expr(inner);
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.rename_expr(lhs);
                self.rename_expr(rhs);
            }
            Expr::Unary { expr, .. } => {
                self.rename_expr(expr);
            }
            Expr::ArrayIndexing { lhs, index } => {
                self.rename_expr(lhs);
                self.rename_expr(index);
            }
            Expr::Swizzle { lhs, params, .. } => {
                self.rename_expr(lhs);
                if let Some(args) = params {
                    for a in args {
                        self.rename_expr(a);
                    }
                }
            }
            Expr::Cast { lhs, .. } => {
                self.rename_expr(lhs);
            }
            Expr::FnCall { params, .. } => {
                for p in params {
                    self.rename_expr(p);
                }
            }
            Expr::Struct { fields, .. } => {
                for f in fields {
                    self.rename_expr(&mut f.expr);
                }
            }
            Expr::FieldAccess { base, .. } => {
                self.rename_expr(base);
            }
            Expr::ZeroValueArray { len, .. } => {
                self.rename_expr(len);
            }
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
