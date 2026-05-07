//! Mutable visitor over the parse-side AST.
//!
//! This module defines a single trait, [`ParseVisitorMut`], and a family
//! of `walk_*` free functions that perform the standard recursive
//! descent. Each walker in `monomorphize.rs` is a struct that carries
//! its own context and overrides only the `visit_*` methods relevant to
//! its job.
//!
//! Why use `&mut` for a trait that some walkers don't need? Several
//! walkers do mutate (substitution, name mangling, cross-module
//! collection); the rest are read-only. Rather than maintaining two
//! parallel trait hierarchies (the `syn` crate's approach), we use a
//! single `&mut` trait — read-only walkers simply don't write through
//! the references they receive. This halves the boilerplate at the
//! cost of a small loss of compile-time guarantees.
//!
//! All visitors return `Result<(), parse::Error>`. Infallible visitors
//! return `Ok(())` from every method.
//!
//! # Default behavior
//!
//! The default `visit_*` implementations call the matching `walk_*`
//! function, which means a visitor that overrides nothing performs a
//! full recursive walk that does nothing. To inject custom behavior
//! at a node, override the relevant `visit_*` method and call
//! `walk_*` from inside it to continue descent (or skip the call to
//! prune the traversal at that point).

use crate::parse::{
    Block, CaseSelector, ElseBody, Error, Expr, Field, Item, ItemConst, ItemFn, ItemImpl,
    ItemStruct, Stmt, StmtIf, Type,
};

/// A mutable visitor over the parse AST.
pub(crate) trait ParseVisitorMut {
    /// Visit a top-level [`Item`]. The default implementation dispatches
    /// to the variant-specific entry points (`visit_fn`, `visit_impl`,
    /// `visit_struct`, `visit_const`) and falls back to a no-op for
    /// items that don't carry traversable structure (use, mod, etc.).
    fn visit_item(&mut self, item: &mut Item) -> Result<(), Error> {
        walk_item(self, item)
    }

    /// Visit an [`ItemFn`]. The default implementation walks the
    /// argument types, return type, and body block.
    fn visit_fn(&mut self, f: &mut ItemFn) -> Result<(), Error> {
        walk_fn(self, f)
    }

    /// Visit an [`ItemImpl`] block. The default implementation
    /// dispatches each impl item to the appropriate `visit_*` method.
    fn visit_impl(&mut self, i: &mut ItemImpl) -> Result<(), Error> {
        walk_impl(self, i)
    }

    /// Visit an [`ItemStruct`]. The default implementation walks each
    /// field's type.
    fn visit_struct(&mut self, s: &mut ItemStruct) -> Result<(), Error> {
        walk_struct(self, s)
    }

    /// Visit an [`ItemConst`]. The default implementation walks the
    /// type and the initializer expression.
    fn visit_const(&mut self, c: &mut ItemConst) -> Result<(), Error> {
        walk_const(self, c)
    }

    /// Visit a single struct [`Field`]. The default implementation
    /// walks the field's type.
    fn visit_field(&mut self, f: &mut Field) -> Result<(), Error> {
        walk_field(self, f)
    }

    /// Visit a [`Block`]. The default implementation walks each
    /// statement.
    fn visit_block(&mut self, b: &mut Block) -> Result<(), Error> {
        walk_block(self, b)
    }

    /// Visit a [`Stmt`]. The default implementation dispatches into
    /// the appropriate sub-walker based on the variant.
    fn visit_stmt(&mut self, s: &mut Stmt) -> Result<(), Error> {
        walk_stmt(self, s)
    }

    /// Visit an [`StmtIf`] (handles the full `if`/`else if`/`else`
    /// chain by recursing through nested else-if branches).
    fn visit_if(&mut self, i: &mut StmtIf) -> Result<(), Error> {
        walk_if(self, i)
    }

    /// Visit an [`Expr`]. The default implementation recurses into
    /// sub-expressions and embedded types.
    fn visit_expr(&mut self, e: &mut Expr) -> Result<(), Error> {
        walk_expr(self, e)
    }

    /// Visit a [`Type`]. The default implementation recurses into
    /// composite types (arrays, atomics, pointers, generic struct
    /// type args).
    fn visit_type(&mut self, t: &mut Type) -> Result<(), Error> {
        walk_type(self, t)
    }
}

// ===== walk_* functions =====

/// Recursive-descent walker for [`Item`]. Dispatches to the
/// variant-specific `visit_*` entry points on the visitor; falls back
/// to a no-op for items without traversable structure.
pub(crate) fn walk_item<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    item: &mut Item,
) -> Result<(), Error> {
    match item {
        Item::Fn(f) => v.visit_fn(f),
        Item::Impl(i) => v.visit_impl(i),
        Item::Struct(s) => v.visit_struct(s),
        Item::Const(c) => v.visit_const(c),
        Item::Uniform(u) => v.visit_type(&mut u.ty),
        Item::Storage(s) => v.visit_type(&mut s.ty),
        Item::Workgroup(w) => v.visit_type(&mut w.ty),
        Item::Sampler(s) => v.visit_type(&mut s.ty),
        Item::Texture(t) => v.visit_type(&mut t.ty),
        // Items without traversable bodies.
        Item::Mod(_)
        | Item::Use(_)
        | Item::Enum(_)
        | Item::MacroRules
        | Item::Trait
        | Item::Ignored => Ok(()),
    }
}

/// Walk a function's argument types, return type, and body.
pub(crate) fn walk_fn<V: ParseVisitorMut + ?Sized>(v: &mut V, f: &mut ItemFn) -> Result<(), Error> {
    for input in f.inputs.iter_mut() {
        v.visit_type(&mut input.ty)?;
    }
    if let crate::parse::ReturnType::Type { ty, .. } = &mut f.return_type {
        v.visit_type(ty)?;
    }
    v.visit_block(&mut f.block)?;
    Ok(())
}

/// Walk every item inside an impl block.
pub(crate) fn walk_impl<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    i: &mut ItemImpl,
) -> Result<(), Error> {
    for ii in &mut i.items {
        match ii {
            crate::parse::ImplItem::Fn(f) => v.visit_fn(f)?,
            crate::parse::ImplItem::Const(c) => v.visit_const(c)?,
        }
    }
    Ok(())
}

/// Walk the field types of a struct definition.
pub(crate) fn walk_struct<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    s: &mut ItemStruct,
) -> Result<(), Error> {
    for pair in s.fields.named.iter_mut() {
        v.visit_field(pair)?;
    }
    Ok(())
}

/// Walk a single struct field's type.
pub(crate) fn walk_field<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    f: &mut Field,
) -> Result<(), Error> {
    v.visit_type(&mut f.ty)
}

/// Walk a const item's type and initializer expression.
pub(crate) fn walk_const<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    c: &mut ItemConst,
) -> Result<(), Error> {
    v.visit_type(&mut c.ty)?;
    v.visit_expr(&mut c.expr)?;
    Ok(())
}

/// Walk all statements in a block.
pub(crate) fn walk_block<V: ParseVisitorMut + ?Sized>(
    v: &mut V,
    b: &mut Block,
) -> Result<(), Error> {
    for stmt in &mut b.stmt {
        v.visit_stmt(stmt)?;
    }
    Ok(())
}

/// Walk a single statement, recursing into its sub-expressions, types,
/// and nested blocks.
pub(crate) fn walk_stmt<V: ParseVisitorMut + ?Sized>(v: &mut V, s: &mut Stmt) -> Result<(), Error> {
    match s {
        Stmt::Local(local) => {
            if let Some((_, ty)) = &mut local.ty {
                v.visit_type(ty)?;
            }
            if let Some(init) = &mut local.init {
                v.visit_expr(&mut init.expr)?;
            }
        }
        Stmt::Const(c) => {
            v.visit_const(c)?;
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            v.visit_expr(lhs)?;
            v.visit_expr(rhs)?;
        }
        Stmt::While {
            condition, body, ..
        } => {
            v.visit_expr(condition)?;
            v.visit_block(body)?;
        }
        Stmt::Loop { body, .. } => {
            v.visit_block(body)?;
        }
        Stmt::Expr { expr, .. } => {
            v.visit_expr(expr)?;
        }
        Stmt::If(if_stmt) => {
            v.visit_if(if_stmt)?;
        }
        Stmt::For(for_loop) => {
            if let Some((_, ty)) = &mut for_loop.ty {
                v.visit_type(ty)?;
            }
            v.visit_expr(&mut for_loop.from)?;
            v.visit_expr(&mut for_loop.to)?;
            v.visit_block(&mut for_loop.body)?;
        }
        Stmt::Switch(switch) => {
            v.visit_expr(&mut switch.selector)?;
            for arm in &mut switch.arms {
                for sel in &mut arm.selectors {
                    if let CaseSelector::Expr(e) = sel {
                        v.visit_expr(e)?;
                    }
                }
                v.visit_block(&mut arm.body)?;
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                v.visit_expr(e)?;
            }
        }
        Stmt::Block(block) => {
            v.visit_block(block)?;
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            v.visit_expr(slab)?;
            v.visit_expr(offset)?;
            v.visit_expr(dest)?;
            v.visit_expr(size)?;
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            v.visit_expr(slab)?;
            v.visit_expr(offset)?;
            v.visit_expr(src)?;
            if let Some(s) = size {
                v.visit_expr(s)?;
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
    Ok(())
}

/// Walk an `if`/`else if`/`else` chain in full. Recurses through the
/// nested `else if` chain so every branch is visited.
pub(crate) fn walk_if<V: ParseVisitorMut + ?Sized>(v: &mut V, i: &mut StmtIf) -> Result<(), Error> {
    v.visit_expr(&mut i.condition)?;
    v.visit_block(&mut i.then_block)?;
    if let Some(else_branch) = &mut i.else_branch {
        match &mut else_branch.body {
            ElseBody::Block(block) => v.visit_block(block)?,
            ElseBody::If(nested) => v.visit_if(nested)?,
        }
    }
    Ok(())
}

/// Walk an expression's sub-expressions and any embedded types.
pub(crate) fn walk_expr<V: ParseVisitorMut + ?Sized>(v: &mut V, e: &mut Expr) -> Result<(), Error> {
    match e {
        Expr::FnCall {
            type_args, params, ..
        } => {
            for ta in type_args.iter_mut() {
                v.visit_type(ta)?;
            }
            for p in params.iter_mut() {
                v.visit_expr(p)?;
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            v.visit_expr(lhs)?;
            v.visit_expr(rhs)?;
        }
        Expr::Unary { expr, .. } => {
            v.visit_expr(expr)?;
        }
        Expr::Paren { inner, .. } => {
            v.visit_expr(inner)?;
        }
        Expr::Array { elems, .. } => {
            for elem in elems.iter_mut() {
                v.visit_expr(elem)?;
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            v.visit_expr(lhs)?;
            v.visit_expr(index)?;
        }
        Expr::Swizzle { lhs, params, .. } => {
            v.visit_expr(lhs)?;
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    v.visit_expr(p)?;
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            v.visit_expr(lhs)?;
            v.visit_type(ty)?;
        }
        Expr::Struct {
            type_args, fields, ..
        } => {
            for ta in type_args.iter_mut() {
                v.visit_type(ta)?;
            }
            for field in fields.iter_mut() {
                v.visit_expr(&mut field.expr)?;
            }
        }
        Expr::FieldAccess { base, .. } => {
            v.visit_expr(base)?;
        }
        Expr::Reference { expr, .. } => {
            v.visit_expr(expr)?;
        }
        Expr::ZeroValueArray { elem_type, len, .. } => {
            v.visit_type(elem_type)?;
            v.visit_expr(len)?;
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
    Ok(())
}

/// Walk a type's sub-types and any embedded expressions (e.g. array
/// length expressions).
pub(crate) fn walk_type<V: ParseVisitorMut + ?Sized>(v: &mut V, t: &mut Type) -> Result<(), Error> {
    match t {
        Type::Array { elem, len, .. } => {
            v.visit_type(elem)?;
            v.visit_expr(len)?;
        }
        Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } | Type::Ptr { elem, .. } => {
            v.visit_type(elem)?;
        }
        Type::Struct { type_args, .. } => {
            for ta in type_args.iter_mut() {
                v.visit_type(ta)?;
            }
        }
        Type::Scalar { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::Sampler { .. }
        | Type::SamplerComparison { .. }
        | Type::Texture { .. }
        | Type::TextureDepth { .. }
        | Type::TypeParam { .. } => {}
    }
    Ok(())
}
