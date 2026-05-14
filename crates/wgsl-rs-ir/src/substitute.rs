//! Generic type-parameter substitution over the IR.
//!
//! [`substitute_types`] walks an IR module (or any sub-tree) and replaces
//! every [`Type::TypeParam`] with the corresponding concrete type from a
//! caller-supplied map. The replacement is applied recursively into all
//! nested types, expressions, and statements.

use std::collections::HashMap;

use crate::*;

/// Substitute every [`Type::TypeParam`] in `module` whose name appears in
/// `subst` with the corresponding concrete type. Mutates in place.
pub fn substitute_types(module: &mut Module, subst: &HashMap<String, Type>) {
    for item in &mut module.items {
        sub_item(item, subst);
    }
}

/// Substitute every [`Type::TypeParam`] in a slice of items (e.g. a
/// generic template's body).
pub fn substitute_items(items: &mut [Item], subst: &HashMap<String, Type>) {
    for item in items {
        sub_item(item, subst);
    }
}

/// Rename every reference to `from` (as a struct / function name) to
/// `to` across the supplied IR items. Used at template instantiation
/// time to mangle generic names like `Pair` → `Pair_f32` and `id` →
/// `id_f32` after type substitution.
///
/// The rename touches:
/// * `Item::Struct.name` and `Item::Impl.self_ty`
/// * Top-level `Item::Fn.name` (and impl-method names through
///   `Item::Impl.self_ty`)
/// * `Type::Struct.name` references
/// * `FnPath::Ident` and `FnPath::TypeMethod.ty` references in expressions
/// * `Expr::Struct.name` references
pub fn rename_items(items: &mut [Item], from: &str, to: &str) {
    for item in items {
        rename_item(item, from, to);
    }
}

fn rename_item(item: &mut Item, from: &str, to: &str) {
    match item {
        Item::Const(c) => rename_const(c, from, to),
        Item::Uniform(u) => rename_type(&mut u.ty, from, to),
        Item::Storage(s) => rename_type(&mut s.ty, from, to),
        Item::Workgroup(w) => rename_type(&mut w.ty, from, to),
        Item::Sampler(sa) => rename_type(&mut sa.ty, from, to),
        Item::Texture(t) => rename_type(&mut t.ty, from, to),
        Item::Fn(f) => {
            if f.name == from {
                f.name = to.to_string();
            }
            rename_fn(f, from, to);
        }
        Item::Struct(s) => {
            if s.name == from {
                s.name = to.to_string();
            }
            for f in &mut s.fields {
                rename_type(&mut f.ty, from, to);
            }
        }
        Item::Impl(i) => {
            if i.self_ty == from {
                i.self_ty = to.to_string();
            }
            for ii in &mut i.items {
                match ii {
                    ImplItem::Fn(f) => rename_fn(f, from, to),
                    ImplItem::Const(c) => rename_const(c, from, to),
                }
            }
        }
        Item::Enum(_) => {}
    }
}

fn rename_const(c: &mut ItemConst, from: &str, to: &str) {
    rename_type(&mut c.ty, from, to);
    rename_expr(&mut c.expr, from, to);
}

fn rename_fn(f: &mut ItemFn, from: &str, to: &str) {
    for arg in &mut f.inputs {
        rename_type(&mut arg.ty, from, to);
    }
    if let ReturnType::Type { ty, .. } = &mut f.return_type {
        rename_type(ty, from, to);
    }
    rename_block(&mut f.block, from, to);
}

fn rename_type(ty: &mut Type, from: &str, to: &str) {
    match ty {
        Type::Struct { name, type_args } => {
            if name == from {
                *name = to.to_string();
            }
            for ta in type_args {
                rename_type(ta, from, to);
            }
        }
        Type::Array { elem, len } => {
            rename_type(elem, from, to);
            rename_expr(len, from, to);
        }
        Type::RuntimeArray { elem } | Type::Atomic { elem } | Type::Ptr { elem, .. } => {
            rename_type(elem, from, to)
        }
        _ => {}
    }
}

fn rename_expr(e: &mut Expr, from: &str, to: &str) {
    match e {
        Expr::Lit(_) => {}
        Expr::Ident(name) => {
            if name == from {
                *name = to.to_string();
            }
        }
        Expr::TypePath { ty, .. } => {
            if ty == from {
                *ty = to.to_string();
            }
        }
        Expr::Array { elems } => {
            for x in elems {
                rename_expr(x, from, to);
            }
        }
        Expr::Paren(inner) | Expr::Reference(inner) => rename_expr(inner, from, to),
        Expr::Binary { lhs, rhs, .. } => {
            rename_expr(lhs, from, to);
            rename_expr(rhs, from, to);
        }
        Expr::Unary { expr, .. } => rename_expr(expr, from, to),
        Expr::ArrayIndexing { lhs, index } => {
            rename_expr(lhs, from, to);
            rename_expr(index, from, to);
        }
        Expr::Swizzle { lhs, params, .. } => {
            rename_expr(lhs, from, to);
            if let Some(args) = params {
                for a in args {
                    rename_expr(a, from, to);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            rename_expr(lhs, from, to);
            rename_type(ty, from, to);
        }
        Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            match path {
                FnPath::Ident(name) => {
                    if name == from {
                        *name = to.to_string();
                    }
                }
                FnPath::TypeMethod { ty, .. } => {
                    if ty == from {
                        *ty = to.to_string();
                    }
                }
            }
            for ta in type_args {
                rename_type(ta, from, to);
            }
            for p in params {
                rename_expr(p, from, to);
            }
        }
        Expr::Struct {
            name,
            type_args,
            fields,
        } => {
            if name == from {
                *name = to.to_string();
            }
            for ta in type_args {
                rename_type(ta, from, to);
            }
            for f in fields {
                rename_expr(&mut f.expr, from, to);
            }
        }
        Expr::FieldAccess { base, .. } => rename_expr(base, from, to),
        Expr::ZeroValueArray { elem_type, len } => {
            rename_type(elem_type, from, to);
            rename_expr(len, from, to);
        }
    }
}

fn rename_block(b: &mut Block, from: &str, to: &str) {
    for s in &mut b.stmts {
        rename_stmt(s, from, to);
    }
}

fn rename_stmt(s: &mut Stmt, from: &str, to: &str) {
    match s {
        Stmt::Local(l) => {
            if let Some(t) = &mut l.ty {
                rename_type(t, from, to);
            }
            if let Some(e) = &mut l.init {
                rename_expr(e, from, to);
            }
        }
        Stmt::Const(c) => rename_const(c, from, to),
        Stmt::Assignment { lhs, rhs } => {
            rename_expr(lhs, from, to);
            rename_expr(rhs, from, to);
        }
        Stmt::CompoundAssignment { lhs, rhs, .. } => {
            rename_expr(lhs, from, to);
            rename_expr(rhs, from, to);
        }
        Stmt::While { condition, body } => {
            rename_expr(condition, from, to);
            rename_block(body, from, to);
        }
        Stmt::Loop { body } => rename_block(body, from, to),
        Stmt::Expr { expr, .. } => rename_expr(expr, from, to),
        Stmt::If(i) => rename_if(i, from, to),
        Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        Stmt::Return(e) => {
            if let Some(e) = e {
                rename_expr(e, from, to);
            }
        }
        Stmt::For(f) => {
            if let Some(t) = &mut f.var_ty {
                rename_type(t, from, to);
            }
            rename_expr(&mut f.from, from, to);
            rename_expr(&mut f.to, from, to);
            rename_block(&mut f.body, from, to);
        }
        Stmt::Switch(sw) => {
            rename_expr(&mut sw.selector, from, to);
            for arm in &mut sw.arms {
                for sel in &mut arm.selectors {
                    if let CaseSelector::Expr(e) = sel {
                        rename_expr(e, from, to);
                    }
                }
                rename_block(&mut arm.body, from, to);
            }
        }
        Stmt::Block(b) => rename_block(b, from, to),
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
        } => {
            rename_expr(slab, from, to);
            rename_expr(offset, from, to);
            rename_expr(dest, from, to);
            rename_expr(size, from, to);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
        } => {
            rename_expr(slab, from, to);
            rename_expr(offset, from, to);
            rename_expr(src, from, to);
            if let Some(sz) = size {
                rename_expr(sz, from, to);
            }
        }
    }
}

fn rename_if(i: &mut StmtIf, from: &str, to: &str) {
    rename_expr(&mut i.condition, from, to);
    rename_block(&mut i.then_block, from, to);
    if let Some(eb) = &mut i.else_branch {
        match eb {
            ElseBranch::Block(b) => rename_block(b, from, to),
            ElseBranch::If(inner) => rename_if(inner, from, to),
        }
    }
}

// ===== Items =====

fn sub_item(item: &mut Item, s: &HashMap<String, Type>) {
    match item {
        Item::Const(c) => sub_const(c, s),
        Item::Uniform(u) => sub_type(&mut u.ty, s),
        Item::Storage(st) => sub_type(&mut st.ty, s),
        Item::Workgroup(w) => sub_type(&mut w.ty, s),
        Item::Sampler(sa) => sub_type(&mut sa.ty, s),
        Item::Texture(t) => sub_type(&mut t.ty, s),
        Item::Fn(f) => sub_fn(f, s),
        Item::Struct(st) => {
            for f in &mut st.fields {
                sub_type(&mut f.ty, s);
            }
        }
        Item::Impl(i) => {
            for ii in &mut i.items {
                match ii {
                    ImplItem::Fn(f) => sub_fn(f, s),
                    ImplItem::Const(c) => sub_const(c, s),
                }
            }
        }
        Item::Enum(_) => {}
    }
}

fn sub_const(c: &mut ItemConst, s: &HashMap<String, Type>) {
    sub_type(&mut c.ty, s);
    sub_expr(&mut c.expr, s);
}

fn sub_fn(f: &mut ItemFn, s: &HashMap<String, Type>) {
    for arg in &mut f.inputs {
        sub_type(&mut arg.ty, s);
    }
    if let ReturnType::Type { ty, .. } = &mut f.return_type {
        sub_type(ty, s);
    }
    sub_block(&mut f.block, s);
}

// ===== Types =====

fn sub_type(ty: &mut Type, s: &HashMap<String, Type>) {
    // Outer match first to detect a TypeParam at the top of the tree.
    if let Type::TypeParam { name } = ty {
        if let Some(replacement) = s.get(name) {
            *ty = replacement.clone();
            // Recurse into the replacement in case it itself contains
            // type parameters that should be further substituted.
            sub_type(ty, s);
            return;
        } else {
            // No mapping for this type param: leave it unchanged.
            return;
        }
    }
    match ty {
        Type::Scalar(_)
        | Type::Sampler
        | Type::SamplerComparison
        | Type::Texture { .. }
        | Type::TextureDepth { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. } => {}
        Type::Array { elem, len } => {
            sub_type(elem, s);
            sub_expr(len, s);
        }
        Type::RuntimeArray { elem } | Type::Atomic { elem } => sub_type(elem, s),
        Type::Struct { type_args, .. } => {
            for ta in type_args {
                sub_type(ta, s);
            }
        }
        Type::Ptr { elem, .. } => sub_type(elem, s),
        Type::TypeParam { .. } => unreachable!("handled above"),
    }
}

// ===== Expressions =====

fn sub_expr(e: &mut Expr, s: &HashMap<String, Type>) {
    match e {
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
        Expr::Array { elems } => {
            for x in elems {
                sub_expr(x, s);
            }
        }
        Expr::Paren(inner) | Expr::Reference(inner) => sub_expr(inner, s),
        Expr::Binary { lhs, rhs, .. } => {
            sub_expr(lhs, s);
            sub_expr(rhs, s);
        }
        Expr::Unary { expr, .. } => sub_expr(expr, s),
        Expr::ArrayIndexing { lhs, index } => {
            sub_expr(lhs, s);
            sub_expr(index, s);
        }
        Expr::Swizzle { lhs, params, .. } => {
            sub_expr(lhs, s);
            if let Some(args) = params {
                for a in args {
                    sub_expr(a, s);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            sub_expr(lhs, s);
            sub_type(ty, s);
        }
        Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            // `Type::method()` calls where `Type` is a type param need
            // their `ty` rewritten too. The IR keeps the path as a String,
            // so we convert the concrete substitution to its identifier
            // form (mangled name).
            if let FnPath::TypeMethod { ty, .. } = path
                && let Some(concrete) = s.get(ty.as_str())
            {
                *ty = type_to_ident(concrete);
            }
            for ta in type_args {
                sub_type(ta, s);
            }
            for p in params {
                sub_expr(p, s);
            }
        }
        Expr::Struct {
            name,
            type_args,
            fields,
        } => {
            // Struct construction expressions like `T { ... }` where `T`
            // is a type param: rewrite the name to the substituted form.
            if let Some(concrete) = s.get(name.as_str()) {
                *name = type_to_ident(concrete);
            }
            for ta in type_args {
                sub_type(ta, s);
            }
            for f in fields {
                sub_expr(&mut f.expr, s);
            }
        }
        Expr::FieldAccess { base, .. } => sub_expr(base, s),
        Expr::ZeroValueArray { elem_type, len } => {
            sub_type(elem_type, s);
            sub_expr(len, s);
        }
    }
}

/// Convert a concrete IR type to an identifier string suitable for use
/// where a struct / function name is expected (i.e. as the `ty` field of
/// a `FnPath::TypeMethod`, or the `name` of an `Expr::Struct`). For
/// scalar types this is the WGSL name (`f32`); for user structs the
/// struct name; for compound types it's a mangled identifier like
/// `array_f32_4`.
pub fn type_to_ident(t: &Type) -> String {
    match t {
        Type::Scalar(s) => match s {
            ScalarType::I32 => "i32".to_string(),
            ScalarType::U32 => "u32".to_string(),
            ScalarType::F32 => "f32".to_string(),
            ScalarType::Bool => "bool".to_string(),
        },
        Type::Vector {
            elements,
            scalar_ty,
        } => match scalar_ty {
            Some(s) => format!("vec{}{}", elements, scalar_short(*s)),
            None => format!("vec{}", elements),
        },
        Type::Matrix {
            columns,
            rows,
            scalar_ty,
        } => match scalar_ty {
            Some(s) => format!("mat{}x{}{}", columns, rows, scalar_short(*s)),
            None => format!("mat{}x{}f", columns, rows),
        },
        Type::Array { elem, .. } => format!("array_{}", type_to_ident(elem)),
        Type::RuntimeArray { elem } => format!("array_{}", type_to_ident(elem)),
        Type::Atomic { elem } => format!("atomic_{}", type_to_ident(elem)),
        Type::Struct { name, .. } => name.clone(),
        Type::Ptr { elem, .. } => format!("ptr_{}", type_to_ident(elem)),
        Type::Sampler => "sampler".to_string(),
        Type::SamplerComparison => "sampler_comparison".to_string(),
        Type::Texture { .. } => "texture".to_string(),
        Type::TextureDepth { .. } => "texture_depth".to_string(),
        Type::TypeParam { name } => name.clone(),
    }
}

fn scalar_short(s: ScalarType) -> &'static str {
    match s {
        ScalarType::I32 => "i",
        ScalarType::U32 => "u",
        ScalarType::F32 => "f",
        ScalarType::Bool => "b",
    }
}

// ===== Statements / blocks =====

fn sub_block(b: &mut Block, s: &HashMap<String, Type>) {
    for stmt in &mut b.stmts {
        sub_stmt(stmt, s);
    }
}

fn sub_stmt(st: &mut Stmt, s: &HashMap<String, Type>) {
    match st {
        Stmt::Local(l) => {
            if let Some(t) = &mut l.ty {
                sub_type(t, s);
            }
            if let Some(e) = &mut l.init {
                sub_expr(e, s);
            }
        }
        Stmt::Const(c) => sub_const(c, s),
        Stmt::Assignment { lhs, rhs } => {
            sub_expr(lhs, s);
            sub_expr(rhs, s);
        }
        Stmt::CompoundAssignment { lhs, rhs, .. } => {
            sub_expr(lhs, s);
            sub_expr(rhs, s);
        }
        Stmt::While { condition, body } => {
            sub_expr(condition, s);
            sub_block(body, s);
        }
        Stmt::Loop { body } => sub_block(body, s),
        Stmt::Expr { expr, .. } => sub_expr(expr, s),
        Stmt::If(i) => sub_if(i, s),
        Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        Stmt::Return(e) => {
            if let Some(e) = e {
                sub_expr(e, s);
            }
        }
        Stmt::For(f) => {
            if let Some(t) = &mut f.var_ty {
                sub_type(t, s);
            }
            sub_expr(&mut f.from, s);
            sub_expr(&mut f.to, s);
            sub_block(&mut f.body, s);
        }
        Stmt::Switch(sw) => {
            sub_expr(&mut sw.selector, s);
            for arm in &mut sw.arms {
                for sel in &mut arm.selectors {
                    if let CaseSelector::Expr(e) = sel {
                        sub_expr(e, s);
                    }
                }
                sub_block(&mut arm.body, s);
            }
        }
        Stmt::Block(b) => sub_block(b, s),
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
        } => {
            sub_expr(slab, s);
            sub_expr(offset, s);
            sub_expr(dest, s);
            sub_expr(size, s);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
        } => {
            sub_expr(slab, s);
            sub_expr(offset, s);
            sub_expr(src, s);
            if let Some(sz) = size {
                sub_expr(sz, s);
            }
        }
    }
}

fn sub_if(i: &mut StmtIf, s: &HashMap<String, Type>) {
    sub_expr(&mut i.condition, s);
    sub_block(&mut i.then_block, s);
    if let Some(eb) = &mut i.else_branch {
        match eb {
            ElseBranch::Block(b) => sub_block(b, s),
            ElseBranch::If(inner) => sub_if(inner, s),
        }
    }
}
