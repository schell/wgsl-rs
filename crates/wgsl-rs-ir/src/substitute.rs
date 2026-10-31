//! Generic type-parameter substitution over the IR.
//!
//! [`substitute_types`] walks an IR module (or any sub-tree) and replaces
//! every [`Type::TypeParam`] with the corresponding concrete type from a
//! caller-supplied map. The replacement is applied recursively into all
//! nested types, expressions, and statements.
//!
//! [`substitute_consts`] is the parallel pass for `const` generic
//! parameters: it replaces every `Expr::Ident(name)` referencing a const
//! param with a concrete `Expr::Lit` integer literal.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

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

/// Substitute every const-param reference (`Expr::Ident(name)`) in
/// `module` whose name appears in `consts` with a concrete
/// `Expr::Lit(Lit::Int { ... })`. Mutates in place.
///
/// This is the const-generic parallel of [`substitute_types`]. It walks
/// the same IR tree but rewrites array length expressions (and any other
/// `Expr::Ident` naming a const param) to integer literals. The two passes
/// are independent and can be run in either order.
pub fn substitute_consts(module: &mut Module, consts: &HashMap<String, i64>) {
    for item in &mut module.items {
        sub_item_const(item, consts);
    }
}

/// Like [`substitute_consts`] but over a slice of items (e.g. a generic
/// template's body, as used by the cross-module template machinery).
pub fn substitute_consts_in_items(items: &mut [Item], consts: &HashMap<String, i64>) {
    for item in items {
        sub_item_const(item, consts);
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
                f.name = Cow::Owned(to.to_string());
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
                    ImplItem::Type(t) => rename_type(&mut t.ty, from, to),
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
        Type::Phantom { elem } => rename_type(elem, from, to),
        Type::AssocType { ty, .. } => rename_type(ty, from, to),
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
        Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            rename_expr(src, from, to);
            rename_expr(src_offset, from, to);
            rename_expr(dest, from, to);
            rename_expr(dest_offset, from, to);
            rename_expr(size, from, to);
        }
        Stmt::Macro { .. } => {}
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
                    ImplItem::Type(t) => sub_type(&mut t.ty, s),
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
        | Type::TextureStorage { .. }
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
        Type::Phantom { elem } => sub_type(elem, s),
        Type::AssocType { ty, .. } => sub_type(ty, s),
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
        Type::TextureStorage { format, .. } => format!("texture_storage_{}", format.wgsl_name()),
        Type::TypeParam { name } => name.clone(),
        Type::Phantom { elem } => format!("phantom_{}", type_to_ident(elem)),
        Type::AssocType { ty, member } => format!("{}_{}", type_to_ident(ty), member),
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
        Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            sub_expr(src, s);
            sub_expr(src_offset, s);
            sub_expr(dest, s);
            sub_expr(dest_offset, s);
            sub_expr(size, s);
        }
        Stmt::Macro { .. } => {}
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

// ===== Const substitution =====
//
// Parallel to the type substitution walkers above, but replaces
// `Expr::Ident(name)` with `Expr::Lit(Lit::Int { ... })` when `name`
// names a const generic parameter. The substitution target is always a
// bare ident because stable Rust only allows bare idents (or literals)
// as const generic arguments in type position.

fn sub_item_const(item: &mut Item, consts: &HashMap<String, i64>) {
    match item {
        Item::Const(c) => {
            sub_type_const(&mut c.ty, consts);
            sub_expr_const(&mut c.expr, consts);
        }
        Item::Uniform(u) => sub_type_const(&mut u.ty, consts),
        Item::Storage(st) => sub_type_const(&mut st.ty, consts),
        Item::Workgroup(w) => sub_type_const(&mut w.ty, consts),
        Item::Sampler(sa) => sub_type_const(&mut sa.ty, consts),
        Item::Texture(t) => sub_type_const(&mut t.ty, consts),
        Item::Fn(f) => sub_fn_const(f, consts),
        Item::Struct(st) => {
            for f in &mut st.fields {
                sub_type_const(&mut f.ty, consts);
            }
        }
        Item::Impl(i) => {
            for ii in &mut i.items {
                match ii {
                    ImplItem::Fn(f) => sub_fn_const(f, consts),
                    ImplItem::Const(c) => {
                        sub_type_const(&mut c.ty, consts);
                        sub_expr_const(&mut c.expr, consts);
                    }
                    ImplItem::Type(t) => sub_type_const(&mut t.ty, consts),
                }
            }
        }
        Item::Enum(_) => {}
    }
}

fn sub_fn_const(f: &mut ItemFn, consts: &HashMap<String, i64>) {
    for arg in &mut f.inputs {
        sub_type_const(&mut arg.ty, consts);
    }
    if let ReturnType::Type { ty, .. } = &mut f.return_type {
        sub_type_const(ty, consts);
    }
    sub_block_const(&mut f.block, consts);
}

fn sub_type_const(ty: &mut Type, consts: &HashMap<String, i64>) {
    match ty {
        Type::Scalar(_)
        | Type::Sampler
        | Type::SamplerComparison
        | Type::Texture { .. }
        | Type::TextureDepth { .. }
        | Type::TextureStorage { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::TypeParam { .. } => {}
        Type::Array { elem, len } => {
            expect_nonneg_const(len, consts);
            sub_type_const(elem, consts);
            sub_expr_const(len, consts);
        }
        Type::RuntimeArray { elem } | Type::Atomic { elem } => sub_type_const(elem, consts),
        Type::Struct { type_args, .. } => {
            for ta in type_args {
                sub_type_const(ta, consts);
            }
        }
        Type::Ptr { elem, .. } => sub_type_const(elem, consts),
        Type::Phantom { elem } => sub_type_const(elem, consts),
        Type::AssocType { ty, .. } => sub_type_const(ty, consts),
    }
}

/// Build the IR expression for a concrete const-generic value.
///
/// Non-negative values render as a bare abstract-int literal —
/// byte-identical to the output before signed const support. Negative
/// values are wrapped in parentheses so they stay unambiguous in every
/// expression context (e.g. `2 - (-4)` rather than `2 - -4`, and
/// `-(-4)` rather than `--4`).
fn const_value_expr(n: i64) -> Expr {
    let lit = Expr::Lit(Lit::Int {
        digits: n.unsigned_abs().to_string(),
        suffix: String::new(),
    });
    if n < 0 {
        Expr::Paren(Box::new(Expr::Unary {
            op: UnOp::Neg,
            expr: Box::new(lit),
        }))
    } else {
        lit
    }
}

/// Guard an array-length position against a negative const value. WGSL
/// requires positive element counts, so a negative must never render
/// there. This is unreachable through wgsl-rs macros — rustc rejects
/// non-`usize` array lengths, so an `i32` const param can't occupy this
/// position in valid Rust — but hand-rolled IR (extensions) could
/// produce it, and the failure would otherwise surface as an opaque
/// WGSL validation error far from the cause.
fn expect_nonneg_const(len: &Expr, consts: &HashMap<String, i64>) {
    if let Expr::Ident(name) = len
        && let Some(&n) = consts.get(name)
        && n < 0
    {
        panic!(
            "negative const generic value {n} substituted into an array length; WGSL element \
             counts must be positive (unreachable through wgsl-rs macros, which only allow \
             `usize` array lengths; this indicates hand-rolled IR)"
        );
    }
}

fn sub_expr_const(e: &mut Expr, consts: &HashMap<String, i64>) {
    match e {
        Expr::Ident(name) => {
            if let Some(&n) = consts.get(name) {
                *e = const_value_expr(n);
            }
        }
        Expr::Lit(_) | Expr::TypePath { .. } => {}
        Expr::Array { elems } => {
            for x in elems {
                sub_expr_const(x, consts);
            }
        }
        Expr::Paren(inner) | Expr::Reference(inner) => sub_expr_const(inner, consts),
        Expr::Binary { lhs, rhs, .. } => {
            sub_expr_const(lhs, consts);
            sub_expr_const(rhs, consts);
        }
        Expr::Unary { expr, .. } => sub_expr_const(expr, consts),
        Expr::ArrayIndexing { lhs, index } => {
            sub_expr_const(lhs, consts);
            sub_expr_const(index, consts);
        }
        Expr::Swizzle { lhs, params, .. } => {
            sub_expr_const(lhs, consts);
            if let Some(args) = params {
                for a in args {
                    sub_expr_const(a, consts);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            sub_expr_const(lhs, consts);
            sub_type_const(ty, consts);
        }
        Expr::FnCall {
            path: _,
            type_args,
            params,
        } => {
            // Note: do NOT rewrite `FnPath::TypeMethod.ty` here — that's
            // a type name, not a const-param reference. Const
            // substitution targets only bare `Expr::Ident` nodes (e.g.
            // `N` in an array length `[u32; N]`), which are handled at the
            // top of `sub_expr_const`. Rewriting type/method paths here
            // would corrupt them if a type name happened to match a
            // const-param name.
            for ta in type_args {
                sub_type_const(ta, consts);
            }
            for p in params {
                sub_expr_const(p, consts);
            }
        }
        Expr::Struct {
            name: _,
            type_args,
            fields,
        } => {
            // Note: do NOT rewrite `name` here — it's a struct type
            // name, not a const-param reference (see the note on
            // `Expr::FnCall` above).
            for ta in type_args {
                sub_type_const(ta, consts);
            }
            for f in fields {
                sub_expr_const(&mut f.expr, consts);
            }
        }
        Expr::FieldAccess { base, .. } => sub_expr_const(base, consts),
        Expr::ZeroValueArray { elem_type, len } => {
            expect_nonneg_const(len, consts);
            sub_type_const(elem_type, consts);
            sub_expr_const(len, consts);
        }
    }
}

fn sub_block_const(b: &mut Block, consts: &HashMap<String, i64>) {
    for stmt in &mut b.stmts {
        sub_stmt_const(stmt, consts);
    }
}

fn sub_stmt_const(st: &mut Stmt, consts: &HashMap<String, i64>) {
    match st {
        Stmt::Local(l) => {
            if let Some(t) = &mut l.ty {
                sub_type_const(t, consts);
            }
            if let Some(e) = &mut l.init {
                sub_expr_const(e, consts);
            }
        }
        Stmt::Const(c) => {
            sub_type_const(&mut c.ty, consts);
            sub_expr_const(&mut c.expr, consts);
        }
        Stmt::Assignment { lhs, rhs } => {
            sub_expr_const(lhs, consts);
            sub_expr_const(rhs, consts);
        }
        Stmt::CompoundAssignment { lhs, rhs, .. } => {
            sub_expr_const(lhs, consts);
            sub_expr_const(rhs, consts);
        }
        Stmt::While { condition, body } => {
            sub_expr_const(condition, consts);
            sub_block_const(body, consts);
        }
        Stmt::Loop { body } => sub_block_const(body, consts),
        Stmt::Expr { expr, .. } => sub_expr_const(expr, consts),
        Stmt::If(i) => sub_if_const(i, consts),
        Stmt::Break | Stmt::Continue | Stmt::Discard => {}
        Stmt::Return(e) => {
            if let Some(e) = e {
                sub_expr_const(e, consts);
            }
        }
        Stmt::For(f) => {
            if let Some(t) = &mut f.var_ty {
                sub_type_const(t, consts);
            }
            sub_expr_const(&mut f.from, consts);
            sub_expr_const(&mut f.to, consts);
            sub_block_const(&mut f.body, consts);
        }
        Stmt::Switch(sw) => {
            sub_expr_const(&mut sw.selector, consts);
            for arm in &mut sw.arms {
                for sel in &mut arm.selectors {
                    if let CaseSelector::Expr(e) = sel {
                        sub_expr_const(e, consts);
                    }
                }
                sub_block_const(&mut arm.body, consts);
            }
        }
        Stmt::Block(b) => sub_block_const(b, consts),
        Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            sub_expr_const(src, consts);
            sub_expr_const(src_offset, consts);
            sub_expr_const(dest, consts);
            sub_expr_const(dest_offset, consts);
            sub_expr_const(size, consts);
        }
        Stmt::Macro { .. } => {}
    }
}

fn sub_if_const(i: &mut StmtIf, consts: &HashMap<String, i64>) {
    sub_expr_const(&mut i.condition, consts);
    sub_block_const(&mut i.then_block, consts);
    if let Some(eb) = &mut i.else_branch {
        match eb {
            ElseBranch::Block(b) => sub_block_const(b, consts),
            ElseBranch::If(inner) => sub_if_const(inner, consts),
        }
    }
}

// ===== Unresolved template reference detection =====
//
// Companion to the substitution and rename passes: scans IR that is about
// to be rendered for references to generic template *names* that were
// never resolved to a concrete instantiation (wgsl-rs#237). A bare call
// `Id::new(...)` or an un-rewritten `Id::<T>::new(...)` in a generic
// context renders the un-mangled name and fails WGSL validation far from
// the authoring mistake; the runtime source-assembly path uses this scan
// to report the offense with an actionable message instead.

/// A reference to a generic template that was not resolved to a concrete
/// instantiation before rendering (wgsl-rs#237).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedTemplateRef {
    /// The referenced template's name (e.g. `Id` in `Id::new(...)`).
    pub template: String,
    /// The referenced member for method calls, associated constants, and
    /// field types (e.g. `new`); `None` for a bare call to a generic
    /// function template.
    pub member: Option<String>,
    /// Whether the reference carried explicit type arguments that could
    /// not be resolved (e.g. `Id::<T>::new(...)` inside a generic
    /// context). `false` for references with no type arguments at all.
    pub has_type_args: bool,
}

/// Scan IR items for the first reference to any name in `template_names`
/// that would render without a concrete instantiation. Returns `None`
/// when every reference is resolved.
///
/// Covered positions: calls to generic function templates without
/// turbofish, calls to generic struct methods (with or without leftover
/// turbofish), associated-constant paths, and type references. Bare
/// struct-constructor expressions are deliberately not flagged: they
/// are a documented known limitation with let-annotation workarounds,
/// not a call-site mistake.
pub fn find_unresolved_template_refs(
    items: &[Item],
    template_names: &HashSet<String>,
) -> Option<UnresolvedTemplateRef> {
    for item in items {
        if let Some(offense) = ref_item(item, template_names) {
            return Some(offense);
        }
    }
    None
}

fn ref_item(item: &Item, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    match item {
        Item::Const(c) => ref_const(c, t),
        Item::Uniform(u) => ref_type(&u.ty, t),
        Item::Storage(s) => ref_type(&s.ty, t),
        Item::Workgroup(w) => ref_type(&w.ty, t),
        Item::Sampler(sa) => ref_type(&sa.ty, t),
        Item::Texture(tx) => ref_type(&tx.ty, t),
        Item::Fn(f) => ref_fn(f, t),
        Item::Struct(s) => {
            for f in &s.fields {
                if let Some(o) = ref_type(&f.ty, t) {
                    return Some(o);
                }
            }
            None
        }
        Item::Impl(i) => {
            if t.contains(i.self_ty.as_str()) {
                return Some(UnresolvedTemplateRef {
                    template: i.self_ty.to_string(),
                    member: None,
                    has_type_args: false,
                });
            }
            for ii in &i.items {
                let offense = match ii {
                    ImplItem::Fn(f) => ref_fn(f, t),
                    ImplItem::Const(c) => ref_const(c, t),
                    ImplItem::Type(a) => ref_type(&a.ty, t),
                };
                if let Some(o) = offense {
                    return Some(o);
                }
            }
            None
        }
        Item::Enum(_) => None,
    }
}

fn ref_fn(f: &ItemFn, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    for arg in &f.inputs {
        if let Some(o) = ref_type(&arg.ty, t) {
            return Some(o);
        }
    }
    if let ReturnType::Type { ty, .. } = &f.return_type
        && let Some(o) = ref_type(ty, t)
    {
        return Some(o);
    }
    ref_block(&f.block, t)
}

fn ref_const(c: &ItemConst, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    ref_type(&c.ty, t).or_else(|| ref_expr(&c.expr, t))
}

fn ref_type(ty: &Type, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    match ty {
        Type::Struct { name, type_args } => {
            if t.contains(name.as_str()) {
                return Some(UnresolvedTemplateRef {
                    template: name.to_string(),
                    member: None,
                    has_type_args: !type_args.is_empty(),
                });
            }
            for ta in type_args {
                if let Some(o) = ref_type(ta, t) {
                    return Some(o);
                }
            }
            None
        }
        Type::Array { elem, len } => ref_type(elem, t).or_else(|| ref_expr(len, t)),
        Type::RuntimeArray { elem } | Type::Atomic { elem } => ref_type(elem, t),
        Type::Ptr { elem, .. } => ref_type(elem, t),
        Type::Phantom { elem } => ref_type(elem, t),
        Type::AssocType { ty, .. } => ref_type(ty, t),
        Type::Scalar(_)
        | Type::Sampler
        | Type::SamplerComparison
        | Type::Texture { .. }
        | Type::TextureDepth { .. }
        | Type::TextureStorage { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::TypeParam { .. } => None,
    }
}

fn ref_expr(e: &Expr, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    match e {
        Expr::Lit(_) | Expr::Ident(_) => None,
        Expr::TypePath { ty, member } => {
            if t.contains(ty.as_str()) {
                return Some(UnresolvedTemplateRef {
                    template: ty.to_string(),
                    member: Some(member.to_string()),
                    has_type_args: false,
                });
            }
            None
        }
        Expr::Array { elems } => elems.iter().find_map(|x| ref_expr(x, t)),
        Expr::Paren(inner) | Expr::Reference(inner) => ref_expr(inner, t),
        Expr::Binary { lhs, rhs, .. } => ref_expr(lhs, t).or_else(|| ref_expr(rhs, t)),
        Expr::Unary { expr, .. } => ref_expr(expr, t),
        Expr::ArrayIndexing { lhs, index } => ref_expr(lhs, t).or_else(|| ref_expr(index, t)),
        Expr::Swizzle { lhs, params, .. } => ref_expr(lhs, t).or_else(|| {
            params
                .as_ref()
                .and_then(|args| args.iter().find_map(|a| ref_expr(a, t)))
        }),
        Expr::Cast { lhs, ty } => ref_expr(lhs, t).or_else(|| ref_type(ty, t)),
        Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            let direct = match path {
                FnPath::Ident(name) if type_args.is_empty() && t.contains(name.as_str()) => {
                    Some(UnresolvedTemplateRef {
                        template: name.to_string(),
                        member: None,
                        has_type_args: false,
                    })
                }
                FnPath::TypeMethod { ty, method } if t.contains(ty.as_str()) => {
                    Some(UnresolvedTemplateRef {
                        template: ty.to_string(),
                        member: Some(method.to_string()),
                        has_type_args: !type_args.is_empty(),
                    })
                }
                _ => None,
            };
            direct
                .or_else(|| type_args.iter().find_map(|ta| ref_type(ta, t)))
                .or_else(|| params.iter().find_map(|p| ref_expr(p, t)))
        }
        // Bare struct-constructor expressions (`Pair { a, b }`) are a
        // documented known limitation with let-annotation workarounds;
        // they are not flagged here. Their *field* expressions may still
        // contain flagged references.
        Expr::Struct { fields, .. } => fields.iter().find_map(|f| ref_expr(&f.expr, t)),
        Expr::FieldAccess { base, .. } => ref_expr(base, t),
        Expr::ZeroValueArray { elem_type, len } => {
            ref_type(elem_type, t).or_else(|| ref_expr(len, t))
        }
    }
}

fn ref_block(b: &Block, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    for s in &b.stmts {
        if let Some(o) = ref_stmt(s, t) {
            return Some(o);
        }
    }
    None
}

fn ref_stmt(s: &Stmt, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    match s {
        Stmt::Local(l) => {
            if let Some(ty) = &l.ty
                && let Some(o) = ref_type(ty, t)
            {
                return Some(o);
            }
            if let Some(init) = &l.init {
                return ref_expr(init, t);
            }
            None
        }
        Stmt::Const(c) => ref_const(c, t),
        Stmt::Assignment { lhs, rhs } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            ref_expr(lhs, t).or_else(|| ref_expr(rhs, t))
        }
        Stmt::While { condition, body } => ref_expr(condition, t).or_else(|| ref_block(body, t)),
        Stmt::Loop { body } => ref_block(body, t),
        Stmt::Expr { expr, .. } => ref_expr(expr, t),
        Stmt::If(i) => ref_if(i, t),
        Stmt::Break | Stmt::Continue | Stmt::Discard => None,
        Stmt::Return(e) => e.as_ref().and_then(|e| ref_expr(e, t)),
        Stmt::For(f) => {
            if let Some(ty) = &f.var_ty
                && let Some(o) = ref_type(ty, t)
            {
                return Some(o);
            }
            ref_expr(&f.from, t)
                .or_else(|| ref_expr(&f.to, t))
                .or_else(|| ref_block(&f.body, t))
        }
        Stmt::Switch(sw) => {
            if let Some(o) = ref_expr(&sw.selector, t) {
                return Some(o);
            }
            for arm in &sw.arms {
                for sel in &arm.selectors {
                    if let CaseSelector::Expr(e) = sel
                        && let Some(o) = ref_expr(e, t)
                    {
                        return Some(o);
                    }
                }
                if let Some(o) = ref_block(&arm.body, t) {
                    return Some(o);
                }
            }
            None
        }
        Stmt::Block(b) => ref_block(b, t),
        Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => ref_expr(src, t)
            .or_else(|| ref_expr(src_offset, t))
            .or_else(|| ref_expr(dest, t))
            .or_else(|| ref_expr(dest_offset, t))
            .or_else(|| ref_expr(size, t)),
        Stmt::Macro { .. } => None,
    }
}

fn ref_if(i: &StmtIf, t: &HashSet<String>) -> Option<UnresolvedTemplateRef> {
    ref_expr(&i.condition, t)
        .or_else(|| ref_block(&i.then_block, t))
        .or_else(|| {
            i.else_branch.as_ref().and_then(|eb| match eb {
                ElseBranch::Block(b) => ref_block(b, t),
                ElseBranch::If(inner) => ref_if(inner, t),
            })
        })
}
