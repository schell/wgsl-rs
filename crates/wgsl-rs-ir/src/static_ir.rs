//! Borrowed (`&'a`) mirror of the owned IR types, for emitting IR as
//! data instead of code (wgsl-rs#221).
//!
//! The `#[wgsl]` proc macro historically emitted a constructor
//! *function* that builds the owned IR at runtime. For a large shader
//! that function is enormous (a ~1000-line shader expands to a
//! ~12,000-line ctor) and LLVM burns ~14s per incremental rebuild at
//! `opt-level=1`. The fix is to emit the IR as a static literal built
//! from these borrowed types plus a tiny shim that converts to the
//! owned IR with the [`From`] impls in this module. The heavy code
//! lives once in this library; the per-module expansion is only data.
//!
//! Mirroring rules. Every type here is the `crate::types` namesake
//! with, recursively:
//!
//! * `String` / `Cow<'static, str>` → `&'a str`
//! * `Vec<T>` → `&'a [T]`
//! * `Box<T>` → `&'a T`
//! * `Option<T>` and inline enum payloads stay inline
//!
//! Field names and enum variant names are identical to the owned IR,
//! so the two can be read side by side. Types that are pure const data
//! (e.g. [`ScalarType`], [`FnAttrs`], [`InterStageIo`],
//! [`ReturnTypeAnnotation`], [`WorkgroupSize`]) are *reused* from
//! `crate::types` rather than mirrored.
//!
//! The owned IR's `Module::name` is `&'static str` and `ItemFn::name`
//! is `Cow<'static, str>` (borrowed for fresh, non-monomorphized
//! functions; linkage code relies on staying `Borrowed`), so the
//! converters for [`StaticModule`], [`StaticItemFn`],
//! [`StaticImplItem`], and [`StaticItem`] are restricted to `'static`.
//! Every other converter is generic over `'a`. The macro always uses
//! these types at `'static` (they are emitted as static literals).
//!
//! ## Keeping the mirror in sync
//!
//! Rust has no reflection, so sync is enforced where a mistake is a
//! compile error rather than a runtime bug:
//!
//! * the `wgsl-rs-macros` owned-to-borrowed emission matches every owned enum
//!   exhaustively, so a new IR variant without a mirror arm fails *this
//!   workspace's* build, never a user's shader;
//! * the round-trip test at the bottom of this module converts a kitchen-sink
//!   fixture through both directions and compares exactly, so any field dropped
//!   or mistranslated by either direction fails CI.

use std::borrow::Cow;

use crate::types::*;

/// Convert a borrowed slice to an owned `Vec`, element-wise via the
/// element's [`From`] impl.
fn conv<T, U>(slice: &[T]) -> Vec<U>
where
    U: for<'x> From<&'x T>,
{
    slice.iter().map(U::from).collect()
}

/// Borrowed mirror of [`Attribute`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticAttribute<'a> {
    pub path: &'a str,
    pub args: &'a [&'a str],
}

/// Owned copy of the borrowed attribute.
impl<'a> From<&StaticAttribute<'a>> for Attribute {
    fn from(s: &StaticAttribute<'a>) -> Self {
        Attribute {
            path: s.path.to_string(),
            args: s.args.iter().map(|arg| arg.to_string()).collect(),
        }
    }
}

/// Borrowed mirror of [`Module`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticModule<'a> {
    pub name: &'a str,
    pub items: &'a [StaticItem<'a>],
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed module.
///
/// Restricted to `'static`: the owned `Module::name` is `&'static str`.
impl From<&StaticModule<'static>> for Module {
    fn from(s: &StaticModule<'static>) -> Self {
        Module {
            name: s.name,
            items: conv(s.items),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`Block`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBlock<'a> {
    pub stmts: &'a [StaticStmt<'a>],
}

/// Owned copy of the borrowed block.
impl<'a> From<&StaticBlock<'a>> for Block {
    fn from(s: &StaticBlock<'a>) -> Self {
        Block {
            stmts: conv(s.stmts),
        }
    }
}

/// Borrowed mirror of [`Type`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticType<'a> {
    Scalar(ScalarType),
    Vector {
        elements: u8,
        scalar_ty: Option<ScalarType>,
    },
    Matrix {
        columns: u8,
        rows: u8,
        scalar_ty: Option<ScalarType>,
    },
    Array {
        elem: &'a StaticType<'a>,
        len: StaticExpr<'a>,
    },
    RuntimeArray {
        elem: &'a StaticType<'a>,
    },
    Atomic {
        elem: &'a StaticType<'a>,
    },
    Struct {
        name: &'a str,
        type_args: &'a [StaticType<'a>],
    },
    Ptr {
        address_space: AddressSpace,
        elem: &'a StaticType<'a>,
    },
    Sampler,
    SamplerComparison,
    Texture {
        kind: TextureKind,
        sampled_type: ScalarType,
    },
    TextureDepth {
        kind: TextureDepthKind,
    },
    TextureStorage {
        kind: TextureStorageKind,
        format: TexelFormat,
        access: StorageTextureAccess,
    },
    TypeParam {
        name: &'a str,
    },
    Phantom {
        elem: &'a StaticType<'a>,
    },
    AssocType {
        ty: &'a StaticType<'a>,
        member: &'a str,
    },
}

/// Owned copy of the borrowed type expression.
impl<'a> From<&StaticType<'a>> for Type {
    fn from(s: &StaticType<'a>) -> Self {
        match s {
            StaticType::Scalar(scalar) => Type::Scalar(*scalar),
            StaticType::Vector {
                elements,
                scalar_ty,
            } => Type::Vector {
                elements: *elements,
                scalar_ty: *scalar_ty,
            },
            StaticType::Matrix {
                columns,
                rows,
                scalar_ty,
            } => Type::Matrix {
                columns: *columns,
                rows: *rows,
                scalar_ty: *scalar_ty,
            },
            StaticType::Array { elem, len } => Type::Array {
                elem: Box::new(Type::from(*elem)),
                len: Expr::from(len),
            },
            StaticType::RuntimeArray { elem } => Type::RuntimeArray {
                elem: Box::new(Type::from(*elem)),
            },
            StaticType::Atomic { elem } => Type::Atomic {
                elem: Box::new(Type::from(*elem)),
            },
            StaticType::Struct { name, type_args } => Type::Struct {
                name: name.to_string(),
                type_args: conv(type_args),
            },
            StaticType::Ptr {
                address_space,
                elem,
            } => Type::Ptr {
                address_space: *address_space,
                elem: Box::new(Type::from(*elem)),
            },
            StaticType::Sampler => Type::Sampler,
            StaticType::SamplerComparison => Type::SamplerComparison,
            StaticType::Texture { kind, sampled_type } => Type::Texture {
                kind: *kind,
                sampled_type: *sampled_type,
            },
            StaticType::TextureDepth { kind } => Type::TextureDepth { kind: *kind },
            StaticType::TextureStorage {
                kind,
                format,
                access,
            } => Type::TextureStorage {
                kind: *kind,
                format: *format,
                access: *access,
            },
            StaticType::TypeParam { name } => Type::TypeParam {
                name: name.to_string(),
            },
            StaticType::Phantom { elem } => Type::Phantom {
                elem: Box::new(Type::from(*elem)),
            },
            StaticType::AssocType { ty, member } => Type::AssocType {
                ty: Box::new(Type::from(*ty)),
                member: member.to_string(),
            },
        }
    }
}

/// Borrowed mirror of [`Lit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticLit<'a> {
    Bool(bool),
    Int { digits: &'a str, suffix: &'a str },
    Float { text: &'a str },
}

/// Owned copy of the borrowed literal.
impl<'a> From<&StaticLit<'a>> for Lit {
    fn from(s: &StaticLit<'a>) -> Self {
        match s {
            StaticLit::Bool(b) => Lit::Bool(*b),
            StaticLit::Int { digits, suffix } => Lit::Int {
                digits: digits.to_string(),
                suffix: suffix.to_string(),
            },
            StaticLit::Float { text } => Lit::Float {
                text: text.to_string(),
            },
        }
    }
}

/// Borrowed mirror of [`FnPath`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticFnPath<'a> {
    Ident(&'a str),
    TypeMethod { ty: &'a str, method: &'a str },
}

/// Owned copy of the borrowed function path.
impl<'a> From<&StaticFnPath<'a>> for FnPath {
    fn from(s: &StaticFnPath<'a>) -> Self {
        match s {
            StaticFnPath::Ident(name) => FnPath::Ident(name.to_string()),
            StaticFnPath::TypeMethod { ty, method } => FnPath::TypeMethod {
                ty: ty.to_string(),
                method: method.to_string(),
            },
        }
    }
}

/// Borrowed mirror of [`FieldValue`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticFieldValue<'a> {
    pub member: &'a str,
    pub expr: StaticExpr<'a>,
}

/// Owned copy of the borrowed field value.
impl<'a> From<&StaticFieldValue<'a>> for FieldValue {
    fn from(s: &StaticFieldValue<'a>) -> Self {
        FieldValue {
            member: s.member.to_string(),
            expr: Expr::from(&s.expr),
        }
    }
}

/// Borrowed mirror of [`Expr`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticExpr<'a> {
    Lit(StaticLit<'a>),
    Ident(&'a str),
    Array {
        elems: &'a [StaticExpr<'a>],
    },
    Paren(&'a StaticExpr<'a>),
    Binary {
        lhs: &'a StaticExpr<'a>,
        op: BinOp,
        rhs: &'a StaticExpr<'a>,
    },
    Unary {
        op: UnOp,
        expr: &'a StaticExpr<'a>,
    },
    ArrayIndexing {
        lhs: &'a StaticExpr<'a>,
        index: &'a StaticExpr<'a>,
    },
    Swizzle {
        lhs: &'a StaticExpr<'a>,
        swizzle: &'a str,
        params: Option<&'a [StaticExpr<'a>]>,
    },
    Cast {
        lhs: &'a StaticExpr<'a>,
        ty: &'a StaticType<'a>,
    },
    FnCall {
        path: StaticFnPath<'a>,
        type_args: &'a [StaticType<'a>],
        params: &'a [StaticExpr<'a>],
    },
    Struct {
        name: &'a str,
        type_args: &'a [StaticType<'a>],
        fields: &'a [StaticFieldValue<'a>],
    },
    FieldAccess {
        base: &'a StaticExpr<'a>,
        field: &'a str,
    },
    TypePath {
        ty: &'a str,
        member: &'a str,
    },
    Reference(&'a StaticExpr<'a>),
    ZeroValueArray {
        elem_type: &'a StaticType<'a>,
        len: &'a StaticExpr<'a>,
    },
}

/// Owned copy of the borrowed expression.
impl<'a> From<&StaticExpr<'a>> for Expr {
    fn from(s: &StaticExpr<'a>) -> Self {
        match s {
            StaticExpr::Lit(lit) => Expr::Lit(Lit::from(lit)),
            StaticExpr::Ident(name) => Expr::Ident(name.to_string()),
            StaticExpr::Array { elems } => Expr::Array { elems: conv(elems) },
            StaticExpr::Paren(expr) => Expr::Paren(Box::new(Expr::from(*expr))),
            StaticExpr::Binary { lhs, op, rhs } => Expr::Binary {
                lhs: Box::new(Expr::from(*lhs)),
                op: *op,
                rhs: Box::new(Expr::from(*rhs)),
            },
            StaticExpr::Unary { op, expr } => Expr::Unary {
                op: *op,
                expr: Box::new(Expr::from(*expr)),
            },
            StaticExpr::ArrayIndexing { lhs, index } => Expr::ArrayIndexing {
                lhs: Box::new(Expr::from(*lhs)),
                index: Box::new(Expr::from(*index)),
            },
            StaticExpr::Swizzle {
                lhs,
                swizzle,
                params,
            } => Expr::Swizzle {
                lhs: Box::new(Expr::from(*lhs)),
                swizzle: swizzle.to_string(),
                params: params.as_ref().map(|ps| conv(ps)),
            },
            StaticExpr::Cast { lhs, ty } => Expr::Cast {
                lhs: Box::new(Expr::from(*lhs)),
                ty: Box::new(Type::from(*ty)),
            },
            StaticExpr::FnCall {
                path,
                type_args,
                params,
            } => Expr::FnCall {
                path: FnPath::from(path),
                type_args: conv(type_args),
                params: conv(params),
            },
            StaticExpr::Struct {
                name,
                type_args,
                fields,
            } => Expr::Struct {
                name: name.to_string(),
                type_args: conv(type_args),
                fields: conv(fields),
            },
            StaticExpr::FieldAccess { base, field } => Expr::FieldAccess {
                base: Box::new(Expr::from(*base)),
                field: field.to_string(),
            },
            StaticExpr::TypePath { ty, member } => Expr::TypePath {
                ty: ty.to_string(),
                member: member.to_string(),
            },
            StaticExpr::Reference(expr) => Expr::Reference(Box::new(Expr::from(*expr))),
            StaticExpr::ZeroValueArray { elem_type, len } => Expr::ZeroValueArray {
                elem_type: Box::new(Type::from(*elem_type)),
                len: Box::new(Expr::from(*len)),
            },
        }
    }
}

/// Borrowed mirror of [`Local`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticLocal<'a> {
    pub mutable: bool,
    pub name: &'a str,
    pub ty: Option<StaticType<'a>>,
    pub init: Option<StaticExpr<'a>>,
}

/// Owned copy of the borrowed local declaration.
impl<'a> From<&StaticLocal<'a>> for Local {
    fn from(s: &StaticLocal<'a>) -> Self {
        Local {
            mutable: s.mutable,
            name: s.name.to_string(),
            ty: s.ty.as_ref().map(Type::from),
            init: s.init.as_ref().map(Expr::from),
        }
    }
}

/// Borrowed mirror of [`ForLoop`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticForLoop<'a> {
    pub var: &'a str,
    pub var_ty: Option<StaticType<'a>>,
    pub from: StaticExpr<'a>,
    pub to: StaticExpr<'a>,
    pub inclusive: bool,
    pub body: StaticBlock<'a>,
}

/// Owned copy of the borrowed for loop.
impl<'a> From<&StaticForLoop<'a>> for ForLoop {
    fn from(s: &StaticForLoop<'a>) -> Self {
        ForLoop {
            var: s.var.to_string(),
            var_ty: s.var_ty.as_ref().map(Type::from),
            from: Expr::from(&s.from),
            to: Expr::from(&s.to),
            inclusive: s.inclusive,
            body: Block::from(&s.body),
        }
    }
}

/// Borrowed mirror of [`StmtIf`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticStmtIf<'a> {
    pub condition: StaticExpr<'a>,
    pub then_block: StaticBlock<'a>,
    pub else_branch: Option<StaticElseBranch<'a>>,
}

/// Owned copy of the borrowed if statement.
impl<'a> From<&StaticStmtIf<'a>> for StmtIf {
    fn from(s: &StaticStmtIf<'a>) -> Self {
        StmtIf {
            condition: Expr::from(&s.condition),
            then_block: Block::from(&s.then_block),
            else_branch: s.else_branch.as_ref().map(ElseBranch::from),
        }
    }
}

/// Borrowed mirror of [`ElseBranch`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticElseBranch<'a> {
    Block(StaticBlock<'a>),
    If(&'a StaticStmtIf<'a>),
}

/// Owned copy of the borrowed else branch.
impl<'a> From<&StaticElseBranch<'a>> for ElseBranch {
    fn from(s: &StaticElseBranch<'a>) -> Self {
        match s {
            StaticElseBranch::Block(block) => ElseBranch::Block(Block::from(block)),
            StaticElseBranch::If(if_stmt) => ElseBranch::If(Box::new(StmtIf::from(*if_stmt))),
        }
    }
}

/// Borrowed mirror of [`StmtSwitch`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticStmtSwitch<'a> {
    pub selector: StaticExpr<'a>,
    pub arms: &'a [StaticSwitchArm<'a>],
    pub has_explicit_default: bool,
}

/// Owned copy of the borrowed switch statement.
impl<'a> From<&StaticStmtSwitch<'a>> for StmtSwitch {
    fn from(s: &StaticStmtSwitch<'a>) -> Self {
        StmtSwitch {
            selector: Expr::from(&s.selector),
            arms: conv(s.arms),
            has_explicit_default: s.has_explicit_default,
        }
    }
}

/// Borrowed mirror of [`SwitchArm`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticSwitchArm<'a> {
    pub selectors: &'a [StaticCaseSelector<'a>],
    pub body: StaticBlock<'a>,
}

/// Owned copy of the borrowed switch arm.
impl<'a> From<&StaticSwitchArm<'a>> for SwitchArm {
    fn from(s: &StaticSwitchArm<'a>) -> Self {
        SwitchArm {
            selectors: conv(s.selectors),
            body: Block::from(&s.body),
        }
    }
}

/// Borrowed mirror of [`CaseSelector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticCaseSelector<'a> {
    Literal(StaticLit<'a>),
    Expr(StaticExpr<'a>),
    Default,
}

/// Owned copy of the borrowed case selector.
impl<'a> From<&StaticCaseSelector<'a>> for CaseSelector {
    fn from(s: &StaticCaseSelector<'a>) -> Self {
        match s {
            StaticCaseSelector::Literal(lit) => CaseSelector::Literal(Lit::from(lit)),
            StaticCaseSelector::Expr(expr) => CaseSelector::Expr(Expr::from(expr)),
            StaticCaseSelector::Default => CaseSelector::Default,
        }
    }
}

/// Borrowed mirror of [`Stmt`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticStmt<'a> {
    Local(StaticLocal<'a>),
    Const(StaticItemConst<'a>),
    Assignment {
        lhs: StaticExpr<'a>,
        rhs: StaticExpr<'a>,
    },
    CompoundAssignment {
        lhs: StaticExpr<'a>,
        op: CompoundOp,
        rhs: StaticExpr<'a>,
    },
    While {
        condition: StaticExpr<'a>,
        body: StaticBlock<'a>,
    },
    Loop {
        body: StaticBlock<'a>,
    },
    Expr {
        expr: StaticExpr<'a>,
        has_semi: bool,
    },
    If(StaticStmtIf<'a>),
    Break,
    Continue,
    Return(Option<StaticExpr<'a>>),
    For(StaticForLoop<'a>),
    Switch(StaticStmtSwitch<'a>),
    Block(StaticBlock<'a>),
    SlabCopy {
        src: StaticExpr<'a>,
        src_offset: StaticExpr<'a>,
        dest: StaticExpr<'a>,
        dest_offset: StaticExpr<'a>,
        size: StaticExpr<'a>,
    },
    Discard,
    Macro {
        name: &'a str,
        args: &'a str,
    },
}

/// Owned copy of the borrowed statement.
impl<'a> From<&StaticStmt<'a>> for Stmt {
    fn from(s: &StaticStmt<'a>) -> Self {
        match s {
            StaticStmt::Local(local) => Stmt::Local(Local::from(local)),
            StaticStmt::Const(item) => Stmt::Const(ItemConst::from(item)),
            StaticStmt::Assignment { lhs, rhs } => Stmt::Assignment {
                lhs: Expr::from(lhs),
                rhs: Expr::from(rhs),
            },
            StaticStmt::CompoundAssignment { lhs, op, rhs } => Stmt::CompoundAssignment {
                lhs: Expr::from(lhs),
                op: *op,
                rhs: Expr::from(rhs),
            },
            StaticStmt::While { condition, body } => Stmt::While {
                condition: Expr::from(condition),
                body: Block::from(body),
            },
            StaticStmt::Loop { body } => Stmt::Loop {
                body: Block::from(body),
            },
            StaticStmt::Expr { expr, has_semi } => Stmt::Expr {
                expr: Expr::from(expr),
                has_semi: *has_semi,
            },
            StaticStmt::If(if_stmt) => Stmt::If(StmtIf::from(if_stmt)),
            StaticStmt::Break => Stmt::Break,
            StaticStmt::Continue => Stmt::Continue,
            StaticStmt::Return(expr) => Stmt::Return(expr.as_ref().map(Expr::from)),
            StaticStmt::For(for_loop) => Stmt::For(ForLoop::from(for_loop)),
            StaticStmt::Switch(switch) => Stmt::Switch(StmtSwitch::from(switch)),
            StaticStmt::Block(block) => Stmt::Block(Block::from(block)),
            StaticStmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
            } => Stmt::SlabCopy {
                src: Expr::from(src),
                src_offset: Expr::from(src_offset),
                dest: Expr::from(dest),
                dest_offset: Expr::from(dest_offset),
                size: Expr::from(size),
            },
            StaticStmt::Discard => Stmt::Discard,
            StaticStmt::Macro { name, args } => Stmt::Macro {
                name: name.to_string(),
                args: args.to_string(),
            },
        }
    }
}

/// Borrowed mirror of [`ReturnType`].
#[derive(Clone, Debug, PartialEq)]
pub enum StaticReturnType<'a> {
    Default,
    Type {
        annotation: ReturnTypeAnnotation,
        ty: StaticType<'a>,
    },
}

/// Owned copy of the borrowed return type.
impl<'a> From<&StaticReturnType<'a>> for ReturnType {
    fn from(s: &StaticReturnType<'a>) -> Self {
        match s {
            StaticReturnType::Default => ReturnType::Default,
            StaticReturnType::Type { annotation, ty } => ReturnType::Type {
                annotation: annotation.clone(),
                ty: Type::from(ty),
            },
        }
    }
}

/// Borrowed mirror of [`FnArg`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticFnArg<'a> {
    pub inter_stage_io: &'a [InterStageIo],
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed function argument.
impl<'a> From<&StaticFnArg<'a>> for FnArg {
    fn from(s: &StaticFnArg<'a>) -> Self {
        FnArg {
            inter_stage_io: s.inter_stage_io.to_vec(),
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemConst`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemConst<'a> {
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub expr: StaticExpr<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed const item.
impl<'a> From<&StaticItemConst<'a>> for ItemConst {
    fn from(s: &StaticItemConst<'a>) -> Self {
        ItemConst {
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            expr: Expr::from(&s.expr),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemUniform`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemUniform<'a> {
    pub group: u32,
    pub binding: u32,
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed uniform linkage.
impl<'a> From<&StaticItemUniform<'a>> for ItemUniform {
    fn from(s: &StaticItemUniform<'a>) -> Self {
        ItemUniform {
            group: s.group,
            binding: s.binding,
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemStorage`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemStorage<'a> {
    pub group: u32,
    pub binding: u32,
    pub access: StorageAccess,
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed storage linkage.
impl<'a> From<&StaticItemStorage<'a>> for ItemStorage {
    fn from(s: &StaticItemStorage<'a>) -> Self {
        ItemStorage {
            group: s.group,
            binding: s.binding,
            access: s.access,
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemWorkgroup`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemWorkgroup<'a> {
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed workgroup declaration.
impl<'a> From<&StaticItemWorkgroup<'a>> for ItemWorkgroup {
    fn from(s: &StaticItemWorkgroup<'a>) -> Self {
        ItemWorkgroup {
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`Filterability`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StaticFilterability<'a> {
    /// A concrete binding-layout value.
    Concrete(bool),
    /// A binding-owned const parameter to resolve during instantiation.
    Param(&'a str),
}

impl From<StaticFilterability<'_>> for Filterability {
    fn from(value: StaticFilterability<'_>) -> Self {
        match value {
            StaticFilterability::Concrete(value) => Self::Concrete(value),
            StaticFilterability::Param(name) => Self::Param(name.to_string()),
        }
    }
}

/// Borrowed mirror of [`ItemSampler`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemSampler<'a> {
    pub group: u32,
    pub binding: u32,
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub filterable: StaticFilterability<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed sampler linkage.
impl<'a> From<&StaticItemSampler<'a>> for ItemSampler {
    fn from(s: &StaticItemSampler<'a>) -> Self {
        ItemSampler {
            group: s.group,
            binding: s.binding,
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            filterable: Filterability::from(s.filterable),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemTexture`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemTexture<'a> {
    pub group: u32,
    pub binding: u32,
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub filterable: StaticFilterability<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed texture linkage.
impl<'a> From<&StaticItemTexture<'a>> for ItemTexture {
    fn from(s: &StaticItemTexture<'a>) -> Self {
        ItemTexture {
            group: s.group,
            binding: s.binding,
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            filterable: Filterability::from(s.filterable),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemFn`].
///
/// Not `Copy`: it holds the non-`Copy` reused [`FnAttrs`] inline.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticItemFn<'a> {
    pub type_params: &'a [&'a str],
    pub const_params: &'a [&'a str],
    pub fn_attrs: FnAttrs,
    pub name: &'a str,
    pub inputs: &'a [StaticFnArg<'a>],
    pub return_type: StaticReturnType<'a>,
    pub block: StaticBlock<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed function definition.
///
/// Restricted to `'static`: the owned name is a `Cow<'static, str>`
/// and must stay `Borrowed` for fresh functions, which linkage code
/// relies on when extracting `&'static str` names without copying.
impl From<&StaticItemFn<'static>> for ItemFn {
    fn from(s: &StaticItemFn<'static>) -> Self {
        ItemFn {
            type_params: s.type_params.iter().map(|p| p.to_string()).collect(),
            const_params: s.const_params.iter().map(|p| p.to_string()).collect(),
            fn_attrs: s.fn_attrs.clone(),
            name: Cow::Borrowed(s.name),
            inputs: conv(s.inputs),
            return_type: ReturnType::from(&s.return_type),
            block: Block::from(&s.block),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`Field`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticField<'a> {
    pub inter_stage_io: &'a [InterStageIo],
    pub name: &'a str,
    pub ty: StaticType<'a>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed struct field.
impl<'a> From<&StaticField<'a>> for Field {
    fn from(s: &StaticField<'a>) -> Self {
        Field {
            inter_stage_io: s.inter_stage_io.to_vec(),
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemStruct`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemStruct<'a> {
    pub type_params: &'a [&'a str],
    pub const_params: &'a [&'a str],
    pub name: &'a str,
    pub fields: &'a [StaticField<'a>],
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed struct definition.
impl<'a> From<&StaticItemStruct<'a>> for ItemStruct {
    fn from(s: &StaticItemStruct<'a>) -> Self {
        ItemStruct {
            type_params: s.type_params.iter().map(|p| p.to_string()).collect(),
            const_params: s.const_params.iter().map(|p| p.to_string()).collect(),
            name: s.name.to_string(),
            fields: conv(s.fields),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemTypeAlias`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemTypeAlias<'a> {
    pub name: &'a str,
    pub ty: StaticType<'a>,
}

/// Owned copy of the borrowed associated type alias.
impl<'a> From<&StaticItemTypeAlias<'a>> for ItemTypeAlias {
    fn from(s: &StaticItemTypeAlias<'a>) -> Self {
        ItemTypeAlias {
            name: s.name.to_string(),
            ty: Type::from(&s.ty),
        }
    }
}

/// Borrowed mirror of [`ImplItem`].
///
/// Not `Copy`: the `Fn` variant holds the non-`Copy`
/// [`StaticItemFn`] inline.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticImplItem<'a> {
    Fn(StaticItemFn<'a>),
    Const(StaticItemConst<'a>),
    Type(StaticItemTypeAlias<'a>),
}

/// Owned copy of the borrowed impl item.
impl From<&StaticImplItem<'static>> for ImplItem {
    fn from(s: &StaticImplItem<'static>) -> Self {
        match s {
            StaticImplItem::Fn(item_fn) => ImplItem::Fn(ItemFn::from(item_fn)),
            StaticImplItem::Const(item_const) => ImplItem::Const(ItemConst::from(item_const)),
            StaticImplItem::Type(item_alias) => ImplItem::Type(ItemTypeAlias::from(item_alias)),
        }
    }
}

/// Borrowed mirror of [`ItemImpl`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemImpl<'a> {
    pub type_params: &'a [&'a str],
    pub const_params: &'a [&'a str],
    pub self_ty: &'a str,
    pub items: &'a [StaticImplItem<'a>],
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed impl block.
///
/// Restricted to `'static` because the contained functions are (see
/// [`From<&StaticItemFn<'static>> for ItemFn`]).
impl From<&StaticItemImpl<'static>> for ItemImpl {
    fn from(s: &StaticItemImpl<'static>) -> Self {
        ItemImpl {
            type_params: s.type_params.iter().map(|p| p.to_string()).collect(),
            const_params: s.const_params.iter().map(|p| p.to_string()).collect(),
            self_ty: s.self_ty.to_string(),
            items: conv(s.items),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`EnumVariant`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticEnumVariant<'a> {
    pub name: &'a str,
    pub discriminant: Option<u32>,
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed enum variant.
impl<'a> From<&StaticEnumVariant<'a>> for EnumVariant {
    fn from(s: &StaticEnumVariant<'a>) -> Self {
        EnumVariant {
            name: s.name.to_string(),
            discriminant: s.discriminant,
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`ItemEnum`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticItemEnum<'a> {
    pub name: &'a str,
    pub variants: &'a [StaticEnumVariant<'a>],
    pub attrs: &'a [StaticAttribute<'a>],
}

/// Owned copy of the borrowed enum definition.
impl<'a> From<&StaticItemEnum<'a>> for ItemEnum {
    fn from(s: &StaticItemEnum<'a>) -> Self {
        ItemEnum {
            name: s.name.to_string(),
            variants: conv(s.variants),
            attrs: conv(s.attrs),
        }
    }
}

/// Borrowed mirror of [`Item`].
///
/// Not `Copy`: the `Fn` and `Impl` variants hold non-`Copy` mirrors
/// inline.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticItem<'a> {
    Const(StaticItemConst<'a>),
    Uniform(StaticItemUniform<'a>),
    Storage(StaticItemStorage<'a>),
    Workgroup(StaticItemWorkgroup<'a>),
    Sampler(StaticItemSampler<'a>),
    Texture(StaticItemTexture<'a>),
    Fn(StaticItemFn<'a>),
    Struct(StaticItemStruct<'a>),
    Impl(StaticItemImpl<'a>),
    Enum(StaticItemEnum<'a>),
}

/// Owned copy of the borrowed module item.
///
/// Restricted to `'static` because the `Fn` and `Impl` variants are
/// (see [`From<&StaticItemFn<'static>> for ItemFn`]).
impl From<&StaticItem<'static>> for Item {
    fn from(s: &StaticItem<'static>) -> Self {
        match s {
            StaticItem::Const(item) => Item::Const(ItemConst::from(item)),
            StaticItem::Uniform(item) => Item::Uniform(ItemUniform::from(item)),
            StaticItem::Storage(item) => Item::Storage(ItemStorage::from(item)),
            StaticItem::Workgroup(item) => Item::Workgroup(ItemWorkgroup::from(item)),
            StaticItem::Sampler(item) => Item::Sampler(ItemSampler::from(item)),
            StaticItem::Texture(item) => Item::Texture(ItemTexture::from(item)),
            StaticItem::Fn(item) => Item::Fn(ItemFn::from(item)),
            StaticItem::Struct(item) => Item::Struct(ItemStruct::from(item)),
            StaticItem::Impl(item) => Item::Impl(ItemImpl::from(item)),
            StaticItem::Enum(item) => Item::Enum(ItemEnum::from(item)),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Round-trip coverage for the borrowed mirror.
    //!
    //! Strategy: one owned kitchen-sink fixture covering every mirrored
    //! variant (and every variant of the reused fieldless enums),
    //! converted to the mirror by the test-side `to_static_*` helpers
    //! (the in-memory analog of the macro's literal emission), then
    //! converted back with the library `From` impls and compared for
    //! exact equality. Any field dropped, renamed, or mistranslated by
    //! either direction fails the assert.
    //!
    //! The `to_static_*` helpers leak their allocations (`Box::leak`)
    //! to produce the `'static` data the `'static`-restricted
    //! converters require. Tests are short-lived; leaks die with the
    //! test process.

    use super::*;

    // ===== exhaustive lists for the reused enums =====

    const ALL_BIN_OPS: &[BinOp] = &[
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Lt,
        BinOp::Le,
        BinOp::Gt,
        BinOp::Ge,
        BinOp::And,
        BinOp::Or,
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
    ];
    const ALL_COMPOUND_OPS: &[CompoundOp] = &[
        CompoundOp::AddAssign,
        CompoundOp::SubAssign,
        CompoundOp::MulAssign,
        CompoundOp::DivAssign,
        CompoundOp::RemAssign,
        CompoundOp::BitAndAssign,
        CompoundOp::BitOrAssign,
        CompoundOp::BitXorAssign,
        CompoundOp::ShlAssign,
        CompoundOp::ShrAssign,
    ];
    const ALL_UN_OPS: &[UnOp] = &[UnOp::Not, UnOp::Complement, UnOp::Neg, UnOp::Deref];
    const ALL_SCALARS: &[ScalarType] = &[
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::F32,
        ScalarType::Bool,
    ];
    const ALL_ADDRESS_SPACES: &[AddressSpace] = &[
        AddressSpace::Function,
        AddressSpace::Private,
        AddressSpace::Workgroup,
    ];
    const ALL_TEXTURE_KINDS: &[TextureKind] = &[
        TextureKind::Texture1D,
        TextureKind::Texture2D,
        TextureKind::Texture2DArray,
        TextureKind::Texture3D,
        TextureKind::TextureCube,
        TextureKind::TextureCubeArray,
        TextureKind::TextureMultisampled2D,
    ];
    const ALL_DEPTH_KINDS: &[TextureDepthKind] = &[
        TextureDepthKind::Depth2D,
        TextureDepthKind::Depth2DArray,
        TextureDepthKind::DepthCube,
        TextureDepthKind::DepthCubeArray,
        TextureDepthKind::DepthMultisampled2D,
    ];
    const ALL_STORAGE_KINDS: &[TextureStorageKind] = &[
        TextureStorageKind::Storage1D,
        TextureStorageKind::Storage2D,
        TextureStorageKind::Storage2DArray,
        TextureStorageKind::Storage3D,
    ];
    const ALL_STORAGE_TEXTURE_ACCESS: &[StorageTextureAccess] = &[
        StorageTextureAccess::Read,
        StorageTextureAccess::Write,
        StorageTextureAccess::ReadWrite,
    ];
    const ALL_TEXEL_FORMATS: &[TexelFormat] = &[
        TexelFormat::Rgba8unorm,
        TexelFormat::Rgba8snorm,
        TexelFormat::Rgba8uint,
        TexelFormat::Rgba8sint,
        TexelFormat::Rgba16uint,
        TexelFormat::Rgba16sint,
        TexelFormat::Rgba16float,
        TexelFormat::R32uint,
        TexelFormat::R32sint,
        TexelFormat::R32float,
        TexelFormat::Rg32uint,
        TexelFormat::Rg32sint,
        TexelFormat::Rg32float,
        TexelFormat::Rgba32uint,
        TexelFormat::Rgba32sint,
        TexelFormat::Rgba32float,
        TexelFormat::Bgra8unorm,
        TexelFormat::Rgba16unorm,
        TexelFormat::Rgba16snorm,
        TexelFormat::Rg8unorm,
        TexelFormat::Rg8snorm,
        TexelFormat::Rg8uint,
        TexelFormat::Rg8sint,
        TexelFormat::Rg16unorm,
        TexelFormat::Rg16snorm,
        TexelFormat::Rg16uint,
        TexelFormat::Rg16sint,
        TexelFormat::Rg16float,
        TexelFormat::R8unorm,
        TexelFormat::R8snorm,
        TexelFormat::R8uint,
        TexelFormat::R8sint,
        TexelFormat::R16unorm,
        TexelFormat::R16snorm,
        TexelFormat::R16uint,
        TexelFormat::R16sint,
        TexelFormat::R16float,
        TexelFormat::Rgb10a2unorm,
        TexelFormat::Rgb10a2uint,
        TexelFormat::Rg11b10ufloat,
    ];
    const ALL_BUILT_INS: &[BuiltIn] = &[
        BuiltIn::VertexIndex,
        BuiltIn::InstanceIndex,
        BuiltIn::Position,
        BuiltIn::FrontFacing,
        BuiltIn::FragDepth,
        BuiltIn::SampleIndex,
        BuiltIn::SampleMask,
        BuiltIn::LocalInvocationId,
        BuiltIn::LocalInvocationIndex,
        BuiltIn::GlobalInvocationId,
        BuiltIn::WorkgroupId,
        BuiltIn::NumWorkgroups,
        BuiltIn::SubgroupInvocationId,
        BuiltIn::SubgroupSize,
        BuiltIn::PrimitiveIndex,
        BuiltIn::SubgroupId,
        BuiltIn::NumSubgroups,
    ];
    const ALL_INTERP_TYPES: &[InterpolationType] = &[
        InterpolationType::Perspective,
        InterpolationType::Linear,
        InterpolationType::Flat,
    ];
    const ALL_INTERP_SAMPLINGS: &[InterpolationSampling] = &[
        InterpolationSampling::Center,
        InterpolationSampling::Centroid,
        InterpolationSampling::Sample,
        InterpolationSampling::First,
        InterpolationSampling::Either,
    ];

    // ===== small owned-fixture helpers =====

    fn attr(path: &str, args: &[&str]) -> Attribute {
        Attribute {
            path: path.into(),
            args: args.iter().map(|arg| arg.to_string()).collect(),
        }
    }

    fn block(stmts: Vec<Stmt>) -> Block {
        Block { stmts }
    }

    fn scalar(scalar: ScalarType) -> Type {
        Type::Scalar(scalar)
    }

    fn vec4f() -> Type {
        Type::Vector {
            elements: 4,
            scalar_ty: Some(ScalarType::F32),
        }
    }

    fn ident(name: &str) -> Expr {
        Expr::Ident(name.into())
    }

    fn lit_int(digits: &str, suffix: &str) -> Expr {
        Expr::Lit(Lit::Int {
            digits: digits.into(),
            suffix: suffix.into(),
        })
    }

    fn lit_float(text: &str) -> Expr {
        Expr::Lit(Lit::Float { text: text.into() })
    }

    fn lit_bool(value: bool) -> Expr {
        Expr::Lit(Lit::Bool(value))
    }

    // ===== the owned kitchen-sink fixture =====

    /// Every item kind, every statement, every expression, every type,
    /// and every reused-enum variant, in one (semantically silly but
    /// structurally complete) module. No validation runs on this; it
    /// exists to exercise the mirror and its converters.
    fn kitchen_sink_module() -> Module {
        Module {
            name: "kitchen_sink",
            attrs: vec![attr("wgsl", &["doc"])],
            items: vec![
                Item::Const(ItemConst {
                    name: "MAX".into(),
                    ty: scalar(ScalarType::U32),
                    expr: lit_int("255", "u32"),
                    attrs: vec![attr("cfg", &["test"])],
                }),
                Item::Uniform(ItemUniform {
                    group: 0,
                    binding: 1,
                    name: "FRAME".into(),
                    ty: Type::Struct {
                        name: "Frame".into(),
                        type_args: vec![],
                    },
                    attrs: vec![],
                }),
                Item::Storage(ItemStorage {
                    group: 0,
                    binding: 2,
                    access: StorageAccess::Read,
                    name: "RO_BINS".into(),
                    ty: Type::RuntimeArray {
                        elem: Box::new(scalar(ScalarType::U32)),
                    },
                    attrs: vec![],
                }),
                Item::Storage(ItemStorage {
                    group: 0,
                    binding: 3,
                    access: StorageAccess::ReadWrite,
                    name: "RW_BINS".into(),
                    ty: Type::RuntimeArray {
                        elem: Box::new(scalar(ScalarType::F32)),
                    },
                    attrs: vec![],
                }),
                Item::Workgroup(ItemWorkgroup {
                    name: "TILE".into(),
                    ty: Type::Array {
                        elem: Box::new(scalar(ScalarType::F32)),
                        len: lit_int("16", "u32"),
                    },
                    attrs: vec![],
                }),
                Item::Sampler(ItemSampler {
                    group: 0,
                    binding: 4,
                    name: "SMP".into(),
                    ty: Type::Sampler,
                    filterable: false.into(),
                    attrs: vec![],
                }),
                Item::Texture(ItemTexture {
                    group: 0,
                    binding: 5,
                    name: "TEX".into(),
                    ty: Type::Texture {
                        kind: TextureKind::Texture2D,
                        sampled_type: ScalarType::F32,
                    },
                    filterable: true.into(),
                    attrs: vec![],
                }),
                Item::Fn(kitchen_fn()),
                Item::Fn(small_fn(
                    "vertex_main",
                    FnAttrs::Vertex,
                    ReturnType::Type {
                        annotation: ReturnTypeAnnotation::Location(0),
                        ty: vec4f(),
                    },
                )),
                Item::Fn(small_fn(
                    "fragment_main",
                    FnAttrs::Fragment,
                    ReturnType::Type {
                        annotation: ReturnTypeAnnotation::DefaultBuiltInPosition,
                        ty: vec4f(),
                    },
                )),
                Item::Fn(small_fn(
                    "helper",
                    FnAttrs::None,
                    ReturnType::Type {
                        annotation: ReturnTypeAnnotation::DefaultLocation,
                        ty: scalar(ScalarType::F32),
                    },
                )),
                Item::Fn(small_fn(
                    "typed",
                    FnAttrs::None,
                    ReturnType::Type {
                        annotation: ReturnTypeAnnotation::None,
                        ty: scalar(ScalarType::U32),
                    },
                )),
                Item::Fn(small_fn("void", FnAttrs::None, ReturnType::Default)),
                Item::Struct(struct_item()),
                Item::Impl(impl_item()),
                Item::Enum(enum_item()),
            ],
        }
    }

    fn small_fn(name: &'static str, fn_attrs: FnAttrs, return_type: ReturnType) -> ItemFn {
        ItemFn {
            type_params: vec![],
            const_params: vec![],
            fn_attrs,
            name: Cow::Borrowed(name),
            inputs: vec![],
            return_type,
            block: block(vec![]),
            attrs: vec![],
        }
    }

    fn kitchen_fn() -> ItemFn {
        ItemFn {
            type_params: vec!["T".into()],
            const_params: vec!["N".into()],
            fn_attrs: FnAttrs::Compute {
                workgroup_size: WorkgroupSize {
                    x: 8,
                    y: Some(4),
                    z: None,
                },
            },
            name: "kitchen".into(),
            inputs: kitchen_inputs(),
            return_type: ReturnType::Type {
                annotation: ReturnTypeAnnotation::BuiltIn(BuiltIn::Position),
                ty: vec4f(),
            },
            block: kitchen_block(),
            attrs: vec![],
        }
    }

    fn kitchen_inputs() -> Vec<FnArg> {
        let mut inputs: Vec<FnArg> = ALL_BUILT_INS
            .iter()
            .enumerate()
            .map(|(i, builtin)| FnArg {
                inter_stage_io: vec![InterStageIo::BuiltIn(*builtin)],
                name: format!("b{i}"),
                ty: scalar(ScalarType::U32),
                attrs: vec![],
            })
            .collect();
        inputs.push(FnArg {
            inter_stage_io: vec![
                InterStageIo::Location(0),
                InterStageIo::Interpolate(Interpolate {
                    ty: InterpolationType::Flat,
                    sampling: None,
                }),
            ],
            name: "loc".into(),
            ty: vec4f(),
            attrs: vec![attr("cfg", &["test"])],
        });
        inputs.push(FnArg {
            inter_stage_io: vec![InterStageIo::BlendSrc(1)],
            name: "blend".into(),
            ty: vec4f(),
            attrs: vec![],
        });
        inputs.push(FnArg {
            inter_stage_io: vec![InterStageIo::Invariant],
            name: "inv".into(),
            ty: vec4f(),
            attrs: vec![],
        });
        inputs
    }

    fn all_types() -> Vec<Type> {
        let mut types: Vec<Type> = ALL_SCALARS
            .iter()
            .map(|scalar| Type::Scalar(*scalar))
            .chain([
                Type::Vector {
                    elements: 3,
                    scalar_ty: Some(ScalarType::F32),
                },
                Type::Vector {
                    elements: 4,
                    scalar_ty: None,
                },
                Type::Matrix {
                    columns: 4,
                    rows: 4,
                    scalar_ty: Some(ScalarType::F32),
                },
                Type::Matrix {
                    columns: 2,
                    rows: 3,
                    scalar_ty: None,
                },
                Type::Array {
                    elem: Box::new(scalar(ScalarType::F32)),
                    len: lit_int("4", "u32"),
                },
                Type::RuntimeArray {
                    elem: Box::new(scalar(ScalarType::U32)),
                },
                Type::Atomic {
                    elem: Box::new(scalar(ScalarType::U32)),
                },
                Type::Struct {
                    name: "Frame".into(),
                    type_args: vec![],
                },
                Type::Struct {
                    name: "Pair".into(),
                    type_args: vec![scalar(ScalarType::F32)],
                },
                Type::Sampler,
                Type::SamplerComparison,
                Type::TypeParam { name: "T".into() },
                Type::Phantom {
                    elem: Box::new(Type::TypeParam { name: "T".into() }),
                },
                Type::AssocType {
                    ty: Box::new(Type::TypeParam {
                        name: "Self".into(),
                    }),
                    member: "Array".into(),
                },
            ])
            .collect();
        for space in ALL_ADDRESS_SPACES {
            types.push(Type::Ptr {
                address_space: *space,
                elem: Box::new(scalar(ScalarType::Bool)),
            });
        }
        for kind in ALL_TEXTURE_KINDS {
            types.push(Type::Texture {
                kind: *kind,
                sampled_type: ScalarType::F32,
            });
        }
        for kind in ALL_DEPTH_KINDS {
            types.push(Type::TextureDepth { kind: *kind });
        }
        for kind in ALL_STORAGE_KINDS {
            types.push(Type::TextureStorage {
                kind: *kind,
                format: TexelFormat::Rgba8unorm,
                access: StorageTextureAccess::Read,
            });
        }
        for access in ALL_STORAGE_TEXTURE_ACCESS {
            types.push(Type::TextureStorage {
                kind: TextureStorageKind::Storage2D,
                format: TexelFormat::R32float,
                access: *access,
            });
        }
        for format in ALL_TEXEL_FORMATS {
            types.push(Type::TextureStorage {
                kind: TextureStorageKind::Storage3D,
                format: *format,
                access: StorageTextureAccess::ReadWrite,
            });
        }
        types
    }

    fn kitchen_block() -> Block {
        let mut stmts = vec![
            Stmt::Local(Local {
                mutable: true,
                name: "v".into(),
                ty: Some(vec4f()),
                init: Some(ident("a")),
            }),
            Stmt::Local(Local {
                mutable: false,
                name: "w".into(),
                ty: None,
                init: Some(lit_int("1", "u32")),
            }),
            Stmt::Local(Local {
                mutable: true,
                name: "u".into(),
                ty: Some(scalar(ScalarType::U32)),
                init: None,
            }),
        ];
        // Every type variant as a typed local with no initializer.
        for (i, ty) in all_types().into_iter().enumerate() {
            stmts.push(Stmt::Local(Local {
                mutable: false,
                name: format!("t{i}"),
                ty: Some(ty),
                init: None,
            }));
        }
        // In-body const item.
        stmts.push(Stmt::Const(ItemConst {
            name: "INNER".into(),
            ty: scalar(ScalarType::F32),
            expr: lit_float("1.5"),
            attrs: vec![],
        }));
        // Assignment plus every binary operator.
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: ident("w"),
        });
        for op in ALL_BIN_OPS {
            stmts.push(Stmt::Assignment {
                lhs: ident("v"),
                rhs: Expr::Binary {
                    lhs: Box::new(ident("v")),
                    op: *op,
                    rhs: Box::new(lit_int("2", "")),
                },
            });
        }
        // Every unary operator.
        for op in ALL_UN_OPS {
            stmts.push(Stmt::Assignment {
                lhs: ident("v"),
                rhs: Expr::Unary {
                    op: *op,
                    expr: Box::new(ident("u")),
                },
            });
        }
        // Every compound-assignment operator.
        for op in ALL_COMPOUND_OPS {
            stmts.push(Stmt::CompoundAssignment {
                lhs: ident("v"),
                op: *op,
                rhs: lit_float("1.0"),
            });
        }
        // The remaining expression forms.
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Array {
                elems: vec![ident("v"), lit_bool(true)],
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Paren(Box::new(ident("v"))),
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::ArrayIndexing {
                lhs: Box::new(ident("arr")),
                index: Box::new(lit_int("0", "u32")),
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Swizzle {
                lhs: Box::new(ident("v")),
                swizzle: "xyz".into(),
                params: None,
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Swizzle {
                lhs: Box::new(ident("v")),
                swizzle: "x".into(),
                params: Some(vec![lit_int("0", "u32")]),
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Cast {
                lhs: Box::new(ident("u")),
                ty: Box::new(scalar(ScalarType::F32)),
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::FnCall {
                path: FnPath::Ident("max".into()),
                type_args: vec![],
                params: vec![ident("v"), ident("w")],
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::FnCall {
                path: FnPath::TypeMethod {
                    ty: "MyType".into(),
                    method: "get".into(),
                },
                type_args: vec![scalar(ScalarType::F32)],
                params: vec![],
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Struct {
                name: "Pair".into(),
                type_args: vec![scalar(ScalarType::F32)],
                fields: vec![FieldValue {
                    member: "x".into(),
                    expr: lit_float("1.0"),
                }],
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::FieldAccess {
                base: Box::new(ident("p")),
                field: "x".into(),
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::TypePath {
                ty: "MyType".into(),
                member: "COUNT".into(),
            },
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::Reference(Box::new(ident("v"))),
        });
        stmts.push(Stmt::Assignment {
            lhs: ident("v"),
            rhs: Expr::ZeroValueArray {
                elem_type: Box::new(scalar(ScalarType::F32)),
                len: Box::new(lit_int("4", "u32")),
            },
        });
        // Control flow.
        stmts.push(Stmt::While {
            condition: lit_bool(true),
            body: block(vec![Stmt::Break]),
        });
        stmts.push(Stmt::Loop {
            body: block(vec![Stmt::Continue]),
        });
        stmts.push(Stmt::Expr {
            expr: ident("v"),
            has_semi: true,
        });
        stmts.push(Stmt::Expr {
            expr: ident("v"),
            has_semi: false,
        });
        stmts.push(Stmt::If(StmtIf {
            condition: lit_bool(true),
            then_block: block(vec![Stmt::Break]),
            else_branch: Some(ElseBranch::Block(block(vec![Stmt::Continue]))),
        }));
        stmts.push(Stmt::If(StmtIf {
            condition: lit_bool(false),
            then_block: block(vec![]),
            else_branch: Some(ElseBranch::If(Box::new(StmtIf {
                condition: lit_bool(true),
                then_block: block(vec![Stmt::Return(Some(ident("v")))]),
                else_branch: None,
            }))),
        }));
        stmts.push(Stmt::If(StmtIf {
            condition: lit_bool(true),
            then_block: block(vec![]),
            else_branch: None,
        }));
        stmts.push(Stmt::Return(None));
        stmts.push(Stmt::Return(Some(ident("v"))));
        stmts.push(Stmt::For(ForLoop {
            var: "i".into(),
            var_ty: Some(scalar(ScalarType::U32)),
            from: lit_int("0", "u32"),
            to: lit_int("4", "u32"),
            inclusive: true,
            body: block(vec![Stmt::Assignment {
                lhs: ident("v"),
                rhs: ident("v"),
            }]),
        }));
        stmts.push(Stmt::Switch(StmtSwitch {
            selector: ident("u"),
            arms: vec![
                SwitchArm {
                    selectors: vec![CaseSelector::Literal(Lit::Int {
                        digits: "0".into(),
                        suffix: "u32".into(),
                    })],
                    body: block(vec![Stmt::Break]),
                },
                SwitchArm {
                    selectors: vec![CaseSelector::Expr(ident("K"))],
                    body: block(vec![]),
                },
                SwitchArm {
                    selectors: vec![CaseSelector::Default],
                    body: block(vec![Stmt::Discard]),
                },
            ],
            has_explicit_default: true,
        }));
        stmts.push(Stmt::Block(block(vec![Stmt::Continue])));
        stmts.push(Stmt::SlabCopy {
            src: ident("src"),
            src_offset: ident("so"),
            dest: ident("dest"),
            dest_offset: ident("doff"),
            size: ident("n"),
        });
        stmts.push(Stmt::Discard);
        stmts.push(Stmt::Macro {
            name: "my_macro".into(),
            args: "1, 2".into(),
        });
        block(stmts)
    }

    fn struct_item() -> ItemStruct {
        let mut fields = vec![
            Field {
                inter_stage_io: vec![InterStageIo::Location(1)],
                name: "x".into(),
                ty: scalar(ScalarType::F32),
                attrs: vec![attr("cfg", &["test"])],
            },
            Field {
                inter_stage_io: vec![],
                name: "phantom".into(),
                ty: Type::Phantom {
                    elem: Box::new(Type::TypeParam { name: "T".into() }),
                },
                attrs: vec![],
            },
            Field {
                inter_stage_io: vec![],
                name: "arr".into(),
                ty: Type::AssocType {
                    ty: Box::new(Type::TypeParam {
                        name: "Self".into(),
                    }),
                    member: "Array".into(),
                },
                attrs: vec![],
            },
        ];
        // Every interpolation type / sampling combination.
        for (i, interp_ty) in ALL_INTERP_TYPES.iter().enumerate() {
            for (j, sampling) in ALL_INTERP_SAMPLINGS.iter().enumerate() {
                fields.push(Field {
                    inter_stage_io: vec![InterStageIo::Interpolate(Interpolate {
                        ty: *interp_ty,
                        sampling: Some(*sampling),
                    })],
                    name: format!("f{i}{j}"),
                    ty: vec4f(),
                    attrs: vec![],
                });
            }
        }
        ItemStruct {
            type_params: vec!["T".into(), "U".into()],
            const_params: vec!["N".into()],
            name: "Pair".into(),
            fields,
            attrs: vec![attr("repr", &["C"])],
        }
    }

    fn impl_item() -> ItemImpl {
        ItemImpl {
            type_params: vec!["T".into()],
            const_params: vec![],
            self_ty: "Pair".into(),
            items: vec![
                ImplItem::Fn(small_fn("get", FnAttrs::None, ReturnType::Default)),
                ImplItem::Const(ItemConst {
                    name: "COUNT".into(),
                    ty: scalar(ScalarType::U32),
                    expr: lit_int("4", "u32"),
                    attrs: vec![],
                }),
                ImplItem::Type(ItemTypeAlias {
                    name: "Array".into(),
                    ty: Type::Array {
                        elem: Box::new(scalar(ScalarType::F32)),
                        len: lit_int("4", "u32"),
                    },
                }),
            ],
            attrs: vec![],
        }
    }

    fn enum_item() -> ItemEnum {
        ItemEnum {
            name: "Color".into(),
            variants: vec![
                EnumVariant {
                    name: "Red".into(),
                    discriminant: None,
                    attrs: vec![],
                },
                EnumVariant {
                    name: "Green".into(),
                    discriminant: Some(1),
                    attrs: vec![attr("default", &[])],
                },
            ],
            attrs: vec![],
        }
    }

    // ===== test-side owned -> borrowed converters (leaking) =====

    fn leak_str(s: &str) -> &'static str {
        Box::leak(s.to_string().into_boxed_str())
    }

    fn leak_box<T>(value: T) -> &'static T {
        Box::leak(Box::new(value))
    }

    fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
        Box::leak(values.into_boxed_slice())
    }

    fn to_static_attribute(a: &Attribute) -> StaticAttribute<'static> {
        StaticAttribute {
            path: leak_str(&a.path),
            args: leak_slice(a.args.iter().map(|arg| leak_str(arg)).collect()),
        }
    }

    fn to_static_module(m: &Module) -> StaticModule<'static> {
        StaticModule {
            name: m.name,
            items: leak_slice(m.items.iter().map(to_static_item).collect()),
            attrs: leak_slice(m.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item(item: &Item) -> StaticItem<'static> {
        match item {
            Item::Const(c) => StaticItem::Const(to_static_item_const(c)),
            Item::Uniform(u) => StaticItem::Uniform(to_static_item_uniform(u)),
            Item::Storage(s) => StaticItem::Storage(to_static_item_storage(s)),
            Item::Workgroup(w) => StaticItem::Workgroup(to_static_item_workgroup(w)),
            Item::Sampler(s) => StaticItem::Sampler(to_static_item_sampler(s)),
            Item::Texture(t) => StaticItem::Texture(to_static_item_texture(t)),
            Item::Fn(f) => StaticItem::Fn(to_static_item_fn(f)),
            Item::Struct(s) => StaticItem::Struct(to_static_item_struct(s)),
            Item::Impl(i) => StaticItem::Impl(to_static_item_impl(i)),
            Item::Enum(e) => StaticItem::Enum(to_static_item_enum(e)),
        }
    }

    fn to_static_item_const(c: &ItemConst) -> StaticItemConst<'static> {
        StaticItemConst {
            name: leak_str(&c.name),
            ty: to_static_type(&c.ty),
            expr: to_static_expr(&c.expr),
            attrs: leak_slice(c.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_uniform(u: &ItemUniform) -> StaticItemUniform<'static> {
        StaticItemUniform {
            group: u.group,
            binding: u.binding,
            name: leak_str(&u.name),
            ty: to_static_type(&u.ty),
            attrs: leak_slice(u.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_storage(s: &ItemStorage) -> StaticItemStorage<'static> {
        StaticItemStorage {
            group: s.group,
            binding: s.binding,
            access: s.access,
            name: leak_str(&s.name),
            ty: to_static_type(&s.ty),
            attrs: leak_slice(s.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_workgroup(w: &ItemWorkgroup) -> StaticItemWorkgroup<'static> {
        StaticItemWorkgroup {
            name: leak_str(&w.name),
            ty: to_static_type(&w.ty),
            attrs: leak_slice(w.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_filterability(filterable: &Filterability) -> StaticFilterability<'static> {
        match filterable {
            Filterability::Concrete(value) => StaticFilterability::Concrete(*value),
            Filterability::Param(name) => StaticFilterability::Param(leak_str(name)),
        }
    }

    fn to_static_item_sampler(s: &ItemSampler) -> StaticItemSampler<'static> {
        StaticItemSampler {
            group: s.group,
            binding: s.binding,
            name: leak_str(&s.name),
            ty: to_static_type(&s.ty),
            filterable: to_static_filterability(&s.filterable),
            attrs: leak_slice(s.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_texture(t: &ItemTexture) -> StaticItemTexture<'static> {
        StaticItemTexture {
            group: t.group,
            binding: t.binding,
            name: leak_str(&t.name),
            ty: to_static_type(&t.ty),
            filterable: to_static_filterability(&t.filterable),
            attrs: leak_slice(t.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_fn(f: &ItemFn) -> StaticItemFn<'static> {
        StaticItemFn {
            type_params: leak_slice(f.type_params.iter().map(|p| leak_str(p)).collect()),
            const_params: leak_slice(f.const_params.iter().map(|p| leak_str(p)).collect()),
            fn_attrs: f.fn_attrs.clone(),
            name: leak_str(f.name.as_ref()),
            inputs: leak_slice(f.inputs.iter().map(to_static_fn_arg).collect()),
            return_type: to_static_return_type(&f.return_type),
            block: to_static_block(&f.block),
            attrs: leak_slice(f.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_struct(s: &ItemStruct) -> StaticItemStruct<'static> {
        StaticItemStruct {
            type_params: leak_slice(s.type_params.iter().map(|p| leak_str(p)).collect()),
            const_params: leak_slice(s.const_params.iter().map(|p| leak_str(p)).collect()),
            name: leak_str(&s.name),
            fields: leak_slice(s.fields.iter().map(to_static_field).collect()),
            attrs: leak_slice(s.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_item_impl(i: &ItemImpl) -> StaticItemImpl<'static> {
        StaticItemImpl {
            type_params: leak_slice(i.type_params.iter().map(|p| leak_str(p)).collect()),
            const_params: leak_slice(i.const_params.iter().map(|p| leak_str(p)).collect()),
            self_ty: leak_str(&i.self_ty),
            items: leak_slice(i.items.iter().map(to_static_impl_item).collect()),
            attrs: leak_slice(i.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_impl_item(item: &ImplItem) -> StaticImplItem<'static> {
        match item {
            ImplItem::Fn(f) => StaticImplItem::Fn(to_static_item_fn(f)),
            ImplItem::Const(c) => StaticImplItem::Const(to_static_item_const(c)),
            ImplItem::Type(t) => StaticImplItem::Type(to_static_item_type_alias(t)),
        }
    }

    fn to_static_item_type_alias(t: &ItemTypeAlias) -> StaticItemTypeAlias<'static> {
        StaticItemTypeAlias {
            name: leak_str(&t.name),
            ty: to_static_type(&t.ty),
        }
    }

    fn to_static_item_enum(e: &ItemEnum) -> StaticItemEnum<'static> {
        StaticItemEnum {
            name: leak_str(&e.name),
            variants: leak_slice(e.variants.iter().map(to_static_enum_variant).collect()),
            attrs: leak_slice(e.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_enum_variant(v: &EnumVariant) -> StaticEnumVariant<'static> {
        StaticEnumVariant {
            name: leak_str(&v.name),
            discriminant: v.discriminant,
            attrs: leak_slice(v.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_field(f: &Field) -> StaticField<'static> {
        StaticField {
            inter_stage_io: leak_slice(f.inter_stage_io.to_vec()),
            name: leak_str(&f.name),
            ty: to_static_type(&f.ty),
            attrs: leak_slice(f.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_fn_arg(a: &FnArg) -> StaticFnArg<'static> {
        StaticFnArg {
            inter_stage_io: leak_slice(a.inter_stage_io.to_vec()),
            name: leak_str(&a.name),
            ty: to_static_type(&a.ty),
            attrs: leak_slice(a.attrs.iter().map(to_static_attribute).collect()),
        }
    }

    fn to_static_return_type(r: &ReturnType) -> StaticReturnType<'static> {
        match r {
            ReturnType::Default => StaticReturnType::Default,
            ReturnType::Type { annotation, ty } => StaticReturnType::Type {
                annotation: annotation.clone(),
                ty: to_static_type(ty),
            },
        }
    }

    fn to_static_block(b: &Block) -> StaticBlock<'static> {
        StaticBlock {
            stmts: leak_slice(b.stmts.iter().map(to_static_stmt).collect()),
        }
    }

    fn to_static_stmt(s: &Stmt) -> StaticStmt<'static> {
        match s {
            Stmt::Local(l) => StaticStmt::Local(to_static_local(l)),
            Stmt::Const(c) => StaticStmt::Const(to_static_item_const(c)),
            Stmt::Assignment { lhs, rhs } => StaticStmt::Assignment {
                lhs: to_static_expr(lhs),
                rhs: to_static_expr(rhs),
            },
            Stmt::CompoundAssignment { lhs, op, rhs } => StaticStmt::CompoundAssignment {
                lhs: to_static_expr(lhs),
                op: *op,
                rhs: to_static_expr(rhs),
            },
            Stmt::While { condition, body } => StaticStmt::While {
                condition: to_static_expr(condition),
                body: to_static_block(body),
            },
            Stmt::Loop { body } => StaticStmt::Loop {
                body: to_static_block(body),
            },
            Stmt::Expr { expr, has_semi } => StaticStmt::Expr {
                expr: to_static_expr(expr),
                has_semi: *has_semi,
            },
            Stmt::If(i) => StaticStmt::If(to_static_stmt_if(i)),
            Stmt::Break => StaticStmt::Break,
            Stmt::Continue => StaticStmt::Continue,
            Stmt::Return(e) => StaticStmt::Return(e.as_ref().map(to_static_expr)),
            Stmt::For(f) => StaticStmt::For(to_static_for_loop(f)),
            Stmt::Switch(sw) => StaticStmt::Switch(to_static_stmt_switch(sw)),
            Stmt::Block(b) => StaticStmt::Block(to_static_block(b)),
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
            } => StaticStmt::SlabCopy {
                src: to_static_expr(src),
                src_offset: to_static_expr(src_offset),
                dest: to_static_expr(dest),
                dest_offset: to_static_expr(dest_offset),
                size: to_static_expr(size),
            },
            Stmt::Discard => StaticStmt::Discard,
            Stmt::Macro { name, args } => StaticStmt::Macro {
                name: leak_str(name),
                args: leak_str(args),
            },
        }
    }

    fn to_static_local(l: &Local) -> StaticLocal<'static> {
        StaticLocal {
            mutable: l.mutable,
            name: leak_str(&l.name),
            ty: l.ty.as_ref().map(to_static_type),
            init: l.init.as_ref().map(to_static_expr),
        }
    }

    fn to_static_for_loop(f: &ForLoop) -> StaticForLoop<'static> {
        StaticForLoop {
            var: leak_str(&f.var),
            var_ty: f.var_ty.as_ref().map(to_static_type),
            from: to_static_expr(&f.from),
            to: to_static_expr(&f.to),
            inclusive: f.inclusive,
            body: to_static_block(&f.body),
        }
    }

    fn to_static_stmt_if(i: &StmtIf) -> StaticStmtIf<'static> {
        StaticStmtIf {
            condition: to_static_expr(&i.condition),
            then_block: to_static_block(&i.then_block),
            else_branch: i.else_branch.as_ref().map(to_static_else_branch),
        }
    }

    fn to_static_else_branch(b: &ElseBranch) -> StaticElseBranch<'static> {
        match b {
            ElseBranch::Block(b) => StaticElseBranch::Block(to_static_block(b)),
            ElseBranch::If(i) => StaticElseBranch::If(leak_box(to_static_stmt_if(i))),
        }
    }

    fn to_static_stmt_switch(sw: &StmtSwitch) -> StaticStmtSwitch<'static> {
        StaticStmtSwitch {
            selector: to_static_expr(&sw.selector),
            arms: leak_slice(sw.arms.iter().map(to_static_switch_arm).collect()),
            has_explicit_default: sw.has_explicit_default,
        }
    }

    fn to_static_switch_arm(arm: &SwitchArm) -> StaticSwitchArm<'static> {
        StaticSwitchArm {
            selectors: leak_slice(arm.selectors.iter().map(to_static_case_selector).collect()),
            body: to_static_block(&arm.body),
        }
    }

    fn to_static_case_selector(c: &CaseSelector) -> StaticCaseSelector<'static> {
        match c {
            CaseSelector::Literal(lit) => StaticCaseSelector::Literal(to_static_lit(lit)),
            CaseSelector::Expr(e) => StaticCaseSelector::Expr(to_static_expr(e)),
            CaseSelector::Default => StaticCaseSelector::Default,
        }
    }

    fn to_static_expr(e: &Expr) -> StaticExpr<'static> {
        match e {
            Expr::Lit(lit) => StaticExpr::Lit(to_static_lit(lit)),
            Expr::Ident(name) => StaticExpr::Ident(leak_str(name)),
            Expr::Array { elems } => StaticExpr::Array {
                elems: leak_slice(elems.iter().map(to_static_expr).collect()),
            },
            Expr::Paren(e) => StaticExpr::Paren(leak_box(to_static_expr(e))),
            Expr::Binary { lhs, op, rhs } => StaticExpr::Binary {
                lhs: leak_box(to_static_expr(lhs)),
                op: *op,
                rhs: leak_box(to_static_expr(rhs)),
            },
            Expr::Unary { op, expr } => StaticExpr::Unary {
                op: *op,
                expr: leak_box(to_static_expr(expr)),
            },
            Expr::ArrayIndexing { lhs, index } => StaticExpr::ArrayIndexing {
                lhs: leak_box(to_static_expr(lhs)),
                index: leak_box(to_static_expr(index)),
            },
            Expr::Swizzle {
                lhs,
                swizzle,
                params,
            } => StaticExpr::Swizzle {
                lhs: leak_box(to_static_expr(lhs)),
                swizzle: leak_str(swizzle),
                params: params
                    .as_ref()
                    .map(|ps| leak_slice(ps.iter().map(to_static_expr).collect())),
            },
            Expr::Cast { lhs, ty } => StaticExpr::Cast {
                lhs: leak_box(to_static_expr(lhs)),
                ty: leak_box(to_static_type(ty)),
            },
            Expr::FnCall {
                path,
                type_args,
                params,
            } => StaticExpr::FnCall {
                path: to_static_fn_path(path),
                type_args: leak_slice(type_args.iter().map(to_static_type).collect()),
                params: leak_slice(params.iter().map(to_static_expr).collect()),
            },
            Expr::Struct {
                name,
                type_args,
                fields,
            } => StaticExpr::Struct {
                name: leak_str(name),
                type_args: leak_slice(type_args.iter().map(to_static_type).collect()),
                fields: leak_slice(fields.iter().map(to_static_field_value).collect()),
            },
            Expr::FieldAccess { base, field } => StaticExpr::FieldAccess {
                base: leak_box(to_static_expr(base)),
                field: leak_str(field),
            },
            Expr::TypePath { ty, member } => StaticExpr::TypePath {
                ty: leak_str(ty),
                member: leak_str(member),
            },
            Expr::Reference(e) => StaticExpr::Reference(leak_box(to_static_expr(e))),
            Expr::ZeroValueArray { elem_type, len } => StaticExpr::ZeroValueArray {
                elem_type: leak_box(to_static_type(elem_type)),
                len: leak_box(to_static_expr(len)),
            },
        }
    }

    fn to_static_lit(lit: &Lit) -> StaticLit<'static> {
        match lit {
            Lit::Bool(b) => StaticLit::Bool(*b),
            Lit::Int { digits, suffix } => StaticLit::Int {
                digits: leak_str(digits),
                suffix: leak_str(suffix),
            },
            Lit::Float { text } => StaticLit::Float {
                text: leak_str(text),
            },
        }
    }

    fn to_static_fn_path(path: &FnPath) -> StaticFnPath<'static> {
        match path {
            FnPath::Ident(name) => StaticFnPath::Ident(leak_str(name)),
            FnPath::TypeMethod { ty, method } => StaticFnPath::TypeMethod {
                ty: leak_str(ty),
                method: leak_str(method),
            },
        }
    }

    fn to_static_field_value(f: &FieldValue) -> StaticFieldValue<'static> {
        StaticFieldValue {
            member: leak_str(&f.member),
            expr: to_static_expr(&f.expr),
        }
    }

    fn to_static_type(t: &Type) -> StaticType<'static> {
        match t {
            Type::Scalar(scalar) => StaticType::Scalar(*scalar),
            Type::Vector {
                elements,
                scalar_ty,
            } => StaticType::Vector {
                elements: *elements,
                scalar_ty: *scalar_ty,
            },
            Type::Matrix {
                columns,
                rows,
                scalar_ty,
            } => StaticType::Matrix {
                columns: *columns,
                rows: *rows,
                scalar_ty: *scalar_ty,
            },
            Type::Array { elem, len } => StaticType::Array {
                elem: leak_box(to_static_type(elem)),
                len: to_static_expr(len),
            },
            Type::RuntimeArray { elem } => StaticType::RuntimeArray {
                elem: leak_box(to_static_type(elem)),
            },
            Type::Atomic { elem } => StaticType::Atomic {
                elem: leak_box(to_static_type(elem)),
            },
            Type::Struct { name, type_args } => StaticType::Struct {
                name: leak_str(name),
                type_args: leak_slice(type_args.iter().map(to_static_type).collect()),
            },
            Type::Ptr {
                address_space,
                elem,
            } => StaticType::Ptr {
                address_space: *address_space,
                elem: leak_box(to_static_type(elem)),
            },
            Type::Sampler => StaticType::Sampler,
            Type::SamplerComparison => StaticType::SamplerComparison,
            Type::Texture { kind, sampled_type } => StaticType::Texture {
                kind: *kind,
                sampled_type: *sampled_type,
            },
            Type::TextureDepth { kind } => StaticType::TextureDepth { kind: *kind },
            Type::TextureStorage {
                kind,
                format,
                access,
            } => StaticType::TextureStorage {
                kind: *kind,
                format: *format,
                access: *access,
            },
            Type::TypeParam { name } => StaticType::TypeParam {
                name: leak_str(name),
            },
            Type::Phantom { elem } => StaticType::Phantom {
                elem: leak_box(to_static_type(elem)),
            },
            Type::AssocType { ty, member } => StaticType::AssocType {
                ty: leak_box(to_static_type(ty)),
                member: leak_str(member),
            },
        }
    }

    // ===== the tests =====

    /// The kitchen-sink fixture survives a full round trip through the
    /// borrowed mirror: owned -> borrowed -> owned is exactly equal.
    #[test]
    fn kitchen_sink_round_trips() {
        let module = kitchen_sink_module();
        let borrowed = to_static_module(&module);
        assert_eq!(Module::from(&borrowed), module);
    }

    /// A small hand-written borrowed literal converts to the expected
    /// owned values, independently of the `to_static_*` helpers (so a
    /// shared mistake in both directions cannot cancel out).
    #[test]
    fn hand_written_literal_converts() {
        let borrowed: StaticModule<'static> = StaticModule {
            name: "tiny",
            attrs: &[StaticAttribute {
                path: "wgsl",
                args: &["doc"],
            }],
            items: &[StaticItem::Fn(StaticItemFn {
                type_params: &["T"],
                const_params: &[],
                fn_attrs: FnAttrs::None,
                name: "id",
                inputs: &[StaticFnArg {
                    inter_stage_io: &[InterStageIo::Location(0)],
                    name: "x",
                    ty: StaticType::Scalar(ScalarType::F32),
                    attrs: &[],
                }],
                return_type: StaticReturnType::Default,
                block: StaticBlock {
                    stmts: &[StaticStmt::Return(Some(StaticExpr::Ident("x")))],
                },
                attrs: &[],
            })],
        };
        let owned = Module::from(&borrowed);
        assert_eq!(owned.name, "tiny");
        assert_eq!(
            owned.attrs,
            vec![Attribute {
                path: "wgsl".into(),
                args: vec!["doc".into()],
            }]
        );
        let Item::Fn(f) = &owned.items[0] else {
            panic!("expected a fn item");
        };
        assert_eq!(f.name, Cow::Borrowed("id"));
        assert_eq!(f.type_params, vec!["T".to_string()]);
        assert_eq!(f.inputs.len(), 1);
        assert_eq!(f.inputs[0].name, "x");
        assert_eq!(f.inputs[0].inter_stage_io, vec![InterStageIo::Location(0)]);
        match f.block.stmts.as_slice() {
            [Stmt::Return(Some(Expr::Ident(name)))] => assert_eq!(name, "x"),
            other => panic!("unexpected block: {other:?}"),
        }
    }

    #[test]
    fn filterability_param_round_trips() {
        let owned = Filterability::Param("FILTERABLE".into());
        let borrowed = to_static_filterability(&owned);
        assert_eq!(borrowed, StaticFilterability::Param("FILTERABLE"));
        assert_eq!(Filterability::from(borrowed), owned);
    }
}
