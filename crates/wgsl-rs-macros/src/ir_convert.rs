//! Conversion from `parse::*` (compile-time, `syn`-based AST) to `ir::*`
//! (owned, `String`/`Vec<T>`-based runtime IR).
//!
//! The conversion is intentionally mechanical: it strips `syn` tokens and
//! spans, converts identifiers to `String`, and parses literal integers
//! into Rust numeric types. It does *not* perform any WGSL-specific
//! transformations (those live in `wgsl-rs-ir/src/render.rs`).

use proc_macro2::Span;
use snafu::prelude::*;
use wgsl_rs_ir as ir;

use crate::parse;

#[derive(Debug, Snafu)]
pub enum ConvertError {
    #[snafu(display("invalid integer literal: {note}"))]
    BadInt { span: Span, note: String },
    #[snafu(display("unsupported during IR conversion: {note}"))]
    Unsupported { span: Span, note: String },
}

impl ConvertError {
    pub fn into_parse(self) -> crate::parse::Error {
        match self {
            ConvertError::BadInt { span, note } => crate::parse::Error::unsupported(span, note),
            ConvertError::Unsupported { span, note } => {
                crate::parse::Error::unsupported(span, note)
            }
        }
    }
}

type Result<T> = std::result::Result<T, ConvertError>;

// ===== Top-level helpers =====

/// Convert a parse module's top-level items to a list of IR items.
pub fn items_from_parse(items: &[parse::Item]) -> Result<Vec<ir::Item>> {
    let mut out = Vec::new();
    for item in items {
        if let Some(ir_item) = item_from_parse(item)? {
            out.push(ir_item);
        }
    }
    Ok(out)
}

// ===== Items =====

fn item_from_parse(item: &parse::Item) -> Result<Option<ir::Item>> {
    Ok(Some(match item {
        parse::Item::Const(c) => ir::Item::Const(item_const(c)?),
        parse::Item::Uniform(u) => ir::Item::Uniform(item_uniform(u)?),
        parse::Item::Storage(s) => ir::Item::Storage(item_storage(s)?),
        parse::Item::Workgroup(w) => ir::Item::Workgroup(item_workgroup(w)?),
        parse::Item::Sampler(s) => ir::Item::Sampler(item_sampler(s)?),
        parse::Item::Texture(t) => ir::Item::Texture(item_texture(t)?),
        parse::Item::Fn(f) => ir::Item::Fn(item_fn(f)?),
        parse::Item::Struct(s) => ir::Item::Struct(item_struct(s)?),
        parse::Item::Impl(i) => ir::Item::Impl(item_impl(i)?),
        parse::Item::Enum(e) => ir::Item::Enum(item_enum(e)),
        // Items that produce no WGSL output are skipped.
        parse::Item::Mod(_)
        | parse::Item::Use(_)
        | parse::Item::MacroRules
        | parse::Item::Trait
        | parse::Item::Ignored => return Ok(None),
    }))
}

fn item_const(c: &parse::ItemConst) -> Result<ir::ItemConst> {
    Ok(ir::ItemConst {
        name: c.ident.to_string(),
        ty: ty_from_parse(&c.ty)?,
        expr: expr_from_parse(&c.expr)?,
        attrs: c.attrs.clone(),
    })
}

fn lit_int_to_u32(lit: &syn::LitInt) -> Result<u32> {
    lit.base10_parse::<u32>().map_err(|e| ConvertError::BadInt {
        span: lit.span(),
        note: e.to_string(),
    })
}

fn item_uniform(u: &parse::ItemUniform) -> Result<ir::ItemUniform> {
    Ok(ir::ItemUniform {
        group: lit_int_to_u32(&u.group)?,
        binding: lit_int_to_u32(&u.binding)?,
        name: u.name.to_string(),
        ty: ty_from_parse(&u.ty)?,
        attrs: u.attrs.clone(),
    })
}

fn item_storage(s: &parse::ItemStorage) -> Result<ir::ItemStorage> {
    Ok(ir::ItemStorage {
        group: lit_int_to_u32(&s.group)?,
        binding: lit_int_to_u32(&s.binding)?,
        access: match s.access {
            parse::StorageAccess::Read => ir::StorageAccess::Read,
            parse::StorageAccess::ReadWrite => ir::StorageAccess::ReadWrite,
        },
        name: s.name.to_string(),
        ty: ty_from_parse(&s.ty)?,
        attrs: s.attrs.clone(),
    })
}

fn item_workgroup(w: &parse::ItemWorkgroup) -> Result<ir::ItemWorkgroup> {
    Ok(ir::ItemWorkgroup {
        name: w.name.to_string(),
        ty: ty_from_parse(&w.ty)?,
        attrs: w.attrs.clone(),
    })
}

fn item_sampler(s: &parse::ItemSampler) -> Result<ir::ItemSampler> {
    Ok(ir::ItemSampler {
        group: lit_int_to_u32(&s.group)?,
        binding: lit_int_to_u32(&s.binding)?,
        name: s.name.to_string(),
        ty: ty_from_parse(&s.ty)?,
        attrs: s.attrs.clone(),
    })
}

fn item_texture(t: &parse::ItemTexture) -> Result<ir::ItemTexture> {
    Ok(ir::ItemTexture {
        group: lit_int_to_u32(&t.group)?,
        binding: lit_int_to_u32(&t.binding)?,
        name: t.name.to_string(),
        ty: ty_from_parse(&t.ty)?,
        attrs: t.attrs.clone(),
    })
}

fn item_fn(f: &parse::ItemFn) -> Result<ir::ItemFn> {
    Ok(ir::ItemFn {
        type_params: f.type_params.iter().map(|i| i.to_string()).collect(),
        fn_attrs: fn_attrs(&f.fn_attrs)?,
        name: f.ident.to_string().into(),
        inputs: f.inputs.iter().map(fn_arg).collect::<Result<Vec<_>>>()?,
        return_type: return_type(&f.return_type)?,
        block: block_from_parse(&f.block)?,
        attrs: f.attrs.clone(),
    })
}

fn item_struct(s: &parse::ItemStruct) -> Result<ir::ItemStruct> {
    Ok(ir::ItemStruct {
        type_params: s.type_params.iter().map(|i| i.to_string()).collect(),
        name: s.ident.to_string(),
        fields: s
            .fields
            .named
            .iter()
            .map(field_from_parse)
            .collect::<Result<Vec<_>>>()?,
        attrs: s.attrs.clone(),
    })
}

fn item_impl(i: &parse::ItemImpl) -> Result<ir::ItemImpl> {
    Ok(ir::ItemImpl {
        type_params: i.type_params.iter().map(|i| i.to_string()).collect(),
        self_ty: i.self_ty.to_string(),
        items: i
            .items
            .iter()
            .map(|ii| {
                Ok(match ii {
                    parse::ImplItem::Fn(f) => ir::ImplItem::Fn(item_fn(f)?),
                    parse::ImplItem::Const(c) => ir::ImplItem::Const(item_const(c)?),
                })
            })
            .collect::<Result<Vec<_>>>()?,
        attrs: i.attrs.clone(),
    })
}

fn item_enum(e: &parse::ItemEnum) -> ir::ItemEnum {
    ir::ItemEnum {
        name: e.ident.to_string(),
        variants: e
            .variants
            .iter()
            .map(|v| ir::EnumVariant {
                name: v.ident.to_string(),
                discriminant: v
                    .discriminant
                    .as_ref()
                    .and_then(|(_, lit)| lit.base10_parse::<u32>().ok()),
            })
            .collect(),
        attrs: e.attrs.clone(),
    }
}

fn field_from_parse(f: &parse::Field) -> Result<ir::Field> {
    Ok(ir::Field {
        inter_stage_io: f
            .inter_stage_io
            .iter()
            .map(inter_stage_io)
            .collect::<Result<Vec<_>>>()?,
        name: f.ident.to_string(),
        ty: ty_from_parse(&f.ty)?,
        attrs: f.attrs.clone(),
    })
}

fn fn_arg(a: &parse::FnArg) -> Result<ir::FnArg> {
    Ok(ir::FnArg {
        inter_stage_io: a
            .inter_stage_io
            .iter()
            .map(inter_stage_io)
            .collect::<Result<Vec<_>>>()?,
        name: a.ident.to_string(),
        ty: ty_from_parse(&a.ty)?,
        attrs: a.attrs.clone(),
    })
}

fn fn_attrs(a: &parse::FnAttrs) -> Result<ir::FnAttrs> {
    Ok(match a {
        parse::FnAttrs::None => ir::FnAttrs::None,
        parse::FnAttrs::Vertex(_) => ir::FnAttrs::Vertex,
        parse::FnAttrs::Fragment(_) => ir::FnAttrs::Fragment,
        parse::FnAttrs::Compute { workgroup_size, .. } => ir::FnAttrs::Compute {
            workgroup_size: ir::WorkgroupSize {
                x: lit_int_to_u32(&workgroup_size.x)?,
                y: workgroup_size
                    .y
                    .as_ref()
                    .map(|(_, lit)| lit_int_to_u32(lit))
                    .transpose()?,
                z: workgroup_size
                    .z
                    .as_ref()
                    .map(|(_, lit)| lit_int_to_u32(lit))
                    .transpose()?,
            },
        },
    })
}

fn return_type(rt: &parse::ReturnType) -> Result<ir::ReturnType> {
    Ok(match rt {
        parse::ReturnType::Default => ir::ReturnType::Default,
        parse::ReturnType::Type { annotation, ty, .. } => ir::ReturnType::Type {
            annotation: return_annotation(annotation)?,
            ty: ty_from_parse(ty)?,
        },
    })
}

fn return_annotation(a: &parse::ReturnTypeAnnotation) -> Result<ir::ReturnTypeAnnotation> {
    Ok(match a {
        parse::ReturnTypeAnnotation::None => ir::ReturnTypeAnnotation::None,
        parse::ReturnTypeAnnotation::BuiltIn(ident) => {
            ir::ReturnTypeAnnotation::BuiltIn(builtin_from_ident(ident)?)
        }
        parse::ReturnTypeAnnotation::DefaultBuiltInPosition => {
            ir::ReturnTypeAnnotation::DefaultBuiltInPosition
        }
        parse::ReturnTypeAnnotation::Location { lit, .. } => {
            ir::ReturnTypeAnnotation::Location(lit_to_u32(lit)?)
        }
        parse::ReturnTypeAnnotation::DefaultLocation => ir::ReturnTypeAnnotation::DefaultLocation,
    })
}

fn lit_to_u32(l: &parse::Lit) -> Result<u32> {
    match l {
        parse::Lit::Int(li) => lit_int_to_u32(li),
        parse::Lit::Bool(b) => Err(ConvertError::Unsupported {
            span: b.span(),
            note: "expected integer literal, got bool".to_string(),
        }),
        parse::Lit::Float(f) => Err(ConvertError::Unsupported {
            span: f.span(),
            note: "expected integer literal, got float".to_string(),
        }),
    }
}

fn inter_stage_io(io: &parse::InterStageIo) -> Result<ir::InterStageIo> {
    Ok(match io {
        parse::InterStageIo::BuiltIn { inner, .. } => {
            ir::InterStageIo::BuiltIn(builtin_from_parse(inner))
        }
        parse::InterStageIo::Location { inner, .. } => {
            ir::InterStageIo::Location(inner.base10_parse::<u32>().map_err(|e| {
                ConvertError::BadInt {
                    span: inner.span(),
                    note: e.to_string(),
                }
            })?)
        }
        parse::InterStageIo::BlendSrc { lit, .. } => {
            ir::InterStageIo::BlendSrc(lit.base10_parse::<u32>().map_err(|e| {
                ConvertError::BadInt {
                    span: lit.span(),
                    note: e.to_string(),
                }
            })?)
        }
        parse::InterStageIo::Interpolate { inner, .. } => {
            ir::InterStageIo::Interpolate(ir::Interpolate {
                ty: match inner.ty {
                    parse::InterpolationType::Perspective(_) => ir::InterpolationType::Perspective,
                    parse::InterpolationType::Linear(_) => ir::InterpolationType::Linear,
                    parse::InterpolationType::Flat(_) => ir::InterpolationType::Flat,
                },
                sampling: inner.sampling.as_ref().map(|s| match s {
                    parse::InterpolationSampling::Center(_) => ir::InterpolationSampling::Center,
                    parse::InterpolationSampling::Centroid(_) => {
                        ir::InterpolationSampling::Centroid
                    }
                    parse::InterpolationSampling::Sample(_) => ir::InterpolationSampling::Sample,
                    parse::InterpolationSampling::First(_) => ir::InterpolationSampling::First,
                    parse::InterpolationSampling::Either(_) => ir::InterpolationSampling::Either,
                }),
            })
        }
        parse::InterStageIo::Invariant(_) => ir::InterStageIo::Invariant,
    })
}

fn builtin_from_parse(b: &parse::BuiltIn) -> ir::BuiltIn {
    match b {
        parse::BuiltIn::VertexIndex(_) => ir::BuiltIn::VertexIndex,
        parse::BuiltIn::InstanceIndex(_) => ir::BuiltIn::InstanceIndex,
        parse::BuiltIn::Position(_) => ir::BuiltIn::Position,
        parse::BuiltIn::FrontFacing(_) => ir::BuiltIn::FrontFacing,
        parse::BuiltIn::FragDepth(_) => ir::BuiltIn::FragDepth,
        parse::BuiltIn::SampleIndex(_) => ir::BuiltIn::SampleIndex,
        parse::BuiltIn::SampleMask(_) => ir::BuiltIn::SampleMask,
        parse::BuiltIn::LocalInvocationId(_) => ir::BuiltIn::LocalInvocationId,
        parse::BuiltIn::LocalInvocationIndex(_) => ir::BuiltIn::LocalInvocationIndex,
        parse::BuiltIn::GlobalInvocationId(_) => ir::BuiltIn::GlobalInvocationId,
        parse::BuiltIn::WorkgroupId(_) => ir::BuiltIn::WorkgroupId,
        parse::BuiltIn::NumWorkgroups(_) => ir::BuiltIn::NumWorkgroups,
        parse::BuiltIn::SubgroupInvocationId(_) => ir::BuiltIn::SubgroupInvocationId,
        parse::BuiltIn::SubgroupSize(_) => ir::BuiltIn::SubgroupSize,
        parse::BuiltIn::PrimitiveIndex(_) => ir::BuiltIn::PrimitiveIndex,
        parse::BuiltIn::SubgroupId(_) => ir::BuiltIn::SubgroupId,
        parse::BuiltIn::NumSubgroups(_) => ir::BuiltIn::NumSubgroups,
    }
}

fn builtin_from_ident(i: &proc_macro2::Ident) -> Result<ir::BuiltIn> {
    Ok(match i.to_string().as_str() {
        "vertex_index" => ir::BuiltIn::VertexIndex,
        "instance_index" => ir::BuiltIn::InstanceIndex,
        "position" => ir::BuiltIn::Position,
        "front_facing" => ir::BuiltIn::FrontFacing,
        "frag_depth" => ir::BuiltIn::FragDepth,
        "sample_index" => ir::BuiltIn::SampleIndex,
        "sample_mask" => ir::BuiltIn::SampleMask,
        "local_invocation_id" => ir::BuiltIn::LocalInvocationId,
        "local_invocation_index" => ir::BuiltIn::LocalInvocationIndex,
        "global_invocation_id" => ir::BuiltIn::GlobalInvocationId,
        "workgroup_id" => ir::BuiltIn::WorkgroupId,
        "num_workgroups" => ir::BuiltIn::NumWorkgroups,
        "subgroup_invocation_id" => ir::BuiltIn::SubgroupInvocationId,
        "subgroup_size" => ir::BuiltIn::SubgroupSize,
        "primitive_index" => ir::BuiltIn::PrimitiveIndex,
        "subgroup_id" => ir::BuiltIn::SubgroupId,
        "num_subgroups" => ir::BuiltIn::NumSubgroups,
        other => UnsupportedSnafu {
            span: i.span(),
            note: format!("'{other}' is not a known builtin"),
        }
        .fail()?,
    })
}

// ===== Types =====

/// Convert a parse type to IR.
pub fn ty_from_parse(t: &parse::Type) -> Result<ir::Type> {
    Ok(match t {
        parse::Type::Scalar { ty, .. } => ir::Type::Scalar(scalar_from_parse(*ty)),
        parse::Type::Vector {
            elements,
            scalar_ty,
            ..
        } => {
            // The parse-side `scalar_ty` is always set (it's the inferred
            // element type for both `Vec4f` and `vec4<f32>` forms). The
            // IR carries it as `Some(...)` and lets the renderer pick
            // the WGSL shorthand (`vec4f`) form.
            ir::Type::Vector {
                elements: *elements,
                scalar_ty: Some(scalar_from_parse(*scalar_ty)),
            }
        }
        parse::Type::Matrix {
            columns,
            rows,
            scalar,
            ..
        } => ir::Type::Matrix {
            columns: *columns,
            rows: *rows,
            scalar_ty: scalar
                .as_ref()
                .map(|(_, ident, _)| scalar_from_ident(ident))
                .transpose()?,
        },
        parse::Type::Array { elem, len, .. } => ir::Type::Array {
            elem: Box::new(ty_from_parse(elem)?),
            len: expr_from_parse(len)?,
        },
        parse::Type::RuntimeArray { elem, .. } => ir::Type::RuntimeArray {
            elem: Box::new(ty_from_parse(elem)?),
        },
        parse::Type::Atomic { elem, .. } => ir::Type::Atomic {
            elem: Box::new(ty_from_parse(elem)?),
        },
        parse::Type::Struct { ident, type_args } => ir::Type::Struct {
            name: ident.to_string(),
            type_args: type_args
                .iter()
                .map(ty_from_parse)
                .collect::<Result<Vec<_>>>()?,
        },
        parse::Type::Ptr {
            address_space,
            elem,
            ..
        } => ir::Type::Ptr {
            address_space: match address_space {
                parse::AddressSpace::Function => ir::AddressSpace::Function,
                parse::AddressSpace::Private => ir::AddressSpace::Private,
                parse::AddressSpace::Workgroup => ir::AddressSpace::Workgroup,
            },
            elem: Box::new(ty_from_parse(elem)?),
        },
        parse::Type::Sampler { .. } => ir::Type::Sampler,
        parse::Type::SamplerComparison { .. } => ir::Type::SamplerComparison,
        parse::Type::Texture {
            kind, sampled_type, ..
        } => ir::Type::Texture {
            kind: tex_kind(*kind),
            sampled_type: scalar_from_parse(*sampled_type),
        },
        parse::Type::TextureDepth { kind, .. } => ir::Type::TextureDepth {
            kind: tex_depth_kind(*kind),
        },
        parse::Type::TypeParam { ident } => ir::Type::TypeParam {
            name: ident.to_string(),
        },
    })
}

fn scalar_from_parse(s: parse::ScalarType) -> ir::ScalarType {
    match s {
        parse::ScalarType::I32 => ir::ScalarType::I32,
        parse::ScalarType::U32 => ir::ScalarType::U32,
        parse::ScalarType::F32 => ir::ScalarType::F32,
        parse::ScalarType::Bool => ir::ScalarType::Bool,
    }
}

fn scalar_from_ident(i: &proc_macro2::Ident) -> Result<ir::ScalarType> {
    Ok(match i.to_string().as_str() {
        "i32" => ir::ScalarType::I32,
        "u32" => ir::ScalarType::U32,
        "f32" => ir::ScalarType::F32,
        "bool" => ir::ScalarType::Bool,
        other => UnsupportedSnafu {
            span: i.span(),
            note: format!("'{other}' is not a scalar type"),
        }
        .fail()?,
    })
}

fn tex_kind(k: parse::TextureKind) -> ir::TextureKind {
    match k {
        parse::TextureKind::Texture1D => ir::TextureKind::Texture1D,
        parse::TextureKind::Texture2D => ir::TextureKind::Texture2D,
        parse::TextureKind::Texture2DArray => ir::TextureKind::Texture2DArray,
        parse::TextureKind::Texture3D => ir::TextureKind::Texture3D,
        parse::TextureKind::TextureCube => ir::TextureKind::TextureCube,
        parse::TextureKind::TextureCubeArray => ir::TextureKind::TextureCubeArray,
        parse::TextureKind::TextureMultisampled2D => ir::TextureKind::TextureMultisampled2D,
    }
}

fn tex_depth_kind(k: parse::TextureDepthKind) -> ir::TextureDepthKind {
    match k {
        parse::TextureDepthKind::Depth2D => ir::TextureDepthKind::Depth2D,
        parse::TextureDepthKind::Depth2DArray => ir::TextureDepthKind::Depth2DArray,
        parse::TextureDepthKind::DepthCube => ir::TextureDepthKind::DepthCube,
        parse::TextureDepthKind::DepthCubeArray => ir::TextureDepthKind::DepthCubeArray,
        parse::TextureDepthKind::DepthMultisampled2D => ir::TextureDepthKind::DepthMultisampled2D,
    }
}

// ===== Expressions =====

/// Convert a parse expression to IR.
pub fn expr_from_parse(e: &parse::Expr) -> Result<ir::Expr> {
    Ok(match e {
        parse::Expr::Lit(l) => ir::Expr::Lit(lit(l)),
        parse::Expr::Ident(i) => ir::Expr::Ident(i.to_string()),
        parse::Expr::Array { elems, .. } => ir::Expr::Array {
            elems: elems
                .iter()
                .map(expr_from_parse)
                .collect::<Result<Vec<_>>>()?,
        },
        parse::Expr::Paren { inner, .. } => ir::Expr::Paren(Box::new(expr_from_parse(inner)?)),
        parse::Expr::Binary { lhs, op, rhs } => ir::Expr::Binary {
            lhs: Box::new(expr_from_parse(lhs)?),
            op: bin_op(op),
            rhs: Box::new(expr_from_parse(rhs)?),
        },
        parse::Expr::Unary { op, expr } => ir::Expr::Unary {
            op: un_op(op),
            expr: Box::new(expr_from_parse(expr)?),
        },
        parse::Expr::ArrayIndexing { lhs, index, .. } => ir::Expr::ArrayIndexing {
            lhs: Box::new(expr_from_parse(lhs)?),
            index: Box::new(expr_from_parse(index)?),
        },
        parse::Expr::Swizzle {
            lhs,
            swizzle,
            params,
            ..
        } => ir::Expr::Swizzle {
            lhs: Box::new(expr_from_parse(lhs)?),
            swizzle: swizzle.to_string(),
            params: params
                .as_ref()
                .map(|ps| ps.iter().map(expr_from_parse).collect::<Result<Vec<_>>>())
                .transpose()?,
        },
        parse::Expr::Cast { lhs, ty } => ir::Expr::Cast {
            lhs: Box::new(expr_from_parse(lhs)?),
            ty: Box::new(ty_from_parse(ty)?),
        },
        parse::Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => ir::Expr::FnCall {
            path: fn_path(path),
            type_args: type_args
                .iter()
                .map(ty_from_parse)
                .collect::<Result<Vec<_>>>()?,
            params: params
                .iter()
                .map(expr_from_parse)
                .collect::<Result<Vec<_>>>()?,
        },
        parse::Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } => ir::Expr::Struct {
            name: ident.to_string(),
            type_args: type_args
                .iter()
                .map(ty_from_parse)
                .collect::<Result<Vec<_>>>()?,
            fields: fields
                .iter()
                .map(|f| {
                    Ok(ir::FieldValue {
                        member: f.member.to_string(),
                        expr: expr_from_parse(&f.expr)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        },
        parse::Expr::FieldAccess { base, field, .. } => ir::Expr::FieldAccess {
            base: Box::new(expr_from_parse(base)?),
            field: field.to_string(),
        },
        parse::Expr::TypePath { ty, member, .. } => ir::Expr::TypePath {
            ty: ty.to_string(),
            member: member.to_string(),
        },
        parse::Expr::Reference { expr, .. } => {
            ir::Expr::Reference(Box::new(expr_from_parse(expr)?))
        }
        parse::Expr::ZeroValueArray { elem_type, len, .. } => ir::Expr::ZeroValueArray {
            elem_type: Box::new(ty_from_parse(elem_type)?),
            len: Box::new(expr_from_parse(len)?),
        },
        parse::Expr::LinkageAccess { ident, .. } => ir::Expr::Ident(ident.to_string()),
    })
}

fn fn_path(p: &parse::FnPath) -> ir::FnPath {
    match p {
        parse::FnPath::Ident(i) => ir::FnPath::Ident(i.to_string()),
        parse::FnPath::TypeMethod { ty, method, .. } => ir::FnPath::TypeMethod {
            ty: ty.to_string(),
            method: method.to_string(),
        },
    }
}

fn lit(l: &parse::Lit) -> ir::Lit {
    match l {
        parse::Lit::Bool(b) => ir::Lit::Bool(b.value()),
        parse::Lit::Int(i) => ir::Lit::Int {
            digits: i.base10_digits().to_string(),
            suffix: i.suffix().to_string(),
        },
        parse::Lit::Float(f) => ir::Lit::Float {
            text: f.to_string(),
        },
    }
}

fn bin_op(op: &parse::BinOp) -> ir::BinOp {
    match op {
        parse::BinOp::Add(_) => ir::BinOp::Add,
        parse::BinOp::Sub(_) => ir::BinOp::Sub,
        parse::BinOp::Mul(_) => ir::BinOp::Mul,
        parse::BinOp::Div(_) => ir::BinOp::Div,
        parse::BinOp::Rem(_) => ir::BinOp::Rem,
        parse::BinOp::Eq(_) => ir::BinOp::Eq,
        parse::BinOp::Ne(_) => ir::BinOp::Ne,
        parse::BinOp::Lt(_) => ir::BinOp::Lt,
        parse::BinOp::Le(_) => ir::BinOp::Le,
        parse::BinOp::Gt(_) => ir::BinOp::Gt,
        parse::BinOp::Ge(_) => ir::BinOp::Ge,
        parse::BinOp::And(_) => ir::BinOp::And,
        parse::BinOp::Or(_) => ir::BinOp::Or,
        parse::BinOp::BitAnd(_) => ir::BinOp::BitAnd,
        parse::BinOp::BitOr(_) => ir::BinOp::BitOr,
        parse::BinOp::BitXor(_) => ir::BinOp::BitXor,
        parse::BinOp::Shl(_) => ir::BinOp::Shl,
        parse::BinOp::Shr(_) => ir::BinOp::Shr,
    }
}

fn compound_op(op: &parse::CompoundOp) -> ir::CompoundOp {
    match op {
        parse::CompoundOp::AddAssign(_) => ir::CompoundOp::AddAssign,
        parse::CompoundOp::SubAssign(_) => ir::CompoundOp::SubAssign,
        parse::CompoundOp::MulAssign(_) => ir::CompoundOp::MulAssign,
        parse::CompoundOp::DivAssign(_) => ir::CompoundOp::DivAssign,
        parse::CompoundOp::RemAssign(_) => ir::CompoundOp::RemAssign,
        parse::CompoundOp::BitAndAssign(_) => ir::CompoundOp::BitAndAssign,
        parse::CompoundOp::BitOrAssign(_) => ir::CompoundOp::BitOrAssign,
        parse::CompoundOp::BitXorAssign(_) => ir::CompoundOp::BitXorAssign,
        parse::CompoundOp::ShlAssign(_) => ir::CompoundOp::ShlAssign,
        parse::CompoundOp::ShrAssign(_) => ir::CompoundOp::ShrAssign,
    }
}

fn un_op(op: &parse::UnOp) -> ir::UnOp {
    match op {
        parse::UnOp::Not(_) => ir::UnOp::Not,
        parse::UnOp::Neg(_) => ir::UnOp::Neg,
        parse::UnOp::Deref(_) => ir::UnOp::Deref,
    }
}

// ===== Statements / blocks =====

pub fn block_from_parse(b: &parse::Block) -> Result<ir::Block> {
    Ok(ir::Block {
        stmts: b
            .stmt
            .iter()
            .map(stmt_from_parse)
            .collect::<Result<Vec<_>>>()?,
    })
}

fn stmt_from_parse(s: &parse::Stmt) -> Result<ir::Stmt> {
    Ok(match s {
        parse::Stmt::Local(l) => ir::Stmt::Local(ir::Local {
            mutable: l.mutability.is_some(),
            name: l.ident.to_string(),
            ty: l.ty.as_ref().map(|(_, t)| ty_from_parse(t)).transpose()?,
            init: l
                .init
                .as_ref()
                .map(|i| expr_from_parse(&i.expr))
                .transpose()?,
        }),
        parse::Stmt::Const(c) => ir::Stmt::Const(item_const(c)?),
        parse::Stmt::Assignment { lhs, rhs, .. } => ir::Stmt::Assignment {
            lhs: expr_from_parse(lhs)?,
            rhs: expr_from_parse(rhs)?,
        },
        parse::Stmt::CompoundAssignment { lhs, op, rhs, .. } => ir::Stmt::CompoundAssignment {
            lhs: expr_from_parse(lhs)?,
            op: compound_op(op),
            rhs: expr_from_parse(rhs)?,
        },
        parse::Stmt::While {
            condition, body, ..
        } => ir::Stmt::While {
            condition: expr_from_parse(condition)?,
            body: block_from_parse(body)?,
        },
        parse::Stmt::Loop { body, .. } => ir::Stmt::Loop {
            body: block_from_parse(body)?,
        },
        parse::Stmt::Expr { expr, semi_token } => ir::Stmt::Expr {
            expr: expr_from_parse(expr)?,
            has_semi: semi_token.is_some(),
        },
        parse::Stmt::If(i) => ir::Stmt::If(stmt_if(i)?),
        parse::Stmt::Break { .. } => ir::Stmt::Break,
        parse::Stmt::Continue { .. } => ir::Stmt::Continue,
        parse::Stmt::Return { expr, .. } => {
            ir::Stmt::Return(expr.as_ref().map(expr_from_parse).transpose()?)
        }
        parse::Stmt::For(f) => ir::Stmt::For(ir::ForLoop {
            var: f.ident.to_string(),
            var_ty: f.ty.as_ref().map(|(_, t)| ty_from_parse(t)).transpose()?,
            from: expr_from_parse(&f.from)?,
            to: expr_from_parse(&f.to)?,
            inclusive: f.inclusive,
            body: block_from_parse(&f.body)?,
        }),
        parse::Stmt::Switch(s) => ir::Stmt::Switch(ir::StmtSwitch {
            selector: expr_from_parse(&s.selector)?,
            arms: s
                .arms
                .iter()
                .map(|arm| {
                    Ok(ir::SwitchArm {
                        selectors: arm
                            .selectors
                            .iter()
                            .map(case_selector)
                            .collect::<Result<Vec<_>>>()?,
                        body: block_from_parse(&arm.body)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            has_explicit_default: s.has_explicit_default,
        }),
        parse::Stmt::Block(b) => ir::Stmt::Block(block_from_parse(b)?),
        parse::Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => ir::Stmt::SlabRead {
            slab: expr_from_parse(slab)?,
            offset: expr_from_parse(offset)?,
            dest: expr_from_parse(dest)?,
            size: expr_from_parse(size)?,
        },
        parse::Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => ir::Stmt::SlabWrite {
            slab: expr_from_parse(slab)?,
            offset: expr_from_parse(offset)?,
            src: expr_from_parse(src)?,
            size: size.as_ref().map(expr_from_parse).transpose()?,
        },
        parse::Stmt::Discard { .. } => ir::Stmt::Discard,
    })
}

fn stmt_if(i: &parse::StmtIf) -> Result<ir::StmtIf> {
    Ok(ir::StmtIf {
        condition: expr_from_parse(&i.condition)?,
        then_block: block_from_parse(&i.then_block)?,
        else_branch: i
            .else_branch
            .as_ref()
            .map(|eb| match &eb.body {
                parse::ElseBody::Block(b) => Ok(ir::ElseBranch::Block(block_from_parse(b)?)),
                parse::ElseBody::If(inner) => Ok(ir::ElseBranch::If(Box::new(stmt_if(inner)?))),
            })
            .transpose()?,
    })
}

fn case_selector(s: &parse::CaseSelector) -> Result<ir::CaseSelector> {
    Ok(match s {
        parse::CaseSelector::Literal(l) => ir::CaseSelector::Literal(lit(l)),
        parse::CaseSelector::Expr(e) => ir::CaseSelector::Expr(expr_from_parse(e)?),
        parse::CaseSelector::Default(_) => ir::CaseSelector::Default,
    })
}
