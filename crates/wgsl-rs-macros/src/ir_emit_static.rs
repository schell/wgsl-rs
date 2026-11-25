//! Emit static `wgsl_rs_ir::static_ir` data literals for the given owned
//! IR (wgsl-rs#221).
//!
//! This is the `fast-compile` counterpart of `ir_emit`: instead of a
//! function that constructs the owned IR at runtime, the consuming
//! crate gets static data (a `&StaticModule` / `&[StaticItem]`
//! literal) plus a slim shim that converts it via the
//! `wgsl_rs_ir::static_ir` `From` impls. A ~1000-line shader's legacy
//! ctor is a ~12,000-line function that LLVM spends ~14s on per
//! incremental rebuild at `opt-level=1`; static data costs LLVM
//! almost nothing.
//!
//! Every IR enum is matched exhaustively here. A new IR variant
//! without a mirror arm fails *this* crate's build, keeping the
//! `static_ir` mirror and the owned IR in sync as a compile-time
//! guarantee rather than a runtime risk.
//!
//! All types are referenced through the caller-provided `ir_path`
//! (e.g. `wgsl_rs::ir`), so `#[wgsl(crate = "...")]` keeps working.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use wgsl_rs_ir as ir;

/// Emit a `&StaticModule` literal (with leading `&`) for the given
/// module name, items, and attributes, suitable as the initializer of
/// `static X: &ir::static_ir::StaticModule = ...;`.
///
/// Takes the module pieces rather than an owned `ir::Module` so the
/// macro never has to materialize one: `ir::Module::name` is
/// `&'static str`, and building that value in the macro would require
/// leaking the name string into the proc-macro process. The emitted
/// literal carries the name as a plain string literal, which is
/// `&'static str` in the consuming crate for free, exactly like the
/// legacy path's `stringify!(...)`.
pub fn emit_static_module(
    ir_path: &TokenStream,
    name: &str,
    items: &[ir::Item],
    attrs: &[ir::Attribute],
) -> TokenStream {
    let p = ir_path;
    let items_ts = items.iter().map(|i| static_item(p, i));
    let attrs_ts = static_attrs(p, attrs);
    quote! {
        &#p::static_ir::StaticModule {
            name: #name,
            items: &[#(#items_ts),*],
            attrs: #attrs_ts,
        }
    }
}

/// Emit a `&[StaticItem]` literal (with leading `&`) for the given
/// owned items, suitable as the initializer of
/// `static X: &[ir::static_ir::StaticItem] = ...;`.
pub fn emit_static_items(ir_path: &TokenStream, items: &[ir::Item]) -> TokenStream {
    let p = ir_path;
    let xs = items.iter().map(|i| static_item(p, i));
    quote! { &[#(#xs),*] }
}

/// Emit `Some(#inner)` or `None` for already-emitted inner tokens.
fn some_or_none(inner: Option<TokenStream>) -> TokenStream {
    match inner {
        Some(ts) => quote! { ::std::option::Option::Some(#ts) },
        None => quote! { ::std::option::Option::None },
    }
}

/// Emit `Some(#v)` or `None` for an optional value whose type
/// interpolates directly into `quote!` (numbers, bools).
fn opt_tokens<T: ToTokens>(opt: Option<T>) -> TokenStream {
    match opt {
        Some(v) => quote! { ::std::option::Option::Some(#v) },
        None => quote! { ::std::option::Option::None },
    }
}

/// Emit a `&[&str]` literal.
fn str_slice(strings: &[String]) -> TokenStream {
    quote! { &[#(#strings),*] }
}

// ===== Attributes =====

fn static_attribute(p: &TokenStream, a: &ir::Attribute) -> TokenStream {
    let path = &a.path;
    let args = str_slice(&a.args);
    quote! { #p::static_ir::StaticAttribute { path: #path, args: #args } }
}

/// Emit a `&[StaticAttribute]` literal.
fn static_attrs(p: &TokenStream, attrs: &[ir::Attribute]) -> TokenStream {
    let xs = attrs.iter().map(|a| static_attribute(p, a));
    quote! { &[#(#xs),*] }
}

// ===== Reused const-data enums =====

fn static_scalar_type(p: &TokenStream, s: ir::ScalarType) -> TokenStream {
    match s {
        ir::ScalarType::I32 => quote! { #p::ScalarType::I32 },
        ir::ScalarType::U32 => quote! { #p::ScalarType::U32 },
        ir::ScalarType::F32 => quote! { #p::ScalarType::F32 },
        ir::ScalarType::Bool => quote! { #p::ScalarType::Bool },
    }
}

fn static_address_space(p: &TokenStream, s: ir::AddressSpace) -> TokenStream {
    match s {
        ir::AddressSpace::Function => quote! { #p::AddressSpace::Function },
        ir::AddressSpace::Private => quote! { #p::AddressSpace::Private },
        ir::AddressSpace::Workgroup => quote! { #p::AddressSpace::Workgroup },
    }
}

fn static_storage_access(p: &TokenStream, s: ir::StorageAccess) -> TokenStream {
    match s {
        ir::StorageAccess::Read => quote! { #p::StorageAccess::Read },
        ir::StorageAccess::ReadWrite => quote! { #p::StorageAccess::ReadWrite },
    }
}

fn static_texture_kind(p: &TokenStream, k: ir::TextureKind) -> TokenStream {
    match k {
        ir::TextureKind::Texture1D => quote! { #p::TextureKind::Texture1D },
        ir::TextureKind::Texture2D => quote! { #p::TextureKind::Texture2D },
        ir::TextureKind::Texture2DArray => quote! { #p::TextureKind::Texture2DArray },
        ir::TextureKind::Texture3D => quote! { #p::TextureKind::Texture3D },
        ir::TextureKind::TextureCube => quote! { #p::TextureKind::TextureCube },
        ir::TextureKind::TextureCubeArray => quote! { #p::TextureKind::TextureCubeArray },
        ir::TextureKind::TextureMultisampled2D => {
            quote! { #p::TextureKind::TextureMultisampled2D }
        }
    }
}

fn static_texture_depth_kind(p: &TokenStream, k: ir::TextureDepthKind) -> TokenStream {
    match k {
        ir::TextureDepthKind::Depth2D => quote! { #p::TextureDepthKind::Depth2D },
        ir::TextureDepthKind::Depth2DArray => {
            quote! { #p::TextureDepthKind::Depth2DArray }
        }
        ir::TextureDepthKind::DepthCube => quote! { #p::TextureDepthKind::DepthCube },
        ir::TextureDepthKind::DepthCubeArray => {
            quote! { #p::TextureDepthKind::DepthCubeArray }
        }
        ir::TextureDepthKind::DepthMultisampled2D => {
            quote! { #p::TextureDepthKind::DepthMultisampled2D }
        }
    }
}

fn static_texture_storage_kind(p: &TokenStream, k: ir::TextureStorageKind) -> TokenStream {
    match k {
        ir::TextureStorageKind::Storage1D => quote! { #p::TextureStorageKind::Storage1D },
        ir::TextureStorageKind::Storage2D => quote! { #p::TextureStorageKind::Storage2D },
        ir::TextureStorageKind::Storage2DArray => {
            quote! { #p::TextureStorageKind::Storage2DArray }
        }
        ir::TextureStorageKind::Storage3D => quote! { #p::TextureStorageKind::Storage3D },
    }
}

fn static_storage_texture_access(p: &TokenStream, a: ir::StorageTextureAccess) -> TokenStream {
    match a {
        ir::StorageTextureAccess::Read => quote! { #p::StorageTextureAccess::Read },
        ir::StorageTextureAccess::Write => quote! { #p::StorageTextureAccess::Write },
        ir::StorageTextureAccess::ReadWrite => quote! { #p::StorageTextureAccess::ReadWrite },
    }
}

fn static_texel_format(p: &TokenStream, f: ir::TexelFormat) -> TokenStream {
    match f {
        ir::TexelFormat::Rgba8unorm => quote! { #p::TexelFormat::Rgba8unorm },
        ir::TexelFormat::Rgba8snorm => quote! { #p::TexelFormat::Rgba8snorm },
        ir::TexelFormat::Rgba8uint => quote! { #p::TexelFormat::Rgba8uint },
        ir::TexelFormat::Rgba8sint => quote! { #p::TexelFormat::Rgba8sint },
        ir::TexelFormat::Rgba16uint => quote! { #p::TexelFormat::Rgba16uint },
        ir::TexelFormat::Rgba16sint => quote! { #p::TexelFormat::Rgba16sint },
        ir::TexelFormat::Rgba16float => quote! { #p::TexelFormat::Rgba16float },
        ir::TexelFormat::R32uint => quote! { #p::TexelFormat::R32uint },
        ir::TexelFormat::R32sint => quote! { #p::TexelFormat::R32sint },
        ir::TexelFormat::R32float => quote! { #p::TexelFormat::R32float },
        ir::TexelFormat::Rg32uint => quote! { #p::TexelFormat::Rg32uint },
        ir::TexelFormat::Rg32sint => quote! { #p::TexelFormat::Rg32sint },
        ir::TexelFormat::Rg32float => quote! { #p::TexelFormat::Rg32float },
        ir::TexelFormat::Rgba32uint => quote! { #p::TexelFormat::Rgba32uint },
        ir::TexelFormat::Rgba32sint => quote! { #p::TexelFormat::Rgba32sint },
        ir::TexelFormat::Rgba32float => quote! { #p::TexelFormat::Rgba32float },
        ir::TexelFormat::Bgra8unorm => quote! { #p::TexelFormat::Bgra8unorm },
        ir::TexelFormat::Rgba16unorm => quote! { #p::TexelFormat::Rgba16unorm },
        ir::TexelFormat::Rgba16snorm => quote! { #p::TexelFormat::Rgba16snorm },
        ir::TexelFormat::Rg8unorm => quote! { #p::TexelFormat::Rg8unorm },
        ir::TexelFormat::Rg8snorm => quote! { #p::TexelFormat::Rg8snorm },
        ir::TexelFormat::Rg8uint => quote! { #p::TexelFormat::Rg8uint },
        ir::TexelFormat::Rg8sint => quote! { #p::TexelFormat::Rg8sint },
        ir::TexelFormat::Rg16unorm => quote! { #p::TexelFormat::Rg16unorm },
        ir::TexelFormat::Rg16snorm => quote! { #p::TexelFormat::Rg16snorm },
        ir::TexelFormat::Rg16uint => quote! { #p::TexelFormat::Rg16uint },
        ir::TexelFormat::Rg16sint => quote! { #p::TexelFormat::Rg16sint },
        ir::TexelFormat::Rg16float => quote! { #p::TexelFormat::Rg16float },
        ir::TexelFormat::R8unorm => quote! { #p::TexelFormat::R8unorm },
        ir::TexelFormat::R8snorm => quote! { #p::TexelFormat::R8snorm },
        ir::TexelFormat::R8uint => quote! { #p::TexelFormat::R8uint },
        ir::TexelFormat::R8sint => quote! { #p::TexelFormat::R8sint },
        ir::TexelFormat::R16unorm => quote! { #p::TexelFormat::R16unorm },
        ir::TexelFormat::R16snorm => quote! { #p::TexelFormat::R16snorm },
        ir::TexelFormat::R16uint => quote! { #p::TexelFormat::R16uint },
        ir::TexelFormat::R16sint => quote! { #p::TexelFormat::R16sint },
        ir::TexelFormat::R16float => quote! { #p::TexelFormat::R16float },
        ir::TexelFormat::Rgb10a2unorm => quote! { #p::TexelFormat::Rgb10a2unorm },
        ir::TexelFormat::Rgb10a2uint => quote! { #p::TexelFormat::Rgb10a2uint },
        ir::TexelFormat::Rg11b10ufloat => quote! { #p::TexelFormat::Rg11b10ufloat },
    }
}

fn static_bin_op(p: &TokenStream, op: ir::BinOp) -> TokenStream {
    match op {
        ir::BinOp::Add => quote! { #p::BinOp::Add },
        ir::BinOp::Sub => quote! { #p::BinOp::Sub },
        ir::BinOp::Mul => quote! { #p::BinOp::Mul },
        ir::BinOp::Div => quote! { #p::BinOp::Div },
        ir::BinOp::Rem => quote! { #p::BinOp::Rem },
        ir::BinOp::Eq => quote! { #p::BinOp::Eq },
        ir::BinOp::Ne => quote! { #p::BinOp::Ne },
        ir::BinOp::Lt => quote! { #p::BinOp::Lt },
        ir::BinOp::Le => quote! { #p::BinOp::Le },
        ir::BinOp::Gt => quote! { #p::BinOp::Gt },
        ir::BinOp::Ge => quote! { #p::BinOp::Ge },
        ir::BinOp::And => quote! { #p::BinOp::And },
        ir::BinOp::Or => quote! { #p::BinOp::Or },
        ir::BinOp::BitAnd => quote! { #p::BinOp::BitAnd },
        ir::BinOp::BitOr => quote! { #p::BinOp::BitOr },
        ir::BinOp::BitXor => quote! { #p::BinOp::BitXor },
        ir::BinOp::Shl => quote! { #p::BinOp::Shl },
        ir::BinOp::Shr => quote! { #p::BinOp::Shr },
    }
}

fn static_compound_op(p: &TokenStream, op: ir::CompoundOp) -> TokenStream {
    match op {
        ir::CompoundOp::AddAssign => quote! { #p::CompoundOp::AddAssign },
        ir::CompoundOp::SubAssign => quote! { #p::CompoundOp::SubAssign },
        ir::CompoundOp::MulAssign => quote! { #p::CompoundOp::MulAssign },
        ir::CompoundOp::DivAssign => quote! { #p::CompoundOp::DivAssign },
        ir::CompoundOp::RemAssign => quote! { #p::CompoundOp::RemAssign },
        ir::CompoundOp::BitAndAssign => quote! { #p::CompoundOp::BitAndAssign },
        ir::CompoundOp::BitOrAssign => quote! { #p::CompoundOp::BitOrAssign },
        ir::CompoundOp::BitXorAssign => quote! { #p::CompoundOp::BitXorAssign },
        ir::CompoundOp::ShlAssign => quote! { #p::CompoundOp::ShlAssign },
        ir::CompoundOp::ShrAssign => quote! { #p::CompoundOp::ShrAssign },
    }
}

fn static_un_op(p: &TokenStream, op: ir::UnOp) -> TokenStream {
    match op {
        ir::UnOp::Not => quote! { #p::UnOp::Not },
        ir::UnOp::Complement => quote! { #p::UnOp::Complement },
        ir::UnOp::Neg => quote! { #p::UnOp::Neg },
        ir::UnOp::Deref => quote! { #p::UnOp::Deref },
    }
}

fn static_builtin(p: &TokenStream, b: ir::BuiltIn) -> TokenStream {
    match b {
        ir::BuiltIn::VertexIndex => quote! { #p::BuiltIn::VertexIndex },
        ir::BuiltIn::InstanceIndex => quote! { #p::BuiltIn::InstanceIndex },
        ir::BuiltIn::Position => quote! { #p::BuiltIn::Position },
        ir::BuiltIn::FrontFacing => quote! { #p::BuiltIn::FrontFacing },
        ir::BuiltIn::FragDepth => quote! { #p::BuiltIn::FragDepth },
        ir::BuiltIn::SampleIndex => quote! { #p::BuiltIn::SampleIndex },
        ir::BuiltIn::SampleMask => quote! { #p::BuiltIn::SampleMask },
        ir::BuiltIn::LocalInvocationId => quote! { #p::BuiltIn::LocalInvocationId },
        ir::BuiltIn::LocalInvocationIndex => quote! { #p::BuiltIn::LocalInvocationIndex },
        ir::BuiltIn::GlobalInvocationId => quote! { #p::BuiltIn::GlobalInvocationId },
        ir::BuiltIn::WorkgroupId => quote! { #p::BuiltIn::WorkgroupId },
        ir::BuiltIn::NumWorkgroups => quote! { #p::BuiltIn::NumWorkgroups },
        ir::BuiltIn::SubgroupInvocationId => quote! { #p::BuiltIn::SubgroupInvocationId },
        ir::BuiltIn::SubgroupSize => quote! { #p::BuiltIn::SubgroupSize },
        ir::BuiltIn::PrimitiveIndex => quote! { #p::BuiltIn::PrimitiveIndex },
        ir::BuiltIn::SubgroupId => quote! { #p::BuiltIn::SubgroupId },
        ir::BuiltIn::NumSubgroups => quote! { #p::BuiltIn::NumSubgroups },
    }
}

fn static_interpolation_type(p: &TokenStream, t: ir::InterpolationType) -> TokenStream {
    match t {
        ir::InterpolationType::Perspective => quote! { #p::InterpolationType::Perspective },
        ir::InterpolationType::Linear => quote! { #p::InterpolationType::Linear },
        ir::InterpolationType::Flat => quote! { #p::InterpolationType::Flat },
    }
}

fn static_interpolation_sampling(p: &TokenStream, s: ir::InterpolationSampling) -> TokenStream {
    match s {
        ir::InterpolationSampling::Center => quote! { #p::InterpolationSampling::Center },
        ir::InterpolationSampling::Centroid => quote! { #p::InterpolationSampling::Centroid },
        ir::InterpolationSampling::Sample => quote! { #p::InterpolationSampling::Sample },
        ir::InterpolationSampling::First => quote! { #p::InterpolationSampling::First },
        ir::InterpolationSampling::Either => quote! { #p::InterpolationSampling::Either },
    }
}

fn static_interpolate(p: &TokenStream, i: &ir::Interpolate) -> TokenStream {
    let t = static_interpolation_type(p, i.ty);
    let s = some_or_none(i.sampling.map(|s| static_interpolation_sampling(p, s)));
    quote! { #p::Interpolate { ty: #t, sampling: #s } }
}

fn static_inter_stage_io(p: &TokenStream, io: &ir::InterStageIo) -> TokenStream {
    match io {
        ir::InterStageIo::BuiltIn(b) => {
            let b = static_builtin(p, *b);
            quote! { #p::InterStageIo::BuiltIn(#b) }
        }
        ir::InterStageIo::Location(n) => quote! { #p::InterStageIo::Location(#n) },
        ir::InterStageIo::BlendSrc(n) => quote! { #p::InterStageIo::BlendSrc(#n) },
        ir::InterStageIo::Interpolate(i) => {
            let i = static_interpolate(p, i);
            quote! { #p::InterStageIo::Interpolate(#i) }
        }
        ir::InterStageIo::Invariant => quote! { #p::InterStageIo::Invariant },
    }
}

/// Emit a `&[InterStageIo]` literal.
fn static_inter_stage_io_slice(p: &TokenStream, ios: &[ir::InterStageIo]) -> TokenStream {
    let xs = ios.iter().map(|io| static_inter_stage_io(p, io));
    quote! { &[#(#xs),*] }
}

fn static_workgroup_size(p: &TokenStream, w: &ir::WorkgroupSize) -> TokenStream {
    let x = w.x;
    let y = opt_tokens(w.y);
    let z = opt_tokens(w.z);
    quote! { #p::WorkgroupSize { x: #x, y: #y, z: #z } }
}

fn static_fn_attrs(p: &TokenStream, f: &ir::FnAttrs) -> TokenStream {
    match f {
        ir::FnAttrs::None => quote! { #p::FnAttrs::None },
        ir::FnAttrs::Vertex => quote! { #p::FnAttrs::Vertex },
        ir::FnAttrs::Fragment => quote! { #p::FnAttrs::Fragment },
        ir::FnAttrs::Compute { workgroup_size } => {
            let ws = static_workgroup_size(p, workgroup_size);
            quote! { #p::FnAttrs::Compute { workgroup_size: #ws } }
        }
    }
}

fn static_return_type_annotation(p: &TokenStream, a: &ir::ReturnTypeAnnotation) -> TokenStream {
    match a {
        ir::ReturnTypeAnnotation::None => quote! { #p::ReturnTypeAnnotation::None },
        ir::ReturnTypeAnnotation::BuiltIn(b) => {
            let b = static_builtin(p, *b);
            quote! { #p::ReturnTypeAnnotation::BuiltIn(#b) }
        }
        ir::ReturnTypeAnnotation::Location(n) => quote! { #p::ReturnTypeAnnotation::Location(#n) },
        ir::ReturnTypeAnnotation::DefaultBuiltInPosition => {
            quote! { #p::ReturnTypeAnnotation::DefaultBuiltInPosition }
        }
        ir::ReturnTypeAnnotation::DefaultLocation => {
            quote! { #p::ReturnTypeAnnotation::DefaultLocation }
        }
    }
}

// ===== Types =====

fn static_type(p: &TokenStream, t: &ir::Type) -> TokenStream {
    match t {
        ir::Type::Scalar(s) => {
            let s = static_scalar_type(p, *s);
            quote! { #p::static_ir::StaticType::Scalar(#s) }
        }
        ir::Type::Vector {
            elements,
            scalar_ty,
        } => {
            let s = some_or_none(scalar_ty.map(|s| static_scalar_type(p, s)));
            quote! { #p::static_ir::StaticType::Vector { elements: #elements, scalar_ty: #s } }
        }
        ir::Type::Matrix {
            columns,
            rows,
            scalar_ty,
        } => {
            let s = some_or_none(scalar_ty.map(|s| static_scalar_type(p, s)));
            quote! {
                #p::static_ir::StaticType::Matrix {
                    columns: #columns,
                    rows: #rows,
                    scalar_ty: #s,
                }
            }
        }
        ir::Type::Array { elem, len } => {
            let e = static_type(p, elem);
            let l = static_expr(p, len);
            quote! { #p::static_ir::StaticType::Array { elem: &#e, len: #l } }
        }
        ir::Type::RuntimeArray { elem } => {
            let e = static_type(p, elem);
            quote! { #p::static_ir::StaticType::RuntimeArray { elem: &#e } }
        }
        ir::Type::Atomic { elem } => {
            let e = static_type(p, elem);
            quote! { #p::static_ir::StaticType::Atomic { elem: &#e } }
        }
        ir::Type::Struct { name, type_args } => {
            let args: Vec<_> = type_args.iter().map(|t| static_type(p, t)).collect();
            quote! {
                #p::static_ir::StaticType::Struct {
                    name: #name,
                    type_args: &[#(#args),*],
                }
            }
        }
        ir::Type::Ptr {
            address_space,
            elem,
        } => {
            let a = static_address_space(p, *address_space);
            let e = static_type(p, elem);
            quote! { #p::static_ir::StaticType::Ptr { address_space: #a, elem: &#e } }
        }
        ir::Type::Sampler => quote! { #p::static_ir::StaticType::Sampler },
        ir::Type::SamplerComparison => quote! { #p::static_ir::StaticType::SamplerComparison },
        ir::Type::Texture { kind, sampled_type } => {
            let k = static_texture_kind(p, *kind);
            let s = static_scalar_type(p, *sampled_type);
            quote! {
                #p::static_ir::StaticType::Texture {
                    kind: #k,
                    sampled_type: #s,
                }
            }
        }
        ir::Type::TextureDepth { kind } => {
            let k = static_texture_depth_kind(p, *kind);
            quote! { #p::static_ir::StaticType::TextureDepth { kind: #k } }
        }
        ir::Type::TextureStorage {
            kind,
            format,
            access,
        } => {
            let k = static_texture_storage_kind(p, *kind);
            let f = static_texel_format(p, *format);
            let a = static_storage_texture_access(p, *access);
            quote! {
                #p::static_ir::StaticType::TextureStorage {
                    kind: #k,
                    format: #f,
                    access: #a,
                }
            }
        }
        ir::Type::TypeParam { name } => {
            quote! { #p::static_ir::StaticType::TypeParam { name: #name } }
        }
        ir::Type::Phantom { elem } => {
            let e = static_type(p, elem);
            quote! { #p::static_ir::StaticType::Phantom { elem: &#e } }
        }
        ir::Type::AssocType { ty, member } => {
            let t = static_type(p, ty);
            quote! { #p::static_ir::StaticType::AssocType { ty: &#t, member: #member } }
        }
    }
}

// ===== Literals / paths =====

fn static_lit(p: &TokenStream, l: &ir::Lit) -> TokenStream {
    match l {
        ir::Lit::Bool(b) => quote! { #p::static_ir::StaticLit::Bool(#b) },
        ir::Lit::Int { digits, suffix } => {
            quote! { #p::static_ir::StaticLit::Int { digits: #digits, suffix: #suffix } }
        }
        ir::Lit::Float { text } => quote! { #p::static_ir::StaticLit::Float { text: #text } },
    }
}

fn static_fn_path(p: &TokenStream, f: &ir::FnPath) -> TokenStream {
    match f {
        ir::FnPath::Ident(name) => quote! { #p::static_ir::StaticFnPath::Ident(#name) },
        ir::FnPath::TypeMethod { ty, method } => {
            quote! { #p::static_ir::StaticFnPath::TypeMethod { ty: #ty, method: #method } }
        }
    }
}

// ===== Expressions =====

fn static_field_value(p: &TokenStream, f: &ir::FieldValue) -> TokenStream {
    let member = &f.member;
    let e = static_expr(p, &f.expr);
    quote! { #p::static_ir::StaticFieldValue { member: #member, expr: #e } }
}

fn static_expr(p: &TokenStream, e: &ir::Expr) -> TokenStream {
    match e {
        ir::Expr::Lit(l) => {
            let l = static_lit(p, l);
            quote! { #p::static_ir::StaticExpr::Lit(#l) }
        }
        ir::Expr::Ident(name) => quote! { #p::static_ir::StaticExpr::Ident(#name) },
        ir::Expr::Array { elems } => {
            let xs: Vec<_> = elems.iter().map(|e| static_expr(p, e)).collect();
            quote! { #p::static_ir::StaticExpr::Array { elems: &[#(#xs),*] } }
        }
        ir::Expr::Paren(inner) => {
            let i = static_expr(p, inner);
            quote! { #p::static_ir::StaticExpr::Paren(&#i) }
        }
        ir::Expr::Binary { lhs, op, rhs } => {
            let l = static_expr(p, lhs);
            let o = static_bin_op(p, *op);
            let r = static_expr(p, rhs);
            quote! { #p::static_ir::StaticExpr::Binary { lhs: &#l, op: #o, rhs: &#r } }
        }
        ir::Expr::Unary { op, expr } => {
            let o = static_un_op(p, *op);
            let inner = static_expr(p, expr);
            quote! { #p::static_ir::StaticExpr::Unary { op: #o, expr: &#inner } }
        }
        ir::Expr::ArrayIndexing { lhs, index } => {
            let l = static_expr(p, lhs);
            let i = static_expr(p, index);
            quote! { #p::static_ir::StaticExpr::ArrayIndexing { lhs: &#l, index: &#i } }
        }
        ir::Expr::Swizzle {
            lhs,
            swizzle,
            params,
        } => {
            let l = static_expr(p, lhs);
            let ps = some_or_none(params.as_ref().map(|ps| {
                let xs: Vec<_> = ps.iter().map(|e| static_expr(p, e)).collect();
                quote! { &[#(#xs),*] }
            }));
            quote! {
                #p::static_ir::StaticExpr::Swizzle {
                    lhs: &#l,
                    swizzle: #swizzle,
                    params: #ps,
                }
            }
        }
        ir::Expr::Cast { lhs, ty } => {
            let l = static_expr(p, lhs);
            let t = static_type(p, ty);
            quote! { #p::static_ir::StaticExpr::Cast { lhs: &#l, ty: &#t } }
        }
        ir::Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            let pt = static_fn_path(p, path);
            let ta: Vec<_> = type_args.iter().map(|t| static_type(p, t)).collect();
            let ps: Vec<_> = params.iter().map(|e| static_expr(p, e)).collect();
            quote! {
                #p::static_ir::StaticExpr::FnCall {
                    path: #pt,
                    type_args: &[#(#ta),*],
                    params: &[#(#ps),*],
                }
            }
        }
        ir::Expr::Struct {
            name,
            type_args,
            fields,
        } => {
            let ta: Vec<_> = type_args.iter().map(|t| static_type(p, t)).collect();
            let fs: Vec<_> = fields.iter().map(|f| static_field_value(p, f)).collect();
            quote! {
                #p::static_ir::StaticExpr::Struct {
                    name: #name,
                    type_args: &[#(#ta),*],
                    fields: &[#(#fs),*],
                }
            }
        }
        ir::Expr::FieldAccess { base, field } => {
            let b = static_expr(p, base);
            quote! { #p::static_ir::StaticExpr::FieldAccess { base: &#b, field: #field } }
        }
        ir::Expr::TypePath { ty, member } => {
            quote! { #p::static_ir::StaticExpr::TypePath { ty: #ty, member: #member } }
        }
        ir::Expr::Reference(inner) => {
            let i = static_expr(p, inner);
            quote! { #p::static_ir::StaticExpr::Reference(&#i) }
        }
        ir::Expr::ZeroValueArray { elem_type, len } => {
            let et = static_type(p, elem_type);
            let l = static_expr(p, len);
            quote! {
                #p::static_ir::StaticExpr::ZeroValueArray {
                    elem_type: &#et,
                    len: &#l,
                }
            }
        }
    }
}

// ===== Statements =====

fn static_block(p: &TokenStream, b: &ir::Block) -> TokenStream {
    let xs: Vec<_> = b.stmts.iter().map(|s| static_stmt(p, s)).collect();
    quote! { #p::static_ir::StaticBlock { stmts: &[#(#xs),*] } }
}

fn static_local(p: &TokenStream, l: &ir::Local) -> TokenStream {
    let mutable = l.mutable;
    let name = &l.name;
    let t = some_or_none(l.ty.as_ref().map(|t| static_type(p, t)));
    let i = some_or_none(l.init.as_ref().map(|e| static_expr(p, e)));
    quote! {
        #p::static_ir::StaticLocal {
            mutable: #mutable,
            name: #name,
            ty: #t,
            init: #i,
        }
    }
}

fn static_for_loop(p: &TokenStream, f: &ir::ForLoop) -> TokenStream {
    let var = &f.var;
    let inclusive = f.inclusive;
    let vt = some_or_none(f.var_ty.as_ref().map(|t| static_type(p, t)));
    let from = static_expr(p, &f.from);
    let to = static_expr(p, &f.to);
    let body = static_block(p, &f.body);
    quote! {
        #p::static_ir::StaticForLoop {
            var: #var,
            var_ty: #vt,
            from: #from,
            to: #to,
            inclusive: #inclusive,
            body: #body,
        }
    }
}

fn static_stmt_if(p: &TokenStream, i: &ir::StmtIf) -> TokenStream {
    let condition = static_expr(p, &i.condition);
    let then_block = static_block(p, &i.then_block);
    let else_branch = some_or_none(i.else_branch.as_ref().map(|b| static_else_branch(p, b)));
    quote! {
        #p::static_ir::StaticStmtIf {
            condition: #condition,
            then_block: #then_block,
            else_branch: #else_branch,
        }
    }
}

fn static_else_branch(p: &TokenStream, b: &ir::ElseBranch) -> TokenStream {
    match b {
        ir::ElseBranch::Block(block) => {
            let inner = static_block(p, block);
            quote! { #p::static_ir::StaticElseBranch::Block(#inner) }
        }
        ir::ElseBranch::If(if_stmt) => {
            let inner = static_stmt_if(p, if_stmt);
            quote! { #p::static_ir::StaticElseBranch::If(&#inner) }
        }
    }
}

fn static_stmt_switch(p: &TokenStream, sw: &ir::StmtSwitch) -> TokenStream {
    let selector = static_expr(p, &sw.selector);
    let has_explicit_default = sw.has_explicit_default;
    let arms: Vec<_> = sw.arms.iter().map(|a| static_switch_arm(p, a)).collect();
    quote! {
        #p::static_ir::StaticStmtSwitch {
            selector: #selector,
            arms: &[#(#arms),*],
            has_explicit_default: #has_explicit_default,
        }
    }
}

fn static_switch_arm(p: &TokenStream, arm: &ir::SwitchArm) -> TokenStream {
    let selectors: Vec<_> = arm
        .selectors
        .iter()
        .map(|c| static_case_selector(p, c))
        .collect();
    let body = static_block(p, &arm.body);
    quote! {
        #p::static_ir::StaticSwitchArm {
            selectors: &[#(#selectors),*],
            body: #body,
        }
    }
}

fn static_case_selector(p: &TokenStream, c: &ir::CaseSelector) -> TokenStream {
    match c {
        ir::CaseSelector::Literal(lit) => {
            let l = static_lit(p, lit);
            quote! { #p::static_ir::StaticCaseSelector::Literal(#l) }
        }
        ir::CaseSelector::Expr(e) => {
            let inner = static_expr(p, e);
            quote! { #p::static_ir::StaticCaseSelector::Expr(#inner) }
        }
        ir::CaseSelector::Default => quote! { #p::static_ir::StaticCaseSelector::Default },
    }
}

fn static_stmt(p: &TokenStream, s: &ir::Stmt) -> TokenStream {
    match s {
        ir::Stmt::Local(l) => {
            let inner = static_local(p, l);
            quote! { #p::static_ir::StaticStmt::Local(#inner) }
        }
        ir::Stmt::Const(c) => {
            let inner = static_item_const(p, c);
            quote! { #p::static_ir::StaticStmt::Const(#inner) }
        }
        ir::Stmt::Assignment { lhs, rhs } => {
            let l = static_expr(p, lhs);
            let r = static_expr(p, rhs);
            quote! { #p::static_ir::StaticStmt::Assignment { lhs: #l, rhs: #r } }
        }
        ir::Stmt::CompoundAssignment { lhs, op, rhs } => {
            let l = static_expr(p, lhs);
            let o = static_compound_op(p, *op);
            let r = static_expr(p, rhs);
            quote! {
                #p::static_ir::StaticStmt::CompoundAssignment {
                    lhs: #l,
                    op: #o,
                    rhs: #r,
                }
            }
        }
        ir::Stmt::While { condition, body } => {
            let c = static_expr(p, condition);
            let b = static_block(p, body);
            quote! { #p::static_ir::StaticStmt::While { condition: #c, body: #b } }
        }
        ir::Stmt::Loop { body } => {
            let b = static_block(p, body);
            quote! { #p::static_ir::StaticStmt::Loop { body: #b } }
        }
        ir::Stmt::Expr { expr, has_semi } => {
            let e = static_expr(p, expr);
            quote! { #p::static_ir::StaticStmt::Expr { expr: #e, has_semi: #has_semi } }
        }
        ir::Stmt::If(i) => {
            let inner = static_stmt_if(p, i);
            quote! { #p::static_ir::StaticStmt::If(#inner) }
        }
        ir::Stmt::Break => quote! { #p::static_ir::StaticStmt::Break },
        ir::Stmt::Continue => quote! { #p::static_ir::StaticStmt::Continue },
        ir::Stmt::Return(e) => {
            let inner = some_or_none(e.as_ref().map(|e| static_expr(p, e)));
            quote! { #p::static_ir::StaticStmt::Return(#inner) }
        }
        ir::Stmt::For(f) => {
            let inner = static_for_loop(p, f);
            quote! { #p::static_ir::StaticStmt::For(#inner) }
        }
        ir::Stmt::Switch(sw) => {
            let inner = static_stmt_switch(p, sw);
            quote! { #p::static_ir::StaticStmt::Switch(#inner) }
        }
        ir::Stmt::Block(b) => {
            let inner = static_block(p, b);
            quote! { #p::static_ir::StaticStmt::Block(#inner) }
        }
        ir::Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            let s1 = static_expr(p, src);
            let s2 = static_expr(p, src_offset);
            let d1 = static_expr(p, dest);
            let d2 = static_expr(p, dest_offset);
            let sz = static_expr(p, size);
            quote! {
                #p::static_ir::StaticStmt::SlabCopy {
                    src: #s1,
                    src_offset: #s2,
                    dest: #d1,
                    dest_offset: #d2,
                    size: #sz,
                }
            }
        }
        ir::Stmt::Discard => quote! { #p::static_ir::StaticStmt::Discard },
        ir::Stmt::Macro { name, args } => {
            quote! { #p::static_ir::StaticStmt::Macro { name: #name, args: #args } }
        }
    }
}

// ===== Function / field pieces =====

fn static_return_type(p: &TokenStream, r: &ir::ReturnType) -> TokenStream {
    match r {
        ir::ReturnType::Default => quote! { #p::static_ir::StaticReturnType::Default },
        ir::ReturnType::Type { annotation, ty } => {
            let a = static_return_type_annotation(p, annotation);
            let t = static_type(p, ty);
            quote! { #p::static_ir::StaticReturnType::Type { annotation: #a, ty: #t } }
        }
    }
}

fn static_fn_arg(p: &TokenStream, a: &ir::FnArg) -> TokenStream {
    let isi = static_inter_stage_io_slice(p, &a.inter_stage_io);
    let name = &a.name;
    let t = static_type(p, &a.ty);
    let attrs = static_attrs(p, &a.attrs);
    quote! {
        #p::static_ir::StaticFnArg {
            inter_stage_io: #isi,
            name: #name,
            ty: #t,
            attrs: #attrs,
        }
    }
}

fn static_field(p: &TokenStream, f: &ir::Field) -> TokenStream {
    let isi = static_inter_stage_io_slice(p, &f.inter_stage_io);
    let name = &f.name;
    let t = static_type(p, &f.ty);
    let attrs = static_attrs(p, &f.attrs);
    quote! {
        #p::static_ir::StaticField {
            inter_stage_io: #isi,
            name: #name,
            ty: #t,
            attrs: #attrs,
        }
    }
}

// ===== Items =====

fn static_item_const(p: &TokenStream, c: &ir::ItemConst) -> TokenStream {
    let name = &c.name;
    let t = static_type(p, &c.ty);
    let e = static_expr(p, &c.expr);
    let attrs = static_attrs(p, &c.attrs);
    quote! {
        #p::static_ir::StaticItemConst {
            name: #name,
            ty: #t,
            expr: #e,
            attrs: #attrs,
        }
    }
}

fn static_item_uniform(p: &TokenStream, u: &ir::ItemUniform) -> TokenStream {
    let group = u.group;
    let binding = u.binding;
    let name = &u.name;
    let t = static_type(p, &u.ty);
    let attrs = static_attrs(p, &u.attrs);
    quote! {
        #p::static_ir::StaticItemUniform {
            group: #group,
            binding: #binding,
            name: #name,
            ty: #t,
            attrs: #attrs,
        }
    }
}

fn static_item_storage(p: &TokenStream, s: &ir::ItemStorage) -> TokenStream {
    let group = s.group;
    let binding = s.binding;
    let name = &s.name;
    let a = static_storage_access(p, s.access);
    let t = static_type(p, &s.ty);
    let attrs = static_attrs(p, &s.attrs);
    quote! {
        #p::static_ir::StaticItemStorage {
            group: #group,
            binding: #binding,
            access: #a,
            name: #name,
            ty: #t,
            attrs: #attrs,
        }
    }
}

fn static_item_workgroup(p: &TokenStream, w: &ir::ItemWorkgroup) -> TokenStream {
    let name = &w.name;
    let t = static_type(p, &w.ty);
    let attrs = static_attrs(p, &w.attrs);
    quote! {
        #p::static_ir::StaticItemWorkgroup {
            name: #name,
            ty: #t,
            attrs: #attrs,
        }
    }
}

fn static_filterability(p: &TokenStream, filterable: &ir::Filterability) -> TokenStream {
    match filterable {
        ir::Filterability::Concrete(value) => quote! {
            #p::static_ir::StaticFilterability::Concrete(#value)
        },
        ir::Filterability::Param(name) => quote! {
            #p::static_ir::StaticFilterability::Param(#name)
        },
    }
}

fn static_item_sampler(p: &TokenStream, s: &ir::ItemSampler) -> TokenStream {
    let group = s.group;
    let binding = s.binding;
    let name = &s.name;
    let filterable = static_filterability(p, &s.filterable);
    let t = static_type(p, &s.ty);
    let attrs = static_attrs(p, &s.attrs);
    quote! {
        #p::static_ir::StaticItemSampler {
            group: #group,
            binding: #binding,
            name: #name,
            ty: #t,
            filterable: #filterable,
            attrs: #attrs,
        }
    }
}

fn static_item_texture(p: &TokenStream, t: &ir::ItemTexture) -> TokenStream {
    let group = t.group;
    let binding = t.binding;
    let name = &t.name;
    let filterable = static_filterability(p, &t.filterable);
    let ty = static_type(p, &t.ty);
    let attrs = static_attrs(p, &t.attrs);
    quote! {
        #p::static_ir::StaticItemTexture {
            group: #group,
            binding: #binding,
            name: #name,
            ty: #ty,
            filterable: #filterable,
            attrs: #attrs,
        }
    }
}

fn static_item_fn(p: &TokenStream, f: &ir::ItemFn) -> TokenStream {
    let type_params = str_slice(&f.type_params);
    let const_params = str_slice(&f.const_params);
    let fn_attrs = static_fn_attrs(p, &f.fn_attrs);
    let name = f.name.as_ref();
    let inputs: Vec<_> = f.inputs.iter().map(|a| static_fn_arg(p, a)).collect();
    let return_type = static_return_type(p, &f.return_type);
    let block = static_block(p, &f.block);
    let attrs = static_attrs(p, &f.attrs);
    quote! {
        #p::static_ir::StaticItemFn {
            type_params: #type_params,
            const_params: #const_params,
            fn_attrs: #fn_attrs,
            name: #name,
            inputs: &[#(#inputs),*],
            return_type: #return_type,
            block: #block,
            attrs: #attrs,
        }
    }
}

fn static_item_struct(p: &TokenStream, s: &ir::ItemStruct) -> TokenStream {
    let name = &s.name;
    let type_params = str_slice(&s.type_params);
    let const_params = str_slice(&s.const_params);
    let fields: Vec<_> = s.fields.iter().map(|f| static_field(p, f)).collect();
    let attrs = static_attrs(p, &s.attrs);
    quote! {
        #p::static_ir::StaticItemStruct {
            type_params: #type_params,
            const_params: #const_params,
            name: #name,
            fields: &[#(#fields),*],
            attrs: #attrs,
        }
    }
}

fn static_item_type_alias(p: &TokenStream, t: &ir::ItemTypeAlias) -> TokenStream {
    let name = &t.name;
    let ty = static_type(p, &t.ty);
    quote! { #p::static_ir::StaticItemTypeAlias { name: #name, ty: #ty } }
}

fn static_impl_item(p: &TokenStream, i: &ir::ImplItem) -> TokenStream {
    match i {
        ir::ImplItem::Fn(f) => {
            let inner = static_item_fn(p, f);
            quote! { #p::static_ir::StaticImplItem::Fn(#inner) }
        }
        ir::ImplItem::Const(c) => {
            let inner = static_item_const(p, c);
            quote! { #p::static_ir::StaticImplItem::Const(#inner) }
        }
        ir::ImplItem::Type(t) => {
            let inner = static_item_type_alias(p, t);
            quote! { #p::static_ir::StaticImplItem::Type(#inner) }
        }
    }
}

fn static_item_impl(p: &TokenStream, i: &ir::ItemImpl) -> TokenStream {
    let self_ty = &i.self_ty;
    let type_params = str_slice(&i.type_params);
    let const_params = str_slice(&i.const_params);
    let items: Vec<_> = i
        .items
        .iter()
        .map(|item| static_impl_item(p, item))
        .collect();
    let attrs = static_attrs(p, &i.attrs);
    quote! {
        #p::static_ir::StaticItemImpl {
            type_params: #type_params,
            const_params: #const_params,
            self_ty: #self_ty,
            items: &[#(#items),*],
            attrs: #attrs,
        }
    }
}

fn static_enum_variant(p: &TokenStream, v: &ir::EnumVariant) -> TokenStream {
    let name = &v.name;
    let discriminant = opt_tokens(v.discriminant);
    let attrs = static_attrs(p, &v.attrs);
    quote! {
        #p::static_ir::StaticEnumVariant {
            name: #name,
            discriminant: #discriminant,
            attrs: #attrs,
        }
    }
}

fn static_item_enum(p: &TokenStream, e: &ir::ItemEnum) -> TokenStream {
    let name = &e.name;
    let variants: Vec<_> = e
        .variants
        .iter()
        .map(|v| static_enum_variant(p, v))
        .collect();
    let attrs = static_attrs(p, &e.attrs);
    quote! {
        #p::static_ir::StaticItemEnum {
            name: #name,
            variants: &[#(#variants),*],
            attrs: #attrs,
        }
    }
}

fn static_item(p: &TokenStream, i: &ir::Item) -> TokenStream {
    match i {
        ir::Item::Const(c) => {
            let inner = static_item_const(p, c);
            quote! { #p::static_ir::StaticItem::Const(#inner) }
        }
        ir::Item::Uniform(u) => {
            let inner = static_item_uniform(p, u);
            quote! { #p::static_ir::StaticItem::Uniform(#inner) }
        }
        ir::Item::Storage(s) => {
            let inner = static_item_storage(p, s);
            quote! { #p::static_ir::StaticItem::Storage(#inner) }
        }
        ir::Item::Workgroup(w) => {
            let inner = static_item_workgroup(p, w);
            quote! { #p::static_ir::StaticItem::Workgroup(#inner) }
        }
        ir::Item::Sampler(s) => {
            let inner = static_item_sampler(p, s);
            quote! { #p::static_ir::StaticItem::Sampler(#inner) }
        }
        ir::Item::Texture(t) => {
            let inner = static_item_texture(p, t);
            quote! { #p::static_ir::StaticItem::Texture(#inner) }
        }
        ir::Item::Fn(f) => {
            let inner = static_item_fn(p, f);
            quote! { #p::static_ir::StaticItem::Fn(#inner) }
        }
        ir::Item::Struct(s) => {
            let inner = static_item_struct(p, s);
            quote! { #p::static_ir::StaticItem::Struct(#inner) }
        }
        ir::Item::Impl(i) => {
            let inner = static_item_impl(p, i);
            quote! { #p::static_ir::StaticItem::Impl(#inner) }
        }
        ir::Item::Enum(e) => {
            let inner = static_item_enum(p, e);
            quote! { #p::static_ir::StaticItem::Enum(#inner) }
        }
    }
}
