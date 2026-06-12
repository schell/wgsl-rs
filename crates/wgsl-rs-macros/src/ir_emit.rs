//! Emit a `quote::TokenStream` that, when expanded in the consumer crate,
//! constructs the equivalent IR value at runtime.
//!
//! `ir_emit::module(...)` returns a `TokenStream` whose body is a series
//! of expressions that build a `wgsl_rs_ir::Module` value. The output is
//! intended to be the body of a `fn() -> wgsl_rs_ir::Module` constructor
//! emitted from the proc-macro into the consuming crate.
//!
//! The emitter assumes the consuming crate has `wgsl_rs_ir` available
//! under a path it can name. The path is provided by the caller (typically
//! `wgsl_rs::ir` or similar).

use proc_macro2::TokenStream;
use quote::quote;
use wgsl_rs_ir as ir;

/// Emit an expression that constructs the given module IR at runtime. The
/// emitted code references types under `#ir_path` (e.g. `wgsl_rs::ir`).
#[allow(dead_code)]
pub fn emit_module(ir_path: &TokenStream, m: &ir::Module) -> TokenStream {
    let name = &m.name;
    let items = m.items.iter().map(|i| emit_item(ir_path, i));
    let attrs = emit_attrs(ir_path, &m.attrs);
    quote! {
        #ir_path::Module {
            name: ::std::string::String::from(#name),
            items: ::std::vec![#(#items),*],
            #attrs,
        }
    }
}

/// Emit an expression that constructs a `Vec<ir::Item>` at runtime.
#[allow(dead_code)]
pub fn emit_items(ir_path: &TokenStream, items: &[ir::Item]) -> TokenStream {
    let xs = items.iter().map(|i| emit_item(ir_path, i));
    quote! {
        ::std::vec![#(#xs),*]
    }
}

/// Emit an expression that constructs an `ir::Type` at runtime.
pub fn emit_type(ir_path: &TokenStream, t: &ir::Type) -> TokenStream {
    ty(ir_path, t)
}

// ===== Items =====

pub fn emit_item(p: &TokenStream, i: &ir::Item) -> TokenStream {
    match i {
        ir::Item::Const(c) => {
            let inner = item_const(p, c);
            quote! { #p::Item::Const(#inner) }
        }
        ir::Item::Uniform(u) => {
            let group = u.group;
            let binding = u.binding;
            let n = &u.name;
            let t = ty(p, &u.ty);
            let attrs = emit_attrs(p, &u.attrs);
            quote! {
                #p::Item::Uniform(#p::ItemUniform {
                    group: #group,
                    binding: #binding,
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    #attrs,
                })
            }
        }
        ir::Item::Storage(s) => {
            let group = s.group;
            let binding = s.binding;
            let acc = match s.access {
                ir::StorageAccess::Read => quote! { #p::StorageAccess::Read },
                ir::StorageAccess::ReadWrite => quote! { #p::StorageAccess::ReadWrite },
            };
            let n = &s.name;
            let t = ty(p, &s.ty);
            let attrs = emit_attrs(p, &s.attrs);
            quote! {
                #p::Item::Storage(#p::ItemStorage {
                    group: #group,
                    binding: #binding,
                    access: #acc,
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    #attrs,
                })
            }
        }
        ir::Item::Workgroup(w) => {
            let n = &w.name;
            let t = ty(p, &w.ty);
            let attrs = emit_attrs(p, &w.attrs);
            quote! {
                #p::Item::Workgroup(#p::ItemWorkgroup {
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    #attrs,
                })
            }
        }
        ir::Item::Sampler(s) => {
            let group = s.group;
            let binding = s.binding;
            let n = &s.name;
            let t = ty(p, &s.ty);
            let attrs = emit_attrs(p, &s.attrs);
            quote! {
                #p::Item::Sampler(#p::ItemSampler {
                    group: #group,
                    binding: #binding,
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    #attrs,
                })
            }
        }
        ir::Item::Texture(t_) => {
            let group = t_.group;
            let binding = t_.binding;
            let n = &t_.name;
            let t = ty(p, &t_.ty);
            let attrs = emit_attrs(p, &t_.attrs);
            quote! {
                #p::Item::Texture(#p::ItemTexture {
                    group: #group,
                    binding: #binding,
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    #attrs,
                })
            }
        }
        ir::Item::Fn(f) => {
            let inner = item_fn(p, f);
            quote! { #p::Item::Fn(#inner) }
        }
        ir::Item::Struct(s) => {
            let type_params = string_vec(&s.type_params);
            let n = &s.name;
            let fs = s.fields.iter().map(|f| {
                let isi = inter_stage_io_vec(p, &f.inter_stage_io);
                let nm = &f.name;
                let t = ty(p, &f.ty);
                let field_attrs = emit_attrs(p, &f.attrs);
                quote! {
                    #p::Field {
                        inter_stage_io: #isi,
                        name: ::std::string::String::from(#nm),
                        ty: #t,
                        #field_attrs,
                    }
                }
            });
            let attrs = emit_attrs(p, &s.attrs);
            quote! {
                #p::Item::Struct(#p::ItemStruct {
                    type_params: #type_params,
                    name: ::std::string::String::from(#n),
                    fields: ::std::vec![#(#fs),*],
                    #attrs,
                })
            }
        }
        ir::Item::Impl(i) => {
            let type_params = string_vec(&i.type_params);
            let n = &i.self_ty;
            let xs = i.items.iter().map(|ii| match ii {
                ir::ImplItem::Fn(f) => {
                    let inner = item_fn(p, f);
                    quote! { #p::ImplItem::Fn(#inner) }
                }
                ir::ImplItem::Const(c) => {
                    let inner = item_const(p, c);
                    quote! { #p::ImplItem::Const(#inner) }
                }
            });
            let attrs = emit_attrs(p, &i.attrs);
            quote! {
                #p::Item::Impl(#p::ItemImpl {
                    type_params: #type_params,
                    self_ty: ::std::string::String::from(#n),
                    items: ::std::vec![#(#xs),*],
                    #attrs,
                })
            }
        }
        ir::Item::Enum(e) => {
            let n = &e.name;
            let vs = e.variants.iter().map(|v| {
                let nm = &v.name;
                let d = match v.discriminant {
                    Some(d) => quote! { ::std::option::Option::Some(#d) },
                    None => quote! { ::std::option::Option::None },
                };
                quote! {
                    #p::EnumVariant {
                        name: ::std::string::String::from(#nm),
                        discriminant: #d,
                    }
                }
            });
            let attrs = emit_attrs(p, &e.attrs);
            quote! {
                #p::Item::Enum(#p::ItemEnum {
                    name: ::std::string::String::from(#n),
                    variants: ::std::vec![#(#vs),*],
                    #attrs,
                })
            }
        }
    }
}

fn item_const(p: &TokenStream, c: &ir::ItemConst) -> TokenStream {
    let n = &c.name;
    let t = ty(p, &c.ty);
    let e = expr(p, &c.expr);
    let attrs = emit_attrs(p, &c.attrs);
    quote! {
        #p::ItemConst {
            name: ::std::string::String::from(#n),
            ty: #t,
            expr: #e,
            #attrs,
        }
    }
}

fn item_fn(p: &TokenStream, f: &ir::ItemFn) -> TokenStream {
    let type_params = string_vec(&f.type_params);
    let attrs = fn_attrs(p, &f.fn_attrs);
    let n = &f.name;
    let fn_item_attrs = emit_attrs(p, &f.attrs);
    let inputs = f.inputs.iter().map(|a| {
        let isi = inter_stage_io_vec(p, &a.inter_stage_io);
        let nm = &a.name;
        let t = ty(p, &a.ty);
        let arg_attrs = emit_attrs(p, &a.attrs);
        quote! {
            #p::FnArg {
                inter_stage_io: #isi,
                name: ::std::string::String::from(#nm),
                ty: #t,
                #arg_attrs,
            }
        }
    });
    let rt = return_type(p, &f.return_type);
    let body = block(p, &f.block);
    quote! {
        #p::ItemFn {
            type_params: #type_params,
            fn_attrs: #attrs,
            name: ::std::borrow::Cow::Borrowed(#n),
            inputs: ::std::vec![#(#inputs),*],
            return_type: #rt,
            block: #body,
            #fn_item_attrs,
        }
    }
}

fn fn_attrs(p: &TokenStream, a: &ir::FnAttrs) -> TokenStream {
    match a {
        ir::FnAttrs::None => quote! { #p::FnAttrs::None },
        ir::FnAttrs::Vertex => quote! { #p::FnAttrs::Vertex },
        ir::FnAttrs::Fragment => quote! { #p::FnAttrs::Fragment },
        ir::FnAttrs::Compute { workgroup_size } => {
            let x = workgroup_size.x;
            let y = match workgroup_size.y {
                Some(v) => quote! { ::std::option::Option::Some(#v) },
                None => quote! { ::std::option::Option::None },
            };
            let z = match workgroup_size.z {
                Some(v) => quote! { ::std::option::Option::Some(#v) },
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::FnAttrs::Compute {
                    workgroup_size: #p::WorkgroupSize { x: #x, y: #y, z: #z },
                }
            }
        }
    }
}

fn return_type(p: &TokenStream, rt: &ir::ReturnType) -> TokenStream {
    match rt {
        ir::ReturnType::Default => quote! { #p::ReturnType::Default },
        ir::ReturnType::Type { annotation, ty: t } => {
            let ann = return_annotation(p, annotation);
            let t = ty(p, t);
            quote! { #p::ReturnType::Type { annotation: #ann, ty: #t } }
        }
    }
}

fn return_annotation(p: &TokenStream, a: &ir::ReturnTypeAnnotation) -> TokenStream {
    match a {
        ir::ReturnTypeAnnotation::None => quote! { #p::ReturnTypeAnnotation::None },
        ir::ReturnTypeAnnotation::BuiltIn(b) => {
            let bi = builtin(p, *b);
            quote! { #p::ReturnTypeAnnotation::BuiltIn(#bi) }
        }
        ir::ReturnTypeAnnotation::Location(n) => {
            quote! { #p::ReturnTypeAnnotation::Location(#n) }
        }
        ir::ReturnTypeAnnotation::DefaultBuiltInPosition => {
            quote! { #p::ReturnTypeAnnotation::DefaultBuiltInPosition }
        }
        ir::ReturnTypeAnnotation::DefaultLocation => {
            quote! { #p::ReturnTypeAnnotation::DefaultLocation }
        }
    }
}

fn inter_stage_io_vec(p: &TokenStream, ios: &[ir::InterStageIo]) -> TokenStream {
    let xs = ios.iter().map(|io| inter_stage_io(p, io));
    quote! { ::std::vec![#(#xs),*] }
}

fn inter_stage_io(p: &TokenStream, io: &ir::InterStageIo) -> TokenStream {
    match io {
        ir::InterStageIo::BuiltIn(b) => {
            let bi = builtin(p, *b);
            quote! { #p::InterStageIo::BuiltIn(#bi) }
        }
        ir::InterStageIo::Location(n) => quote! { #p::InterStageIo::Location(#n) },
        ir::InterStageIo::BlendSrc(n) => quote! { #p::InterStageIo::BlendSrc(#n) },
        ir::InterStageIo::Interpolate(i) => {
            let t = match i.ty {
                ir::InterpolationType::Perspective => {
                    quote! { #p::InterpolationType::Perspective }
                }
                ir::InterpolationType::Linear => quote! { #p::InterpolationType::Linear },
                ir::InterpolationType::Flat => quote! { #p::InterpolationType::Flat },
            };
            let s = match i.sampling {
                Some(ir::InterpolationSampling::Center) => {
                    quote! { ::std::option::Option::Some(#p::InterpolationSampling::Center) }
                }
                Some(ir::InterpolationSampling::Centroid) => {
                    quote! { ::std::option::Option::Some(#p::InterpolationSampling::Centroid) }
                }
                Some(ir::InterpolationSampling::Sample) => {
                    quote! { ::std::option::Option::Some(#p::InterpolationSampling::Sample) }
                }
                Some(ir::InterpolationSampling::First) => {
                    quote! { ::std::option::Option::Some(#p::InterpolationSampling::First) }
                }
                Some(ir::InterpolationSampling::Either) => {
                    quote! { ::std::option::Option::Some(#p::InterpolationSampling::Either) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::InterStageIo::Interpolate(#p::Interpolate { ty: #t, sampling: #s })
            }
        }
        ir::InterStageIo::Invariant => quote! { #p::InterStageIo::Invariant },
    }
}

fn builtin(p: &TokenStream, b: ir::BuiltIn) -> TokenStream {
    let v = match b {
        ir::BuiltIn::VertexIndex => quote! { VertexIndex },
        ir::BuiltIn::InstanceIndex => quote! { InstanceIndex },
        ir::BuiltIn::Position => quote! { Position },
        ir::BuiltIn::FrontFacing => quote! { FrontFacing },
        ir::BuiltIn::FragDepth => quote! { FragDepth },
        ir::BuiltIn::SampleIndex => quote! { SampleIndex },
        ir::BuiltIn::SampleMask => quote! { SampleMask },
        ir::BuiltIn::LocalInvocationId => quote! { LocalInvocationId },
        ir::BuiltIn::LocalInvocationIndex => quote! { LocalInvocationIndex },
        ir::BuiltIn::GlobalInvocationId => quote! { GlobalInvocationId },
        ir::BuiltIn::WorkgroupId => quote! { WorkgroupId },
        ir::BuiltIn::NumWorkgroups => quote! { NumWorkgroups },
        ir::BuiltIn::SubgroupInvocationId => quote! { SubgroupInvocationId },
        ir::BuiltIn::SubgroupSize => quote! { SubgroupSize },
        ir::BuiltIn::PrimitiveIndex => quote! { PrimitiveIndex },
        ir::BuiltIn::SubgroupId => quote! { SubgroupId },
        ir::BuiltIn::NumSubgroups => quote! { NumSubgroups },
    };
    quote! { #p::BuiltIn::#v }
}

fn string_vec(xs: &[String]) -> TokenStream {
    let xs = xs
        .iter()
        .map(|s| quote! { ::std::string::String::from(#s) });
    quote! { ::std::vec![#(#xs),*] }
}

fn emit_attribute(p: &TokenStream, a: &ir::Attribute) -> TokenStream {
    let path = &a.path;
    let args = string_vec(&a.args);
    quote! { #p::Attribute { path: ::std::string::String::from(#path), args: #args } }
}

pub fn emit_attrs(p: &TokenStream, attrs: &[ir::Attribute]) -> TokenStream {
    let attrs_ts = attrs.iter().map(|a| emit_attribute(p, a));
    quote! { attrs: ::std::vec![#(#attrs_ts),*] }
}

// ===== Types =====

fn ty(p: &TokenStream, t: &ir::Type) -> TokenStream {
    match t {
        ir::Type::Scalar(s) => {
            let s = scalar(p, *s);
            quote! { #p::Type::Scalar(#s) }
        }
        ir::Type::Vector {
            elements,
            scalar_ty,
        } => {
            let s = match scalar_ty {
                Some(s) => {
                    let s = scalar(p, *s);
                    quote! { ::std::option::Option::Some(#s) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::Type::Vector { elements: #elements, scalar_ty: #s }
            }
        }
        ir::Type::Matrix {
            columns,
            rows,
            scalar_ty,
        } => {
            let s = match scalar_ty {
                Some(s) => {
                    let s = scalar(p, *s);
                    quote! { ::std::option::Option::Some(#s) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::Type::Matrix { columns: #columns, rows: #rows, scalar_ty: #s }
            }
        }
        ir::Type::Array { elem, len } => {
            let e = ty(p, elem);
            let l = expr(p, len);
            quote! {
                #p::Type::Array {
                    elem: ::std::boxed::Box::new(#e),
                    len: #l,
                }
            }
        }
        ir::Type::RuntimeArray { elem } => {
            let e = ty(p, elem);
            quote! {
                #p::Type::RuntimeArray { elem: ::std::boxed::Box::new(#e) }
            }
        }
        ir::Type::Atomic { elem } => {
            let e = ty(p, elem);
            quote! { #p::Type::Atomic { elem: ::std::boxed::Box::new(#e) } }
        }
        ir::Type::Struct { name, type_args } => {
            let xs = type_args.iter().map(|t| ty(p, t));
            quote! {
                #p::Type::Struct {
                    name: ::std::string::String::from(#name),
                    type_args: ::std::vec![#(#xs),*],
                }
            }
        }
        ir::Type::Ptr {
            address_space,
            elem,
        } => {
            let a = match address_space {
                ir::AddressSpace::Function => quote! { #p::AddressSpace::Function },
                ir::AddressSpace::Private => quote! { #p::AddressSpace::Private },
                ir::AddressSpace::Workgroup => quote! { #p::AddressSpace::Workgroup },
            };
            let e = ty(p, elem);
            quote! {
                #p::Type::Ptr {
                    address_space: #a,
                    elem: ::std::boxed::Box::new(#e),
                }
            }
        }
        ir::Type::Sampler => quote! { #p::Type::Sampler },
        ir::Type::SamplerComparison => quote! { #p::Type::SamplerComparison },
        ir::Type::Texture { kind, sampled_type } => {
            let k = tex_kind(p, *kind);
            let s = scalar(p, *sampled_type);
            quote! { #p::Type::Texture { kind: #k, sampled_type: #s } }
        }
        ir::Type::TextureDepth { kind } => {
            let k = tex_depth_kind(p, *kind);
            quote! { #p::Type::TextureDepth { kind: #k } }
        }
        ir::Type::TypeParam { name } => {
            quote! { #p::Type::TypeParam { name: ::std::string::String::from(#name) } }
        }
    }
}

fn scalar(p: &TokenStream, s: ir::ScalarType) -> TokenStream {
    let v = match s {
        ir::ScalarType::I32 => quote! { I32 },
        ir::ScalarType::U32 => quote! { U32 },
        ir::ScalarType::F32 => quote! { F32 },
        ir::ScalarType::Bool => quote! { Bool },
    };
    quote! { #p::ScalarType::#v }
}

fn tex_kind(p: &TokenStream, k: ir::TextureKind) -> TokenStream {
    let v = match k {
        ir::TextureKind::Texture1D => quote! { Texture1D },
        ir::TextureKind::Texture2D => quote! { Texture2D },
        ir::TextureKind::Texture2DArray => quote! { Texture2DArray },
        ir::TextureKind::Texture3D => quote! { Texture3D },
        ir::TextureKind::TextureCube => quote! { TextureCube },
        ir::TextureKind::TextureCubeArray => quote! { TextureCubeArray },
        ir::TextureKind::TextureMultisampled2D => quote! { TextureMultisampled2D },
    };
    quote! { #p::TextureKind::#v }
}

fn tex_depth_kind(p: &TokenStream, k: ir::TextureDepthKind) -> TokenStream {
    let v = match k {
        ir::TextureDepthKind::Depth2D => quote! { Depth2D },
        ir::TextureDepthKind::Depth2DArray => quote! { Depth2DArray },
        ir::TextureDepthKind::DepthCube => quote! { DepthCube },
        ir::TextureDepthKind::DepthCubeArray => quote! { DepthCubeArray },
        ir::TextureDepthKind::DepthMultisampled2D => quote! { DepthMultisampled2D },
    };
    quote! { #p::TextureDepthKind::#v }
}

// ===== Expressions =====

fn expr(p: &TokenStream, e: &ir::Expr) -> TokenStream {
    match e {
        ir::Expr::Lit(l) => {
            let l = lit(p, l);
            quote! { #p::Expr::Lit(#l) }
        }
        ir::Expr::Ident(name) => {
            quote! { #p::Expr::Ident(::std::string::String::from(#name)) }
        }
        ir::Expr::Array { elems } => {
            let xs = elems.iter().map(|e| expr(p, e));
            quote! { #p::Expr::Array { elems: ::std::vec![#(#xs),*] } }
        }
        ir::Expr::Paren(inner) => {
            let i = expr(p, inner);
            quote! { #p::Expr::Paren(::std::boxed::Box::new(#i)) }
        }
        ir::Expr::Binary { lhs, op, rhs } => {
            let l = expr(p, lhs);
            let o = binop(p, *op);
            let r = expr(p, rhs);
            quote! {
                #p::Expr::Binary {
                    lhs: ::std::boxed::Box::new(#l),
                    op: #o,
                    rhs: ::std::boxed::Box::new(#r),
                }
            }
        }
        ir::Expr::Unary { op, expr: inner } => {
            let o = unop(p, *op);
            let i = expr(p, inner);
            quote! {
                #p::Expr::Unary {
                    op: #o,
                    expr: ::std::boxed::Box::new(#i),
                }
            }
        }
        ir::Expr::ArrayIndexing { lhs, index } => {
            let l = expr(p, lhs);
            let i = expr(p, index);
            quote! {
                #p::Expr::ArrayIndexing {
                    lhs: ::std::boxed::Box::new(#l),
                    index: ::std::boxed::Box::new(#i),
                }
            }
        }
        ir::Expr::Swizzle {
            lhs,
            swizzle,
            params,
        } => {
            let l = expr(p, lhs);
            let s = swizzle;
            let ps = match params {
                Some(args) => {
                    let xs = args.iter().map(|e| expr(p, e));
                    quote! { ::std::option::Option::Some(::std::vec![#(#xs),*]) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::Expr::Swizzle {
                    lhs: ::std::boxed::Box::new(#l),
                    swizzle: ::std::string::String::from(#s),
                    params: #ps,
                }
            }
        }
        ir::Expr::Cast { lhs, ty: t } => {
            let l = expr(p, lhs);
            let t = ty(p, t);
            quote! {
                #p::Expr::Cast {
                    lhs: ::std::boxed::Box::new(#l),
                    ty: ::std::boxed::Box::new(#t),
                }
            }
        }
        ir::Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            let path = fn_path(p, path);
            let tas = type_args.iter().map(|t| ty(p, t));
            let prms = params.iter().map(|e| expr(p, e));
            quote! {
                #p::Expr::FnCall {
                    path: #path,
                    type_args: ::std::vec![#(#tas),*],
                    params: ::std::vec![#(#prms),*],
                }
            }
        }
        ir::Expr::Struct {
            name,
            type_args,
            fields,
        } => {
            let tas = type_args.iter().map(|t| ty(p, t));
            let fs = fields.iter().map(|f| {
                let m = &f.member;
                let e = expr(p, &f.expr);
                quote! {
                    #p::FieldValue {
                        member: ::std::string::String::from(#m),
                        expr: #e,
                    }
                }
            });
            quote! {
                #p::Expr::Struct {
                    name: ::std::string::String::from(#name),
                    type_args: ::std::vec![#(#tas),*],
                    fields: ::std::vec![#(#fs),*],
                }
            }
        }
        ir::Expr::FieldAccess { base, field } => {
            let b = expr(p, base);
            quote! {
                #p::Expr::FieldAccess {
                    base: ::std::boxed::Box::new(#b),
                    field: ::std::string::String::from(#field),
                }
            }
        }
        ir::Expr::TypePath { ty: t, member } => {
            quote! {
                #p::Expr::TypePath {
                    ty: ::std::string::String::from(#t),
                    member: ::std::string::String::from(#member),
                }
            }
        }
        ir::Expr::Reference(inner) => {
            let i = expr(p, inner);
            quote! { #p::Expr::Reference(::std::boxed::Box::new(#i)) }
        }
        ir::Expr::ZeroValueArray { elem_type, len } => {
            let t = ty(p, elem_type);
            let l = expr(p, len);
            quote! {
                #p::Expr::ZeroValueArray {
                    elem_type: ::std::boxed::Box::new(#t),
                    len: ::std::boxed::Box::new(#l),
                }
            }
        }
    }
}

fn fn_path(p: &TokenStream, path: &ir::FnPath) -> TokenStream {
    match path {
        ir::FnPath::Ident(name) => {
            quote! { #p::FnPath::Ident(::std::string::String::from(#name)) }
        }
        ir::FnPath::TypeMethod { ty: t, method } => {
            quote! {
                #p::FnPath::TypeMethod {
                    ty: ::std::string::String::from(#t),
                    method: ::std::string::String::from(#method),
                }
            }
        }
    }
}

fn lit(p: &TokenStream, l: &ir::Lit) -> TokenStream {
    match l {
        ir::Lit::Bool(b) => quote! { #p::Lit::Bool(#b) },
        ir::Lit::Int { digits, suffix } => quote! {
            #p::Lit::Int {
                digits: ::std::string::String::from(#digits),
                suffix: ::std::string::String::from(#suffix),
            }
        },
        ir::Lit::Float { text } => quote! {
            #p::Lit::Float {
                text: ::std::string::String::from(#text),
            }
        },
    }
}

fn binop(p: &TokenStream, op: ir::BinOp) -> TokenStream {
    let v = match op {
        ir::BinOp::Add => quote! { Add },
        ir::BinOp::Sub => quote! { Sub },
        ir::BinOp::Mul => quote! { Mul },
        ir::BinOp::Div => quote! { Div },
        ir::BinOp::Rem => quote! { Rem },
        ir::BinOp::Eq => quote! { Eq },
        ir::BinOp::Ne => quote! { Ne },
        ir::BinOp::Lt => quote! { Lt },
        ir::BinOp::Le => quote! { Le },
        ir::BinOp::Gt => quote! { Gt },
        ir::BinOp::Ge => quote! { Ge },
        ir::BinOp::And => quote! { And },
        ir::BinOp::Or => quote! { Or },
        ir::BinOp::BitAnd => quote! { BitAnd },
        ir::BinOp::BitOr => quote! { BitOr },
        ir::BinOp::BitXor => quote! { BitXor },
        ir::BinOp::Shl => quote! { Shl },
        ir::BinOp::Shr => quote! { Shr },
    };
    quote! { #p::BinOp::#v }
}

fn unop(p: &TokenStream, op: ir::UnOp) -> TokenStream {
    let v = match op {
        ir::UnOp::Not => quote! { Not },
        ir::UnOp::Neg => quote! { Neg },
        ir::UnOp::Deref => quote! { Deref },
    };
    quote! { #p::UnOp::#v }
}

fn compound(p: &TokenStream, op: ir::CompoundOp) -> TokenStream {
    let v = match op {
        ir::CompoundOp::AddAssign => quote! { AddAssign },
        ir::CompoundOp::SubAssign => quote! { SubAssign },
        ir::CompoundOp::MulAssign => quote! { MulAssign },
        ir::CompoundOp::DivAssign => quote! { DivAssign },
        ir::CompoundOp::RemAssign => quote! { RemAssign },
        ir::CompoundOp::BitAndAssign => quote! { BitAndAssign },
        ir::CompoundOp::BitOrAssign => quote! { BitOrAssign },
        ir::CompoundOp::BitXorAssign => quote! { BitXorAssign },
        ir::CompoundOp::ShlAssign => quote! { ShlAssign },
        ir::CompoundOp::ShrAssign => quote! { ShrAssign },
    };
    quote! { #p::CompoundOp::#v }
}

// ===== Statements / blocks =====

fn block(p: &TokenStream, b: &ir::Block) -> TokenStream {
    let xs = b.stmts.iter().map(|s| stmt(p, s));
    quote! {
        #p::Block { stmts: ::std::vec![#(#xs),*] }
    }
}

fn stmt(p: &TokenStream, s: &ir::Stmt) -> TokenStream {
    match s {
        ir::Stmt::Local(l) => {
            let mu = l.mutable;
            let n = &l.name;
            let t = match &l.ty {
                Some(t) => {
                    let t = ty(p, t);
                    quote! { ::std::option::Option::Some(#t) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let init = match &l.init {
                Some(e) => {
                    let e = expr(p, e);
                    quote! { ::std::option::Option::Some(#e) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::Stmt::Local(#p::Local {
                    mutable: #mu,
                    name: ::std::string::String::from(#n),
                    ty: #t,
                    init: #init,
                })
            }
        }
        ir::Stmt::Const(c) => {
            let inner = item_const(p, c);
            quote! { #p::Stmt::Const(#inner) }
        }
        ir::Stmt::Assignment { lhs, rhs } => {
            let l = expr(p, lhs);
            let r = expr(p, rhs);
            quote! { #p::Stmt::Assignment { lhs: #l, rhs: #r } }
        }
        ir::Stmt::CompoundAssignment { lhs, op, rhs } => {
            let l = expr(p, lhs);
            let o = compound(p, *op);
            let r = expr(p, rhs);
            quote! { #p::Stmt::CompoundAssignment { lhs: #l, op: #o, rhs: #r } }
        }
        ir::Stmt::While { condition, body } => {
            let c = expr(p, condition);
            let b = block(p, body);
            quote! { #p::Stmt::While { condition: #c, body: #b } }
        }
        ir::Stmt::Loop { body } => {
            let b = block(p, body);
            quote! { #p::Stmt::Loop { body: #b } }
        }
        ir::Stmt::Expr { expr: e, has_semi } => {
            let e = expr(p, e);
            let h = has_semi;
            quote! { #p::Stmt::Expr { expr: #e, has_semi: #h } }
        }
        ir::Stmt::If(i) => {
            let inner = stmt_if(p, i);
            quote! { #p::Stmt::If(#inner) }
        }
        ir::Stmt::Break => quote! { #p::Stmt::Break },
        ir::Stmt::Continue => quote! { #p::Stmt::Continue },
        ir::Stmt::Return(e) => match e {
            Some(e) => {
                let e = expr(p, e);
                quote! { #p::Stmt::Return(::std::option::Option::Some(#e)) }
            }
            None => quote! { #p::Stmt::Return(::std::option::Option::None) },
        },
        ir::Stmt::For(f) => {
            let v = &f.var;
            let vt = match &f.var_ty {
                Some(t) => {
                    let t = ty(p, t);
                    quote! { ::std::option::Option::Some(#t) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let from = expr(p, &f.from);
            let to = expr(p, &f.to);
            let inc = f.inclusive;
            let body = block(p, &f.body);
            quote! {
                #p::Stmt::For(#p::ForLoop {
                    var: ::std::string::String::from(#v),
                    var_ty: #vt,
                    from: #from,
                    to: #to,
                    inclusive: #inc,
                    body: #body,
                })
            }
        }
        ir::Stmt::Switch(s) => {
            let sel = expr(p, &s.selector);
            let arms = s.arms.iter().map(|arm| {
                let sels = arm.selectors.iter().map(|cs| case_selector(p, cs));
                let body = block(p, &arm.body);
                quote! {
                    #p::SwitchArm {
                        selectors: ::std::vec![#(#sels),*],
                        body: #body,
                    }
                }
            });
            let has_def = s.has_explicit_default;
            quote! {
                #p::Stmt::Switch(#p::StmtSwitch {
                    selector: #sel,
                    arms: ::std::vec![#(#arms),*],
                    has_explicit_default: #has_def,
                })
            }
        }
        ir::Stmt::Block(b) => {
            let b = block(p, b);
            quote! { #p::Stmt::Block(#b) }
        }
        ir::Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
        } => {
            let sl = expr(p, slab);
            let o = expr(p, offset);
            let d = expr(p, dest);
            let s = expr(p, size);
            quote! {
                #p::Stmt::SlabRead { slab: #sl, offset: #o, dest: #d, size: #s }
            }
        }
        ir::Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
        } => {
            let sl = expr(p, slab);
            let o = expr(p, offset);
            let sr = expr(p, src);
            let s = match size {
                Some(s) => {
                    let s = expr(p, s);
                    quote! { ::std::option::Option::Some(#s) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                #p::Stmt::SlabWrite { slab: #sl, offset: #o, src: #sr, size: #s }
            }
        }
        ir::Stmt::Discard => quote! { #p::Stmt::Discard },
    }
}

fn stmt_if(p: &TokenStream, i: &ir::StmtIf) -> TokenStream {
    let c = expr(p, &i.condition);
    let t = block(p, &i.then_block);
    let e = match &i.else_branch {
        Some(ir::ElseBranch::Block(b)) => {
            let b = block(p, b);
            quote! { ::std::option::Option::Some(#p::ElseBranch::Block(#b)) }
        }
        Some(ir::ElseBranch::If(inner)) => {
            let inner = stmt_if(p, inner);
            quote! {
                ::std::option::Option::Some(#p::ElseBranch::If(::std::boxed::Box::new(#inner)))
            }
        }
        None => quote! { ::std::option::Option::None },
    };
    quote! {
        #p::StmtIf {
            condition: #c,
            then_block: #t,
            else_branch: #e,
        }
    }
}

fn case_selector(p: &TokenStream, cs: &ir::CaseSelector) -> TokenStream {
    match cs {
        ir::CaseSelector::Literal(l) => {
            let l = lit(p, l);
            quote! { #p::CaseSelector::Literal(#l) }
        }
        ir::CaseSelector::Expr(e) => {
            let e = expr(p, e);
            quote! { #p::CaseSelector::Expr(#e) }
        }
        ir::CaseSelector::Default => quote! { #p::CaseSelector::Default },
    }
}
