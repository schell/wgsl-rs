//! Rust-only resource-pack structs.
//!
//! A resource pack is recognized by binding annotations on its fields. The
//! struct remains an ordinary Rust value, while its fields describe global
//! WGSL resources for later lowering passes.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Ident, Token, spanned::Spanned};

use crate::parse::{self, InterStageIo, ParseContext, StorageAccess, Type};

/// The resource constructor used by an annotated field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResourceKind {
    Uniform,
    Storage,
    Sampler,
    Texture,
}

impl ResourceKind {
    fn from_attr(attr: &syn::Attribute) -> Option<Self> {
        let ident = attr.path().get_ident()?;
        match ident.to_string().as_str() {
            "uniform" => Some(Self::Uniform),
            "storage" => Some(Self::Storage),
            "sampler" => Some(Self::Sampler),
            "texture" => Some(Self::Texture),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Storage => "storage",
            Self::Sampler => "sampler",
            Self::Texture => "texture",
        }
    }
}

/// A literal or const-generic binding address component.
#[derive(Clone, Debug)]
pub(crate) enum AddressValue {
    Literal(syn::LitInt),
    Param(Ident),
}

impl AddressValue {
    fn parse(
        expr: &syn::Expr,
        const_params: &std::collections::HashMap<String, syn::Type>,
    ) -> parse::Result<Self> {
        match expr {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(lit),
                ..
            }) => {
                lit.base10_parse::<u32>().map_err(|_| {
                    parse::Error::unsupported(
                        lit.span(),
                        format!("resource address `{lit}` does not fit in a u32"),
                    )
                })?;
                Ok(Self::Literal(lit.clone()))
            }
            syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let ident = &path.path.segments[0].ident;
                let Some(ty) = const_params.get(&ident.to_string()) else {
                    return Err(parse::Error::unsupported(
                        ident.span(),
                        format!(
                            "resource address `{ident}` must be an integer literal or a const \
                             parameter of this resource pack"
                        ),
                    ));
                };
                let supported = matches!(
                    ty,
                    syn::Type::Path(path)
                        if path.qself.is_none()
                            && (path.path.is_ident("u32") || path.path.is_ident("usize"))
                );
                if !supported {
                    return Err(parse::Error::unsupported(
                        ty.span(),
                        format!(
                            "resource address const parameter `{ident}` must have type `u32` or \
                             `usize`"
                        ),
                    ));
                }
                Ok(Self::Param(ident.clone()))
            }
            other => Err(parse::Error::unsupported(
                other.span(),
                format!(
                    "resource addresses must be integer literals or bare const parameters; got \
                     `{}`",
                    other.to_token_stream()
                ),
            )),
        }
    }

    /// Emit the value passed to a runtime resource constructor.
    fn constructor_value(&self) -> TokenStream {
        match self {
            Self::Literal(lit) => quote!(#lit),
            Self::Param(ident) => quote!((#ident) as u32),
        }
    }
}

/// One annotated resource field.
#[derive(Clone)]
pub(crate) struct ResourceField {
    pub(crate) ident: Ident,
    pub(crate) kind: ResourceKind,
    pub(crate) group: AddressValue,
    pub(crate) binding: AddressValue,
    pub(crate) access: StorageAccess,
    #[allow(dead_code)] // consumed by IR resource generation in the next lowering stage
    pub(crate) ty: Type,
    pub(crate) rust_ty: syn::Type,
}

/// A registered resource-pack struct.
#[derive(Clone)]
pub(crate) struct ItemResourcePack {
    pub(crate) ident: Ident,
    #[allow(dead_code)] // consumed by parameterized IR addresses in the next lowering stage
    pub(crate) const_params: Vec<Ident>,
    pub(crate) fields: Vec<ResourceField>,
}

/// Returns true when any field identifies this struct as a resource pack.
pub(crate) fn is_resource_pack(item: &syn::ItemStruct) -> bool {
    item.fields.iter().any(|field| {
        field
            .attrs
            .iter()
            .any(|attr| ResourceKind::from_attr(attr).is_some())
    })
}

/// Parse and validate a resource-pack struct.
pub(crate) fn parse_resource_pack(item: &syn::ItemStruct) -> parse::Result<ItemResourcePack> {
    if !matches!(item.vis, syn::Visibility::Public(_)) {
        return Err(parse::Error::unsupported(
            item.struct_token.span(),
            "resource-pack structs must be public",
        ));
    }
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(parse::Error::unsupported(
            item.fields.span(),
            "resource packs require named fields",
        ));
    };

    let mut const_params = Vec::new();
    let mut const_param_types = std::collections::HashMap::new();
    for param in &item.generics.params {
        match param {
            syn::GenericParam::Const(param) => {
                const_params.push(param.ident.clone());
                const_param_types.insert(param.ident.to_string(), param.ty.clone());
            }
            syn::GenericParam::Type(param) => {
                return Err(parse::Error::unsupported(
                    param.span(),
                    "resource packs support const generic parameters, not type parameters",
                ));
            }
            syn::GenericParam::Lifetime(param) => {
                return Err(parse::Error::unsupported(
                    param.span(),
                    "resource packs do not support lifetime parameters",
                ));
            }
        }
    }
    let ctx = ParseContext::from_params(std::iter::empty::<Ident>(), const_params.clone());

    let mut parsed_fields = Vec::with_capacity(fields.named.len());
    for field in &fields.named {
        parsed_fields.push(parse_field(field, &ctx, &const_param_types)?);
    }
    Ok(ItemResourcePack {
        ident: item.ident.clone(),
        const_params,
        fields: parsed_fields,
    })
}

/// Parse one field and reject mixed IO/resource annotations and malformed
/// binding metadata at the field that caused the problem.
fn parse_field(
    field: &syn::Field,
    ctx: &ParseContext,
    const_params: &std::collections::HashMap<String, syn::Type>,
) -> parse::Result<ResourceField> {
    let resource_attrs: Vec<&syn::Attribute> = field
        .attrs
        .iter()
        .filter(|attr| ResourceKind::from_attr(attr).is_some())
        .collect();
    let Some(first) = resource_attrs.first().copied() else {
        return Err(parse::Error::unsupported(
            field.span(),
            "every field of a resource pack must have exactly one `#[texture]`, `#[sampler]`, \
             `#[storage]`, or `#[uniform]` annotation; non-resource fields are not supported",
        ));
    };
    if let Some(second) = resource_attrs.get(1) {
        return Err(parse::Error::unsupported(
            second.span(),
            "a resource-pack field cannot have multiple binding annotations",
        ));
    }
    for attr in &field.attrs {
        if parse::is_known_io_ident(attr) {
            InterStageIo::try_from(attr)?;
            return Err(parse::Error::unsupported(
                attr.span(),
                "resource-pack fields cannot combine binding annotations with interstage IO \
                 annotations; move builtin or location data into a separate IO struct",
            ));
        }
        if attr.path().is_ident("workgroup") {
            return Err(parse::Error::unsupported(
                attr.span(),
                "workgroup values are not supported in resource packs",
            ));
        }
    }

    let kind = ResourceKind::from_attr(first).expect("resource attribute was selected above");
    let args = parse_binding_args(first, kind, const_params)?;
    let rust_ty = field.ty.clone();
    let ty = Type::parse(&rust_ty, ctx)?;
    validate_resource_type(field, kind, &ty)?;

    Ok(ResourceField {
        ident: field
            .ident
            .clone()
            .expect("resource-pack parsing requires named fields"),
        kind,
        group: args.group,
        binding: args.binding,
        access: args.access,
        ty,
        rust_ty,
    })
}

struct BindingArgs {
    group: AddressValue,
    binding: AddressValue,
    access: StorageAccess,
}

/// Parse `group(...)`, `binding(...)`, and storage access modifiers in any
/// order, while rejecting duplicates and unknown modifiers.
fn parse_binding_args(
    attr: &syn::Attribute,
    kind: ResourceKind,
    const_params: &std::collections::HashMap<String, syn::Type>,
) -> parse::Result<BindingArgs> {
    let mut group = None;
    let mut binding = None;
    let mut access = StorageAccess::Read;
    let mut saw_access = false;

    attr.parse_args_with(|input: syn::parse::ParseStream| {
        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "group" | "binding" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let expr: syn::Expr = content.parse()?;
                    if !content.is_empty() {
                        return Err(content.error("expected exactly one resource address value"));
                    }
                    let value = AddressValue::parse(&expr, const_params)
                        .map_err(|error| syn::Error::new(error.span(), error.to_string()))?;
                    let slot = if ident == "group" {
                        &mut group
                    } else {
                        &mut binding
                    };
                    if slot.replace(value).is_some() {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!("duplicate `{ident}` in resource binding annotation"),
                        ));
                    }
                }
                "read_only" | "read_write" if kind == ResourceKind::Storage => {
                    if saw_access {
                        return Err(syn::Error::new(
                            ident.span(),
                            "duplicate storage access modifier",
                        ));
                    }
                    saw_access = true;
                    access = if ident == "read_write" {
                        StorageAccess::ReadWrite
                    } else {
                        StorageAccess::Read
                    };
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown `{other}` modifier on `#[{}]`; expected `group(...)` and \
                             `binding(...)`{}",
                            kind.as_str(),
                            if kind == ResourceKind::Storage {
                                ", with optional `read_only` or `read_write`"
                            } else {
                                ""
                            }
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(())
    })
    .map_err(parse::Error::from)?;

    let Some(group) = group else {
        return Err(parse::Error::unsupported(
            attr.span(),
            format!("`#[{}]` requires `group(...)`", kind.as_str()),
        ));
    };
    let Some(binding) = binding else {
        return Err(parse::Error::unsupported(
            attr.span(),
            format!("`#[{}]` requires `binding(...)`", kind.as_str()),
        ));
    };
    Ok(BindingArgs {
        group,
        binding,
        access,
    })
}

/// Ensure the field's Rust type agrees with its binding annotation.
fn validate_resource_type(field: &syn::Field, kind: ResourceKind, ty: &Type) -> parse::Result<()> {
    let valid = match kind {
        ResourceKind::Texture => matches!(
            ty,
            Type::Texture { .. } | Type::TextureDepth { .. } | Type::TextureStorage { .. }
        ),
        ResourceKind::Sampler => {
            matches!(ty, Type::Sampler { .. } | Type::SamplerComparison { .. })
        }
        ResourceKind::Uniform | ResourceKind::Storage => !matches!(
            ty,
            Type::Sampler { .. }
                | Type::SamplerComparison { .. }
                | Type::Texture { .. }
                | Type::TextureDepth { .. }
                | Type::TextureStorage { .. }
                | Type::Ptr { .. }
                | Type::Phantom { .. }
        ),
    };
    if valid {
        Ok(())
    } else {
        Err(parse::Error::unsupported(
            field.ty.span(),
            format!(
                "`#[{}]` field type `{}` is not a supported resource of that kind",
                kind.as_str(),
                field.ty.to_token_stream()
            ),
        ))
    }
}

/// Rewrite resource packs in the emitted Rust module and append their CPU-only
/// constructors. Constructors are added after parsing, so they cannot become
/// WGSL functions or resource values.
pub(crate) fn rewrite_rust_module(
    input_mod: &mut syn::ItemMod,
    module: &parse::ItemMod,
    crate_path: &syn::Path,
) -> parse::Result<()> {
    let Some((_, items)) = input_mod.content.as_mut() else {
        return Ok(());
    };
    let mut output = Vec::with_capacity(items.len() + module.resource_packs().count());
    for mut item in items.drain(..) {
        let syn::Item::Struct(item_struct) = &mut item else {
            output.push(item);
            continue;
        };
        let Some(pack) = module.resource_pack(&item_struct.ident) else {
            output.push(item);
            continue;
        };
        if let syn::Fields::Named(fields) = &mut item_struct.fields {
            for field in &mut fields.named {
                field
                    .attrs
                    .retain(|attr| ResourceKind::from_attr(attr).is_none());
                let resource = pack
                    .fields
                    .iter()
                    .find(|resource| Some(&resource.ident) == field.ident.as_ref())
                    .expect("resource-pack parse and Rust fields have the same order");
                match resource.kind {
                    ResourceKind::Uniform => {
                        let inner = &resource.rust_ty;
                        field.ty = syn::parse_quote!(#crate_path::std::Uniform<#inner>);
                    }
                    ResourceKind::Storage => {
                        let inner = &resource.rust_ty;
                        let access = match resource.access {
                            StorageAccess::Read => quote!(#crate_path::std::Read),
                            StorageAccess::ReadWrite => quote!(#crate_path::std::ReadWrite),
                        };
                        field.ty = syn::parse_quote!(#crate_path::std::Storage<#inner, #access>);
                    }
                    ResourceKind::Sampler | ResourceKind::Texture => {}
                }
            }
        }
        let constructor = generate_constructor(item_struct, pack, crate_path);
        output.push(item);
        output.push(syn::Item::Impl(constructor));
    }
    *items = output;
    Ok(())
}

/// Generate `Pack::new()` using only annotation-owned addresses.
fn generate_constructor(
    item_struct: &syn::ItemStruct,
    pack: &ItemResourcePack,
    crate_path: &syn::Path,
) -> syn::ItemImpl {
    let ident = &item_struct.ident;
    let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();
    let initializers = pack.fields.iter().map(|field| {
        let name = &field.ident;
        let group = field.group.constructor_value();
        let binding = field.binding.constructor_value();
        let value = match field.kind {
            ResourceKind::Texture | ResourceKind::Sampler => {
                let ty = &field.rust_ty;
                quote!(<#ty>::new(#group, #binding))
            }
            ResourceKind::Uniform => {
                let ty = &field.rust_ty;
                quote!(<#crate_path::std::Uniform<#ty>>::new(#group, #binding))
            }
            ResourceKind::Storage => {
                let ty = &field.rust_ty;
                let access = match field.access {
                    StorageAccess::Read => quote!(#crate_path::std::Read),
                    StorageAccess::ReadWrite => quote!(#crate_path::std::ReadWrite),
                };
                quote!(<#crate_path::std::Storage<#ty, #access>>::new(#group, #binding))
            }
        };
        quote!(#name: #value)
    });
    syn::parse_quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            /// Construct this resource pack from its annotated addresses.
            pub fn new() -> Self {
                Self {
                    #(#initializers),*
                }
            }
        }
    }
}
