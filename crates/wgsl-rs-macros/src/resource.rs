//! Rust-only resource-pack structs.
//!
//! A resource pack is recognized by binding annotations on its fields. The
//! struct remains an ordinary Rust value, while its fields describe global
//! WGSL resources for later lowering passes.

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote};
use syn::{Ident, Token, spanned::Spanned};
use wgsl_rs_ir as ir;

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
    pub(crate) ty: Type,
    pub(crate) rust_ty: syn::Type,
}

/// One entry-point use of a resource-pack type. Const arguments have already
/// been positionally encoded by the entry-point parser.
#[derive(Clone)]
pub(crate) struct ResourcePackUse {
    pub(crate) pack: Ident,
    pub(crate) const_args: Vec<parse::PackConstArg>,
    pub(crate) span: Span,
}

/// A registered resource-pack struct.
#[derive(Clone)]
pub(crate) struct ItemResourcePack {
    pub(crate) ident: Ident,
    pub(crate) const_params: Vec<(Ident, syn::Type)>,
    pub(crate) fields: Vec<ResourceField>,
}

/// Deterministic module slot for one const parameter of one pack type.
/// Namespacing by pack keeps identically named consts independent.
pub(crate) fn pack_const_slot_name(pack: &Ident, param: &Ident) -> String {
    format!("{pack}_{param}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackConstKind {
    Shared,
    Fixed(ir::ConstValue),
}

/// The unified instantiation plan for one pack declaration const.
pub(crate) struct PackConstPlan {
    pub(crate) pack: String,
    /// Namespaced key stored in binding IR and `Source::module_type_params`.
    pub(crate) source: String,
    pub(crate) kind: PackConstKind,
    pub(crate) span: Span,
    /// Positional entry-point const keys that read the shared value.
    pub(crate) aliases: Vec<Ident>,
}

/// Unify all entry-point uses of each pack const. A fixed/generic mixture or
/// two different concrete values is a layout conflict at the second use.
pub(crate) fn pack_const_plans(module: &parse::ItemMod) -> parse::Result<Vec<PackConstPlan>> {
    enum PartialKind {
        Shared,
        Fixed(ir::ConstValue),
    }
    struct Partial {
        pack: String,
        source: String,
        kind: Option<PartialKind>,
        span: Span,
        aliases: Vec<Ident>,
        resources: String,
    }

    let mut plans = Vec::<Partial>::new();
    for usage in &module.resource_pack_uses {
        let pack = module
            .resource_pack(&usage.pack)
            .expect("lowering registers the pack used by every entry point");
        let resources = pack
            .fields
            .iter()
            .map(|field| field.ident.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        for ((declared, _), actual) in pack.const_params.iter().zip(&usage.const_args) {
            let pack_name = pack.ident.to_string();
            let source = pack_const_slot_name(&pack.ident, declared);
            let index = plans
                .iter()
                .position(|plan| plan.pack == pack_name && plan.source == source)
                .unwrap_or_else(|| {
                    plans.push(Partial {
                        pack: pack_name.clone(),
                        source: source.clone(),
                        kind: None,
                        span: usage.span,
                        aliases: Vec::new(),
                        resources: resources.clone(),
                    });
                    plans.len() - 1
                });
            let plan = &mut plans[index];
            let conflict = |kind: &str, value: &str| {
                parse::Error::unsupported(
                    usage.span,
                    format!(
                        "conflicting const values for `{declared}` in resource pack `{}`: {kind} \
                         `{value}`; affected resources: {}. All uses of one pack layout must \
                         share compatible values",
                        pack.ident, plan.resources
                    ),
                )
            };
            match actual {
                parse::PackConstArg::Param(target) => match &plan.kind {
                    Some(PartialKind::Fixed(value)) => {
                        return Err(conflict(
                            "generic value versus concrete",
                            &format!("{value:?}"),
                        ));
                    }
                    _ => {
                        plan.kind = Some(PartialKind::Shared);
                        if !plan.aliases.contains(target) {
                            plan.aliases.push(target.clone());
                        }
                    }
                },
                parse::PackConstArg::Value(value) => match &plan.kind {
                    Some(PartialKind::Shared) => {
                        return Err(conflict(
                            "concrete value versus generic",
                            &format!("{value:?}"),
                        ));
                    }
                    Some(PartialKind::Fixed(previous)) if previous != value => {
                        return Err(conflict(
                            "previous and new values",
                            &format!("{previous:?} versus {value:?}"),
                        ));
                    }
                    _ => plan.kind = Some(PartialKind::Fixed(*value)),
                },
            }
        }
    }
    Ok(plans
        .into_iter()
        .map(|plan| PackConstPlan {
            pack: plan.pack,
            source: plan.source,
            kind: match plan.kind.expect("every plan has at least one use") {
                PartialKind::Shared => PackConstKind::Shared,
                PartialKind::Fixed(value) => PackConstKind::Fixed(value),
            },
            aliases: plan.aliases,
            span: plan.span,
        })
        .collect())
}

/// True when this positional entry-point const reads a shared pack slot.
pub(crate) fn is_shared_pack_alias(plans: &[PackConstPlan], encoded: &str) -> bool {
    plans.iter().any(|plan| {
        matches!(plan.kind, PackConstKind::Shared)
            && plan.aliases.iter().any(|alias| alias == encoded)
    })
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
                const_params.push((param.ident.clone(), param.ty.clone()));
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
    let ctx = ParseContext::from_params(
        std::iter::empty::<Ident>(),
        const_params.iter().map(|(ident, _)| ident.clone()),
    );

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

/// Validate source-local resource names and concrete addresses, including
/// collisions between standalone declarations and pack fields.
pub(crate) fn validate_resources(module: &parse::ItemMod) -> parse::Result<()> {
    let mut names = std::collections::HashMap::<String, Span>::new();
    let mut addresses = std::collections::HashMap::<(u32, u32), Span>::new();

    fn record(
        ident: &Ident,
        group: Option<u32>,
        binding: Option<u32>,
        names: &mut std::collections::HashMap<String, Span>,
        addresses: &mut std::collections::HashMap<(u32, u32), Span>,
    ) -> parse::Result<()> {
        if names.insert(ident.to_string(), ident.span()).is_some() {
            return Err(parse::Error::unsupported(
                ident.span(),
                format!(
                    "duplicate resource name `{ident}`; every field or declaration must have a \
                     unique global resource name"
                ),
            ));
        }
        if let (Some(group), Some(binding)) = (group, binding)
            && addresses.insert((group, binding), ident.span()).is_some()
        {
            return Err(parse::Error::unsupported(
                ident.span(),
                format!(
                    "duplicate resource address @group({group}) @binding({binding}) on `{ident}`"
                ),
            ));
        }
        Ok(())
    }

    fn literal(lit: &syn::LitInt) -> Option<u32> {
        lit.base10_parse::<u32>().ok()
    }

    for item in &module.content {
        match item {
            parse::Item::Uniform(item) => record(
                &item.name,
                literal(&item.group),
                literal(&item.binding),
                &mut names,
                &mut addresses,
            )?,
            parse::Item::Storage(item) => record(
                &item.name,
                literal(&item.group),
                literal(&item.binding),
                &mut names,
                &mut addresses,
            )?,
            parse::Item::Sampler(item) => record(
                &item.name,
                literal(&item.group),
                literal(&item.binding),
                &mut names,
                &mut addresses,
            )?,
            parse::Item::Texture(item) => record(
                &item.name,
                literal(&item.group),
                literal(&item.binding),
                &mut names,
                &mut addresses,
            )?,
            parse::Item::ResourcePack(pack) => {
                for field in &pack.fields {
                    let group = match &field.group {
                        AddressValue::Literal(lit) => literal(lit),
                        AddressValue::Param(_) => None,
                    };
                    let binding = match &field.binding {
                        AddressValue::Literal(lit) => literal(lit),
                        AddressValue::Param(_) => None,
                    };
                    record(&field.ident, group, binding, &mut names, &mut addresses)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
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

// ===== Function and expression lowering =====

#[derive(Clone)]
struct PackParameter {
    param: Ident,
    pack: ItemResourcePack,
    const_args: Vec<parse::PackConstArg>,
    input_index: usize,
}

#[derive(Clone, Default)]
struct FnResourceSignature {
    input_count: usize,
    packs: Vec<PackParameter>,
    const_renames: Vec<(Ident, Ident)>,
}

impl FnResourceSignature {
    fn parameters(&self) -> std::collections::HashMap<String, PackParameter> {
        self.packs
            .iter()
            .map(|param| (param.param.to_string(), param.clone()))
            .collect()
    }
}

/// Remove pack parameters and lower their direct field borrows to globals.
pub(crate) fn lower_module(module: &mut parse::ItemMod) -> parse::Result<()> {
    let registry: std::collections::HashMap<String, ItemResourcePack> = module
        .resource_packs()
        .map(|pack| (pack.ident.to_string(), pack.clone()))
        .collect();
    let mut signatures = std::collections::HashMap::<String, FnResourceSignature>::new();

    // First collect every signature, so calls can be lowered independently of
    // function declaration order.
    for item in &mut module.content {
        match item {
            parse::Item::Fn(f) => {
                let signature = lower_fn_signature(f, &registry, &mut module.resource_pack_uses)?;
                signatures.insert(f.ident.to_string(), signature);
            }
            parse::Item::Impl(implementation) => {
                for item in &mut implementation.items {
                    let parse::ImplItem::Fn(f) = item else {
                        continue;
                    };
                    let key = impl_fn_key(&implementation.self_ty, &f.ident)?;
                    let signature =
                        lower_fn_signature(f, &registry, &mut module.resource_pack_uses)?;
                    signatures.insert(key, signature);
                }
            }
            _ => {}
        }
    }

    for item in &mut module.content {
        match item {
            parse::Item::Fn(f) => lower_fn_body(f, &signatures, &f.ident.to_string())?,
            parse::Item::Impl(implementation) => {
                for item in &mut implementation.items {
                    let parse::ImplItem::Fn(f) = item else {
                        continue;
                    };
                    let key = impl_fn_key(&implementation.self_ty, &f.ident)?;
                    lower_fn_body(f, &signatures, &key)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn impl_fn_key(self_ty: &Type, method: &Ident) -> parse::Result<String> {
    let base = crate::monomorphize::mangle_type(self_ty)?;
    Ok(format!("{base}::{method}"))
}

/// Recognize and remove a function's pack parameters, retaining their original
/// positions for call-argument removal.
fn lower_fn_signature(
    f: &mut parse::ItemFn,
    registry: &std::collections::HashMap<String, ItemResourcePack>,
    uses: &mut Vec<ResourcePackUse>,
) -> parse::Result<FnResourceSignature> {
    let mut signature = FnResourceSignature {
        input_count: f.inputs.len(),
        ..FnResourceSignature::default()
    };
    let mut seen_packs = std::collections::HashSet::new();

    for (input_index, arg) in f.inputs.iter().enumerate() {
        let Type::ResourcePack {
            ident,
            const_args,
            reference,
        } = &arg.ty
        else {
            continue;
        };
        if !reference {
            return Err(parse::Error::unsupported(
                arg.ident.span(),
                format!(
                    "resource-pack parameter `{}` must be borrowed as `&{ident}`",
                    arg.ident
                ),
            ));
        }
        let pack = registry
            .get(&ident.to_string())
            .expect("the parser only creates ResourcePack for registered structs")
            .clone();
        if !seen_packs.insert(ident.to_string()) {
            return Err(parse::Error::unsupported(
                arg.ident.span(),
                format!(
                    "duplicate resource-pack parameter of type `{ident}` in function `{}`",
                    f.ident
                ),
            ));
        }
        if const_args.len() != pack.const_params.len() {
            return Err(parse::Error::unsupported(
                ident.span(),
                format!(
                    "resource pack `{ident}` expects {} const arguments, but `{}` supplies {}",
                    pack.const_params.len(),
                    arg.ident,
                    const_args.len()
                ),
            ));
        }
        if !matches!(f.fn_attrs, parse::FnAttrs::None) {
            uses.push(ResourcePackUse {
                pack: ident.clone(),
                const_args: const_args.clone(),
                span: arg.ident.span(),
            });
        }
        signature.packs.push(PackParameter {
            param: arg.ident.clone(),
            pack,
            const_args: const_args.clone(),
            input_index,
        });
    }

    // A helper's const parameter tied only to a pack becomes a reference to
    // the pack declaration's module-level slot. This avoids manufacturing a
    // generic helper instantiation after its pack argument has disappeared.
    if matches!(f.fn_attrs, parse::FnAttrs::None) {
        let mut tied = std::collections::HashSet::new();
        for param in &signature.packs {
            for ((declared, _), actual) in param.pack.const_params.iter().zip(&param.const_args) {
                let parse::PackConstArg::Param(local) = actual else {
                    continue;
                };
                if f.const_params.contains(local) {
                    if let Some((_, previous)) = signature
                        .const_renames
                        .iter()
                        .find(|(source, _)| source == local)
                    {
                        if previous != declared {
                            return Err(parse::Error::unsupported(
                                local.span(),
                                format!(
                                    "const parameter `{local}` cannot simultaneously drive \
                                     `{previous}` and `{declared}`"
                                ),
                            ));
                        }
                    } else {
                        signature.const_renames.push((
                            local.clone(),
                            Ident::new(
                                &pack_const_slot_name(&param.pack.ident, declared),
                                declared.span(),
                            ),
                        ));
                    }
                    tied.insert(local.clone());
                }
            }
        }
        f.const_params.retain(|param| !tied.contains(param));
    }

    let removed: std::collections::HashSet<usize> = signature
        .packs
        .iter()
        .map(|param| param.input_index)
        .collect();
    let old_inputs = f.inputs.clone();
    let mut inputs = syn::punctuated::Punctuated::new();
    for (index, pair) in old_inputs.pairs().enumerate() {
        if removed.contains(&index) {
            continue;
        }
        inputs.push_value((*pair.value()).clone());
        if let Some(comma) = pair.punct() {
            inputs.push_punct(**comma);
        }
    }
    f.inputs = inputs;
    Ok(signature)
}

fn lower_fn_body(
    f: &mut parse::ItemFn,
    signatures: &std::collections::HashMap<String, FnResourceSignature>,
    key: &str,
) -> parse::Result<()> {
    use crate::parse_visitor::ParseVisitorMut;

    let signature = signatures
        .get(key)
        .expect("every parsed function received a resource signature");
    let mut lowerer = BodyLowerer {
        parameters: signature.parameters(),
        const_renames: &signature.const_renames,
        signatures,
        function: &f.ident,
    };
    if let parse::ReturnType::Type { ty, .. } = &mut f.return_type {
        lowerer.visit_type(ty)?;
    }
    lowerer.visit_block(&mut f.block)
}

struct BodyLowerer<'a> {
    parameters: std::collections::HashMap<String, PackParameter>,
    const_renames: &'a [(Ident, Ident)],
    signatures: &'a std::collections::HashMap<String, FnResourceSignature>,
    function: &'a Ident,
}

impl BodyLowerer<'_> {
    fn lower_call(
        &self,
        path: &parse::FnPath,
        params: &mut syn::punctuated::Punctuated<parse::Expr, Token![,]>,
    ) -> parse::Result<()> {
        let (key, span) = match path {
            parse::FnPath::Ident(ident) => (ident.to_string(), ident.span()),
            parse::FnPath::TypeMethod { ty, method, .. } => {
                (format!("{ty}::{method}"), method.span())
            }
        };
        let Some(signature) = self.signatures.get(&key) else {
            return Ok(());
        };
        if signature.packs.is_empty() {
            return Ok(());
        }
        if params.len() != signature.input_count {
            return Err(parse::Error::unsupported(
                span,
                format!(
                    "call in `{}` supplies {} arguments, but its resource-aware signature expects \
                     {}",
                    self.function,
                    params.len(),
                    signature.input_count
                ),
            ));
        }
        let removed: std::collections::HashSet<usize> = signature
            .packs
            .iter()
            .map(|param| param.input_index)
            .collect();
        let old = params.clone();
        let mut lowered = syn::punctuated::Punctuated::new();
        for (index, pair) in old.pairs().enumerate() {
            if removed.contains(&index) {
                continue;
            }
            lowered.push_value((*pair.value()).clone());
            if let Some(comma) = pair.punct() {
                lowered.push_punct(**comma);
            }
        }
        *params = lowered;
        Ok(())
    }
}

impl crate::parse_visitor::ParseVisitorMut for BodyLowerer<'_> {
    fn visit_expr(&mut self, expr: &mut parse::Expr) -> parse::Result<()> {
        if let parse::Expr::FnCall { path, params, .. } = expr {
            self.lower_call(path, params)?;
        }

        // The complete `&pack.field` expression is replaced, not merely its
        // base. This leaves the global resource identifier expected by WGSL.
        if let parse::Expr::Reference { expr: inner, .. } = expr
            && let parse::Expr::FieldAccess { base, field, .. } = inner.as_ref()
            && let parse::Expr::Ident(base) = base.as_ref()
            && let Some(param) = self.parameters.get(&base.to_string())
        {
            let Some(resource) = param.pack.fields.iter().find(|item| item.ident == *field) else {
                return Err(parse::Error::unsupported(
                    field.span(),
                    format!(
                        "resource pack `{}` has no field `{field}`",
                        param.pack.ident
                    ),
                ));
            };
            *expr = parse::Expr::Ident(resource.ident.clone());
            return Ok(());
        }

        if let parse::Expr::FieldAccess { base, field, .. } = expr
            && let parse::Expr::Ident(base) = base.as_ref()
            && self.parameters.contains_key(&base.to_string())
        {
            return Err(parse::Error::unsupported(
                field.span(),
                "resource-pack fields must be used as direct borrows such as `&pack.field`",
            ));
        }

        if let parse::Expr::Ident(ident) = expr {
            if self.parameters.contains_key(&ident.to_string()) {
                return Err(parse::Error::unsupported(
                    ident.span(),
                    format!(
                        "resource-pack parameter `{ident}` in `{}` can only be passed to a \
                         registered helper or used as `&{ident}.field`; destructuring and local \
                         aliases are not supported",
                        self.function
                    ),
                ));
            }
            if let Some((_, replacement)) = self
                .const_renames
                .iter()
                .find(|(source, _)| source == ident)
            {
                *ident = replacement.clone();
            }
        }
        crate::parse_visitor::walk_expr(self, expr)
    }

    fn visit_type(&mut self, ty: &mut Type) -> parse::Result<()> {
        if let Type::ResourcePack { ident, .. } = ty {
            return Err(parse::Error::unsupported(
                ident.span(),
                "resource-pack types are supported only as borrowed function parameters; return \
                 values, locals, and aliases are not WGSL resources",
            ));
        }
        crate::parse_visitor::walk_type(self, ty)
    }
}
