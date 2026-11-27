//! Rust-only resource-pack structs.
//!
//! A resource pack is recognized by binding annotations on its fields. The
//! struct remains an ordinary Rust value, while its fields describe global
//! WGSL resources for later lowering passes.

use proc_macro2::Span;
use quote::ToTokens;
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
    Workgroup,
}

impl ResourceKind {
    fn from_attr(attr: &syn::Attribute) -> Option<Self> {
        let ident = attr.path().get_ident()?;
        match ident.to_string().as_str() {
            "uniform" => Some(Self::Uniform),
            "storage" => Some(Self::Storage),
            "sampler" => Some(Self::Sampler),
            "texture" => Some(Self::Texture),
            "workgroup" => Some(Self::Workgroup),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Storage => "storage",
            Self::Sampler => "sampler",
            Self::Texture => "texture",
            Self::Workgroup => "workgroup",
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
}

/// One annotated resource field.
#[derive(Clone)]
pub(crate) struct ResourceField {
    pub(crate) ident: Ident,
    pub(crate) kind: ResourceKind,
    pub(crate) group: AddressValue,
    pub(crate) binding: AddressValue,
    pub(crate) access: StorageAccess,
    /// True when the annotation named the access but the field type omitted
    /// the marker; the emitted Rust field type gains the marker.
    pub(crate) insert_marker: bool,
    pub(crate) ty: Type,
}

/// One entry-point use of a resource-pack type. Const arguments have already
/// been positionally encoded by the entry-point parser; type arguments are
/// retained verbatim for the generated `instantiate` constraints.
#[derive(Clone)]
pub(crate) struct ResourcePackUse {
    pub(crate) pack: Ident,
    pub(crate) const_args: Vec<parse::PackConstArg>,
    pub(crate) type_args: Vec<parse::PackTypeArg>,
    pub(crate) fn_name: String,
    pub(crate) span: Span,
}

/// A registered resource-pack struct.
#[derive(Clone)]
pub(crate) struct ItemResourcePack {
    pub(crate) ident: Ident,
    pub(crate) type_params: Vec<syn::TypeParam>,
    pub(crate) const_params: Vec<(Ident, syn::Type)>,
    pub(crate) fields: Vec<ResourceField>,
}

/// Deterministic module slot for one generic parameter of one pack type,
/// shared by const and type parameters alike. Namespacing by pack keeps
/// identically named parameters of different packs independent.
pub(crate) fn pack_slot_name(pack: &Ident, param: &Ident) -> String {
    format!("{pack}_{param}")
}

/// The generated WGSL global name for one pack field: the lowercased pack
/// name joined to the field name with a double underscore. The loud
/// separator marks the pack/field boundary for human readers; the encoding
/// is not injective in pathological corners, so resource-name validation
/// rejects collisions.
pub(crate) fn mangle_resource_name(pack: &Ident, field: &Ident) -> String {
    format!("{}__{}", pack.to_string().to_lowercase(), field)
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
            let source = pack_slot_name(&pack.ident, declared);
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
            if let parse::PackConstArg::Param(target) = actual {
                let reused = plans.iter().any(|other| {
                    !(other.pack == pack_name && other.source == source)
                        && other.aliases.contains(target)
                });
                if reused {
                    return Err(parse::Error::unsupported(
                        usage.span,
                        format!(
                            "entry-point const parameter `{target}` feeds multiple resource-pack \
                             slots; use independent const parameters for each pack layout"
                        ),
                    ));
                }
            }
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
    if item.ident.to_string().to_lowercase().starts_with("__") {
        return Err(parse::Error::unsupported(
            item.ident.span(),
            format!(
                "resource-pack name `{}` mangles to a WGSL identifier starting with `__`, which \
                 the WGSL spec reserves; rename the pack",
                item.ident
            ),
        ));
    }
    let syn::Fields::Named(fields) = &item.fields else {
        return Err(parse::Error::unsupported(
            item.fields.span(),
            "resource packs require named fields",
        ));
    };

    let mut type_params = Vec::new();
    let mut const_params = Vec::new();
    let mut const_param_types = std::collections::HashMap::new();
    for param in &item.generics.params {
        match param {
            syn::GenericParam::Type(param) => type_params.push(param.clone()),
            syn::GenericParam::Const(param) => {
                const_params.push((param.ident.clone(), param.ty.clone()));
                const_param_types.insert(param.ident.to_string(), param.ty.clone());
            }
            syn::GenericParam::Lifetime(param) => {
                return Err(parse::Error::unsupported(
                    param.span(),
                    "resource packs do not support lifetime parameters",
                ));
            }
        }
    }
    // Field types parse under renamed type parameters: a parameter `T`
    // becomes the pack-owned slot `Pack_T`, the same namespacing const
    // parameters use, so one module-level generic substitutes every use
    // of one pack layout. Const parameters keep their source names here;
    // their slots are applied when addresses are lowered.
    let ctx = ParseContext::from_renamed_type_params_with_consts(
        type_params.iter().map(|param| {
            (
                param.ident.to_string(),
                pack_slot_name(&item.ident, &param.ident),
            )
        }),
        const_params
            .iter()
            .map(|(ident, _)| (ident.to_string(), ident.to_string())),
    );

    let mut parsed_fields = Vec::with_capacity(fields.named.len());
    for field in &fields.named {
        parsed_fields.push(parse_field(field, &ctx, &const_param_types)?);
    }
    for param in &type_params {
        let slot = pack_slot_name(&item.ident, &param.ident);
        let used = parsed_fields
            .iter()
            .any(|field| type_references_param(&field.ty, &slot));
        if !used {
            return Err(parse::Error::unsupported(
                param.span(),
                format!(
                    "resource-pack type parameter `{}` is not used by any field type; remove it \
                     or reference it from an annotated field",
                    param.ident
                ),
            ));
        }
    }
    Ok(ItemResourcePack {
        ident: item.ident.clone(),
        type_params,
        const_params,
        fields: parsed_fields,
    })
}

/// Whether `ty` contains a `Type::TypeParam` named `name`, reaching through
/// every type that nests other types. Used to require that each pack type
/// parameter is referenced by at least one annotated field, and to keep tied
/// helper parameters out of non-pack positions.
fn type_references_param(ty: &Type, name: &str) -> bool {
    match ty {
        Type::TypeParam { ident } => ident == name,
        Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } => {
            type_references_param(elem, name)
        }
        Type::Struct { type_args, .. } => {
            type_args.iter().any(|ty| type_references_param(ty, name))
        }
        Type::Ptr { elem, .. } | Type::Phantom { elem, .. } => type_references_param(elem, name),
        Type::AssocType { ty, .. } => type_references_param(ty, name),
        Type::Scalar { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::Sampler { .. }
        | Type::SamplerComparison { .. }
        | Type::Texture { .. }
        | Type::TextureDepth { .. }
        | Type::TextureStorage { .. }
        | Type::ResourcePack { .. } => false,
    }
}

/// Parse one field and reject mixed IO/resource annotations and malformed
/// binding metadata at the field that caused the problem.
fn parse_field(
    field: &syn::Field,
    ctx: &ParseContext,
    const_params: &std::collections::HashMap<String, syn::Type>,
) -> parse::Result<ResourceField> {
    if matches!(field.vis, syn::Visibility::Inherited) {
        return Err(parse::Error::unsupported(
            field.span(),
            "resource-pack fields must be `pub` or `pub(crate)` so hosts can construct packs by \
             hand",
        ));
    }
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
    }

    let kind = ResourceKind::from_attr(first).expect("resource attribute was selected above");
    // `#[workgroup]` takes no arguments: workgroup variables have no group or
    // binding. The dummy addresses below never reach the IR or the WGSL; the
    // Workgroup arm of the IR conversion ignores them.
    let args = match kind {
        ResourceKind::Workgroup => {
            if !matches!(first.meta, syn::Meta::Path(_)) {
                return Err(parse::Error::unsupported(
                    first.span(),
                    "`#[workgroup]` takes no arguments; workgroup variables have no group or \
                     binding",
                ));
            }
            BindingArgs {
                group: AddressValue::Literal(syn::LitInt::new("0", Span::call_site())),
                binding: AddressValue::Literal(syn::LitInt::new("0", Span::call_site())),
                access: None,
            }
        }
        _ => parse_binding_args(first, kind, const_params)?,
    };

    // Uniform, storage, and workgroup fields declare the runtime wrapper
    // types (`Uniform<T>`, `Storage<T, AM>`, `Workgroup<T>`); the WGSL inner
    // type is extracted from the wrapper's first generic argument. Textures
    // and samplers are their own runtime types and pass through unchanged.
    let (inner_ty, marker_access) = match kind {
        ResourceKind::Uniform => extract_wrapper(field, kind, "Uniform", false)?,
        ResourceKind::Storage => extract_wrapper(field, kind, "Storage", true)?,
        ResourceKind::Workgroup => extract_wrapper(field, kind, "Workgroup", false)?,
        ResourceKind::Texture | ResourceKind::Sampler => (field.ty.clone(), None),
    };
    let ty = Type::parse(&inner_ty, ctx)?;
    validate_resource_type(field, kind, &ty)?;

    // Effective storage access: the annotation and the type marker must
    // agree when both name one; otherwise whichever names one wins, with
    // WGSL's `read` as the default. When the annotation names the access
    // and the marker is absent, the emitted Rust field type gains the
    // marker so CPU capability follows the annotation.
    let (access, insert_marker) = match (args.access, marker_access) {
        (Some(StorageAccess::Read), Some(StorageAccess::ReadWrite))
        | (Some(StorageAccess::ReadWrite), Some(StorageAccess::Read)) => {
            return Err(parse::Error::unsupported(
                field.ty.span(),
                format!(
                    "storage access mismatch on `{}`: the annotation and the `Storage<T, _>` type \
                     marker must agree; remove one or make them match",
                    field
                        .ident
                        .as_ref()
                        .expect("resource-pack parsing requires named fields"),
                ),
            ));
        }
        (Some(annotation), Some(_)) => (annotation, false),
        (Some(annotation), None) => (annotation, true),
        (None, Some(marker)) => (marker, false),
        (None, None) => (StorageAccess::Read, false),
    };

    Ok(ResourceField {
        ident: field
            .ident
            .clone()
            .expect("resource-pack parsing requires named fields"),
        kind,
        group: args.group,
        binding: args.binding,
        access,
        insert_marker,
        ty,
    })
}

/// Split a wrapper-typed field into its WGSL inner type and optional
/// access marker. `Uniform<T>` takes exactly one type argument;
/// `Storage<T, AM>` takes the inner type first and the access marker
/// (`Read`/`ReadWrite`) second, optionally.
fn extract_wrapper(
    field: &syn::Field,
    kind: ResourceKind,
    wrapper: &str,
    marker_allowed: bool,
) -> parse::Result<(syn::Type, Option<StorageAccess>)> {
    let wrong_shape = || {
        parse::Error::unsupported(
            field.ty.span(),
            format!(
                "`#[{}]` field types must use the `{wrapper}<T>` wrapper type; the WGSL type is \
                 extracted from its type argument, got `{}`",
                kind.as_str(),
                field.ty.to_token_stream(),
            ),
        )
    };
    let syn::Type::Path(type_path) = &field.ty else {
        return Err(wrong_shape());
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return Err(wrong_shape());
    }
    let segment = type_path
        .path
        .segments
        .first()
        .expect("length checked above");
    if segment.ident != wrapper {
        return Err(wrong_shape());
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return Err(wrong_shape());
    };
    let max = if marker_allowed { 2 } else { 1 };
    if args.args.is_empty() || args.args.len() > max {
        return Err(wrong_shape());
    }
    let syn::GenericArgument::Type(inner) = args.args.first().expect("length checked above") else {
        return Err(wrong_shape());
    };
    let inner = inner.clone();
    let marker = if args.args.len() == 2 {
        let syn::GenericArgument::Type(syn::Type::Path(path)) =
            args.args.last().expect("length checked above")
        else {
            return Err(wrong_shape());
        };
        match path
            .path
            .get_ident()
            .map(|ident| ident.to_string())
            .as_deref()
        {
            Some("Read") => Some(StorageAccess::Read),
            Some("ReadWrite") => Some(StorageAccess::ReadWrite),
            _ => {
                return Err(parse::Error::unsupported(
                    field.ty.span(),
                    "the `Storage<T, AM>` access marker must be `Read` or `ReadWrite`",
                ));
            }
        }
    } else {
        None
    };
    Ok((inner, marker))
}

struct BindingArgs {
    group: AddressValue,
    binding: AddressValue,
    /// The annotation-named access mode, if present.
    access: Option<StorageAccess>,
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
    let mut access: Option<StorageAccess> = None;

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
                    if access.is_some() {
                        return Err(syn::Error::new(
                            ident.span(),
                            "duplicate storage access modifier",
                        ));
                    }
                    access = Some(
                        if ident == "read_write" {
                            StorageAccess::ReadWrite
                        } else {
                            StorageAccess::Read
                        },
                    );
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
        ResourceKind::Uniform | ResourceKind::Storage | ResourceKind::Workgroup => !matches!(
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
                    // Workgroup fields carry no addresses; their dummy
                    // literals must not collide with real bindings.
                    let (group, binding) = if matches!(field.kind, ResourceKind::Workgroup) {
                        (None, None)
                    } else {
                        let group = match &field.group {
                            AddressValue::Literal(lit) => literal(lit),
                            AddressValue::Param(_) => None,
                        };
                        let binding = match &field.binding {
                            AddressValue::Literal(lit) => literal(lit),
                            AddressValue::Param(_) => None,
                        };
                        (group, binding)
                    };
                    let name = Ident::new(
                        &mangle_resource_name(&pack.ident, &field.ident),
                        field.ident.span(),
                    );
                    record(&name, group, binding, &mut names, &mut addresses)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validate that pack-owned parameter slots are unambiguous across the module.
///
/// [`pack_slot_name`] encodes `(pack, param)` as `{pack}_{param}`, which is
/// not injective: `PackA_B::<const C>` and `PackA::<const B_C>` both produce
/// `PackA_B_C`. Distinct packs sharing a slot key would collapse to a
/// single substitution entry in the generated `instantiate`, silently
/// pointing one pack's addresses, filterability, or field types at the
/// other pack's values. Re-encoding for injectivity would give up the
/// readable, user-facing slot names (surfaced through
/// `Source::module_type_params`), so the collision is rejected at the second
/// pack instead.
///
/// Standalone binding consts (legacy `texture!`/`sampler!` filterability
/// parameters) share the same dedup-by-string namespace: a standalone const
/// spelled `{pack}_{param}` would silently adopt the pack slot's argument
/// (and vice versa) while the two differ in type (`bool` vs the pack's
/// const type). Those collisions are rejected here too.
pub(crate) fn validate_pack_param_slots(module: &parse::ItemMod) -> parse::Result<()> {
    let mut slots = std::collections::HashMap::<String, (&Ident, &Ident)>::new();
    for pack in module.resource_packs() {
        let params = pack
            .type_params
            .iter()
            .map(|param| &param.ident)
            .chain(pack.const_params.iter().map(|(param, _)| param));
        for param in params {
            let slot = pack_slot_name(&pack.ident, param);
            if let Some((other_pack, other_param)) =
                slots.insert(slot.clone(), (&pack.ident, param))
            {
                return Err(parse::Error::unsupported(
                    param.span(),
                    format!(
                        "resource-pack slot `{slot}` is generated by both \
                         `{other_pack}::<{other_param}>` and `{}::<{param}>`; rename one pack or \
                         parameter so each slot stays unique",
                        pack.ident
                    ),
                ));
            }
        }
    }
    for (name, _ty, span, owner) in parse::binding_const_params(module) {
        if owner.is_some() {
            continue;
        }
        if let Some((other_pack, other_param)) = slots.get(&name) {
            return Err(parse::Error::unsupported(
                span,
                format!(
                    "binding const parameter `{name}` collides with the resource-pack slot \
                     generated by `{other_pack}::<{other_param}>`; use distinct names"
                ),
            ));
        }
    }
    Ok(())
}

/// Strip binding attributes from emitted pack structs and insert access
/// markers the annotation supplied but the field type omitted. Everything
/// else about the user's struct is emitted verbatim: hosts construct packs
/// by hand, so there are no generated constructors and no field-type
/// rewrites.
pub(crate) fn rewrite_rust_module(
    input_mod: &mut syn::ItemMod,
    module: &parse::ItemMod,
) -> parse::Result<()> {
    let Some((_, items)) = input_mod.content.as_mut() else {
        return Ok(());
    };
    for item in items.iter_mut() {
        let syn::Item::Struct(item_struct) = item else {
            continue;
        };
        let Some(pack) = module.resource_pack(&item_struct.ident) else {
            continue;
        };
        let syn::Fields::Named(fields) = &mut item_struct.fields else {
            continue;
        };
        for field in &mut fields.named {
            field
                .attrs
                .retain(|attr| ResourceKind::from_attr(attr).is_none());
            let Some(resource) = pack
                .fields
                .iter()
                .find(|resource| Some(&resource.ident) == field.ident.as_ref())
            else {
                continue;
            };
            if resource.insert_marker {
                insert_access_marker(&mut field.ty, resource.access);
            }
        }
    }
    Ok(())
}

/// Append the access marker to a `Storage<T>` field type that omitted it,
/// so CPU capability follows the annotation.
fn insert_access_marker(ty: &mut syn::Type, access: StorageAccess) {
    let syn::Type::Path(type_path) = ty else {
        return;
    };
    let Some(segment) = type_path.path.segments.iter_mut().next() else {
        return;
    };
    let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments else {
        return;
    };
    let marker: syn::Type = match access {
        StorageAccess::Read => syn::parse_quote!(Read),
        StorageAccess::ReadWrite => syn::parse_quote!(ReadWrite),
    };
    args.args.push_punct(Default::default());
    args.args.push(syn::GenericArgument::Type(marker));
}

// ===== Function and expression lowering =====

#[derive(Clone)]
struct PackParameter {
    param: Ident,
    pack: ItemResourcePack,
    const_args: Vec<parse::PackConstArg>,
    type_args: Vec<parse::PackTypeArg>,
    input_index: usize,
}

#[derive(Clone, Default)]
struct FnResourceSignature {
    input_count: usize,
    packs: Vec<PackParameter>,
    const_renames: Vec<(Ident, Ident)>,
    type_renames: Vec<(Ident, Ident)>,
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
            type_args,
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
        if const_args.len() + type_args.len() != pack.const_params.len() + pack.type_params.len() {
            return Err(parse::Error::unsupported(
                ident.span(),
                format!(
                    "resource pack `{ident}` expects {} generic arguments, but `{}` supplies {}",
                    pack.const_params.len() + pack.type_params.len(),
                    arg.ident,
                    const_args.len() + type_args.len()
                ),
            ));
        }
        if !matches!(f.fn_attrs, parse::FnAttrs::None) {
            uses.push(ResourcePackUse {
                pack: ident.clone(),
                const_args: const_args.clone(),
                type_args: type_args.clone(),
                fn_name: f.ident.to_string(),
                span: arg.ident.span(),
            });
        }
        signature.packs.push(PackParameter {
            param: arg.ident.clone(),
            pack,
            const_args: const_args.clone(),
            type_args: type_args.clone(),
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
                                &pack_slot_name(&param.pack.ident, declared),
                                declared.span(),
                            ),
                        ));
                    }
                    tied.insert(local.clone());
                }
            }
        }
        f.const_params.retain(|param| !tied.contains(param));

        // A helper's type parameter tied only to a pack parameter becomes a
        // reference to the pack declaration's module-level slot, mirroring
        // the const treatment above. A tied type parameter must not appear
        // in any other parameter or return position: those flow through
        // same-module monomorphization and would conflate the per-call-site
        // generic with the module-level slot.
        let mut tied_types = std::collections::HashSet::new();
        for param in &signature.packs {
            for (declared, actual) in param.pack.type_params.iter().zip(&param.type_args) {
                let parse::PackTypeArg::Param(local) = actual else {
                    continue;
                };
                if !f.type_params.contains(local) {
                    continue;
                }
                if let Some((_, previous)) = signature
                    .type_renames
                    .iter()
                    .find(|(source, _)| source == local)
                {
                    if previous != &declared.ident {
                        return Err(parse::Error::unsupported(
                            local.span(),
                            format!(
                                "type parameter `{local}` cannot simultaneously drive \
                                 `{previous}` and `{}`",
                                declared.ident
                            ),
                        ));
                    }
                } else {
                    signature.type_renames.push((
                        local.clone(),
                        Ident::new(
                            &pack_slot_name(&param.pack.ident, &declared.ident),
                            declared.ident.span(),
                        ),
                    ));
                }
                tied_types.insert(local.clone());
            }
        }
        if !tied_types.is_empty() {
            let pack_indices: std::collections::HashSet<usize> = signature
                .packs
                .iter()
                .map(|param| param.input_index)
                .collect();
            let offense = f
                .inputs
                .iter()
                .enumerate()
                .filter(|(index, _)| !pack_indices.contains(index))
                .find_map(|(_, arg)| {
                    tied_types
                        .iter()
                        .find(|local| type_references_param(&arg.ty, &local.to_string()))
                        .map(|local| (local.clone(), arg.ident.span()))
                })
                .or_else(|| {
                    let parse::ReturnType::Type { ty, .. } = &f.return_type else {
                        return None;
                    };
                    tied_types
                        .iter()
                        .find(|local| type_references_param(ty, &local.to_string()))
                        .map(|local| (local.clone(), local.span()))
                });
            if let Some((local, span)) = offense {
                return Err(parse::Error::unsupported(
                    span,
                    format!(
                        "helper type parameter `{local}` is tied to a resource pack but also used \
                         in another parameter or return position; pack-tied type parameters must \
                         appear only in the pack parameter"
                    ),
                ));
            }
            f.type_params.retain(|param| !tied_types.contains(param));
        }
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
        type_renames: &signature.type_renames,
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
    type_renames: &'a [(Ident, Ident)],
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
        // `*get!(pack.field)` / `*get_mut!(pack.field) = value`: elide the
        // deref by replacing the unary with its access operand; the
        // LinkageAccess arm below resolves it, and WGSL reads and assigns
        // storage and workgroup values directly. Parenthesized operands
        // peel the same way.
        if let parse::Expr::Unary {
            op: parse::UnOp::Deref(_),
            expr: inner,
            ..
        } = expr
            && matches!(
                inner.peel_parens(),
                parse::Expr::LinkageAccess {
                    pack_field: Some(_),
                    ..
                }
            )
        {
            *expr = (**inner).clone();
        }

        // `get!`/`get_mut!`/`load!` on a pack field borrow resolve to the
        // mangled global resource, with the access checked against the
        // field's kind and storage access.
        if let parse::Expr::LinkageAccess {
            kind,
            pack_field: Some((pack_param, field)),
            type_arg,
            ..
        } = expr
        {
            let Some(param) = self.parameters.get(&pack_param.to_string()) else {
                return Err(parse::Error::unsupported(
                    pack_param.span(),
                    format!(
                        "`{pack_param}` is not a resource-pack parameter of `{}`; \
                         get!/get_mut!/load! targets must be a pack parameter's field",
                        self.function
                    ),
                ));
            };
            let Some(resource) = param.pack.fields.iter().find(|item| item.ident == *field) else {
                return Err(parse::Error::unsupported(
                    field.span(),
                    format!(
                        "resource pack `{}` has no field `{field}`",
                        param.pack.ident
                    ),
                ));
            };
            match resource.kind {
                ResourceKind::Texture | ResourceKind::Sampler => {
                    return Err(parse::Error::unsupported(
                        field.span(),
                        "textures and samplers are handles in WGSL, not loadable values; pass \
                         them to the texture builtins as `&pack.field`",
                    ));
                }
                ResourceKind::Uniform => {
                    if matches!(kind, parse::LinkageKind::GetMut) {
                        return Err(parse::Error::unsupported(
                            field.span(),
                            "uniform values are read-only from a shader; get_mut! requires a \
                             read_write storage or workgroup field",
                        ));
                    }
                }
                ResourceKind::Storage => {
                    if matches!(kind, parse::LinkageKind::GetMut)
                        && matches!(resource.access, parse::StorageAccess::Read)
                    {
                        return Err(parse::Error::unsupported(
                            field.span(),
                            "get_mut! requires a read_write storage field; this field is read_only",
                        ));
                    }
                }
                ResourceKind::Workgroup => {}
            }
            if type_arg.is_some() {
                return Err(parse::Error::unsupported(
                    field.span(),
                    "resource-pack fields hold concrete types at their declaration; the `, T` \
                     form of get!/get_mut!/load! exists only for generic module statics",
                ));
            }
            if matches!(kind, parse::LinkageKind::Load) {
                match &resource.ty {
                    parse::Type::Atomic { .. } => {
                        return Err(parse::Error::unsupported(
                            field.span(),
                            "load! on an atomic field — atomics must be read with atomicLoad; \
                             write atomic_load(&get!(...))",
                        ));
                    }
                    parse::Type::RuntimeArray { .. } => {
                        return Err(parse::Error::unsupported(
                            field.span(),
                            "load! on a runtime-sized array field — runtime-sized arrays are not \
                             copyable WGSL values; index into get!(...) or copy elements with \
                             slab_copy!",
                        ));
                    }
                    _ => {}
                }
            }
            *expr = parse::Expr::Ident(Ident::new(
                &mangle_resource_name(&param.pack.ident, &resource.ident),
                resource.ident.span(),
            ));
            return Ok(());
        }

        if let parse::Expr::FnCall { path, params, .. } = expr {
            // WGSL arrayLength is the one supported builtin here that needs
            // a pointer to the global storage value rather than its value.
            if matches!(path, parse::FnPath::Ident(ident) if ident == "array_length") {
                for param_expr in params.iter_mut() {
                    let parse::Expr::Reference { expr: inner, .. } = param_expr else {
                        continue;
                    };
                    let parse::Expr::FieldAccess { base, field, .. } = inner.as_ref() else {
                        continue;
                    };
                    let parse::Expr::Ident(base) = base.as_ref() else {
                        continue;
                    };
                    let Some(param) = self.parameters.get(&base.to_string()) else {
                        continue;
                    };
                    let Some(resource) = param.pack.fields.iter().find(|item| item.ident == *field)
                    else {
                        continue;
                    };
                    **inner = parse::Expr::Ident(Ident::new(
                        &mangle_resource_name(&param.pack.ident, &resource.ident),
                        resource.ident.span(),
                    ));
                }
            }
            self.lower_call(path, params)?;
        }

        // The complete `&pack.field` expression is replaced, not merely its
        // base. Handle-typed resources (textures, samplers) render as value
        // handles in WGSL, so their borrows drop the `&`; value resources
        // (storage, uniform, workgroup) keep it: WGSL pointers to those
        // globals are written `&name`, e.g. for `workgroupUniformLoad` and
        // `arrayLength`.
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
            let mangled = Ident::new(
                &mangle_resource_name(&param.pack.ident, &resource.ident),
                resource.ident.span(),
            );
            match resource.kind {
                ResourceKind::Texture | ResourceKind::Sampler => {
                    *expr = parse::Expr::Ident(mangled);
                }
                _ => {
                    **inner = parse::Expr::Ident(mangled);
                }
            }
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
        // A helper's pack-tied type parameter reads the pack declaration's
        // module-level slot, the same rename the const treatment applies to
        // identifiers.
        if let Type::TypeParam { ident } = ty
            && let Some((_, replacement)) =
                self.type_renames.iter().find(|(source, _)| source == ident)
        {
            *ident = replacement.clone();
        }
        crate::parse_visitor::walk_type(self, ty)
    }
}
