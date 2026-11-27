//! `#[derive(BindGroup)]`: typed, string-free bind-group creation.
//!
//! The derive is shared by `#[wgsl]`-module packs and standalone annotated
//! Rust structs: both are annotated the same way, and both emit the same
//! generated shape - a companion `{Pack}Resources<'a>` struct plus the
//! `wgsl_rs_linkage_wgpu::BindGroup` impl. Inside `#[wgsl]` the macro keeps
//! the field annotations on the emitted struct when it sees this derive, so
//! the derive re-reads them here.
//!
//! The `#[proc_macro_derive]` entry point lives at the crate root (a
//! proc-macro crate restriction); this module holds the expansion.

use quote::{format_ident, quote};
use syn::spanned::Spanned;

use crate::resource::{AddressValue, ResourceKind, parse_binding_args};

/// One bindable field's contribution to the generated code.
struct FieldPlan {
    field: syn::Ident,
    /// The `"Pack::field"` key.
    key: String,
    /// The `Resources` field type.
    resource_ty: proc_macro2::TokenStream,
    /// Wraps the resource field as a `wgpu::BindingResource`.
    wrap: proc_macro2::TokenStream,
    group: AddressValue,
}

/// Expands `#[derive(BindGroup)]`: typed resources, zero string keys.
///
/// The generated `Resources` struct carries one **named** wgpu resource per
/// bindable field, mapped by annotation kind - `#[storage]`/`#[uniform]` →
/// `wgpu::BufferBinding`, `#[texture]` → `&wgpu::TextureView`,
/// `#[sampler]` → `&wgpu::Sampler` - while `#[workgroup]` fields take no
/// host resource and are skipped. Bindings resolve through the generated
/// `"Pack::field"` keys and the field attributes' binding numbers, never
/// through declaration order.
///
/// Every annotated field must select the same group (one literal, or one
/// shared const parameter, whose value the impl carries), because one
/// `create_bind_group` call builds exactly one bind group.
pub(crate) fn expand(input: &syn::DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let syn::Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            input.span(),
            "`BindGroup` requires a struct with named fields",
        ));
    };
    let syn::Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new(
            input.span(),
            "`BindGroup` requires a struct with named fields",
        ));
    };

    let const_params: std::collections::HashMap<String, syn::Type> = input
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Const(param) => Some((param.ident.to_string(), param.ty.clone())),
            _ => None,
        })
        .collect();

    let mut plans: Vec<FieldPlan> = Vec::new();
    for field in &fields.named {
        let Some(attr) = field
            .attrs
            .iter()
            .find(|attr| ResourceKind::from_attr(attr).is_some())
        else {
            return Err(syn::Error::new(
                field.ident.span(),
                "`BindGroup` requires every field to carry a resource annotation (`#[uniform]`, \
                 `#[storage]`, `#[texture]`, `#[sampler]`, or `#[workgroup]`)",
            ));
        };
        let kind = ResourceKind::from_attr(attr).expect("checked above");
        if kind == ResourceKind::Workgroup {
            // Workgroup storage is workgroup-scoped: no host resource.
            continue;
        }
        let args = parse_binding_args(attr, kind, &const_params)
            .map_err(|error| syn::Error::new(error.span(), error.to_string()))?;
        let ident = field.ident.clone().expect("named fields");
        let key = format!("{}::{}", input.ident, ident);
        let (resource_ty, wrap) = match kind {
            ResourceKind::Storage | ResourceKind::Uniform => (
                quote! { ::wgsl_rs_linkage_wgpu::wgpu::BufferBinding<'a> },
                quote! { ::wgsl_rs_linkage_wgpu::wgpu::BindingResource::Buffer },
            ),
            ResourceKind::Texture => (
                quote! { &'a ::wgsl_rs_linkage_wgpu::wgpu::TextureView },
                quote! { ::wgsl_rs_linkage_wgpu::wgpu::BindingResource::TextureView },
            ),
            ResourceKind::Sampler => (
                quote! { &'a ::wgsl_rs_linkage_wgpu::wgpu::Sampler },
                quote! { ::wgsl_rs_linkage_wgpu::wgpu::BindingResource::Sampler },
            ),
            ResourceKind::Workgroup => unreachable!("skipped above"),
        };
        plans.push(FieldPlan {
            field: ident,
            key,
            resource_ty,
            wrap,
            group: args.group,
        });
    }

    if plans.is_empty() {
        return Err(syn::Error::new(
            input.span(),
            "`BindGroup` requires at least one bindable field (`#[workgroup]` fields take no host \
             resource)",
        ));
    }

    // One `create_bind_group` call builds exactly one bind group, so every
    // field must select the same group.
    let first_group = plans[0].group.clone();
    for plan in &plans {
        if plan.group != first_group {
            return Err(syn::Error::new(
                plan.field.span(),
                "`BindGroup` requires every field to select the same group; split multi-group \
                 packs into one pack per group",
            ));
        }
    }
    let group_expr = match &first_group {
        AddressValue::Literal(lit) => quote! { #lit },
        AddressValue::Param(param) => quote! { #param },
    };

    let ident = &input.ident;
    let vis = &input.vis;
    let resources_ident = format_ident!("{}Resources", ident);
    let resource_fields = plans.iter().map(|plan| {
        let field = &plan.field;
        let ty = &plan.resource_ty;
        quote! { pub #field: #ty, }
    });
    let entries = plans.iter().map(|plan| {
        let field = &plan.field;
        let key = &plan.key;
        let wrap = &plan.wrap;
        quote! { (#key, #wrap(resources.#field)), }
    });
    let pack_name = ident.to_string();

    // The impl mirrors the pack's generics (const params select the group,
    // layout-only consts ride along) with a leading `'a` for the borrowed
    // resources. The type arguments exclude `'a`: the pack type itself
    // stays as written.
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut impl_generics_source = input.generics.clone();
    impl_generics_source
        .params
        .insert(0, syn::GenericParam::Lifetime(syn::parse_quote!('a)));
    let (impl_generics, _, _) = impl_generics_source.split_for_impl();

    Ok(quote! {
        #[allow(dead_code)]
        #vis struct #resources_ident<'a> {
            #(#resource_fields)*
        }

        #[allow(dead_code)]
        #[automatically_derived]
        impl #impl_generics ::wgsl_rs_linkage_wgpu::BindGroup<'a> for #ident #ty_generics
        #where_clause
        {
            type Resources = #resources_ident<'a>;

            fn create_bind_group(
                linkage: &mut ::wgsl_rs_linkage_wgpu::WgpuLinkage,
                device: &::wgsl_rs_linkage_wgpu::wgpu::Device,
                resources: Self::Resources,
                label: Option<&str>,
            ) -> ::wgsl_rs_linkage_wgpu::wgpu::BindGroup {
                ::wgsl_rs_linkage_wgpu::WgpuLinkage::create_bind_group_named_with_label(
                    linkage,
                    #group_expr,
                    device,
                    label,
                    &[
                        #(#entries)*
                    ],
                )
                .unwrap_or_else(|error| {
                    panic!("bind group creation for `{}` failed: {error}", #pack_name)
                })
            }
        }
    })
}
