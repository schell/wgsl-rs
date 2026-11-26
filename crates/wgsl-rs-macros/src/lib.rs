#![cfg_attr(nightly, feature(proc_macro_diagnostic))]

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use snafu::prelude::*;
use syn::{
    DeriveInput,
    visit_mut::{self, VisitMut},
};

use crate::parse::InterStageIo;

mod builder;
mod builtins;
mod deref_validation;
mod ir_convert;
mod ir_emit;
mod ir_emit_static;
mod load_validation;
mod module_id;
mod monomorphize;
mod parse;
mod parse_visitor;
mod ptr;
mod sampler;
mod storage;
mod swizzle;
mod texture;
mod uniform;
mod vector_cmp;
mod workgroup;

/// Visitor that strips `#[wgsl_allow(...)]` attributes from expressions.
/// These attributes are used by wgsl-rs during parsing but should not appear
/// in the emitted Rust code (since statement-level attributes require nightly).
struct StripWgslAllowAttrs;

impl VisitMut for StripWgslAllowAttrs {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        // Strip wgsl_allow attributes from any expression that has them
        strip_wgsl_allow_attrs(expr);
        // Continue visiting nested expressions
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Strips `#[wgsl_allow(...)]` attributes from an expression.
fn strip_wgsl_allow_attrs(expr: &mut syn::Expr) {
    let attrs = match expr {
        syn::Expr::ForLoop(e) => &mut e.attrs,
        syn::Expr::While(e) => &mut e.attrs,
        syn::Expr::Loop(e) => &mut e.attrs,
        syn::Expr::If(e) => &mut e.attrs,
        syn::Expr::Match(e) => &mut e.attrs,
        syn::Expr::Block(e) => &mut e.attrs,
        _ => return,
    };

    attrs.retain(|attr| !attr.path().is_ident("wgsl_allow"));
}

/// Visitor that strips inter-stage IO attributes (`#[builtin(...)]`,
/// `#[location(N)]`, `#[interpolate(...)]`, `#[blend_src(N)]`, `#[invariant]`)
/// from struct fields and function parameters.
///
/// These attributes are read during WGSL parsing, but they aren't valid Rust
/// attributes in the emitted code. The `#[wgsl]` macro strips them after
/// parsing so the module compiles as plain Rust, while the IO information has
/// already been captured for WGSL output.
struct StripIoAttrs;

impl StripIoAttrs {
    /// Retains attributes that are not wgsl-rs inter-stage IO annotations.
    fn strip(attrs: &mut Vec<syn::Attribute>) {
        attrs.retain(|attr| InterStageIo::try_from(attr).is_err());
    }
}

impl VisitMut for StripIoAttrs {
    fn visit_field_mut(&mut self, field: &mut syn::Field) {
        Self::strip(&mut field.attrs);
        visit_mut::visit_field_mut(self, field);
    }

    fn visit_fn_arg_mut(&mut self, arg: &mut syn::FnArg) {
        if let syn::FnArg::Typed(pat_type) = arg {
            Self::strip(&mut pat_type.attrs);
        }
        visit_mut::visit_fn_arg_mut(self, arg);
    }
}

#[derive(Default)]
struct Attrs {
    /// Present if the `wgsl` macro is of the form:
    /// #[wgsl(crate_path = path::to::crate)]
    ///
    /// Otherwise this is `None`.
    crate_path: Option<syn::Path>,

    /// If true, skip the auto-generated `__validate_wgsl` test.
    ///
    /// Set via `#[wgsl(skip_validation)]`.
    skip_validation: bool,

    /// Concrete type lists for validating template (generic) modules.
    ///
    /// Each occurrence of `validate_with_instantiation_types(T1, T2, ...)`
    /// provides a set of concrete types to instantiate the module with and
    /// validate the resulting WGSL. The types must match the order and
    /// number of type parameters in the module's `instantiate` function.
    ///
    /// For non-template modules, this has no effect (they are always
    /// validated via `WGSL_SOURCE.validate()`).
    validate_instantiations: Vec<Vec<syn::Type>>,

    /// Extension types that implement `WgslExtension`.
    ///
    /// Each path must refer to a type that implements the
    /// `wgsl_rs::WgslExtension` trait. The `modify_ir` method of each
    /// extension will be called on the IR module after construction
    /// but before type instantiation.
    ///
    /// Syntax: `extensions = [path1::Ext1, path2::Ext2]`
    extensions: Vec<syn::Path>,
}

struct InstantiationTypes {
    types: syn::punctuated::Punctuated<syn::Type, syn::Token![,]>,
}

struct ExtensionsList {
    paths: syn::punctuated::Punctuated<syn::Path, syn::Token![,]>,
}

impl syn::parse::Parse for ExtensionsList {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        syn::bracketed!(content in input);
        let paths = content.parse_terminated(syn::Path::parse, syn::Token![,])?;
        Ok(Self { paths })
    }
}

impl syn::parse::Parse for InstantiationTypes {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let types = content.parse_terminated(syn::Type::parse, syn::Token![,])?;
        Ok(Self { types })
    }
}

impl syn::parse::Parse for Attrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::default());
        }

        let mut attrs = Self::default();

        loop {
            let ident: syn::Ident = input.parse()?;
            match ident.to_string().as_str() {
                "crate_path" => {
                    let _eq: syn::Token![=] = input.parse()?;
                    let path: syn::Path = input.parse()?;
                    attrs.crate_path = Some(path);
                }
                "skip_validation" => {
                    attrs.skip_validation = true;
                }
                "validate_with_instantiation_types" => {
                    let types: InstantiationTypes = input.parse()?;
                    attrs
                        .validate_instantiations
                        .push(types.types.into_iter().collect());
                }
                "extensions" => {
                    let _eq: syn::Token![=] = input.parse()?;
                    let ext: ExtensionsList = input.parse()?;
                    attrs.extensions.extend(ext.paths);
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "Unknown attribute '{other}', expected 'crate_path', \
                             'skip_validation', 'validate_with_instantiation_types', or \
                             'extensions'"
                        ),
                    ));
                }
            }

            // Check for comma separator or end
            if input.is_empty() {
                break;
            }
            let _comma: syn::Token![,] = input.parse()?;
            if input.is_empty() {
                break;
            }
        }

        Ok(attrs)
    }
}

impl Attrs {
    fn crate_path(&self) -> syn::Path {
        self.crate_path
            .as_ref()
            .cloned()
            .unwrap_or_else(|| syn::Path {
                leading_colon: None,
                segments: syn::punctuated::Punctuated::from_iter(Some(syn::PathSegment {
                    ident: quote::format_ident!("wgsl_rs"),
                    arguments: syn::PathArguments::None,
                })),
            })
    }
}

#[derive(Debug, Snafu)]
enum WgslGenError {
    RustParse { source: syn::Error },

    WgslParse { source: parse::Error },
}

impl From<syn::Error> for WgslGenError {
    fn from(source: syn::Error) -> Self {
        Self::RustParse { source }
    }
}

impl From<parse::Error> for WgslGenError {
    fn from(source: parse::Error) -> Self {
        Self::WgslParse { source }
    }
}

/// Collect the encoded names of all module-level parameters in the order they
/// appear in `Source::module_type_params` and the generated `instantiate`.
///
/// Three sources contribute:
///
/// 1. Linkage variables declared with `impl Trait` syntax.
/// 2. Binding-owned filterability const parameters. Repeated names share one
///    slot, so a texture and its sampler can use the same flag.
/// 3. Entry-point type and const parameters, encoded positionally.
///
/// Order follows those groups: generic type linkages in declaration order,
/// binding consts in first-occurrence declaration order, then entry-point
/// parameters in their existing order.
fn collect_module_params(wgsl_module: &parse::ItemMod) -> Result<Vec<String>, WgslGenError> {
    use parse::{FnAttrs, Item};
    use std::collections::HashSet;

    let mut params: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    let mut type_slots = HashSet::new();

    // Pass 1: linkage variables declared with `impl Trait`.
    for item in &wgsl_module.content {
        let name = match item {
            Item::Uniform(u) if u.impl_bounds.is_some() => u.name.to_string(),
            Item::Storage(s) if s.impl_bounds.is_some() => s.name.to_string(),
            Item::Workgroup(w) if w.impl_bounds.is_some() => w.name.to_string(),
            _ => continue,
        };
        type_slots.insert(name.clone());
        if seen.insert(name.clone()) {
            params.push(name);
        }
    }

    // Pass 2: binding-owned filterability consts. A name may repeat only
    // because another binding intentionally shares the same const.
    let mut binding_slots = HashSet::new();
    for (name, span) in parse::binding_filterability_params(wgsl_module) {
        if type_slots.contains(&name) {
            return Err(parse::Error::unsupported(
                span,
                format!(
                    "binding filterability parameter `{name}` collides with the `impl Trait` type \
                     linkage slot of the same name; use distinct names"
                ),
            )
            .into());
        }
        binding_slots.insert(name.clone());
        if seen.insert(name.clone()) {
            params.push(name);
        }
    }

    // Pass 3: entry point parameters, encoded positionally.
    for item in &wgsl_module.content {
        let Item::Fn(f) = item else { continue };
        if matches!(f.fn_attrs, FnAttrs::None) {
            continue;
        }
        let fn_name = f.ident.to_string();
        for (i, _tp) in f.type_params.iter().enumerate() {
            // Entry-point slot encoding `{fn}_{i}` is intentionally not
            // run through `wgsl_rs_ir::mangle`; see issue #112.
            let encoded = format!("{fn_name}_{i}");
            if binding_slots.contains(&encoded) {
                return Err(parse::Error::unsupported(
                    f.ident.span(),
                    format!(
                        "binding filterability parameter `{encoded}` collides with an entry-point \
                         parameter; rename the binding parameter"
                    ),
                )
                .into());
            }
            if seen.insert(encoded.clone()) {
                params.push(encoded);
            }
        }
        // Entry-point const params use a separate `{fn}_c{i}` namespace.
        for (i, _cp) in f.const_params.iter().enumerate() {
            let encoded = format!("{fn_name}_c{i}");
            if binding_slots.contains(&encoded) {
                return Err(parse::Error::unsupported(
                    f.ident.span(),
                    format!(
                        "binding filterability parameter `{encoded}` collides with an entry-point \
                         parameter; rename the binding parameter"
                    ),
                )
                .into());
            }
            if seen.insert(encoded.clone()) {
                params.push(encoded);
            }
        }
    }

    Ok(params)
}

/// Rewrite generic linkage declarations in the input Rust module to CPU-side
/// types that compile without module generics.
///
/// `impl Trait` uniform/storage/workgroup declarations become type-erased
/// statics. Texture/sampler declarations with binding-owned filterability
/// parameters instead retain their macro shape but substitute `true` for the
/// unresolved const. The parsed IR is not modified.
fn rewrite_generic_linkages(
    input_mod: &mut syn::ItemMod,
    wgsl_module: &parse::ItemMod,
    crate_path: &syn::Path,
) -> Result<(), WgslGenError> {
    use parse::Item;

    // Collect the names of generic linkage items by scanning the parsed
    // `wgsl_module`. A linkage variable is "generic" if the user declared
    // it with `impl Trait` syntax (e.g. `FRAME: impl Convert<f32>`); we
    // detect this via the `impl_bounds` field on each linkage item.
    let mut generic_uniforms: std::collections::BTreeMap<String, (syn::LitInt, syn::LitInt)> =
        Default::default();
    let mut generic_storages: std::collections::BTreeMap<
        String,
        (syn::LitInt, syn::LitInt, parse::StorageAccess),
    > = Default::default();
    let mut generic_workgroups: std::collections::BTreeSet<String> = Default::default();
    type GenericFilterability = (syn::LitInt, syn::LitInt, syn::Type);
    let mut generic_textures: std::collections::BTreeMap<String, GenericFilterability> =
        Default::default();
    let mut generic_samplers: std::collections::BTreeMap<String, GenericFilterability> =
        Default::default();

    for item in &wgsl_module.content {
        match item {
            Item::Uniform(u) if u.impl_bounds.is_some() => {
                generic_uniforms.insert(u.name.to_string(), (u.group.clone(), u.binding.clone()));
            }
            Item::Storage(s) if s.impl_bounds.is_some() => {
                generic_storages.insert(
                    s.name.to_string(),
                    (s.group.clone(), s.binding.clone(), s.access),
                );
            }
            Item::Workgroup(w) if w.impl_bounds.is_some() => {
                generic_workgroups.insert(w.name.to_string());
            }
            Item::Texture(t)
                if matches!(
                    &t.ty,
                    parse::Type::Texture {
                        filterable: wgsl_rs_ir::Filterability::Param(_),
                        ..
                    }
                ) =>
            {
                generic_textures.insert(
                    t.name.to_string(),
                    (
                        t.group.clone(),
                        t.binding.clone(),
                        with_default_filterability(&t.rust_ty),
                    ),
                );
            }
            Item::Sampler(s)
                if matches!(
                    &s.ty,
                    parse::Type::Sampler {
                        filterable: wgsl_rs_ir::Filterability::Param(_),
                        ..
                    }
                ) =>
            {
                generic_samplers.insert(
                    s.name.to_string(),
                    (
                        s.group.clone(),
                        s.binding.clone(),
                        with_default_filterability(&s.rust_ty),
                    ),
                );
            }
            _ => {}
        }
    }

    if generic_uniforms.is_empty()
        && generic_storages.is_empty()
        && generic_workgroups.is_empty()
        && generic_textures.is_empty()
        && generic_samplers.is_empty()
    {
        return Ok(());
    }

    let Some((_, items)) = input_mod.content.as_mut() else {
        return Ok(());
    };

    for item in items.iter_mut() {
        let syn::Item::Macro(item_macro) = item else {
            continue;
        };
        let Some(macro_ident) = item_macro.mac.path.get_ident() else {
            continue;
        };
        let macro_name = macro_ident.to_string();

        // Find the variable name that this macro declares (the bare ident
        // after the last `,` and before the `:`).
        let var_name = match macro_name.as_str() {
            "uniform" | "storage" | "workgroup" | "texture" | "sampler" => {
                extract_linkage_var_name(item_macro.mac.tokens.clone())
            }
            _ => None,
        };
        let Some(var_name) = var_name else {
            continue;
        };

        // Const-generic filterability is not in scope for the CPU static.
        // Rewrite only the Rust-side macro input to the default `true`; the
        // already-parsed IR retains the parameter for template substitution.
        if matches!(macro_name.as_str(), "texture" | "sampler") {
            let generic = if macro_name == "texture" {
                generic_textures.get(&var_name)
            } else {
                generic_samplers.get(&var_name)
            };
            if let Some((group, binding, default_ty)) = generic {
                let name = quote::format_ident!("{}", var_name);
                item_macro.mac.tokens =
                    quote!(group(#group), binding(#binding), #name: #default_ty);
            }
            continue;
        }

        let replacement: Option<proc_macro2::TokenStream> = match macro_name.as_str() {
            "uniform" => generic_uniforms.get(&var_name).map(|(group, binding)| {
                let name = quote::format_ident!("{}", var_name);
                quote! {
                    pub static #name: #crate_path::std::Uniform =
                        #crate_path::std::Uniform::new(#group, #binding);
                }
            }),
            "storage" => generic_storages
                .get(&var_name)
                .map(|(group, binding, access)| {
                    let name = quote::format_ident!("{}", var_name);
                    let access_mode = match access {
                        parse::StorageAccess::Read => quote!(#crate_path::std::Read),
                        parse::StorageAccess::ReadWrite => quote!(#crate_path::std::ReadWrite),
                    };
                    quote! {
                        pub static #name: #crate_path::std::Storage<
                            #crate_path::std::WgslTypeVariable,
                            #access_mode,
                        > = #crate_path::std::Storage::new(#group, #binding);
                    }
                }),
            "workgroup" => generic_workgroups.get(&var_name).map(|_| {
                let name = quote::format_ident!("{}", var_name);
                quote! {
                    pub static #name: #crate_path::std::Workgroup =
                        #crate_path::std::Workgroup::new();
                }
            }),
            _ => None,
        };

        if let Some(tokens) = replacement {
            *item = syn::parse2::<syn::Item>(tokens)?;
        }
    }

    Ok(())
}

/// Replace a binding-owned filterability parameter with `true` in the Rust
/// type used for its CPU static. The WGSL-facing parse tree is not modified.
fn with_default_filterability(ty: &syn::Type) -> syn::Type {
    let mut default_ty = ty.clone();
    let syn::Type::Path(type_path) = &mut default_ty else {
        unreachable!("filterability parsing requires a path type")
    };
    let segment = type_path
        .path
        .segments
        .iter_mut()
        .next()
        .expect("filterability types have one segment");
    let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments else {
        unreachable!("a filterability parameter has angle-bracketed arguments")
    };
    let name = segment.ident.to_string();
    let index = if name == "Sampler" { 0 } else { 1 };
    let arg = args
        .args
        .iter_mut()
        .nth(index)
        .expect("the unresolved filterability argument is present");
    *arg = syn::parse_quote!(true);
    default_ty
}

/// Extracts the variable name from a binding macro's token stream. Returns
/// `None` if the tokens do not contain the expected `NAME:` separator.
fn extract_linkage_var_name(tokens: proc_macro2::TokenStream) -> Option<String> {
    use proc_macro2::TokenTree;
    let mut last_ident: Option<String> = None;
    for tt in tokens {
        if let TokenTree::Punct(p) = &tt
            && p.as_char() == ':'
        {
            return last_ident;
        }
        if let TokenTree::Ident(id) = &tt {
            last_ident = Some(id.to_string());
        }
    }
    None
}

/// Construct a token stream that names the IR crate via the consuming
/// crate's wgsl-rs root path (e.g. `wgsl_rs::ir`).
fn ir_path(crate_path: &syn::Path) -> proc_macro2::TokenStream {
    quote! { #crate_path::ir }
}

#[allow(clippy::too_many_arguments)]
fn gen_wgsl_module(
    name: &syn::Ident,
    crate_path: &syn::Path,
    imports: &[proc_macro2::TokenStream],
    reexports: &[proc_macro2::TokenStream],
    named_exports: &[(String, proc_macro2::TokenStream)],
    wgsl_module: &parse::ItemMod,
    mono_result: &monomorphize::MonoResult,
    module_type_params: &[String],
    extensions: &[syn::Path],
) -> Result<proc_macro2::TokenStream, WgslGenError> {
    fn is_wgsl_std_import(crate_path: &syn::Path, path: &syn::Path) -> bool {
        let wgsl_std = {
            let mut std = crate_path.clone();
            if !std.segments.empty_or_trailing() {
                std.segments.push_punct(syn::token::PathSep::default());
            }
            std.segments.push_value(syn::PathSegment {
                ident: quote::format_ident!("std"),
                arguments: syn::PathArguments::None,
            });
            std
        };
        wgsl_std.into_token_stream().to_string() == path.into_token_stream().to_string()
    }

    let ir_p = ir_path(crate_path);
    let module_name_ident = &wgsl_module.ident;

    // Convert the parse module's items to IR items once; both emission
    // paths consume the same owned IR.
    let ir_items = ir_convert::items_from_parse(&wgsl_module.content).map_err(|e| {
        WgslGenError::WgslParse {
            source: e.into_parse(),
        }
    })?;

    let module_ctor_ident = quote::format_ident!("__wgsl_module_ctor");

    // The canonical `static_ir` rendering of the module: the emitted
    // static data under `fast-compile`, and the content-hash input for
    // the deterministic module id in both emission modes. The pieces
    // are passed directly instead of assembling an owned `ir::Module`:
    // its `name` field is `&'static str`, and materializing one here
    // would mean leaking the name string in the proc-macro process.
    let module_static_data = ir_emit_static::emit_static_module(
        &ir_p,
        &module_name_ident.to_string(),
        &ir_items,
        &wgsl_module.attrs,
    );
    let module_descriptor = module_static_data.to_string();

    // The module IR constructor and, under `fast-compile`, the static
    // data it reads. `fast-compile` emits the IR as a static literal of
    // the borrowed `static_ir` mirror plus a slim convert-at-runtime
    // shim (wgsl-rs#221); the legacy path emits a constructor function
    // that builds the IR expression-by-expression, which for a large
    // shader is a ~12,000-line function LLVM spends ~14s on per
    // incremental rebuild at opt-level=1.
    let (module_ctor, module_statics): (proc_macro2::TokenStream, proc_macro2::TokenStream) = if cfg!(
        feature = "fast-compile"
    ) {
        let ir_module_static_ident = quote::format_ident!("__WGSL_IR_MODULE");
        let ctor_body = if extensions.is_empty() {
            quote! { #ir_p::Module::from(#ir_module_static_ident) }
        } else {
            let extension_calls: Vec<proc_macro2::TokenStream> = extensions
                .iter()
                .map(|ext_path| {
                    quote! {
                        <#ext_path as #crate_path::WgslExtension>::modify_ir(&mut m);
                    }
                })
                .collect();
            quote! {
                let mut m = #ir_p::Module::from(#ir_module_static_ident);
                #(#extension_calls)*
                m
            }
        };
        (
            quote! {
                fn #module_ctor_ident() -> #ir_p::Module { #ctor_body }
            },
            quote! {
                static #ir_module_static_ident: &#ir_p::static_ir::StaticModule = #module_static_data;
            },
        )
    } else {
        let item_exprs: Vec<proc_macro2::TokenStream> = ir_items
            .iter()
            .map(|i| ir_emit::emit_item(&ir_p, i))
            .collect();
        let module_attrs = ir_emit::emit_attrs(&ir_p, &wgsl_module.attrs);
        let module_struct = quote! {
            #ir_p::Module {
                name: stringify!(#module_name_ident),
                items: ::std::vec![#(#item_exprs),*],
                #module_attrs,
            }
        };
        let ctor_body = if extensions.is_empty() {
            module_struct
        } else {
            let extension_calls: Vec<proc_macro2::TokenStream> = extensions
                .iter()
                .map(|ext_path| {
                    quote! {
                        <#ext_path as #crate_path::WgslExtension>::modify_ir(&mut m);
                    }
                })
                .collect();
            quote! {
                let mut m = #module_struct;
                #(#extension_calls)*
                m
            }
        };
        (
            quote! {
                fn #module_ctor_ident() -> #ir_p::Module { #ctor_body }
            },
            quote! {},
        )
    };

    // Generate template entries for generic functions / structs defined in
    // this module. The IR is built lazily by an emitted constructor
    // function.
    let mut template_constructors: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut template_statics: Vec<proc_macro2::TokenStream> = Vec::new();
    // Canonical content descriptors of each template, for the
    // deterministic module id.
    let mut template_descriptors: Vec<String> = Vec::new();
    let template_entries: Vec<proc_macro2::TokenStream> = mono_result
        .template_macros
        .iter()
        .enumerate()
        .map(|(idx, tmpl)| {
            let name_lit = &tmpl.fn_name;
            let params: Vec<&str> = tmpl.type_param_names.iter().map(|s| s.as_str()).collect();
            let const_params: Vec<&str> =
                tmpl.const_param_names.iter().map(|s| s.as_str()).collect();

            // Convert the template's items to IR.
            let ir_items =
                ir_convert::items_from_parse(&tmpl.items).map_err(|e| WgslGenError::WgslParse {
                    source: e.into_parse(),
                })?;

            // Canonical `static_ir` rendering of the template: emitted
            // data under `fast-compile`, content-hash input in both
            // modes.
            let tmpl_static_data = ir_emit_static::emit_static_items(&ir_p, &ir_items);
            template_descriptors.push(format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                tmpl.fn_name,
                tmpl.type_param_names.join("\u{1f}"),
                tmpl.const_param_names.join("\u{1f}"),
                tmpl.dependencies
                    .iter()
                    .map(|dep| format!(
                        "{}:{}",
                        dep.callee,
                        dep.type_param_mapping
                            .iter()
                            .map(|mapping| mapping.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    ))
                    .collect::<Vec<_>>()
                    .join("\u{1f}"),
                tmpl_static_data,
            ));

            let ctor_ident = quote::format_ident!("__wgsl_template_{}_ctor", idx);
            if cfg!(feature = "fast-compile") {
                // Static IR data plus a slim conversion shim; the legacy
                // path below builds the `vec![...]` constructor body.
                let static_ident = quote::format_ident!("__wgsl_template_{}", idx);
                template_statics.push(quote! {
                    static #static_ident: &[#ir_p::static_ir::StaticItem] = #tmpl_static_data;
                });
                template_constructors.push(quote! {
                    fn #ctor_ident() -> ::std::vec::Vec<#ir_p::Item> {
                        #static_ident
                            .iter()
                            .map(|item| #ir_p::Item::from(item))
                            .collect()
                    }
                });
            } else {
                let body_items: Vec<proc_macro2::TokenStream> = ir_items
                    .iter()
                    .map(|i| ir_emit::emit_item(&ir_p, i))
                    .collect();
                template_constructors.push(quote! {
                    fn #ctor_ident() -> ::std::vec::Vec<#ir_p::Item> {
                        ::std::vec![#(#body_items),*]
                    }
                });
            }

            let dep_entries: Vec<proc_macro2::TokenStream> = tmpl
                .dependencies
                .iter()
                .map(|dep| {
                    let callee = &dep.callee;
                    let mapping = &dep.type_param_mapping;
                    quote! {
                        #crate_path::TemplateDependency {
                            callee: #callee,
                            type_param_mapping: &[#(#mapping),*],
                        }
                    }
                })
                .collect();

            Ok::<_, WgslGenError>(quote! {
                #crate_path::GenericTemplate {
                    name: #name_lit,
                    type_params: &[#(#params),*],
                    const_params: &[#(#const_params),*],
                    ir_constructor: #ctor_ident,
                    dependencies: &[#(#dep_entries),*],
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Generate instantiation entries for cross-module generic calls. Each
    // entry's `type_args_constructor` produces the concrete IR types at
    // runtime; the mangled identifier-safe names are stored as static
    // string slices for deduplication.
    let mut inst_constructors: Vec<proc_macro2::TokenStream> = Vec::new();
    // Canonical content descriptors of each cross-module instantiation,
    // for the deterministic module id.
    let mut inst_descriptors: Vec<String> = Vec::new();
    let inst_entries: Vec<proc_macro2::TokenStream> = mono_result
        .cross_module_instantiations
        .iter()
        .enumerate()
        .map(|(idx, inst)| {
            let import_paths = &inst.import_paths;
            let tmpl_name_lit = &inst.fn_name;
            let mangled_args: Vec<&str> =
                inst.mangled_type_args.iter().map(|s| s.as_str()).collect();
            let mangled_const_args: Vec<&str> =
                inst.mangled_const_args.iter().map(|s| s.as_str()).collect();

            inst_descriptors.push(format!(
                "{}\u{1f}{}\u{1f}{}\u{1f}{}",
                inst.import_paths
                    .iter()
                    .map(|path| path.to_token_stream().to_string())
                    .collect::<Vec<_>>()
                    .join("\u{1f}"),
                inst.fn_name,
                inst.mangled_type_args.join("\u{1f}"),
                inst.mangled_const_args.join("\u{1f}"),
            ));

            // Build IR types for each type argument.
            let ir_args: Vec<wgsl_rs_ir::Type> = inst
                .type_args
                .iter()
                .map(ir_convert::ty_from_parse)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| WgslGenError::WgslParse {
                    source: e.into_parse(),
                })?;
            let arg_exprs: Vec<proc_macro2::TokenStream> = ir_args
                .iter()
                .map(|t| ir_emit::emit_type(&ir_p, t))
                .collect();

            // Build the const-args constructor. Const values are known
            // at macro time (parsed from literals in turbofish: integer,
            // possibly negative, or bool), so the constructor is a simple
            // `vec![ConstValue::Int(4), ...]` literal.
            let const_vals: Vec<proc_macro2::TokenStream> = inst
                .const_args
                .iter()
                .map(|v| ir_emit::emit_const_value(&ir_p, *v))
                .collect();

            let ctor_ident = quote::format_ident!("__wgsl_inst_{}_ctor", idx);
            inst_constructors.push(quote! {
                fn #ctor_ident() -> ::std::vec::Vec<#ir_p::Type> {
                    ::std::vec![#(#arg_exprs),*]
                }
            });

            let const_ctor_ident = quote::format_ident!("__wgsl_inst_const_{}_ctor", idx);
            inst_constructors.push(quote! {
                fn #const_ctor_ident() -> ::std::vec::Vec<#ir_p::ConstValue> {
                    ::std::vec![#(#const_vals),*]
                }
            });

            let modules: Vec<proc_macro2::TokenStream> = import_paths
                .iter()
                .filter(|path| !is_wgsl_std_import(crate_path, path))
                .map(|path| quote! { &#path::WGSL_SOURCE })
                .collect();

            Ok::<_, WgslGenError>(quote! {
                #crate_path::TemplateInstantiation {
                    modules: &[#(#modules),*],
                    template_name: #tmpl_name_lit,
                    type_args_constructor: #ctor_ident,
                    mangled_type_args: &[#(#mangled_args),*],
                    const_args_constructor: #const_ctor_ident,
                    mangled_const_args: &[#(#mangled_const_args),*],
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let module_type_params_lits: Vec<&str> =
        module_type_params.iter().map(|s| s.as_str()).collect();

    // Deterministic module id (wgsl-rs#221): a stable hash of the
    // module's full content descriptor — IR rendering, templates,
    // instantiations, imports, module type params, and the
    // `extensions = [...]` list — mixed with the consuming unit's
    // identity. Replaces the per-process counter, which shifted every
    // later module's id whenever sibling `#[wgsl]` mods were added,
    // removed, or reordered, invalidating builds that did not change.
    let mut source_descriptor = module_descriptor;
    for part in template_descriptors.iter().chain(inst_descriptors.iter()) {
        source_descriptor.push('\0');
        source_descriptor.push_str(part);
    }
    for import in imports {
        source_descriptor.push('\0');
        source_descriptor.push_str(&import.to_string());
    }
    // Glob re-export edges ARE part of the descriptor: two modules
    // identical except for a glob's visibility (`use defs::*` vs
    // `pub use defs::*`) must not hash to one id — the id-keyed
    // visited sets of the resolution and harvest walks would collapse
    // the pair, and if the private variant is visited first the
    // public variant's edges are silently dropped, making downstream
    // resolution depend on traversal order (wgsl-rs#114 review,
    // superseding the 2026-10-31 #245 exclusion). This also restores
    // symmetry with named re-export entries below.
    for reexport in reexports {
        source_descriptor.push('\0');
        source_descriptor.push_str(&reexport.to_string());
    }
    // Named re-export alias-table entries, for the same reason and
    // one sharper: two modules importing the same source can
    // re-export different names (`pub use defs::Id;` vs
    // `pub use defs::Hidden;`) and otherwise hash to one id — one
    // alias would be dropped outright.
    for (name, source) in named_exports {
        source_descriptor.push('\0');
        source_descriptor.push_str(name);
        source_descriptor.push('\0');
        source_descriptor.push_str(&source.to_string());
    }

    // The named re-export alias-table entries, interpolated below.
    let named_export_names: Vec<&str> = named_exports
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    let named_export_sources: Vec<&proc_macro2::TokenStream> =
        named_exports.iter().map(|(_, source)| source).collect();
    for type_param in module_type_params {
        source_descriptor.push('\0');
        source_descriptor.push_str(type_param);
    }
    // The extension list changes what `modify_ir` produces from
    // otherwise identical IR, so it must be part of the id: without it,
    // two same-named mods differing only in their extension lists
    // would share an id, and assembly dedup would silently keep one
    // and drop the other's extensions — where previously both sources
    // were kept and the duplicate WGSL definitions failed loudly at
    // validation.
    for extension in extensions {
        source_descriptor.push('\0');
        source_descriptor.push_str(&extension.to_token_stream().to_string());
    }
    let module_id = module_id::deterministic_module_id(&source_descriptor);

    // Concrete modules get an `instantiate()` constructor with the same
    // shape as template modules' `instantiate::<T>()`: the fully
    // assembled IR (imports depth-first + own items + cross-source
    // template instantiations). Consumers must build pipelines from
    // this, never from `(WGSL_SOURCE.ir_constructor)()`, which returns
    // own items only and silently drops imports (wgsl-rs#195).
    let concrete_instantiate = if module_type_params.is_empty() {
        let expect_msg = format!("failed to assemble IR for wgsl module '{name}'");
        quote! {
            /// Returns this module's fully assembled IR: imported
            /// sources first (depth-first, deduplicated by id), then
            /// this module's own items, then cross-source template
            /// instantiations.
            ///
            /// Build pipelines and shader modules from this — not from
            /// `(WGSL_SOURCE.ir_constructor)()`, which returns this
            /// module's own items only and silently drops imports.
            pub fn instantiate() -> #ir_p::Module {
                WGSL_SOURCE.assemble_ir().expect(#expect_msg)
            }
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #module_statics

        #module_ctor

        #(#template_statics)*

        #(#template_constructors)*

        #(#inst_constructors)*

        pub static WGSL_SOURCE: #crate_path::Source = #crate_path::Source {
            id: #module_id,
            name: stringify!(#name),
            imports: &[
                #(&#imports),*
            ],
            reexports: &[
                #(&#reexports),*
            ],
            named_exports: &[
                #(#crate_path::NamedExport { name: #named_export_names, source: &#named_export_sources }),*
            ],
            ir_constructor: #module_ctor_ident,
            templates: &[
                #(#template_entries),*
            ],
            instantiations: &[
                #(#inst_entries),*
            ],
            module_type_params: &[
                #(#module_type_params_lits),*
            ],
        };

        #concrete_instantiate
    })
}

/// Generates a `#[test]` function that validates the assembled WGSL source.
///
/// Emitted for non-template `#[wgsl]` modules (unless `skip_validation` is
/// set). The validation is deferred to `cargo test` time, where it calls
/// `WGSL_SOURCE.validate()`.
///
/// The test is only generated when the `validation` feature is enabled
/// (the default). Without this feature, `Module::validate()` is unavailable,
/// so the test would not compile.
fn gen_validation_test(module_ident: &syn::Ident) -> proc_macro2::TokenStream {
    if !cfg!(feature = "validation") {
        return quote! {};
    }
    let error_msg = format!("WGSL validation failed for module '{module_ident}'");
    quote! {
        #[cfg(test)]
        #[test]
        fn __validate_wgsl() {
            WGSL_SOURCE.validate().expect(#error_msg);
        }
    }
}

/// Generates `#[test]` functions that validate instantiated WGSL source for
/// template (generic) modules.
///
/// Each occurrence of `validate_with_instantiation_types(T1, T2, ...)`
/// produces a test that calls `instantiate::<T1, T2, ...>()`, renders the
/// IR to WGSL through the canonical deshadowing entry (`Module::wgsl_source`,
/// matching the concrete-module route), and validates the result with naga.
///
/// The tests are only generated when the `validation` feature is enabled.
fn gen_instantiated_validation_tests(
    module_ident: &syn::Ident,
    crate_path: &syn::Path,
    instantiations: &[Vec<syn::Type>],
) -> proc_macro2::TokenStream {
    if !cfg!(feature = "validation") {
        return quote! {};
    }
    let validate_fn = quote! { #crate_path::validate_wgsl_source };
    let tests: Vec<proc_macro2::TokenStream> = instantiations
        .iter()
        .enumerate()
        .map(|(i, types)| {
            let test_name = format_ident!("__validate_wgsl_instantiated_{i}");
            let error_msg =
                format!("WGSL validation failed for instantiated module '{module_ident}'");
            quote! {
                #[cfg(test)]
                #[test]
                fn #test_name() {
                    let __m = instantiate::<#(#types),*>();
                    // Render through the canonical deshadowing entry, not
                    // the raw renderer: the raw render skips deshadow, so
                    // same-scope shadowing (legal Rust, invalid raw WGSL)
                    // would fail validation here while rendering correctly
                    // at runtime.
                    let __src = __m.wgsl_source();
                    #validate_fn(&__src).expect(#error_msg);
                }
            }
        })
        .collect();
    quote! { #(#tests)* }
}

fn go_wgsl(attr: TokenStream, mut input_mod: syn::ItemMod) -> Result<TokenStream, WgslGenError> {
    // Parse Attrs from attr TokenStream
    let attrs: Attrs = syn::parse(attr)?;
    let crate_path = attrs.crate_path();

    let mut wgsl_module = parse::ItemMod::try_from(&input_mod)?;
    // `WGSL_SOURCE` and `instantiate` are the macro-emitted items; a
    // user item with either name collides with them and would fail
    // with a duplicate-definition error deep in the consumer crate.
    // Fail here with a targeted diagnostic instead.
    if let Some((span, kind)) = wgsl_module.reserved_name_conflict() {
        return Err(WgslGenError::WgslParse {
            source: parse::Error::unsupported(
                span,
                format!(
                    "a {kind} named `instantiate` or `WGSL_SOURCE` collides with the items the \
                     `#[wgsl]` macro emits into this module: the `WGSL_SOURCE` static and the \
                     `instantiate` IR constructor. Rename the item"
                ),
            ),
        });
    }
    let mut mono_result = monomorphize::run(&mut wgsl_module)?;
    // Resolve associated type projections (e.g. `Self::Array`, `T::Array`)
    // to concrete types using the impl-block index.
    monomorphize::resolve_assoc_types(&mut wgsl_module)?;
    // Lower vector `==`/`!=` to `all(...)` wraps (#164). Runs after mono so
    // monomorphized functions have concrete param types.
    vector_cmp::rewrite(&mut wgsl_module.content);
    // Validate `load!` targets against the module's declarations
    // (textures/samplers, atomics, runtime arrays, unknown idents). The
    // declaration table comes from the module body; monomorphized
    // template bodies carry their own copies of generic functions and
    // are checked against the same table.
    let load_table = load_validation::ModuleVarTable::from_items(&wgsl_module.content, &crate_path);
    load_validation::validate_items(&load_table, &mut wgsl_module.content)?;
    for tmpl in mono_result.template_macros.iter_mut() {
        load_validation::validate_items(&load_table, &mut tmpl.items)?;
    }
    // Error on derefs of locals that provably hold module variable values
    // (e.g. `let u = get!(U); *u`) — they render invalid WGSL (wgsl-rs#153).
    // Derefs of linkage accessors are elided at IR conversion; real
    // pointers (`&x`, `ptr!`) pass through unchanged.
    deref_validation::validate_items(&mut wgsl_module.content)?;
    for tmpl in mono_result.template_macros.iter_mut() {
        deref_validation::validate_items(&mut tmpl.items)?;
    }
    let parse::ImportEdges {
        imports,
        reexports,
        named_exports,
    } = wgsl_module.imports_and_reexports(&crate_path);

    // Rewrite generic linkage declarations to CPU-compilable types. The
    // parsed WGSL-facing module retains all placeholders for instantiation.
    rewrite_generic_linkages(&mut input_mod, &wgsl_module, &crate_path)?;

    // Compute module-level parameters from generic type linkages,
    // binding-owned filterability consts, and entry points. This same order
    // drives `Source::module_type_params` and generated `instantiate`.
    let module_type_params: Vec<String> = collect_module_params(&wgsl_module)?;

    let module_fragment = gen_wgsl_module(
        &wgsl_module.ident,
        &crate_path,
        &imports,
        &reexports,
        &named_exports,
        &wgsl_module,
        &mono_result,
        &module_type_params,
        &attrs.extensions,
    )?;

    // Generate validation tests.
    //
    // Non-template modules get a `__validate_wgsl` test that calls
    // `WGSL_SOURCE.validate()`. Template (generic) modules cannot be
    // validated standalone — their placeholders aren't valid WGSL. Instead,
    // each `validate_with_instantiation_types(T1, T2, ...)` attribute
    // produces a test that instantiates with those concrete types, renders
    // the IR, and validates with naga.
    let is_template = !module_type_params.is_empty();
    let validation_test = if !attrs.skip_validation && !is_template {
        gen_validation_test(&input_mod.ident)
    } else {
        quote! {}
    };
    let instantiated_validation_tests =
        if !attrs.skip_validation && is_template && !attrs.validate_instantiations.is_empty() {
            gen_instantiated_validation_tests(
                &input_mod.ident,
                &crate_path,
                &attrs.validate_instantiations,
            )
        } else {
            quote! {}
        };

    // Warn if a template module has no validate_with_instantiation_types
    // and no skip_validation. On stable this is a compile error; on nightly
    // it's a diagnostic warning. Either way, the user should either provide
    // concrete types for validation or explicitly opt out with skip_validation.
    if is_template && !attrs.skip_validation && attrs.validate_instantiations.is_empty() {
        let warning = parse::Warning {
            name: parse::WarningName::MissingValidationTypes,
            spans: vec![input_mod.ident.span()],
        };
        if cfg!(nightly) {
            parse::emit_warning(&warning);
        } else {
            return Err(WgslGenError::WgslParse {
                source: parse::Error::SuppressableWarning { warning },
            });
        }
    }

    // Every module gets an `instantiate` constructor alongside
    // `WGSL_SOURCE`. Template modules (those with module-level type
    // parameters) get the generic form here, which uses
    // `wgsl_rs::linkage::Type<Is = ...>` constraints to enforce at
    // compile time that every linkage variable's concrete type is
    // consistent across all entry points that use it. Concrete modules
    // get the zero-argument form from `gen_wgsl_module` itself.
    let builder_fragment = if module_type_params.is_empty() {
        quote! {}
    } else {
        builder::gen_builder(&crate_path, &wgsl_module)
    };

    if let Some((_, content)) = input_mod.content.as_mut() {
        // The module fragment now contains multiple items (constructor
        // fns + the WGSL_SOURCE static), so parse it as a list of items
        // by wrapping in a synthetic `mod __wgsl_emit { ... }` and
        // splicing its contents.
        let wrapper_tokens = quote! {
            mod __wgsl_emit_wrapper {
                #module_fragment
                #builder_fragment
            }
        };
        let wrapper_mod: syn::ItemMod = syn::parse2(wrapper_tokens)?;
        if let Some((_, wrapper_content)) = wrapper_mod.content {
            content.extend(wrapper_content);
        }

        // Add validation test function(s) if generated
        if !validation_test.is_empty() {
            let test_item: syn::Item = syn::parse2(validation_test)?;
            content.push(test_item);
        }
        if !instantiated_validation_tests.is_empty() {
            let wrapper = quote! {
                mod __inst_validate_wrapper {
                    #instantiated_validation_tests
                }
            };
            let wrapper_mod: syn::ItemMod = syn::parse2(wrapper)?;
            if let Some((_, wrapper_content)) = wrapper_mod.content {
                content.extend(wrapper_content);
            }
        }
    }

    // NOTE: Compile-time WGSL validation has been removed in favor of
    // runtime validation. All non-template modules get an auto-generated
    // `__validate_wgsl` test. Template (generic) modules get an
    // auto-generated test for each `validate_with_instantiation_types(...)`
    // attribute, which instantiates the module with the provided concrete
    // types and validates the resulting WGSL source with naga.
    let _ = &attrs;

    // Emit compile-time `const` checks verifying every `Stmt::Macro` name
    // in the module is claimed by at least one listed extension's `MACROS`
    // const. This catches typos and missing extensions at compile time
    // (E0080), not at runtime.
    let macro_checks = gen_macro_claim_checks(&wgsl_module, &attrs.extensions, &crate_path);
    if let Some((_, content)) = input_mod.content.as_mut()
        && let Ok(item) = syn::parse2::<syn::Item>(macro_checks)
    {
        content.push(item);
    }

    // Strip #[wgsl_allow] attributes before emitting Rust code.
    // These attributes are used during parsing but must be removed from the
    // output because statement-level attributes require the unstable
    // stmt_expr_attributes feature.
    StripWgslAllowAttrs.visit_item_mod_mut(&mut input_mod);

    // Strip inter-stage IO attributes (#[builtin], #[location], etc.) from
    // struct fields and function parameters. They are read during WGSL parsing
    // but aren't valid Rust attributes in the emitted module.
    StripIoAttrs.visit_item_mod_mut(&mut input_mod);

    Ok(input_mod.into_token_stream().into())
}

/// Collect all unique `Stmt::Macro` names from function bodies in the
/// parse tree, then emit a `const _: () = { ... }` block that verifies
/// each name is claimed by at least one listed extension's `MACROS` const.
///
/// If there are no `Stmt::Macro` invocations or no extensions listed,
/// this returns an empty token stream (no checks needed).
fn gen_macro_claim_checks(
    wgsl_module: &parse::ItemMod,
    extensions: &[syn::Path],
    crate_path: &syn::Path,
) -> proc_macro2::TokenStream {
    if extensions.is_empty() {
        return quote! {};
    }

    // Collect unique Stmt::Macro names from all function bodies.
    let mut macro_names: Vec<String> = Vec::new();
    for item in &wgsl_module.content {
        if let parse::Item::Fn(f) = item {
            collect_macro_names(&f.block, &mut macro_names);
        }
    }

    if macro_names.is_empty() {
        return quote! {};
    }

    // Deduplicate (preserve order for deterministic output).
    macro_names.dedup();
    let unique_names: Vec<&str> = macro_names.iter().map(|s| s.as_str()).collect();

    // Build the extension MACROS array references.
    let ext_macros_refs: Vec<proc_macro2::TokenStream> = extensions
        .iter()
        .map(|ext| {
            quote! { <#ext as #crate_path::WgslExtension>::MACROS }
        })
        .collect();

    // Build one assert per unique macro name.
    let checks: Vec<proc_macro2::TokenStream> = unique_names
        .iter()
        .map(|name| {
            let panic_msg = format!(
                "no extension claims the `{name}!` statement macro; ensure the extension that \
                 handles it is listed in `extensions = [...]`"
            );
            quote! {
                if !#crate_path::any_ext_claims(exts, #name) {
                    panic!(#panic_msg);
                }
            }
        })
        .collect();

    quote! {
        const _: () = {
            let exts: &[&[&str]] = &[
                #(#ext_macros_refs),*
            ];
            #(#checks)*
        };
    }
}

/// Recursively collect `Stmt::Macro` names from a block's statements.
fn collect_macro_names(block: &parse::Block, names: &mut Vec<String>) {
    for stmt in &block.stmt {
        collect_macro_names_from_stmt(stmt, names);
    }
}

fn collect_macro_names_from_stmt(stmt: &parse::Stmt, names: &mut Vec<String>) {
    match stmt {
        parse::Stmt::Macro { name, .. } => {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        parse::Stmt::While { body, .. } | parse::Stmt::Loop { body, .. } => {
            collect_macro_names(body, names);
        }
        parse::Stmt::If(i) => {
            collect_macro_names(&i.then_block, names);
            if let Some(parse::ElseBranch { body, .. }) = &i.else_branch {
                match body {
                    parse::ElseBody::Block(b) => collect_macro_names(b, names),
                    parse::ElseBody::If(inner) => {
                        collect_macro_names_from_stmt(&parse::Stmt::If(inner.clone()), names);
                    }
                }
            }
        }
        parse::Stmt::For(f) => {
            collect_macro_names(&f.body, names);
        }
        parse::Stmt::Switch(sw) => {
            for arm in &sw.arms {
                collect_macro_names(&arm.body, names);
            }
        }
        parse::Stmt::Block(b) => {
            collect_macro_names(b, names);
        }
        _ => {}
    }
}

/// Transpiles a Rust module to WGSL.
///
/// Apply `#[wgsl]` to a `mod` item to generate a `WGSL_SOURCE` static
/// alongside any user-defined fns. The Rust code in the module runs on the
/// CPU, while the WGSL produced by `WGSL_SOURCE.wgsl_source()` (or
/// `WGSL_SOURCE.instantiate(...)` for generic sources) runs on the GPU.
///
/// # Module Options
///
/// | Syntax | Description |
/// |--------|-------------|
/// | `#[wgsl]` | Transpile the module with default settings. |
/// | `#[wgsl(skip_validation)]` | Skip the auto-generated `__validate_wgsl` test. |
/// | `#[wgsl(crate_path = path::to::crate)]` | Override the path to the `wgsl_rs` crate. |
/// | `#[wgsl(validate_with_instantiation_types(T1, T2, ...))]` | For template (generic) modules: validate WGSL output after instantiating with the given concrete types. Can be repeated. |
/// | `#[wgsl(extensions = [path1::Ext1, path2::Ext2])]` | Call `Ext1::modify_ir(&mut module)` etc. on the IR after construction. Each path must implement `wgsl_rs::WgslExtension`. Extensions that declare `const MACROS` in their impl also enable compile-time verification that every `Stmt::Macro` in the module is claimed by at least one listed extension. |
///
/// Options can be combined: `#[wgsl(crate_path = my_crate::wgsl_rs,
/// skip_validation)]`.
///
/// # Auto-generated Validation Tests
///
/// Every non-template `#[wgsl]` module gets an auto-generated `__validate_wgsl`
/// test that calls `WGSL_SOURCE.validate()` at `cargo test` time, provided the
/// `validation` feature is enabled (the default). Without this feature,
/// `Module::validate()` is unavailable, so no test is generated.
///
/// Template (generic) modules cannot be validated standalone because their type
/// placeholders aren't valid WGSL; instead, use
/// `validate_with_instantiation_types(T1, T2, ...)` to specify concrete types
/// for instantiation:
///
/// ```ignore
/// #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, f32))]
/// pub mod my_shader { ... }
/// ```
///
/// Each occurrence generates a `__validate_wgsl_instantiated_N` test that
/// instantiates the module, renders the IR to WGSL, and validates with naga.
///
/// # Entry Points
///
/// Mark `pub fn` items with an entry point attribute to declare shader stages.
///
/// | Attribute | WGSL Output | Notes |
/// |-----------|------------|-------|
/// | `#[vertex]` | `@vertex` | |
/// | `#[fragment]` | `@fragment` | |
/// | `#[compute]` | `@compute` | Must also have `#[workgroup_size(...)]`. |
/// | `#[workgroup_size(X)]` | `@workgroup_size(X)` | 1D workgroup. |
/// | `#[workgroup_size(X, Y)]` | `@workgroup_size(X, Y)` | 2D workgroup. |
/// | `#[workgroup_size(X, Y, Z)]` | `@workgroup_size(X, Y, Z)` | 3D workgroup. |
///
/// **Auto-inference rules:**
/// - A `#[vertex]` function returning `Vec4f` automatically gets
///   `@builtin(position)` on its return type.
/// - A `#[fragment]` function returning `Vec4f` automatically gets
///   `@location(0)` on its return type.
///
/// ```ignore
/// #[vertex]
/// pub fn vs_main(#[builtin(vertex_index)] vi: u32) -> Vec4f {
///     // return type automatically gets @builtin(position)
///     vec4f(0.0, 0.0, 0.0, 1.0)
/// }
///
/// #[compute]
/// #[workgroup_size(64)]
/// pub fn cs_main(#[builtin(global_invocation_id)] gid: Vec3u) {
///     // ...
/// }
/// ```
///
/// # Built-in Values
///
/// Use `#[builtin(...)]` on function parameters or struct fields to bind
/// to WGSL built-in values. Each emits `@builtin(name)` in the WGSL output.
///
/// | Built-in Name | Stage | Direction | Type |
/// |---------------|-------|-----------|------|
/// | `vertex_index` | vertex | input | `u32` |
/// | `instance_index` | vertex | input | `u32` |
/// | `position` | vertex / fragment | output / input | `Vec4f` |
/// | `front_facing` | fragment | input | `bool` |
/// | `frag_depth` | fragment | output | `f32` |
/// | `sample_index` | fragment | input | `u32` |
/// | `sample_mask` | fragment | input+output | `u32` |
/// | `local_invocation_id` | compute | input | `Vec3u` |
/// | `local_invocation_index` | compute | input | `u32` |
/// | `global_invocation_id` | compute | input | `Vec3u` |
/// | `workgroup_id` | compute | input | `Vec3u` |
/// | `num_workgroups` | compute | input | `Vec3u` |
/// | `primitive_index` | fragment | input | `u32` |
/// | `subgroup_invocation_id` | compute+fragment | input | `u32` |
/// | `subgroup_size` | compute+fragment | input | `u32` |
/// | `subgroup_id` | compute | input | `u32` |
/// | `num_subgroups` | compute | input | `u32` |
///
/// # Inter-Stage IO Attributes
///
/// These attributes can be placed on function parameters or struct fields
/// to control data flow between shader stages.
///
/// ## `#[location(N)]`
///
/// Assigns a user-defined IO location. Emits `@location(N)` in WGSL.
/// Location indices must be unique within a given set of inputs or outputs
/// but do not need to be contiguous or ordered.
///
/// ## `#[blend_src(N)]`
///
/// For dual-source blending. Emits `@blend_src(N)` in WGSL. Only valid
/// on output struct fields with `@location(0)`. Must have exactly two
/// entries: `@blend_src(0)` and `@blend_src(1)`.
///
/// ## `#[interpolate(type)]` / `#[interpolate(type, sampling)]`
///
/// Controls how values are interpolated between vertex and fragment stages.
/// Emits `@interpolate(type)` or `@interpolate(type, sampling)` in WGSL.
///
/// **Interpolation types:** `perspective`, `linear`, `flat`
///
/// **Sampling modes:** `center`, `centroid`, `sample`, `first`, `either`
///
/// ## `#[invariant]`
///
/// Marks a `@builtin(position)` output as invariant. Emits `@invariant`
/// in WGSL.
///
/// # IO Structs
///
/// Inside a `#[wgsl]` module, place IO annotations directly on struct
/// fields. The `#[wgsl]` macro automatically strips these annotations from
/// the emitted Rust code so the struct compiles as plain Rust, while
/// preserving them in the WGSL output. The same struct can be used as
/// both a vertex shader output and a fragment shader input.
///
/// ```ignore
/// pub struct VertexOutput {
///     #[builtin(position)]
///     pub pos: Vec4f,
///     #[location(0)]
///     pub color: Vec4f,
///     #[location(1)]
///     #[interpolate(flat)]
///     pub material_id: u32,
/// }
/// ```
///
/// Transpiles to:
///
/// ```wgsl
/// struct VertexOutput {
///     @builtin(position) pos: vec4<f32>,
///     @location(0) color: vec4<f32>,
///     @location(1) @interpolate(flat) material_id: u32,
/// }
/// ```
///
/// # Resource Declarations
///
/// Declare GPU resources using macro invocations inside the module.
///
/// ## `uniform!`
///
/// ```ignore
/// uniform!(group(0), binding(0), MY_UNIFORM: MyStruct);
/// ```
///
/// Emits `@group(0) @binding(0) var<uniform> MY_UNIFORM: MyStruct;` in
/// WGSL. On the Rust side, generates a thread-safe static that can be
/// accessed with `get!` and `get_mut!`.
///
/// ## `storage!`
///
/// ```ignore
/// storage!(group(0), binding(1), MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_only, MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_write, MY_BUFFER: MyStruct);
/// ```
///
/// Emits `@group(G) @binding(B) var<storage, read>` or
/// `var<storage, read_write>` in WGSL. The default access mode is
/// `read_only`.
///
/// ## `sampler!`
///
/// ```ignore
/// sampler!(group(0), binding(2), TEX_SAMPLER: Sampler);
/// sampler!(group(0), binding(3), SHADOW_SAMPLER: SamplerComparison);
/// ```
///
/// Emits `@group(G) @binding(B) var NAME: sampler;` or
/// `var NAME: sampler_comparison;` in WGSL.
///
/// ## `texture!`
///
/// ```ignore
/// texture!(group(0), binding(4), DIFFUSE: Texture2D<f32>);
/// texture!(group(0), binding(5), SHADOW_MAP: TextureDepth2D);
/// texture!(group(0), binding(6), OUT: TextureStorage2D<Rgba8unorm, Write>);
/// ```
///
/// **Sampled textures** (with type parameter `f32`, `i32`, or `u32`):
/// `Texture1D`, `Texture2D`, `Texture2DArray`, `Texture3D`, `TextureCube`,
/// `TextureCubeArray`, `TextureMultisampled2D`.
///
/// **Depth textures** (no type parameter):
/// `TextureDepth2D`, `TextureDepth2DArray`, `TextureDepthCube`,
/// `TextureDepthCubeArray`, `TextureDepthMultisampled2D`.
///
/// **Storage textures** (with texel format and access mode parameters):
/// `TextureStorage1D`, `TextureStorage2D`, `TextureStorage2DArray`,
/// `TextureStorage3D`. The first type parameter is a texel format marker
/// (e.g. `Rgba8unorm`, `R32float` — see WGSL §6.6.1) and the second is an
/// access mode (`Read`, `Write`, `ReadWrite`).
///
/// ## `workgroup!`
///
/// ```ignore
/// workgroup!(SHARED_DATA: [f32; 64]);
/// workgroup!(COUNTER: Atomic<u32>);
/// ```
///
/// Emits `var<workgroup> NAME: TYPE;` in WGSL. Only usable in compute
/// shaders.
///
/// # Pointer Types
///
/// Use `ptr!(address_space, Type)` in function parameter types to declare
/// WGSL pointers. On the Rust side this expands to `&mut T`.
///
/// **Supported address spaces:** `function`, `private`, `workgroup`.
///
/// ```ignore
/// fn increment(p: ptr!(function, i32)) {
///     *p += 1;
/// }
/// ```
///
/// Emits `fn increment(p: ptr<function, i32>)` in WGSL.
///
/// # Enums
///
/// Enums must have `#[repr(u32)]` and become type aliases with constants
/// in WGSL.
///
/// ```ignore
/// #[repr(u32)]
/// pub enum Material {
///     Metal = 0,
///     Wood = 1,
///     Glass = 2,
/// }
/// ```
///
/// Transpiles to:
///
/// ```wgsl
/// alias Material = u32;
/// const Material_Metal: u32 = 0u;
/// const Material_Wood: u32 = 1u;
/// const Material_Glass: u32 = 2u;
/// ```
///
/// # Statement Macros
///
/// These macros are available inside function bodies within `#[wgsl]`
/// modules.
///
/// ## `get!` / `get_mut!`
///
/// Access module-level variables declared with `uniform!`, `storage!`, or
/// `workgroup!`. In Rust, these acquire a read or write lock on the
/// underlying thread-safe storage. In WGSL, they are stripped and the
/// variable name is emitted directly.
///
/// ```ignore
/// let value = get!(MY_UNIFORM).field;
/// get_mut!(MY_BUFFER).field = 42;
/// ```
///
/// ## `slab_copy!`
///
/// Bulk copy between a storage buffer slab and a local array (or between
/// two arrays). Bidirectional: pass a slab as the first argument to read,
/// or as the third argument to write.
///
/// ```ignore
/// slab_copy!(get!(SLAB), offset, dest_array, 0, count);
/// slab_copy!(src_array, 0, get_mut!(SLAB), offset, count);
/// ```
///
/// Expands to an indexed `for` loop in WGSL of the form
/// `dest[dest_offset + i] = src[src_offset + i]`.
///
/// # Warning Suppression
///
/// Use `#[wgsl_allow(...)]` on loop or match expressions to suppress
/// specific transpiler warnings.
///
/// | Warning Name | Description |
/// |-------------|-------------|
/// | `non_literal_loop_bounds` | Allow `for i in 0..n` where `n` is not a literal. |
/// | `non_literal_match_statement_patterns` | Allow match arms with non-literal patterns. |
///
/// ```ignore
/// #[wgsl_allow(non_literal_loop_bounds)]
/// for i in 0..n {
///     // ...
/// }
/// ```
///
/// # Complete Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// #[wgsl]
/// pub mod my_shader {
///     use super::*;
///
///     pub struct Uniforms {
///         pub projection: Mat4x4f,
///         pub modelview: Mat4x4f,
///     }
///
///     uniform!(group(0), binding(0), UNIFORMS: Uniforms);
///
///     pub struct VertexOutput {
///         #[builtin(position)]
///         pub clip_position: Vec4f,
///         #[location(0)]
///         pub color: Vec4f,
///     }
///
///     #[vertex]
///     pub fn vs_main(
///         #[builtin(vertex_index)] vertex_index: u32,
///         #[location(0)] position: Vec3f,
///         #[location(1)] color: Vec4f,
///     ) -> VertexOutput {
///         let u = get!(UNIFORMS);
///         let clip = u.projection * u.modelview * vec4f(position.x, position.y, position.z, 1.0);
///         VertexOutput {
///             clip_position: clip,
///             color,
///         }
///     }
///
///     #[fragment]
///     pub fn fs_main(
///         #[location(0)] color: Vec4f,
///     ) -> Vec4f {
///         color
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn wgsl(attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input_mod = syn::parse_macro_input!(token_stream as syn::ItemMod);
    match go_wgsl(attr, input_mod) {
        Ok(tokens) => tokens,
        Err(e) => match e {
            WgslGenError::RustParse { source } => source.to_compile_error().into(),
            WgslGenError::WgslParse { source } => {
                syn::Error::from(source).to_compile_error().into()
            }
        },
    }
}

#[proc_macro]
pub fn swizzle(token_stream: TokenStream) -> TokenStream {
    swizzle::swizzle(token_stream)
}

/// Marks a function as a vertex shader entry point.
///
/// Emits `@vertex` in WGSL. If the return type is `Vec4f`, the output
/// automatically gets `@builtin(position)`. Use `#[builtin(...)]` on
/// parameters to bind vertex-stage built-in inputs.
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn vertex(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // For now we don't do any transformations except for pulling out the
    // #[builtin(...)]s
    let mut item_fn: syn::ItemFn = syn::parse_macro_input!(token_stream);
    for arg in item_fn.sig.inputs.iter_mut() {
        if let syn::FnArg::Typed(pat_type) = arg {
            pat_type.attrs.retain(|attr| {
                if let Some(ident) = attr.path().get_ident() {
                    !matches!(ident.to_string().as_str(), "builtin")
                } else {
                    true
                }
            });
        }
    }
    item_fn.into_token_stream().into()
}

/// Marks a function as a fragment shader entry point.
///
/// Emits `@fragment` in WGSL. If the return type is `Vec4f`, the output
/// automatically gets `@location(0)`. Use `#[builtin(...)]` on parameters
/// to bind fragment-stage built-in inputs (e.g., `position`, `front_facing`).
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Strip #[builtin(...)] attributes from function arguments, mirroring
    // `#[vertex]` and `#[compute]`. Without this, `#[builtin(position)]` on a
    // fragment parameter is passed through to the Rust compiler, which rejects
    // it as a non-macro attribute (wgsl-rs#84).
    let mut item_fn: syn::ItemFn = syn::parse_macro_input!(token_stream);
    for arg in item_fn.sig.inputs.iter_mut() {
        if let syn::FnArg::Typed(pat_type) = arg {
            pat_type.attrs.retain(|attr| {
                if let Some(ident) = attr.path().get_ident() {
                    !matches!(ident.to_string().as_str(), "builtin")
                } else {
                    true
                }
            });
        }
    }
    item_fn.into_token_stream().into()
}

/// Marks a function as a compute shader entry point.
///
/// Emits `@compute` in WGSL. Must be paired with `#[workgroup_size(...)]`.
/// Use `#[builtin(...)]` on parameters to bind compute-stage built-in
/// inputs (e.g., `global_invocation_id`, `local_invocation_id`).
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn compute(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Strip #[builtin(...)] attributes from function arguments
    let mut item_fn: syn::ItemFn = syn::parse_macro_input!(token_stream);
    for arg in item_fn.sig.inputs.iter_mut() {
        if let syn::FnArg::Typed(pat_type) = arg {
            pat_type.attrs.retain(|attr| {
                if let Some(ident) = attr.path().get_ident() {
                    !matches!(ident.to_string().as_str(), "builtin")
                } else {
                    true
                }
            });
        }
    }
    item_fn.into_token_stream().into()
}

/// Specifies the workgroup dimensions for a `#[compute]` entry point.
///
/// Accepts 1, 2, or 3 integer literal arguments. Emits
/// `@workgroup_size(X)`, `@workgroup_size(X, Y)`, or
/// `@workgroup_size(X, Y, Z)` in WGSL.
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn workgroup_size(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Binds a function parameter or struct field to a WGSL built-in value.
///
/// Emits `@builtin(name)` in WGSL. Stripped from the Rust output during
/// compilation. See [`wgsl`] for the full list of supported built-in names
/// and which shader stages they belong to.
///
/// ```ignore
/// #[vertex]
/// pub fn main(#[builtin(vertex_index)] vi: u32) -> Vec4f { /* ... */ }
/// ```
#[proc_macro_attribute]
pub fn builtin(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Suppresses specific wgsl-rs warnings/errors on annotated statements.
///
/// Use this attribute to acknowledge cases where the transpiler cannot
/// guarantee correctness during macro expansion, but you know the code
/// is valid.
///
/// # Available Warnings
///
/// - `non_literal_loop_bounds`: Suppresses the error for for-loops with
///   non-literal bounds (e.g., `for i in 0..n` where `n` is a variable). WGSL
///   only supports ascending iteration, so the loop may fail at runtime if the
///   range is descending.
///
/// - `non_literal_match_statement_patterns`: Suppresses the warning for match
///   statements with non-literal case selectors (e.g., constants or
///   identifiers). WGSL requires case selectors to be const-expressions, which
///   the transpiler cannot always verify.
///
/// # Examples
///
/// ```ignore
/// pub fn sum_to_n(n: i32) -> i32 {
///     let mut total = 0;
///     #[wgsl_allow(non_literal_loop_bounds)]
///     for i in 0..n {
///         total += i;
///     }
///     total
/// }
/// ```
///
/// ```ignore
/// const LOW: i32 = 0;
/// const HIGH: i32 = 1;
///
/// pub fn with_const_patterns(level: i32) -> f32 {
///     let mut result = 0.0;
///     #[wgsl_allow(non_literal_match_statement_patterns)]
///     match level {
///         LOW => { result = 0.1; }
///         HIGH => { result = 1.0; }
///         _ => {}
///     }
///     result
/// }
/// ```
#[proc_macro_attribute]
pub fn wgsl_allow(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Ignores any item following the attribute during parsing, skipping WGSL code
/// generation.
///
/// This provides an escape hatch to use arbitrary Rust code within your
/// module without having a WGSL representation, but only from the Rust side,
/// as the code annotated with `#[wgsl_ignore]` has no WGSL representation.
#[proc_macro_attribute]
pub fn wgsl_ignore(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Declares a uniform buffer binding.
///
/// Emits `@group(G) @binding(B) var<uniform> NAME: TYPE;` in WGSL.
/// On the Rust side, generates a thread-safe static accessible via
/// `get!` and `get_mut!`.
///
/// ```ignore
/// uniform!(group(0), binding(0), MY_UNIFORM: MyStruct);
/// ```
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro]
pub fn uniform(input: TokenStream) -> TokenStream {
    uniform::uniform(input)
}

/// Declares a storage buffer binding.
///
/// Emits `@group(G) @binding(B) var<storage, read>` or
/// `var<storage, read_write>` in WGSL depending on the access mode.
/// The default is `read_only`.
///
/// ```ignore
/// storage!(group(0), binding(1), MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_write, MY_BUFFER: MyStruct);
/// ```
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro]
pub fn storage(input: TokenStream) -> TokenStream {
    storage::storage(input)
}

/// Defines a sampler or comparison sampler for texture sampling operations.
///
/// # Syntax
/// ```ignore
/// sampler!(group(G), binding(B), NAME: Sampler);
/// sampler!(group(G), binding(B), NAME: Sampler<false>);
/// sampler!(group(G), binding(B), NAME: Sampler<FILTERABLE>);
/// sampler!(group(G), binding(B), NAME: SamplerComparison);
/// ```
///
/// # Description
/// Samplers control how textures are sampled in shaders, including filtering
/// modes and address wrapping behavior. Comparison samplers are used for
/// depth texture sampling operations like shadow mapping.
///
/// # WGSL Output
/// The macro transpiles to:
/// - `@group(G) @binding(B) var NAME: sampler;` for regular samplers
/// - `@group(G) @binding(B) var NAME: sampler_comparison;` for comparison
///   samplers
///
/// `Sampler<false>` does not change the WGSL output. It selects
/// `SamplerBindingType::NonFiltering` on the generated wgpu bind group layout
/// entry, so the sampler can pair with unfilterable texture bindings such as
/// `R32Float`. The const defaults to `true`. Inside `#[wgsl]`, a bare
/// identifier such as `Sampler<FILTERABLE>` introduces a binding-owned module
/// const parameter resolved by `instantiate::<bool>()`. Reusing the identifier
/// on multiple bindings shares the same argument.
///
/// The older trailing `, unfilterable` modifier remains accepted temporarily
/// as a compatibility alias for `Sampler<false>`.
///
/// # Rust Expansion
/// On the Rust side, the macro generates:
/// - A static `Sampler` or `SamplerComparison` instance
/// - A `SamplerDescriptor` constant for creating the sampler
/// - A convenience function to create the sampler
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);
/// sampler!(group(0), binding(2), SHADOW_SAMPLER: SamplerComparison);
///
/// #[fragment]
/// pub fn main() -> Vec4f {
///     // Use samplers for texture sampling...
///     Vec4f::ZERO
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// @group(0) @binding(1) var TEX_SAMPLER: sampler;
/// @group(0) @binding(2) var SHADOW_SAMPLER: sampler_comparison;
///
/// @fragment
/// fn main() -> vec4<f32> {
///     // Use samplers for texture sampling...
///     return vec4<f32>(0.0);
/// }
/// ```
#[proc_macro]
pub fn sampler(input: TokenStream) -> TokenStream {
    sampler::sampler(input)
}

/// Defines a texture or depth texture for sampling operations.
///
/// # Syntax
/// ```ignore
/// // Sampled textures (with type parameter: f32, i32, or u32)
/// texture!(group(G), binding(B), NAME: Texture2D<f32>);
/// texture!(group(G), binding(B), NAME: TextureCube<i32>);
///
/// // Unfilterable f32-sampled textures (non-filterable formats, e.g. R32Float)
/// texture!(group(G), binding(B), NAME: Texture2D<f32, false>);
///
/// // Binding-owned template filterability inside `#[wgsl]`
/// texture!(group(G), binding(B), NAME: Texture2D<f32, FILTERABLE>);
///
/// // Depth textures (no type parameter)
/// texture!(group(G), binding(B), NAME: TextureDepth2D);
/// texture!(group(G), binding(B), NAME: TextureDepthCube);
///
/// // Storage textures (texel format + access mode)
/// texture!(group(G), binding(B), NAME: TextureStorage2D<Rgba8unorm, Write>);
/// texture!(group(G), binding(B), NAME: TextureStorage2D<R32float, ReadWrite>);
/// ```
///
/// # Supported Texture Types
///
/// ## Sampled Textures
/// - `Texture1D<T>` - 1D texture
/// - `Texture2D<T>` - 2D texture
/// - `Texture2DArray<T>` - 2D texture array
/// - `Texture3D<T>` - 3D texture
/// - `TextureCube<T>` - Cube texture
/// - `TextureCubeArray<T>` - Cube texture array
/// - `TextureMultisampled2D<T>` - Multisampled 2D texture
///
/// Where `T` is one of `f32`, `i32`, or `u32`. F32-sampled textures accept an
/// optional second bool const for host-side filterability, defaulting to
/// `true`. Integer-sampled textures have no filterability parameter.
///
/// ## Depth Textures
/// - `TextureDepth2D` - 2D depth texture
/// - `TextureDepth2DArray` - 2D depth texture array
/// - `TextureDepthCube` - Cube depth texture
/// - `TextureDepthCubeArray` - Cube depth texture array
/// - `TextureDepthMultisampled2D` - Multisampled 2D depth texture
///
/// ## Storage Textures
/// - `TextureStorage1D<F, A>` - 1D storage texture
/// - `TextureStorage2D<F, A>` - 2D storage texture
/// - `TextureStorage2DArray<F, A>` - 2D array storage texture
/// - `TextureStorage3D<F, A>` - 3D storage texture
///
/// Where `F` is a texel format marker (e.g. `Rgba8unorm`, `R32float`,
/// `Rgba32uint`) and `A` is an access mode (`Read`, `Write`, `ReadWrite`). See
/// WGSL §6.6.1 for the full list of texel formats and §6.6.5 for storage
/// texture types.
///
/// # WGSL Output
/// The macro transpiles to:
/// - `@group(G) @binding(B) var NAME: texture_2d<f32>;` for sampled textures
/// - `@group(G) @binding(B) var NAME: texture_depth_2d;` for depth textures
/// - `@group(G) @binding(B) var NAME: texture_storage_2d<rgba8unorm, write>;`
///   for storage textures
///
/// A `false` filterability const does not change the WGSL output. It only sets
/// `filterable: false` on the generated wgpu bind group layout entry for
/// textures backed by non-filterable formats such as `R32Float`. A bare
/// identifier inside `#[wgsl]` introduces a binding-owned module const
/// parameter resolved by `instantiate::<bool>()`. Reusing an identifier on
/// multiple bindings shares one argument.
///
/// The older trailing `, unfilterable` modifier remains accepted temporarily
/// as a compatibility alias for `Texture2D<f32, false>` and its equivalents.
///
/// # Rust Expansion
/// On the Rust side, the macro generates:
/// - A static texture handle instance
/// - A `TextureViewDescriptor` constant for creating views
/// - A convenience function to create a texture view
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
/// texture!(group(0), binding(1), SHADOW_MAP: TextureDepth2D);
/// sampler!(group(0), binding(2), TEX_SAMPLER: Sampler);
///
/// #[fragment]
/// pub fn main() -> Vec4f {
///     // Sample the diffuse texture...
///     Vec4f::ZERO
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// @group(0) @binding(0) var DIFFUSE_TEX: texture_2d<f32>;
/// @group(0) @binding(1) var SHADOW_MAP: texture_depth_2d;
/// @group(0) @binding(2) var TEX_SAMPLER: sampler;
///
/// @fragment
/// fn main() -> vec4<f32> {
///     // Sample the diffuse texture...
///     return vec4<f32>(0.0);
/// }
/// ```
#[proc_macro]
pub fn texture(input: TokenStream) -> TokenStream {
    texture::texture(input)
}

/// Defines a workgroup-scoped variable shared between invocations in a compute
/// shader workgroup.
///
/// # Syntax
/// ```ignore
/// workgroup!(NAME: TYPE);
/// ```
///
/// # Description
/// Workgroup variables are shared between all invocations in a compute shader
/// workgroup. They can only be used in compute shaders and are useful for
/// inter-invocation communication and shared temporary storage.
///
/// # WGSL Output
/// The macro transpiles to `var<workgroup> NAME: TYPE;` in WGSL.
///
/// # Rust Expansion
/// On the Rust side, the variable is backed by a thread-safe `RwLock` to
/// simulate the shared nature of workgroup memory.
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// workgroup!(SHARED_COUNTER: Atomic<u32>);
/// workgroup!(TEMP_DATA: [f32; 64]);
///
/// #[compute]
/// #[workgroup_size(64)]
/// pub fn main(#[builtin(local_invocation_id)] local_id: Vec3u) {
///     // Access workgroup variables...
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// var<workgroup> SHARED_COUNTER: atomic<u32>;
/// var<workgroup> TEMP_DATA: array<f32, 64>;
///
/// @compute @workgroup_size(64)
/// fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
///     // Access workgroup variables...
/// }
/// ```
#[proc_macro]
pub fn workgroup(input: TokenStream) -> TokenStream {
    workgroup::workgroup(input)
}

/// Defines a WGSL pointer type for function parameters.
///
/// # Syntax
/// ```ignore
/// ptr!(address_space, Type)
/// ```
///
/// # Supported Address Spaces
/// - `function` - For pointers to local function variables
/// - `private` - For pointers to module-scope private variables
///
/// # Rust Expansion
/// The macro expands to `&mut T` in Rust, allowing the code to compile and run
/// on the CPU with mutable reference semantics.
///
/// # WGSL Output
/// During transpilation, `ptr!(function, T)` becomes `ptr<function, T>` in
/// WGSL. The access mode is always `read_write` (the only mode supported by
/// `function` and `private` address spaces) and is not written in the output.
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// fn increment(p: ptr!(function, i32)) {
///     *p += 1;
/// }
///
/// fn test() {
///     let mut x: i32 = 5;
///     increment(&mut x);
///     // x is now 6
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// fn increment(p: ptr<function, i32>) {
///     *p += 1;
/// }
/// ```
#[proc_macro]
pub fn ptr(input: TokenStream) -> TokenStream {
    ptr::ptr(input)
}

/// Derives a `Wgsl` implementation for a type.
///
/// The generated `to_ir()` method returns:
/// - `wgsl_rs_ir::Type::Struct { name, type_args }` for structs (where
///   `type_args` is built from each generic type parameter via `T::to_ir()`).
/// - `wgsl_rs_ir::Type::Scalar(ScalarType::U32)` for enums (since enums render
///   as `u32` aliases in WGSL).
///
/// By default the macro references the IR re-export at `::wgsl_rs::ir`
/// and the trait at `::wgsl_rs::std::Wgsl`. Crates that consume `wgsl_rs`
/// under a different path (e.g. `wgsl-rs` itself, when running its own
/// tests) can override the path with the `#[wgsl_path(...)]` helper
/// attribute:
///
/// ```ignore
/// #[derive(Wgsl)]
/// #[wgsl_path(crate)]
/// pub struct Foo { ... }
/// ```
#[proc_macro_derive(Wgsl, attributes(wgsl_path))]
pub fn derive_wgsl(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse_macro_input!(input);
    let ident = input.ident;
    let (impl_generics, ty_generics, where_generics) = input.generics.split_for_impl();

    // Resolve the path to the `wgsl_rs` crate. Defaults to `::wgsl_rs`
    // but can be overridden with `#[wgsl_path(<path>)]`.
    let crate_path: syn::Path = input
        .attrs
        .iter()
        .find(|a| a.path().is_ident("wgsl_path"))
        .and_then(|a| a.parse_args::<syn::Path>().ok())
        .unwrap_or_else(|| syn::parse_quote!(::wgsl_rs));
    let ir_path = quote::quote! { #crate_path::ir };
    let wgsl_trait = quote::quote! { #crate_path::std::Wgsl };

    let tys = input
        .generics
        .type_params()
        .map(|param| {
            let pident = &param.ident;
            quote::quote! { <#pident as #wgsl_trait>::to_ir() }
        })
        .collect::<Vec<_>>();

    match input.data {
        syn::Data::Struct(_) => quote::quote! {
            impl #impl_generics #wgsl_trait for #ident #ty_generics #where_generics {
                fn to_ir() -> #ir_path::Type {
                    #ir_path::Type::Struct {
                        name: stringify!(#ident).to_string(),
                        type_args: ::std::vec![
                            #(#tys),*
                        ]
                    }
                }
            }
        }
        .into(),
        syn::Data::Enum(_) => quote::quote! {
            impl #impl_generics #wgsl_trait for #ident #ty_generics #where_generics {
                fn to_ir() -> #ir_path::Type {
                    #ir_path::Type::Scalar(#ir_path::ScalarType::U32)
                }
            }
        }
        .into(),
        syn::Data::Union(_) => quote::quote! {
            compile_error!("derive Wgsl doesn't support Unions")
        }
        .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two modules identical except for a glob's visibility
    /// (`use defs::*` vs `pub use defs::*`) must get DISTINCT ids: the
    /// id-keyed visited sets of the resolution and harvest walks would
    /// otherwise collapse the pair, and with the private variant
    /// visited first the public variant's edges are silently dropped
    /// (wgsl-rs#114 review, superseding the 2026-10-31 #245 exclusion).
    #[test]
    fn module_id_distinguishes_reexport_edges() {
        fn gen_module_id(reexports: &[proc_macro2::TokenStream]) -> u64 {
            let item_mod: syn::ItemMod = syn::parse_quote! {
                mod m {
                    use super::defs::*;

                    pub fn f() -> u32 {
                        1u32
                    }
                }
            };
            let mut wgsl_module = parse::ItemMod::try_from(&item_mod).unwrap();
            let mono_result = monomorphize::run(&mut wgsl_module).unwrap();
            monomorphize::resolve_assoc_types(&mut wgsl_module).unwrap();
            let imports = vec![quote! { super::defs::WGSL_SOURCE }];
            let tokens = gen_wgsl_module(
                &wgsl_module.ident,
                &syn::parse_quote! { wgsl_rs },
                &imports,
                reexports,
                &[],
                &wgsl_module,
                &mono_result,
                &[],
                &[],
            )
            .unwrap();
            let src = tokens.to_string();
            let marker = src
                .find("id :")
                .or_else(|| src.find("id:"))
                .expect("emitted Source must contain an id field");
            let digits: String = src[marker..]
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().expect("the id field is a u64 literal")
        }

        let private = gen_module_id(&[]);
        let reexported = gen_module_id(&[quote! { super::defs::WGSL_SOURCE }]);
        assert_ne!(
            private, reexported,
            "a pub glob re-export must distinguish the module's id from the private spelling"
        );
        // Stability: identical re-export sets hash to the same id.
        assert_eq!(
            reexported,
            gen_module_id(&[quote! { super::defs::WGSL_SOURCE }])
        );
    }

    /// Two modules whose content differs only in WHICH name they
    /// re-export must get distinct ids (wgsl-rs#114 review): the
    /// id-keyed visited sets of the resolution and harvest walks would
    /// otherwise collapse the two modules and silently drop one
    /// alias, making downstream resolution depend on traversal order.
    #[test]
    fn module_id_distinguishes_named_exports() {
        fn gen_module_id(named_exports: &[(String, proc_macro2::TokenStream)]) -> u64 {
            let item_mod: syn::ItemMod = syn::parse_quote! {
                mod m {
                    use super::defs::*;

                    pub fn f() -> u32 {
                        1u32
                    }
                }
            };
            let mut wgsl_module = parse::ItemMod::try_from(&item_mod).unwrap();
            let mono_result = monomorphize::run(&mut wgsl_module).unwrap();
            monomorphize::resolve_assoc_types(&mut wgsl_module).unwrap();
            let imports = vec![quote! { super::defs::WGSL_SOURCE }];
            let tokens = gen_wgsl_module(
                &wgsl_module.ident,
                &syn::parse_quote! { wgsl_rs },
                &imports,
                &[],
                named_exports,
                &wgsl_module,
                &mono_result,
                &[],
                &[],
            )
            .unwrap();
            let src = tokens.to_string();
            let marker = src
                .find("id :")
                .or_else(|| src.find("id:"))
                .expect("emitted Source must contain an id field");
            let digits: String = src[marker..]
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse().expect("the id field is a u64 literal")
        }

        let defs = quote! { super::defs::WGSL_SOURCE };
        let id_of_id = gen_module_id(&[("Id".to_string(), defs.clone())]);
        let id_of_hidden = gen_module_id(&[("Hidden".to_string(), defs.clone())]);
        assert_ne!(
            id_of_id, id_of_hidden,
            "two modules re-exporting different names from the same source must not share an id"
        );
        // Stability: the same named exports hash to the same id.
        assert_eq!(id_of_id, gen_module_id(&[("Id".to_string(), defs)]));
    }
}
