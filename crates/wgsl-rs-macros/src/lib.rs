#![cfg_attr(nightly, feature(proc_macro_diagnostic))]

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use snafu::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use syn::{
    DeriveInput,
    visit_mut::{self, VisitMut},
};

use crate::parse::InterStageIo;

static NEXT_MODULE_ID: AtomicU64 = AtomicU64::new(0);

mod builder;
mod builtins;
mod ir_convert;
mod ir_emit;
#[cfg(feature = "linkage-wgpu")]
mod linkage;
mod monomorphize;
mod parse;
mod parse_visitor;
mod ptr;
mod sampler;
mod storage;
mod swizzle;
mod texture;
mod uniform;
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
/// from struct fields.
///
/// These attributes are read during WGSL parsing (in `Field::parse_with_ctx`),
/// but they aren't valid Rust attributes on struct fields. The `#[wgsl]` macro
/// strips them from the emitted Rust code so the struct compiles as plain
/// Rust, while the IO information has already been captured for WGSL output.
struct StripIoAttrs;

impl VisitMut for StripIoAttrs {
    fn visit_field_mut(&mut self, field: &mut syn::Field) {
        field
            .attrs
            .retain(|attr| InterStageIo::try_from(attr).is_err());
        visit_mut::visit_field_mut(self, field);
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
    /// validated via `WGSL_MODULE.validate()`).
    validate_instantiations: Vec<Vec<syn::Type>>,
}

struct InstantiationTypes {
    types: syn::punctuated::Punctuated<syn::Type, syn::Token![,]>,
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
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "Unknown attribute '{other}', expected 'crate_path', \
                             'skip_validation', or 'validate_with_instantiation_types'"
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


/// Collect the encoded names of all module-level type parameters in the
/// parsed module, in the order they should appear in the generated
/// `WGSL_MODULE.module_type_params` slice.
///
/// Two sources contribute, both encoded so that no two distinct
/// declarations can share an IR name:
///
/// 1. Linkage variables declared with `impl Trait` syntax. Each one contributes
///    a single name equal to the variable's identifier (e.g. `"FRAME"` for
///    `uniform!(..., FRAME: impl Convert<f32>)`).
/// 2. Type parameters of shader entry points (`#[vertex]`, `#[fragment]`,
///    `#[compute]`). Each one contributes a positional name of the form
///    `"<fn_name>_<index>"` (e.g. `"frag_main_0"`, `"frag_main_1"`).
///    Source-level names like `T` and `S` are intentionally not used so that
///    two entry points using the same letter don't collide.
///
/// Order: linkage variables in declaration order, followed by entry
/// point parameters in declaration order, in turn following each entry
/// point's parameter list left-to-right.
fn collect_entry_point_type_params(wgsl_module: &parse::ItemMod) -> Vec<String> {
    use parse::{FnAttrs, Item};
    let mut params: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Pass 1: linkage variables declared with `impl Trait`.
    for item in &wgsl_module.content {
        let name = match item {
            Item::Uniform(u) if u.impl_bounds.is_some() => u.name.to_string(),
            Item::Storage(s) if s.impl_bounds.is_some() => s.name.to_string(),
            Item::Workgroup(w) if w.impl_bounds.is_some() => w.name.to_string(),
            _ => continue,
        };
        if seen.insert(name.clone()) {
            params.push(name);
        }
    }

    // Pass 2: entry point type parameters, encoded positionally.
    for item in &wgsl_module.content {
        let Item::Fn(f) = item else { continue };
        if matches!(f.fn_attrs, FnAttrs::None) {
            continue;
        }
        let fn_name = f.ident.to_string();
        for (i, _tp) in f.type_params.iter().enumerate() {
            let encoded = format!("{fn_name}_{i}");
            if seen.insert(encoded.clone()) {
                params.push(encoded);
            }
        }
    }

    params
}

/// Replace any `uniform!`/`storage!`/`workgroup!` macro item in the input
/// Rust module whose declared type is a *type parameter* of one of the
/// module's entry points with the pre-expanded "type-erased" form
/// (e.g. `pub static FRAME: Uniform = Uniform::new(0, 0);`).
///
/// The default of `Uniform` / `Storage` / `Workgroup` is `WgslTypeVariable`,
/// which keeps a runtime `TypeId`-keyed map of values; `get!(VAR, T)` /
/// `get_mut!(VAR, T)` then read/write a value of the requested concrete
/// type.
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
            _ => {}
        }
    }

    if generic_uniforms.is_empty() && generic_storages.is_empty() && generic_workgroups.is_empty() {
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
            "uniform" | "storage" | "workgroup" => {
                extract_linkage_var_name(item_macro.mac.tokens.clone())
            }
            _ => None,
        };
        let Some(var_name) = var_name else {
            continue;
        };

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

/// Extracts the variable name from a `uniform!` / `storage!` / `workgroup!`
/// macro token stream. Returns `None` if the tokens don't match the
/// expected form.
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

fn gen_wgsl_module(
    name: &syn::Ident,
    crate_path: &syn::Path,
    imports: &[proc_macro2::TokenStream],
    wgsl_module: &parse::ItemMod,
    mono_result: &monomorphize::MonoResult,
    module_type_params: &[String],
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
    let module_name_lit = wgsl_module.ident.to_string();

    // Convert the parse module's items to IR items and emit a constructor
    // function body that builds the IR at runtime.
    let ir_items = ir_convert::items_from_parse(&wgsl_module.content).map_err(|e| {
        WgslGenError::WgslParse {
            source: e.into_parse(),
        }
    })?;
    let module_constructor_body = {
        let item_exprs: Vec<proc_macro2::TokenStream> = ir_items
            .iter()
            .map(|i| ir_emit::emit_item(&ir_p, i))
            .collect();
        quote! {
            #ir_p::Module {
                name: ::std::string::String::from(#module_name_lit),
                items: ::std::vec![#(#item_exprs),*],
            }
        }
    };

    // Generate template entries for generic functions / structs defined in
    // this module. The IR is built lazily by an emitted constructor
    // function.
    let mut template_constructors: Vec<proc_macro2::TokenStream> = Vec::new();
    let template_entries: Vec<proc_macro2::TokenStream> = mono_result
        .template_macros
        .iter()
        .enumerate()
        .map(|(idx, tmpl)| {
            let name_lit = &tmpl.fn_name;
            let params: Vec<&str> = tmpl.type_param_names.iter().map(|s| s.as_str()).collect();

            // Convert the template's items to IR.
            let ir_items =
                ir_convert::items_from_parse(&tmpl.items).map_err(|e| WgslGenError::WgslParse {
                    source: e.into_parse(),
                })?;
            let body_items: Vec<proc_macro2::TokenStream> = ir_items
                .iter()
                .map(|i| ir_emit::emit_item(&ir_p, i))
                .collect();

            let ctor_ident = quote::format_ident!("__wgsl_template_{}_ctor", idx);
            template_constructors.push(quote! {
                fn #ctor_ident() -> ::std::vec::Vec<#ir_p::Item> {
                    ::std::vec![#(#body_items),*]
                }
            });

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
    let inst_entries: Vec<proc_macro2::TokenStream> = mono_result
        .cross_module_instantiations
        .iter()
        .enumerate()
        .map(|(idx, inst)| {
            let import_paths = &inst.import_paths;
            let tmpl_name_lit = &inst.fn_name;
            let mangled_args: Vec<&str> =
                inst.mangled_type_args.iter().map(|s| s.as_str()).collect();

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

            let ctor_ident = quote::format_ident!("__wgsl_inst_{}_ctor", idx);
            inst_constructors.push(quote! {
                fn #ctor_ident() -> ::std::vec::Vec<#ir_p::Type> {
                    ::std::vec![#(#arg_exprs),*]
                }
            });

            let modules: Vec<proc_macro2::TokenStream> = import_paths
                .iter()
                .filter(|path| !is_wgsl_std_import(crate_path, path))
                .map(|path| quote! { &#path::WGSL_MODULE })
                .collect();

            Ok::<_, WgslGenError>(quote! {
                #crate_path::TemplateInstantiation {
                    modules: &[#(#modules),*],
                    template_name: #tmpl_name_lit,
                    type_args_constructor: #ctor_ident,
                    mangled_type_args: &[#(#mangled_args),*],
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let module_type_params_lits: Vec<&str> =
        module_type_params.iter().map(|s| s.as_str()).collect();

    let module_ctor_ident = quote::format_ident!("__wgsl_module_ctor");

    let module_id = NEXT_MODULE_ID.fetch_add(1, Ordering::Relaxed);

    Ok(quote! {
        fn #module_ctor_ident() -> #ir_p::Module {
            #module_constructor_body
        }

        #(#template_constructors)*

        #(#inst_constructors)*

        pub static WGSL_MODULE: #crate_path::Module = #crate_path::Module {
            id: #module_id,
            name: stringify!(#name),
            imports: &[
                #(&#imports),*
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
    })
}

/// Generates a `#[test]` function that validates the assembled WGSL source.
///
/// Emitted for non-template `#[wgsl]` modules (unless `skip_validation` is
/// set). The validation is deferred to `cargo test` time, where it calls
/// `WGSL_MODULE.validate()`.
///
/// The `validate()` method is available on `Module` when the `wgsl-rs` crate
/// has the `validation` feature enabled (the default). We gate the generated
/// test on `#[cfg(test)]` only — not `feature = "validation"` — because the
/// feature belongs to `wgsl-rs`, not the consuming crate, and checking it
/// here would produce an "unexpected cfg" warning in downstream crates.
fn gen_validation_test(module_ident: &syn::Ident) -> proc_macro2::TokenStream {
    let error_msg = format!("WGSL validation failed for module '{module_ident}'");
    quote! {
        #[cfg(test)]
        #[test]
        fn __validate_wgsl() {
            WGSL_MODULE.validate().expect(#error_msg);
        }
    }
}

/// Generates `#[test]` functions that validate instantiated WGSL source for
/// template (generic) modules.
///
/// Each occurrence of `validate_with_instantiation_types(T1, T2, ...)`
/// produces a test that calls `instantiate::<T1, T2, ...>()`, renders the
/// IR to WGSL, and validates the result with naga.
fn gen_instantiated_validation_tests(
    module_ident: &syn::Ident,
    crate_path: &syn::Path,
    instantiations: &[Vec<syn::Type>],
) -> proc_macro2::TokenStream {
    let ir_p = quote! { #crate_path::ir };
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
                    let __src = #ir_p::render_module(&__m);
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
    let mono_result = monomorphize::run(&mut wgsl_module)?;
    let imports = wgsl_module.imports(&crate_path);

    // Rewrite any `uniform!`/`storage!`/`workgroup!` declarations in the
    // input Rust module that have a generic type (e.g.
    // `uniform!(group(0), binding(0), FRAME: impl Convert<f32>)`). The
    // generated `pub static` for such a declaration must use the
    // type-erased default (`Uniform<WgslTypeVariable>`, i.e. just
    // `Uniform`) because the concrete type is not known at module level.
    rewrite_generic_linkages(&mut input_mod, &wgsl_module, &crate_path)?;

    // Compute the union of type parameters across all entry points in this
    // module. These become the module-level type parameters for the generated
    // `Module::module_type_params` field.
    let module_type_params: Vec<String> = collect_entry_point_type_params(&wgsl_module);

    let module_fragment = gen_wgsl_module(
        &wgsl_module.ident,
        &crate_path,
        &imports,
        &wgsl_module,
        &mono_result,
        &module_type_params,
    )?;

    // Generate validation tests.
    //
    // Non-template modules get a `__validate_wgsl` test that calls
    // `WGSL_MODULE.validate()`. Template (generic) modules cannot be
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

    // Generate linkage module when feature is enabled.
    //
    // Template modules (with module-level type parameters) skip linkage
    // generation: the WGSL `shader_source()` is a template with unresolved
    // placeholders, so a `wgpu::ShaderModule` can't be built from it
    // directly. Users must instantiate the template first, then construct
    // their own pipeline / bind groups manually for now.
    #[cfg(feature = "linkage-wgpu")]
    let linkage_fragment = if module_type_params.is_empty() {
        let linkage_info =
            linkage::LinkageInfo::from_item_mod(input_mod.ident.clone(), &wgsl_module);
        linkage::generate_linkage_module(&linkage_info)
    } else {
        quote! {}
    };

    // For template modules (those with module-level type parameters),
    // emit an `instantiate` function alongside `WGSL_MODULE`. The
    // function uses `wgsl_rs::linkage::Type<Is = ...>` constraints to
    // enforce at compile time that every linkage variable's concrete type
    // is consistent across all entry points that use it.
    let builder_fragment = if module_type_params.is_empty() {
        quote! {}
    } else {
        builder::gen_builder(&crate_path, &wgsl_module)
    };

    if let Some((_, content)) = input_mod.content.as_mut() {
        // The module fragment now contains multiple items (constructor
        // fns + the WGSL_MODULE static), so parse it as a list of items
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

        // Add linkage if the feature is set (skipped for template modules
        // — see the linkage_fragment construction above).
        #[cfg(feature = "linkage-wgpu")]
        if !linkage_fragment.is_empty() {
            let linkage_item: syn::Item = syn::parse2(linkage_fragment)?;
            content.push(linkage_item);
        }
    }

    // NOTE: Compile-time WGSL validation has been removed in favor of
    // runtime validation. All non-template modules get an auto-generated
    // `__validate_wgsl` test. Template (generic) modules get an
    // auto-generated test for each `validate_with_instantiation_types(...)`
    // attribute, which instantiates the module with the provided concrete
    // types and validates the resulting WGSL source with naga.
    let _ = &attrs;

    // Strip #[wgsl_allow] attributes before emitting Rust code.
    // These attributes are used during parsing but must be removed from the output
    // because statement-level attributes require the unstable stmt_expr_attributes
    // feature.
    StripWgslAllowAttrs.visit_item_mod_mut(&mut input_mod);

    // Strip inter-stage IO attributes (#[builtin], #[location], etc.) from
    // struct fields. These aren't valid Rust attributes on fields, but they
    // are read during WGSL parsing.
    StripIoAttrs.visit_item_mod_mut(&mut input_mod);

    Ok(input_mod.into_token_stream().into())
}

/// Transpiles a Rust module to WGSL.
///
/// Apply `#[wgsl]` to a `mod` item to generate a `WGSL_MODULE` static
/// containing the module's IR constructor and metadata. The Rust code
/// inside the module remains fully functional and can be executed on the
/// CPU, while the WGSL produced by `WGSL_MODULE.wgsl_source()` (or
/// `WGSL_MODULE.instantiate(...)` for generic modules) runs on the GPU.
///
/// # Module Options
///
/// | Syntax | Description |
/// |--------|-------------|
/// | `#[wgsl]` | Transpile the module with default settings. |
/// | `#[wgsl(skip_validation)]` | Skip the auto-generated `__validate_wgsl` test. |
/// | `#[wgsl(crate_path = path::to::crate)]` | Override the path to the `wgsl_rs` crate. |
/// | `#[wgsl(validate_with_instantiation_types(T1, T2, ...))]` | For template (generic) modules: validate WGSL output after instantiating with the given concrete types. Can be repeated. |
///
/// Options can be combined: `#[wgsl(crate_path = my_crate::wgsl_rs,
/// skip_validation)]`.
///
/// # Auto-generated Validation Tests
///
/// Every non-template `#[wgsl]` module gets an auto-generated `__validate_wgsl`
/// test that calls `WGSL_MODULE.validate()` at `cargo test` time. Template
/// (generic) modules cannot be validated standalone because their type
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
/// ## `slab_read_array!` / `slab_write_array!`
///
/// Bulk read/write operations on storage buffer slabs.
///
/// ```ignore
/// slab_read_array!(get!(SLAB), offset, dest_array, count);
/// slab_write_array!(get_mut!(SLAB), offset, src_array, count);
/// ```
///
/// These expand to indexed for-loops in WGSL.
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
    token_stream
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
/// // Depth textures (no type parameter)
/// texture!(group(G), binding(B), NAME: TextureDepth2D);
/// texture!(group(G), binding(B), NAME: TextureDepthCube);
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
/// Where `T` is one of `f32`, `i32`, or `u32`.
///
/// ## Depth Textures
/// - `TextureDepth2D` - 2D depth texture
/// - `TextureDepth2DArray` - 2D depth texture array
/// - `TextureDepthCube` - Cube depth texture
/// - `TextureDepthCubeArray` - Cube depth texture array
/// - `TextureDepthMultisampled2D` - Multisampled 2D depth texture
///
/// # WGSL Output
/// The macro transpiles to:
/// - `@group(G) @binding(B) var NAME: texture_2d<f32>;` for sampled textures
/// - `@group(G) @binding(B) var NAME: texture_depth_2d;` for depth textures
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
