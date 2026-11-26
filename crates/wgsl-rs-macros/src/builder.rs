//! `instantiate` function code generation for generic `#[wgsl]` modules.
//!
//! When a `#[wgsl]` module has module-level type parameters (from `impl Trait`
//! linkage variables and/or generic entry points), the proc macro emits an
//! `instantiate` function alongside `WGSL_SOURCE`. The function uses
//! `wgsl_rs::linkage::Type<Is = ...>` trait constraints to enforce at compile
//! time that every linkage variable's concrete type is consistent across all
//! entry points that use it.
//!
//! For a module like
//!
//! ```ignore
//! #[wgsl]
//! pub mod my_shader {
//!     uniform!(group(0), binding(0), FRAME: impl Convert<f32>);
//!
//!     #[fragment]
//!     pub fn frag_main<T: Convert<f32> + Clone>() -> Vec4f {
//!         vec4f(1.0, sin(f32(get!(FRAME, T)) / 128.0), 0.0, 1.0)
//!     }
//! }
//! ```
//!
//! the proc macro emits:
//!
//! ```ignore
//! pub fn instantiate<T, FRAME>() -> wgsl_rs::ir::Module
//! where
//!     T: Convert<f32> + Clone + wgsl_rs::std::Wgsl,
//!     FRAME: Convert<f32> + wgsl_rs::std::Wgsl,
//!     FRAME: wgsl_rs::linkage::Type<Is = T>,
//! {
//!     let mut ir_module = WGSL_SOURCE
//!         .assemble_ir()
//!         .expect("failed to assemble IR for wgsl module 'my_shader'");
//!     let __subst: ::std::collections::HashMap<::std::string::String, wgsl_rs::ir::Type> = [
//!         ("FRAME".to_string(), <FRAME as wgsl_rs::std::Wgsl>::to_ir()),
//!         ("frag_main_0".to_string(), <T as wgsl_rs::std::Wgsl>::to_ir()),
//!     ].into_iter().collect();
//!     wgsl_rs::ir::substitute_types(&mut ir_module, &__subst);
//!     ir_module
//! }
//! ```
//!
//! The `FRAME: Type<Is = T>` constraint means the Rust compiler will reject
//! any call where `FRAME`'s concrete type differs from `T`'s concrete type.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse;

/// A linkage variable slot: an `impl Trait` declaration (uniform, storage, or
/// workgroup) that becomes a generic type parameter on `instantiate`.
struct LinkageSlot {
    /// The variable name (e.g. `FRAME`, `BINS`). Used as the generic type
    /// parameter ident.
    name: syn::Ident,
    /// The `impl Trait1 + Trait2 + ...` bounds from the declaration.
    bounds: syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    /// The module-level type parameter name. For linkage variables this is
    /// the same as the variable name (e.g. `"FRAME"`).
    encoded_name: String,
}

/// A binding-owned const parameter used by an address or filterability value.
struct LinkageConstSlot {
    /// The const parameter ident used on `instantiate`.
    name: syn::Ident,
    /// The substitution key stored in `Source::module_type_params`.
    encoded_name: String,
    /// The declared const type (`bool`, `u32`, or `usize`).
    ty: syn::Type,
}

/// A resource-pack type-parameter slot. Each type parameter of each
/// registered pack becomes one generic type parameter on `instantiate`,
/// substituted into the pack's field types by the type substitution map.
/// Entry-point type arguments tie their concrete types to the slot through
/// `linkage::Type<Is = ...>` constraints, mirroring `get!`/`get_mut!`.
struct PackTypeSlot {
    /// The pack-owned slot ident (e.g. `FramePack_T`).
    name: syn::Ident,
    /// The bounds from the pack struct's declaration, replayed on
    /// `instantiate`.
    bounds: syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
}

/// An entry-point type-parameter slot. Each type parameter of each entry
/// point function becomes a generic type parameter on `instantiate`.
#[allow(dead_code)]
struct EntryPointSlot {
    /// The type parameter ident used on the `instantiate` function.
    /// This may be disambiguated from the original name if there are
    /// collisions (e.g. `T_main_array` instead of `T`).
    name: syn::Ident,
    /// The original type parameter ident from the source code (e.g. `T`).
    /// Used to find bounds in the function's generics.
    original_name: syn::Ident,
    /// The function's full `syn::Generics` (for replaying bounds).
    generics: syn::Generics,
    /// The positional encoded name (e.g. `"frag_main_0"`).
    encoded_name: String,
    /// The parent function's name (for deduplication / diagnostics).
    fn_name: String,
}

/// An entry-point const-parameter slot. Each supported const parameter
/// (`const N: u32`, `usize`, `i32`, or `bool`) of each entry point function
/// becomes a generic const parameter on `instantiate`. Const params use a
/// separate positional namespace (`{fn}_c{i}`) from type params
/// (`{fn}_{i}`) so they don't collide.
#[allow(dead_code)]
struct EntryPointConstSlot {
    /// The const parameter ident used on the `instantiate` function.
    name: syn::Ident,
    /// The positional encoded name (e.g. `"frag_main_c0"`).
    encoded_name: String,
    /// The parent function's name (for diagnostics).
    #[allow(dead_code)]
    fn_name: String,
    /// The declared const-param type (`u32`, `usize`, `i32`, or `bool`),
    /// replayed verbatim on `instantiate` so rustc type-checks
    /// instantiation values at the call site.
    ty: syn::Type,
}

/// Generate an `instantiate` function for the given module. Returns an empty
/// token stream when the module has no module-level type parameters (i.e.
/// no `impl Trait` linkage variables and no generic entry points).
pub(crate) fn gen_builder(crate_path: &syn::Path, wgsl_module: &parse::ItemMod) -> TokenStream {
    let ir_p = quote! { #crate_path::ir };
    let module_name = wgsl_module.ident.to_string();
    let expect_msg = format!("failed to assemble IR for wgsl module '{module_name}'");
    let std_p = quote! { #crate_path::std };
    let linkage_p = quote! { #crate_path::linkage };

    let mut linkage_slots: Vec<LinkageSlot> = Vec::new();
    let mut linkage_const_slots: Vec<LinkageConstSlot> = Vec::new();
    let mut pack_type_slots: Vec<PackTypeSlot> = Vec::new();
    let mut ep_slots: Vec<EntryPointSlot> = Vec::new();
    let mut ep_const_slots: Vec<EntryPointConstSlot> = Vec::new();

    // Collect linkage variable slots.
    for item in &wgsl_module.content {
        let (name, bounds_opt) = match item {
            parse::Item::Uniform(u) => (u.name.clone(), u.impl_bounds.as_ref()),
            parse::Item::Storage(s) => (s.name.clone(), s.impl_bounds.as_ref()),
            parse::Item::Workgroup(w) => (w.name.clone(), w.impl_bounds.as_ref()),
            _ => continue,
        };
        let Some(bounds) = bounds_opt else { continue };
        linkage_slots.push(LinkageSlot {
            name: name.clone(),
            bounds: bounds.clone(),
            encoded_name: name.to_string(),
        });
    }

    // Collect resource-pack type-parameter slots in pack declaration order,
    // then parameter declaration order. Every type parameter of a
    // registered pack becomes one module-level generic; `parse_resource_pack`
    // guarantees each is referenced by a field, so the slot always carries a
    // substitution through the field's type.
    for pack in wgsl_module.resource_packs() {
        for param in &pack.type_params {
            let slot = crate::resource::pack_slot_name(&pack.ident, &param.ident);
            pack_type_slots.push(PackTypeSlot {
                name: syn::Ident::new(&slot, param.ident.span()),
                bounds: param.bounds.clone(),
            });
        }
    }

    // Collect binding-owned consts in first-occurrence declaration order.
    // Repeated standalone filterability names intentionally share one slot.
    let pack_plans = crate::resource::pack_const_plans(wgsl_module)
        .expect("resource-pack const uses were validated during lowering");
    let mut seen_const_slots = std::collections::HashSet::new();
    for (name, ty, _span, pack) in parse::binding_const_params(wgsl_module) {
        if pack.as_ref().is_some_and(|pack_name| {
            pack_plans.iter().any(|plan| {
                plan.pack == *pack_name
                    && plan.source == name
                    && matches!(plan.kind, crate::resource::PackConstKind::Fixed(_))
            })
        }) {
            continue;
        }
        if !seen_const_slots.insert(name.clone()) {
            continue;
        }
        linkage_const_slots.push(LinkageConstSlot {
            name: format_ident!("{name}"),
            encoded_name: name,
            ty,
        });
    }

    // Collect entry-point type-parameter slots.
    // Always suffix type params with their function name so the names
    // are predictable and readable (e.g. `T_main_array` rather than
    // bare `T` that becomes `T_main_zeroable` on collision).
    for item in &wgsl_module.content {
        let parse::Item::Fn(f) = item else { continue };
        if matches!(f.fn_attrs, parse::FnAttrs::None) {
            continue;
        }
        let Some(syn_generics) = &f.syn_generics else {
            continue;
        };
        let fn_name_str = f.ident.to_string();
        for (i, tp) in f.type_params.iter().enumerate() {
            // Entry-point slot encoding `{fn}_{i}` is intentionally not
            // run through `wgsl_rs_ir::mangle`; see comment in
            // `parse.rs` near the matching site and issue #112.
            let encoded = format!("{fn_name_str}_{i}");
            let name = format_ident!("{}_{}", tp, fn_name_str);
            ep_slots.push(EntryPointSlot {
                name,
                original_name: tp.clone(),
                generics: syn_generics.clone(),
                encoded_name: encoded,
                fn_name: fn_name_str.clone(),
            });
        }
        // Collect const-param slots. Const params use a separate
        // `{fn}_c{i}` namespace so they don't collide with type params
        // on the same function.
        for (i, cp) in f.const_params.iter().enumerate() {
            let encoded = format!("{fn_name_str}_c{i}");
            // Uppercase the const param name to satisfy
            // `non_upper_case_globals`. The generated ident is
            // `{CP}_C{i}_{FN}` (e.g. `N_C0_MAIN` for `const N: usize` on
            // `main`).
            let upper_cp = cp.to_string().to_uppercase();
            let upper_fn = fn_name_str.to_uppercase();
            let name = format_ident!("{}_C{}_{}", upper_cp, i, upper_fn);
            // Replay the declared const-param type (`u32`, `usize`,
            // `i32`, or `bool`) verbatim; see the `ty` field on
            // `EntryPointConstSlot`. The lookup can only fail if
            // `const_params` and `syn_generics` disagree, which would be
            // a parser bug — not something to paper over with a
            // fallback type.
            let ty = f
                .syn_generics
                .as_ref()
                .and_then(|g| {
                    g.params.iter().find_map(|p| match p {
                        syn::GenericParam::Const(ccp) if ccp.ident == *cp => Some(ccp.ty.clone()),
                        _ => None,
                    })
                })
                .expect("const param ident must appear in the entry point's syn generics");
            ep_const_slots.push(EntryPointConstSlot {
                name,
                encoded_name: encoded,
                fn_name: fn_name_str.clone(),
                ty,
            });
        }
    }

    if linkage_slots.is_empty()
        && linkage_const_slots.is_empty()
        && pack_type_slots.is_empty()
        && ep_slots.is_empty()
        && ep_const_slots.is_empty()
    {
        return quote! {};
    }

    // Collect linkage constraints from get!/get_mut! calls and from the
    // type arguments of entry-point pack parameters.
    let mut constraints = collect_linkage_constraints(wgsl_module);
    constraints.extend(collect_pack_type_constraints(wgsl_module));

    // Build the generic params and where clause.
    let mut generic_params: Vec<syn::GenericParam> = Vec::new();
    let mut where_predicates: Vec<syn::WherePredicate> = Vec::new();

    let wgsl_bound = syn::TypeParamBound::Trait(syn::TraitBound {
        paren_token: None,
        modifier: syn::TraitBoundModifier::None,
        lifetimes: None,
        path: syn::parse2(quote! { #std_p::Wgsl }).expect("Wgsl is a valid path"),
    });

    // Add linkage variable generic params (e.g. <FRAME>).
    for slot in &linkage_slots {
        let mut tp: syn::TypeParam = syn::TypeParam::from(slot.name.clone());
        tp.bounds = slot.bounds.clone();
        push_wgsl_if_missing(&mut tp.bounds, &wgsl_bound);
        generic_params.push(syn::GenericParam::Type(tp));
    }

    // Add resource-pack type parameter generic params (e.g.
    // `<FramePack_T: Convert<f32>>`), replaying the pack struct's bounds.
    for slot in &pack_type_slots {
        let mut tp: syn::TypeParam = syn::TypeParam::from(slot.name.clone());
        tp.bounds = slot.bounds.clone();
        push_wgsl_if_missing(&mut tp.bounds, &wgsl_bound);
        generic_params.push(syn::GenericParam::Type(tp));
    }

    // Add binding-owned bool consts before all entry-point parameters.
    for slot in &linkage_const_slots {
        generic_params.push(syn::GenericParam::Const(syn::ConstParam {
            attrs: Vec::new(),
            const_token: syn::Token![const](proc_macro2::Span::call_site()),
            ident: slot.name.clone(),
            colon_token: syn::Token![:](proc_macro2::Span::call_site()),
            ty: slot.ty.clone(),
            eq_token: None,
            default: None,
        }));
    }

    // Add entry-point type parameter generic params (e.g. <T>).
    for slot in &ep_slots {
        let mut tp: syn::TypeParam = syn::TypeParam::from(slot.name.clone());
        // Replay the bounds from the function's generics, plus Wgsl.
        if let Some(syn::GenericParam::Type(orig_tp)) = slot
            .generics
            .params
            .iter()
            .find(|p| matches!(p, syn::GenericParam::Type(t) if t.ident == slot.original_name))
        {
            tp.bounds = orig_tp.bounds.clone();
        }
        push_wgsl_if_missing(&mut tp.bounds, &wgsl_bound);
        generic_params.push(syn::GenericParam::Type(tp));
    }

    // Add entry-point const parameter generic params (e.g.
    // `<const N_FRAG_MAIN_C0: usize>`). Const params cannot have trait
    // bounds, so no Wgsl bound is added. The declared const-param type
    // (`u32`, `usize`, or `i32`) is replayed verbatim so rustc
    // type-checks instantiation values at the call site; the values are
    // widened to `i64` when building the substitution map below.
    for slot in &ep_const_slots {
        if crate::resource::is_shared_pack_alias(&pack_plans, &slot.encoded_name) {
            continue;
        }
        let cp = syn::ConstParam {
            attrs: Vec::new(),
            const_token: syn::Token![const](proc_macro2::Span::call_site()),
            ident: slot.name.clone(),
            colon_token: syn::Token![:](proc_macro2::Span::call_site()),
            ty: slot.ty.clone(),
            eq_token: None,
            default: None,
        };
        generic_params.push(syn::GenericParam::Const(cp));
    }

    // Add Type<Is = ...> constraints from get!/get_mut! calls.
    // For each constraint, build a rename map scoped to its entry point
    // function so that type params are disambiguated correctly (e.g.
    // `T` in `main_zeroable`'s constraint becomes `T_main_zeroable`,
    // not `T_main_array`).
    for constraint in &constraints {
        let linkage_name = format_ident!("{}", constraint.linkage_name);
        let linkage_ty: syn::Type = syn::parse_quote!(#linkage_name);

        let renames: Vec<(String, syn::Ident)> = ep_slots
            .iter()
            .filter(|slot| slot.fn_name == constraint.fn_name)
            .filter(|slot| slot.name != slot.original_name)
            .map(|slot| (slot.original_name.to_string(), slot.name.clone()))
            .collect();

        let mut rhs_ty = constraint.type_expr.clone();
        rename_type_params_in_syn_type(&mut rhs_ty, &renames);

        where_predicates.push(syn::WherePredicate::Type(syn::PredicateType {
            lifetimes: None,
            bounded_ty: linkage_ty,
            colon_token: syn::Token![:](proc_macro2::Span::call_site()),
            bounds: vec![syn::TypeParamBound::Trait(syn::TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: syn::parse2(quote! { #linkage_p::Type<Is = #rhs_ty> })
                    .expect("Type<Is = ...> should be a valid path"),
            })]
            .into_iter()
            .collect(),
        }));
    }

    // Build the where clause.
    let where_clause = if where_predicates.is_empty() {
        None
    } else {
        let wc = syn::WhereClause {
            where_token: syn::Token![where](proc_macro2::Span::call_site()),
            predicates: where_predicates.into_iter().collect(),
        };
        Some(wc)
    };

    // Build the substitution-map entries.
    let mut subst_entries: Vec<TokenStream> = Vec::new();

    for slot in &linkage_slots {
        let name = &slot.name;
        let encoded_name_str = &slot.encoded_name;
        subst_entries.push(quote! {
            (::std::string::String::from(#encoded_name_str), <#name as #std_p::Wgsl>::to_ir())
        });
    }

    for slot in &pack_type_slots {
        let name = &slot.name;
        let encoded_name_str = name.to_string();
        subst_entries.push(quote! {
            (::std::string::String::from(#encoded_name_str), <#name as #std_p::Wgsl>::to_ir())
        });
    }

    for slot in &ep_slots {
        let name = &slot.name;
        let encoded_name_str = &slot.encoded_name;
        subst_entries.push(quote! {
            (::std::string::String::from(#encoded_name_str), <#name as #std_p::Wgsl>::to_ir())
        });
    }

    // Build the const substitution-map entries (parallel to the type
    // subst above). Const params are passed directly and wrapped in
    // `ConstValue` for the substitution map (`Int` via `as i64` for the
    // integer types, `Bool` for `bool`); no trait dispatch is needed
    // (unlike the type-param case which uses `<T as Wgsl>::to_ir()`).
    let mut const_subst_entries: Vec<TokenStream> = Vec::new();
    for slot in &linkage_const_slots {
        let name = &slot.name;
        let encoded_name_str = &slot.encoded_name;
        let value = if matches!(&slot.ty, syn::Type::Path(path) if path.path.is_ident("bool")) {
            quote! { #ir_p::ConstValue::Bool(#name) }
        } else {
            quote! { #ir_p::ConstValue::Int(#name as i64) }
        };
        const_subst_entries.push(quote! {
            (::std::string::String::from(#encoded_name_str), #value)
        });
    }
    for slot in &ep_const_slots {
        let name = &slot.name;
        let encoded_name_str = &slot.encoded_name;
        let is_bool = matches!(
            &slot.ty,
            syn::Type::Path(p) if p.qself.is_none() && p.path.is_ident("bool")
        );
        let shared = pack_plans.iter().find(|plan| {
            matches!(plan.kind, crate::resource::PackConstKind::Shared)
                && plan.aliases.iter().any(|alias| alias == &slot.encoded_name)
        });
        let value_name = shared
            .map(|plan| format_ident!("{}", plan.source))
            .unwrap_or_else(|| name.clone());
        let value = if is_bool {
            quote! { #ir_p::ConstValue::Bool(#value_name) }
        } else {
            quote! { #ir_p::ConstValue::Int(#value_name as i64) }
        };
        const_subst_entries.push(quote! {
            (::std::string::String::from(#encoded_name_str), #value)
        });
    }

    // Assemble the generic params.
    let has_generics = !generic_params.is_empty();
    let generics = syn::Generics {
        lt_token: if has_generics {
            Some(syn::Token![<](proc_macro2::Span::call_site()))
        } else {
            None
        },
        params: generic_params.into_iter().collect(),
        gt_token: if has_generics {
            Some(syn::Token![>](proc_macro2::Span::call_site()))
        } else {
            None
        },
        where_clause,
    };
    let (impl_generics, _type_generics, split_where_clause) = generics.split_for_impl();

    let validate_pack_bindings = if wgsl_module.resource_packs().next().is_some() {
        quote! {
            if let Err(error) = #ir_p::validate_binding_addresses(&__ir_module) {
                panic!("{error}");
            }
        }
    } else {
        quote! {}
    };

    quote! {
        /// Instantiate this module's generic type parameters with concrete types.
        ///
        /// The `where` clause enforces at compile time that every linkage
        /// variable's concrete type is consistent across all entry points
        /// that use it via [`wgsl_rs::linkage::Type`] constraints.
        #[allow(non_camel_case_types, non_upper_case_globals)]
        pub fn instantiate #impl_generics () -> #ir_p::Module
        #split_where_clause
        {
            let mut __ir_module = WGSL_SOURCE
                .assemble_ir()
                .expect(#expect_msg);
            let __subst: ::std::collections::HashMap<::std::string::String, #ir_p::Type> = [
                #(#subst_entries),*
            ].into_iter().collect();
            #ir_p::substitute_types(&mut __ir_module, &__subst);
            let __const_subst: ::std::collections::HashMap<::std::string::String, #ir_p::ConstValue> = [
                #(#const_subst_entries),*
            ].into_iter().collect();
            #ir_p::substitute_consts(&mut __ir_module, &__const_subst);
            #validate_pack_bindings
            #ir_p::suffix_module(&mut __ir_module);
            __ir_module
        }
    }
}

/// Collect the `Type<Is = ...>` constraints contributed by entry-point pack
/// type arguments. Each type argument of each resource-pack parameter ties
/// the pack's module-level slot to the concrete type used at that entry
/// point; the builder renames the argument's function type parameters to
/// their entry-point slot idents when emitting the predicate.
fn collect_pack_type_constraints(wgsl_module: &parse::ItemMod) -> Vec<LinkageConstraint> {
    let mut constraints = Vec::new();
    for usage in &wgsl_module.resource_pack_uses {
        let Some(pack) = wgsl_module.resource_pack(&usage.pack) else {
            continue;
        };
        for (param, arg) in pack.type_params.iter().zip(&usage.type_args) {
            let type_expr = match arg {
                parse::PackTypeArg::Param(ident) => syn::parse_quote!(#ident),
                parse::PackTypeArg::Concrete(ty) => ty.as_ref().clone(),
            };
            constraints.push(LinkageConstraint {
                linkage_name: crate::resource::pack_slot_name(&pack.ident, &param.ident),
                type_expr,
                fn_name: usage.fn_name.clone(),
                is_entry_point: true,
                fn_type_params: Vec::new(),
            });
        }
    }
    constraints
}

/// Collect all `get!(VAR, TYPE)` / `get_mut!(VAR, TYPE)` constraints from
/// entry-point functions in the module.
///
/// Only entry points are included (helper functions are deferred to a future
/// iteration that includes call graph analysis).
fn collect_linkage_constraints(wgsl_module: &parse::ItemMod) -> Vec<LinkageConstraint> {
    use parse::FnAttrs;
    let mut constraints: Vec<LinkageConstraint> = Vec::new();

    for item in &wgsl_module.content {
        let parse::Item::Fn(f) = item else { continue };
        if matches!(f.fn_attrs, FnAttrs::None) {
            continue;
        }
        let fn_name = f.ident.to_string();
        let fn_type_params: Vec<String> = f.type_params.iter().map(|id| id.to_string()).collect();
        walk_expr_for_linkage_constraints(
            &f.block.stmt,
            &fn_name,
            true,
            &fn_type_params,
            &mut constraints,
        );
    }

    constraints
}

/// A type constraint collected from a `get!(VAR, TYPE)` or `get_mut!(VAR,
/// TYPE)` usage within a function body.
#[allow(dead_code)]
struct LinkageConstraint {
    /// The linkage variable name (e.g., "BINS", "FRAME").
    linkage_name: String,
    /// The type expression used at the call site (e.g., `Vec4<T>`, `f32`, `T`).
    /// Stored as `syn::Type` because it's a Rust-side type expression used for
    /// code generation (the `instantiate` function's `where` clause), not a
    /// WGSL type.
    type_expr: syn::Type,
    /// The name of the function where this constraint was found.
    fn_name: String,
    /// Whether the function is an entry point.
    is_entry_point: bool,
    /// Type parameters in scope at this call site.
    fn_type_params: Vec<String>,
}

fn walk_else_for_linkage_constraints(
    else_branch: &parse::ElseBranch,
    fn_name: &str,
    is_entry_point: bool,
    fn_type_params: &[String],
    constraints: &mut Vec<LinkageConstraint>,
) {
    match &else_branch.body {
        parse::ElseBody::Block(block) => {
            walk_expr_for_linkage_constraints(
                &block.stmt,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::ElseBody::If(nested_if) => {
            walk_expr_for_constraints_recursive(
                &nested_if.condition,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
            walk_expr_for_linkage_constraints(
                &nested_if.then_block.stmt,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
            if let Some(nested_else) = &nested_if.else_branch {
                walk_else_for_linkage_constraints(
                    nested_else,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
        }
    }
}

/// Recursively walk a block of statements, collecting `LinkageAccess`
/// constraints.
fn walk_expr_for_linkage_constraints(
    stmts: &[parse::Stmt],
    fn_name: &str,
    is_entry_point: bool,
    fn_type_params: &[String],
    constraints: &mut Vec<LinkageConstraint>,
) {
    use parse::Stmt;
    for stmt in stmts {
        match stmt {
            Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    walk_expr_for_constraints_recursive(
                        &init.expr,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
            Stmt::Expr { expr, .. } => {
                walk_expr_for_constraints_recursive(
                    expr,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::If(parse_if) => {
                walk_expr_for_constraints_recursive(
                    &parse_if.condition,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_linkage_constraints(
                    &parse_if.then_block.stmt,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                if let Some(else_branch) = &parse_if.else_branch {
                    walk_else_for_linkage_constraints(
                        else_branch,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
            Stmt::For(parse_for) => {
                walk_expr_for_constraints_recursive(
                    &parse_for.from,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_constraints_recursive(
                    &parse_for.to,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_linkage_constraints(
                    &parse_for.body.stmt,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::Return { expr, .. } => {
                if let Some(expr) = expr {
                    walk_expr_for_constraints_recursive(
                        expr,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
            Stmt::Assignment { lhs, rhs, .. } => {
                walk_expr_for_constraints_recursive(
                    lhs,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_constraints_recursive(
                    rhs,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::CompoundAssignment { lhs, rhs, .. } => {
                walk_expr_for_constraints_recursive(
                    lhs,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_constraints_recursive(
                    rhs,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::While {
                condition, body, ..
            } => {
                walk_expr_for_constraints_recursive(
                    condition,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                walk_expr_for_linkage_constraints(
                    &body.stmt,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::Loop { body, .. } => {
                walk_expr_for_linkage_constraints(
                    &body.stmt,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::Block(block) => {
                walk_expr_for_linkage_constraints(
                    &block.stmt,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
            Stmt::Switch(switch) => {
                walk_expr_for_constraints_recursive(
                    &switch.selector,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
                for arm in &switch.arms {
                    for sel in &arm.selectors {
                        if let parse::CaseSelector::Expr(e) = sel {
                            walk_expr_for_constraints_recursive(
                                e,
                                fn_name,
                                is_entry_point,
                                fn_type_params,
                                constraints,
                            );
                        }
                    }
                    walk_expr_for_linkage_constraints(
                        &arm.body.stmt,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
            Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::Const(_) // `get!`/`get_mut!` in const initializers would
                            // produce invalid WGSL (const exprs can't access
                            // storage/uniform buffers); the parser rejects this.
            | Stmt::Discard { .. }
            | Stmt::Macro { .. } => {}
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
                ..
            } => {
                for expr in [src, src_offset, dest, dest_offset, size] {
                    walk_expr_for_constraints_recursive(
                        expr,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
        }
    }
}

/// Walk an expression tree, collecting `LinkageAccess` constraints when we find
/// `get!(VAR, TYPE)` or `get_mut!(VAR, TYPE)` with a type argument.
fn walk_expr_for_constraints_recursive(
    expr: &parse::Expr,
    fn_name: &str,
    is_entry_point: bool,
    fn_type_params: &[String],
    constraints: &mut Vec<LinkageConstraint>,
) {
    match expr {
        parse::Expr::LinkageAccess {
            ident,
            type_arg: Some(ty),
            ..
        } => {
            constraints.push(LinkageConstraint {
                linkage_name: ident.to_string(),
                type_expr: ty.clone(),
                fn_name: fn_name.to_string(),
                is_entry_point,
                fn_type_params: fn_type_params.to_vec(),
            });
        }
        parse::Expr::Binary { lhs, rhs, .. } => {
            walk_expr_for_constraints_recursive(
                lhs,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
            walk_expr_for_constraints_recursive(
                rhs,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Unary { expr, .. } => {
            walk_expr_for_constraints_recursive(
                expr,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Paren { inner, .. } => {
            walk_expr_for_constraints_recursive(
                inner,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Array { elems, .. } => {
            for elem in elems.iter() {
                walk_expr_for_constraints_recursive(
                    elem,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
        }
        parse::Expr::ArrayIndexing { lhs, index, .. } => {
            walk_expr_for_constraints_recursive(
                lhs,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
            walk_expr_for_constraints_recursive(
                index,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Swizzle { lhs, params, .. } => {
            walk_expr_for_constraints_recursive(
                lhs,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
            if let Some(params) = params {
                for param in params.iter() {
                    walk_expr_for_constraints_recursive(
                        param,
                        fn_name,
                        is_entry_point,
                        fn_type_params,
                        constraints,
                    );
                }
            }
        }
        parse::Expr::Cast { lhs, .. } => {
            walk_expr_for_constraints_recursive(
                lhs,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::FnCall { params, .. } => {
            for param in params.iter() {
                walk_expr_for_constraints_recursive(
                    param,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
        }
        parse::Expr::Struct { fields, .. } => {
            for field in fields.iter() {
                walk_expr_for_constraints_recursive(
                    &field.expr,
                    fn_name,
                    is_entry_point,
                    fn_type_params,
                    constraints,
                );
            }
        }
        parse::Expr::FieldAccess { base, .. } => {
            walk_expr_for_constraints_recursive(
                base,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Reference { expr, .. } => {
            walk_expr_for_constraints_recursive(
                expr,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::ZeroValueArray { len, .. } => {
            walk_expr_for_constraints_recursive(
                len,
                fn_name,
                is_entry_point,
                fn_type_params,
                constraints,
            );
        }
        parse::Expr::Lit(_) | parse::Expr::Ident(_) | parse::Expr::TypePath { .. } => {}
        parse::Expr::LinkageAccess { type_arg: None, .. } => {}
    }
}

/// Push the `Wgsl` trait bound onto `bounds` only if it isn't already present.
/// This avoids duplicate bounds like `T: Wgsl + crate::std::Wgsl`.
fn push_wgsl_if_missing(
    bounds: &mut syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    wgsl_bound: &syn::TypeParamBound,
) {
    let syn::TypeParamBound::Trait(wgsl_trait) = wgsl_bound else {
        bounds.push(wgsl_bound.clone());
        return;
    };
    let wgsl_path = &wgsl_trait.path;
    let already_has_wgsl = bounds.iter().any(|b| {
        if let syn::TypeParamBound::Trait(tb) = b {
            paths_match(&tb.path, wgsl_path)
        } else {
            false
        }
    });
    if !already_has_wgsl {
        bounds.push(wgsl_bound.clone());
    }
}

/// Check if two syn paths refer to the same trait, comparing only the
/// final segment (the trait name). This handles the case where one path
/// is `Wgsl` (unqualified, from the user's code) and the other is
/// `crate_path::std::Wgsl` (fully qualified, from our code generation).
fn paths_match(a: &syn::Path, b: &syn::Path) -> bool {
    let a_last = a.segments.last();
    let b_last = b.segments.last();
    match (a_last, b_last) {
        (Some(sa), Some(sb)) => sa.ident == sb.ident,
        _ => false,
    }
}

/// Walk a `syn::Type` and replace any path segments whose `ident` matches
/// an entry in `renames` with the disambiguated ident. This ensures that
/// type params like `T` in a constraint expression get replaced by their
/// disambiguated names (e.g. `T_main_array`) when multiple entry points
/// use the same letter.
fn rename_type_params_in_syn_type(ty: &mut syn::Type, renames: &[(String, syn::Ident)]) {
    if renames.is_empty() {
        return;
    }
    match ty {
        syn::Type::Path(type_path) => {
            // A single-segment path like `T` or `Vec4` may be a type param.
            // A multi-segment path like `crate::T` is less likely, but we
            // check the last segment as the most common case.
            for segment in &mut type_path.path.segments {
                for (original, replacement) in renames {
                    if segment.ident == original {
                        segment.ident = replacement.clone();
                    }
                }
                // Also rename type arguments inside angle brackets (e.g. `Vec4<T>`)
                if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            rename_type_params_in_syn_type(inner_ty, renames);
                        }
                    }
                }
            }
        }
        syn::Type::Reference(type_ref) => {
            rename_type_params_in_syn_type(&mut type_ref.elem, renames);
        }
        syn::Type::Paren(type_paren) => {
            rename_type_params_in_syn_type(&mut type_paren.elem, renames);
        }
        syn::Type::Slice(type_slice) => {
            rename_type_params_in_syn_type(&mut type_slice.elem, renames);
        }
        syn::Type::Array(type_array) => {
            rename_type_params_in_syn_type(&mut type_array.elem, renames);
        }
        syn::Type::Group(type_group) => {
            rename_type_params_in_syn_type(&mut type_group.elem, renames);
        }
        syn::Type::Tuple(type_tuple) => {
            for elem in &mut type_tuple.elems {
                rename_type_params_in_syn_type(elem, renames);
            }
        }
        syn::Type::Ptr(type_ptr) => {
            rename_type_params_in_syn_type(&mut type_ptr.elem, renames);
        }
        _ => {}
    }
}
