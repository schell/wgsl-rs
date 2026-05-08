//! Typestate-builder code generation for generic `#[wgsl]` modules.
//!
//! When a `#[wgsl]` module declares one or more `impl Trait` linkage
//! variables and/or generic shader entry points, the proc macro emits a
//! `ModuleBuilder` struct alongside `WGSL_MODULE`. The builder uses a
//! typestate pattern to ensure that every type parameter has been bound
//! to a concrete type before [`build()`] is called.
//!
//! For a module like
//!
//! ```ignore
//! #[wgsl]
//! pub mod my_shader {
//!     uniform!(group(0), binding(0), FRAME: impl Convert<f32>);
//!
//!     #[fragment]
//!     pub fn frag_main<T: Convert<f32> + Clone, S: Wgsl>() -> Vec4f { ... }
//! }
//! ```
//!
//! the proc macro emits:
//!
//! ```ignore
//! pub struct NeedsFRAME;
//! pub struct HasFRAME;
//! pub struct NeedsFragMain;
//! pub struct HasFragMain;
//!
//! pub struct ModuleBuilder<S0 = NeedsFRAME, S1 = NeedsFragMain> { /* ... */ }
//!
//! impl ModuleBuilder<NeedsFRAME, NeedsFragMain> {
//!     pub fn new() -> Self { /* ... */ }
//! }
//!
//! // Linkage var setter — bounds replayed from `impl Convert<f32>`.
//! impl<S1> ModuleBuilder<NeedsFRAME, S1> {
//!     pub fn set_frame<FRAME: Convert<f32> + Wgsl>(self)
//!         -> ModuleBuilder<HasFRAME, S1> { /* ... */ }
//! }
//!
//! // Entry-point instantiator — bounds replayed from the fn signature.
//! impl<S0> ModuleBuilder<S0, NeedsFragMain> {
//!     pub fn instantiate_frag_main<T: Convert<f32> + Clone + Wgsl, S: Wgsl>(self)
//!         -> ModuleBuilder<S0, HasFragMain> { /* ... */ }
//! }
//!
//! // Terminal build only available when every state is satisfied.
//! impl ModuleBuilder<HasFRAME, HasFragMain> {
//!     pub fn build(self) -> wgsl_rs::ir::Module { /* ... */ }
//! }
//! ```
//!
//! `Wgsl` is added as an extra bound on every generic parameter so the
//! builder can call `<T as Wgsl>::to_ir()` to materialise an `ir::Type`
//! at runtime.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse;

/// One slot in the typestate tuple.
struct Slot {
    /// Identifier for the slot's "needs" state marker (e.g. `NeedsFRAME`).
    needs_ty: syn::Ident,
    /// Identifier for the slot's "has" state marker (e.g. `HasFRAME`).
    has_ty: syn::Ident,
    /// Code that, given `self` (the builder) and producing a new builder,
    /// pushes the encoded `(name, ir::Type)` substitution entries for this
    /// slot. Generated as the body of the setter / instantiator method.
    method: TokenStream,
}

/// Emit a `ModuleBuilder` typestate struct and its impls for the given
/// parsed module. Returns an empty token stream when the module has no
/// `impl Trait` linkage variables and no generic entry points.
pub(crate) fn gen_builder(crate_path: &syn::Path, wgsl_module: &parse::ItemMod) -> TokenStream {
    let ir_p = quote! { #crate_path::ir };
    let std_p = quote! { #crate_path::std };

    let mut slots: Vec<Slot> = Vec::new();

    // Slots 0..N — `impl Trait` linkage variables.
    for item in &wgsl_module.content {
        let (var_name, bounds_opt) = match item {
            parse::Item::Uniform(u) => (u.name.clone(), u.impl_bounds.as_ref()),
            parse::Item::Storage(s) => (s.name.clone(), s.impl_bounds.as_ref()),
            parse::Item::Workgroup(w) => (w.name.clone(), w.impl_bounds.as_ref()),
            _ => continue,
        };
        let Some(bounds) = bounds_opt else { continue };

        let needs_ty = format_ident!("Needs{}", var_name);
        let has_ty = format_ident!("Has{}", var_name);
        let setter_name = format_ident!("set_{}", lowercase(&var_name.to_string()));
        let var_name_str = var_name.to_string();
        let type_param_ident = var_name.clone();
        let method = quote! {
            #[doc = concat!(
                "Bind the linkage variable `",
                stringify!(#type_param_ident),
                "` to a concrete type.",
            )]
            pub fn #setter_name<#type_param_ident: #bounds + #std_p::Wgsl>(
                mut self,
            ) -> ModuleBuilder<__MB_REST__> {
                self.subst.push((
                    ::std::string::String::from(#var_name_str),
                    <#type_param_ident as #std_p::Wgsl>::to_ir(),
                ));
                ModuleBuilder {
                    subst: self.subst,
                    _phantom: ::std::marker::PhantomData,
                }
            }
        };
        slots.push(Slot {
            needs_ty,
            has_ty,
            method,
        });
    }

    // Slots N..M — generic entry-point functions.
    for item in &wgsl_module.content {
        let parse::Item::Fn(f) = item else { continue };
        if matches!(f.fn_attrs, parse::FnAttrs::None) {
            continue;
        }
        if f.type_params.is_empty() {
            continue;
        }
        let fn_name = &f.ident;
        let fn_name_str = fn_name.to_string();
        let needs_ty = format_ident!("Needs{}", to_camel_case(&fn_name_str));
        let has_ty = format_ident!("Has{}", to_camel_case(&fn_name_str));
        let method_name = format_ident!("instantiate_{}", fn_name_str);

        // Replay the original syn::Generics on the method, augmenting each
        // type parameter with `+ Wgsl` so we can call `to_ir()` on it.
        let generics = f
            .syn_generics
            .clone()
            .expect("entry-point fn should carry syn_generics");
        let augmented_generics = augment_generics_with_wgsl(generics, &std_p);
        let (impl_generics, _, where_clause) = augmented_generics.split_for_impl();

        // Build the body: push one (encoded_name, T::to_ir()) entry per
        // type parameter.
        let push_stmts: Vec<TokenStream> = f
            .type_params
            .iter()
            .enumerate()
            .map(|(i, tp_ident)| {
                let encoded = format!("{fn_name_str}_{i}");
                quote! {
                    self.subst.push((
                        ::std::string::String::from(#encoded),
                        <#tp_ident as #std_p::Wgsl>::to_ir(),
                    ));
                }
            })
            .collect();

        let method = quote! {
            #[doc = concat!(
                "Instantiate the entry point `",
                stringify!(#fn_name),
                "` with concrete types for its generic parameters.",
            )]
            pub fn #method_name #impl_generics (
                mut self,
            ) -> ModuleBuilder<__MB_REST__>
            #where_clause
            {
                #(#push_stmts)*
                ModuleBuilder {
                    subst: self.subst,
                    _phantom: ::std::marker::PhantomData,
                }
            }
        };
        slots.push(Slot {
            needs_ty,
            has_ty,
            method,
        });
    }

    if slots.is_empty() {
        return quote! {};
    }

    emit(slots, &ir_p)
}

/// Generate the actual builder code given the slot list.
///
/// Each slot lives at a position in a fixed-arity type tuple (encoded as
/// the type parameters of `ModuleBuilder<S0, S1, ...>`). For each slot we
/// emit:
///
/// * a public marker pair `Needs<X> / Has<X>`,
/// * an `impl<other slots...> ModuleBuilder<NeedsX, ...> { fn setter(...) }`
///   block whose return type is the same `ModuleBuilder` with the slot's type
///   changed to `HasX`.
///
/// Finally we emit the `new()` and `build()` methods on the fully-needs
/// and fully-has variants respectively.
fn emit(slots: Vec<Slot>, ir_p: &TokenStream) -> TokenStream {
    let n = slots.len();

    // Marker types (NeedsFRAME, HasFRAME, etc.). We collect them all so
    // the doc-link is unambiguous.
    let markers: Vec<TokenStream> = slots
        .iter()
        .map(|s| {
            let needs = &s.needs_ty;
            let has = &s.has_ty;
            quote! {
                #[doc = "Typestate marker for an unbound builder slot."]
                pub struct #needs;
                #[doc = "Typestate marker for a bound builder slot."]
                pub struct #has;
            }
        })
        .collect();

    // Slot type-parameter idents: `S0, S1, ...`
    let slot_param: Vec<syn::Ident> = (0..n).map(|i| format_ident!("S{i}")).collect();

    // The full default tuple, e.g. `<NeedsFRAME, NeedsFragMain>`.
    let initial_states: Vec<&syn::Ident> = slots.iter().map(|s| &s.needs_ty).collect();
    // The fully-bound tuple, e.g. `<HasFRAME, HasFragMain>`.
    let final_states: Vec<&syn::Ident> = slots.iter().map(|s| &s.has_ty).collect();

    // The struct definition. Defaults the slot type params to the initial
    // states so that `ModuleBuilder` (no turbofish) means "needs all".
    let struct_def = quote! {
        /// Typestate builder for instantiating this module's type
        /// parameters with concrete types.
        ///
        /// Construct via `ModuleBuilder::new()`, call each setter /
        /// `instantiate_*` method exactly once (in any order), and finish
        /// with `build()` to obtain an [`ir::Module`][mod_ir]. The Rust
        /// type system enforces that every slot has been bound before
        /// `build()` is reachable.
        ///
        /// [mod_ir]: ../wgsl_rs_ir/struct.Module.html
        pub struct ModuleBuilder<
            #(#slot_param = #initial_states),*
        > {
            subst: ::std::vec::Vec<(
                ::std::string::String,
                #ir_p::Type,
            )>,
            _phantom: ::std::marker::PhantomData<fn() -> ( #(#slot_param,)* )>,
        }
    };

    // Constructor on the fully-unbound state.
    let constructor = quote! {
        impl ModuleBuilder<#(#initial_states),*> {
            /// Start a fresh builder with no type parameters bound.
            pub fn new() -> Self {
                Self {
                    subst: ::std::vec::Vec::new(),
                    _phantom: ::std::marker::PhantomData,
                }
            }
        }

        impl ::std::default::Default for ModuleBuilder<#(#initial_states),*> {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    // Setter / instantiator impls — one per slot. Each impl block is
    // generic over the *other* slots' states and constrains only its own
    // slot to `Needs<X>`. The return type changes only that slot to
    // `Has<X>`.
    let setter_impls: Vec<TokenStream> = slots
        .iter()
        .enumerate()
        .map(|(idx, slot)| {
            // Build the impl-side and return-side type tuples.
            let mut impl_generics: Vec<TokenStream> = Vec::with_capacity(n - 1);
            let mut self_args: Vec<TokenStream> = Vec::with_capacity(n);
            let mut ret_args: Vec<TokenStream> = Vec::with_capacity(n);
            for (i, other) in slots.iter().enumerate() {
                if i == idx {
                    let needs = &slot.needs_ty;
                    let has = &slot.has_ty;
                    self_args.push(quote! { #needs });
                    ret_args.push(quote! { #has });
                    let _ = other; // silence unused
                } else {
                    let p = &slot_param[i];
                    impl_generics.push(quote! { #p });
                    self_args.push(quote! { #p });
                    ret_args.push(quote! { #p });
                }
            }

            // Substitute the placeholder __MB_REST__ in the slot's method
            // body with the concrete return-type tuple. This avoids each
            // slot having to know its own index when it is constructed.
            let method = substitute_rest(&slot.method, &ret_args);

            quote! {
                impl< #(#impl_generics),* > ModuleBuilder< #(#self_args),* > {
                    #method
                }
            }
        })
        .collect();

    // Terminal `build` method on the fully-bound state.
    let build_impl = quote! {
        impl ModuleBuilder<#(#final_states),*> {
            /// Build the substituted [`ir::Module`][mod_ir] for this
            /// instantiation.
            ///
            /// Calls the module's `ir_constructor`, applies the
            /// substitution map collected by the builder, and returns
            /// the owned IR. The caller can render it to WGSL via
            /// `wgsl_rs::ir::render_module`.
            ///
            /// [mod_ir]: ../wgsl_rs_ir/struct.Module.html
            pub fn build(self) -> #ir_p::Module {
                let mut ir_module = (WGSL_MODULE.ir_constructor)();
                let map: ::std::collections::HashMap<
                    ::std::string::String,
                    #ir_p::Type,
                > = self.subst.into_iter().collect();
                #ir_p::substitute_types(&mut ir_module, &map);
                ir_module
            }
        }
    };

    quote! {
        #(#markers)*
        #struct_def
        #constructor
        #(#setter_impls)*
        #build_impl
    }
}

/// Walk the token stream and replace any standalone `__MB_REST__` ident
/// with a comma-separated expansion of `replacement`. This is a hack to
/// let each slot construct its method body without knowing its own index;
/// the surrounding `emit` function fills in the concrete tuple.
fn substitute_rest(input: &TokenStream, replacement: &[TokenStream]) -> TokenStream {
    use proc_macro2::TokenTree;
    let mut out = TokenStream::new();
    for tt in input.clone() {
        match &tt {
            TokenTree::Ident(id) if id == "__MB_REST__" => {
                let mut first = true;
                for r in replacement {
                    if !first {
                        out.extend(quote! { , });
                    }
                    first = false;
                    out.extend(r.clone());
                }
            }
            TokenTree::Group(g) => {
                let inner = substitute_rest(&g.stream(), replacement);
                let mut new_group = proc_macro2::Group::new(g.delimiter(), inner);
                new_group.set_span(g.span());
                out.extend(std::iter::once(TokenTree::Group(new_group)));
            }
            _ => out.extend(std::iter::once(tt)),
        }
    }
    out
}

/// Add `+ Wgsl` to every type parameter's bounds. This lets the builder
/// call `<T as Wgsl>::to_ir()` regardless of what bounds the user
/// originally wrote.
fn augment_generics_with_wgsl(mut generics: syn::Generics, std_p: &TokenStream) -> syn::Generics {
    let wgsl_path: syn::Path =
        syn::parse2(quote! { #std_p::Wgsl }).expect("std_p::Wgsl is a valid path");
    let wgsl_bound = syn::TypeParamBound::Trait(syn::TraitBound {
        paren_token: None,
        modifier: syn::TraitBoundModifier::None,
        lifetimes: None,
        path: wgsl_path,
    });
    for param in generics.params.iter_mut() {
        if let syn::GenericParam::Type(tp) = param {
            tp.bounds.push(wgsl_bound.clone());
        }
    }
    generics
}

/// Lowercase a name, used to derive setter method names from variable
/// idents (e.g. `FRAME` -> `frame`).
fn lowercase(s: &str) -> String {
    s.to_lowercase()
}

/// Convert a snake_case identifier to CamelCase, used to derive marker
/// type names from entry point function names (e.g. `frag_main` ->
/// `FragMain`).
fn to_camel_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut upper_next = true;
    for ch in s.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}
