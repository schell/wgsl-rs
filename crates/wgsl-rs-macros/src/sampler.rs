//! Provides the `sampler!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{ItemSampler, Type};

pub fn sampler(input: TokenStream) -> TokenStream {
    let ItemSampler {
        group,
        binding,
        name,
        ty,
        ..
    } = parse_macro_input!(input as ItemSampler);

    // Determine if this is a comparison sampler
    let is_comparison = matches!(ty, Type::SamplerComparison { .. });

    // Generate the type based on whether it's a comparison sampler
    let sampler_type = if is_comparison {
        quote! { SamplerComparison }
    } else {
        quote! { Sampler }
    };

    // Generate a hidden inner static and a public const reference.
    // This allows users to pass the sampler directly (without &) to texture
    // functions, while WGSL sees just the variable name without any reference
    // syntax.
    let inner_name = format_ident!("__{}", name);

    // TODO(schell): expand the linkage generated
    let expanded = quote! {
        #[doc(hidden)]
        static #inner_name: #sampler_type = #sampler_type::new(#group, #binding);
        const #name: &'static #sampler_type = &#inner_name;
    };

    expanded.into()
}
