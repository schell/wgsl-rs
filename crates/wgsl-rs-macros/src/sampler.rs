//! Provides the `sampler!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::quote;
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

    // TODO(schell): expand the linkage generated
    let expanded = quote! {
        static #name: #sampler_type = #sampler_type::new(#group, #binding);
    };

    expanded.into()
}
