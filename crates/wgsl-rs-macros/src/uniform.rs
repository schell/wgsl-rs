//! Provides the `uniform!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::UniformArgs;

pub fn uniform(input: TokenStream) -> TokenStream {
    let UniformArgs {
        group,
        binding,
        name,
        colon_token: _,
        ty,
    } = parse_macro_input!(input as UniformArgs);

    let internal_name = format_ident!("__{}", name);

    let expanded = quote! {
        static #internal_name: std::sync::LazyLock<UniformVariable<#ty>> =
            std::sync::LazyLock::new(|| UniformVariable {
                group: #group,
                binding: #binding,
                value: Default::default(),
            });
        static #name: &std::sync::LazyLock<UniformVariable<#ty>> = &#internal_name;
    };

    expanded.into()
}
