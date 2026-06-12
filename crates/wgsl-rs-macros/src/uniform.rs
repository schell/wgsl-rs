//! Provides the `uniform!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;

use crate::parse::ItemUniform;

pub fn uniform(input: TokenStream) -> TokenStream {
    let ItemUniform {
        group,
        binding,
        name,
        rust_ty,
        ..
    } = parse_macro_input!(input as ItemUniform);

    let expanded: TokenStream2 = quote! {
        pub static #name: Uniform<#rust_ty> = Uniform::new(#group, #binding);
    };

    expanded.into()
}
