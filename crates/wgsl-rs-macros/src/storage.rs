//! Provides the `storage!` macro in `wgsl_rs::std`.
//!
//! The Rust-side static is a lazily allocated `ModuleVar`: standalone
//! declarations cannot take construction values, so they defer until the
//! host calls `set`. Values-at-construction live in resource packs.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;

use crate::parse::ItemStorage;

pub fn storage(input: TokenStream) -> TokenStream {
    let ItemStorage { name, rust_ty, .. } = parse_macro_input!(input as ItemStorage);

    let expanded: TokenStream2 = quote! {
        pub static #name: ModuleVar<#rust_ty> = ModuleVar::new();
    };

    expanded.into()
}
