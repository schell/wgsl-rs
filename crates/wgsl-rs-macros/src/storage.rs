//! Provides the `storage!` macro in `wgsl_rs::std`.
//!
//! The `storage!` macro defines the Rust binding as well as the WGSL binding.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;

use crate::parse::{ItemStorage, StorageAccess};

pub fn storage(input: TokenStream) -> TokenStream {
    let ItemStorage {
        group,
        binding,
        access,
        name,
        rust_ty,
        ..
    } = parse_macro_input!(input as ItemStorage);

    let access_mode = if matches!(access, StorageAccess::ReadWrite) {
        quote! { ReadWrite }
    } else {
        quote! { Read }
    };

    let expanded: TokenStream2 = quote! {
        pub static #name: Storage<#rust_ty, #access_mode> = Storage::new(
            #group,
            #binding,
        );
    };

    expanded.into()
}
