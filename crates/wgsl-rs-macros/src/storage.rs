//! Provides the `storage!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
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

    let internal_name = format_ident!("__{}", name);

    let is_read_write = matches!(access, StorageAccess::ReadWrite);

    let expanded = quote! {
        static #internal_name: std::sync::LazyLock<StorageVariable<#rust_ty>> =
            std::sync::LazyLock::new(|| StorageVariable {
                group: #group,
                binding: #binding,
                read_write: #is_read_write,
                value: Default::default(),
            });
        static #name: &std::sync::LazyLock<StorageVariable<#rust_ty>> = &#internal_name;
    };

    expanded.into()
}
