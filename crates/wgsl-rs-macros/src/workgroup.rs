//! Provides the `workgroup!` macro in `wgsl_rs::std`.
//!
//! The `workgroup!` macro defines workgroup-scoped variables that are shared
//! between all invocations in a compute shader workgroup.
//!
//! In WGSL, workgroup variables are declared as `var<workgroup> NAME: TYPE;`.
//! The Rust-side static is a lazily allocated `ModuleVar`: standalone
//! declarations cannot take construction values, so they defer until the host
//! calls `set`. Values-at-construction live in resource packs.
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

use crate::parse::ItemWorkgroup;

pub fn workgroup(input: TokenStream) -> TokenStream {
    let ItemWorkgroup { name, rust_ty, .. } = parse_macro_input!(input as ItemWorkgroup);

    let expanded = quote! {
        pub static #name: ModuleVar<#rust_ty> = ModuleVar::new();
    };

    expanded.into()
}
