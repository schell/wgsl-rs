//! Provides the `workgroup!` macro in `wgsl_rs::std`.
//!
//! The `workgroup!` macro defines workgroup-scoped variables that are shared
//! between all invocations in a compute shader workgroup.
//!
//! In WGSL, workgroup variables are declared as `var<workgroup> NAME: TYPE;`.
//! On the Rust side, they are backed by thread-safe shared state using
//! `RwLock`.
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

use crate::parse::ItemWorkgroup;

pub fn workgroup(input: TokenStream) -> TokenStream {
    let ItemWorkgroup { name, rust_ty, .. } = parse_macro_input!(input as ItemWorkgroup);

    // On the Rust side, workgroup variables are simulated using thread-safe
    // shared state. Since workgroup variables are shared across invocations
    // in a workgroup, we use RwLock for synchronization.
    //
    // Note: The actual WGSL code generation happens in the `wgsl` macro,
    // not here. This macro only generates the Rust-side binding.
    let expanded = quote! {
        pub static #name: Workgroup<#rust_ty> = Workgroup::new();
    };

    expanded.into()
}
