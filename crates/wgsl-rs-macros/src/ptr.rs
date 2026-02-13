//! Provides the `ptr!` macro for WGSL pointer types.
//!
//! In WGSL, pointers are used to pass mutable references to functions.
//! The `ptr!` macro provides a way to express pointer types that works
//! in both Rust and WGSL contexts.
//!
//! # Supported Address Spaces
//! - `function` - Local function variables
//! - `private` - Module-scope private variables
//!
//! # Example
//! ```ignore
//! fn increment(p: ptr!(function, i32)) {
//!     *p += 1;
//! }
//! ```
//!
//! # WGSL Output
//! The above Rust code transpiles to:
//! ```wgsl
//! fn increment(p: ptr<function, i32>) {
//!     *p += 1;
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{Ident, Token, Type, parse::Parse};

/// Arguments parsed from `ptr!(address_space, Type)`.
struct PtrMacroArgs {
    address_space: Ident,
    _comma: Token![,],
    store_type: Type,
}

impl Parse for PtrMacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(PtrMacroArgs {
            address_space: input.parse()?,
            _comma: input.parse()?,
            store_type: input.parse()?,
        })
    }
}

/// Implements the `ptr!` macro.
///
/// This macro expands to `&mut T` in Rust, allowing the code to compile
/// and run on the CPU. During WGSL transpilation, the macro invocation
/// is parsed and converted to `ptr<address_space, T>`.
pub fn ptr(input: TokenStream) -> TokenStream {
    let PtrMacroArgs {
        address_space,
        store_type,
        ..
    } = syn::parse_macro_input!(input as PtrMacroArgs);

    let addr_str = address_space.to_string();
    match addr_str.as_str() {
        "function" | "private" => {
            // Both address spaces have read_write access mode (the only mode they support).
            // In Rust, this maps to &mut T for mutable access.
            quote! { &mut #store_type }.into()
        }
        "workgroup" => {
            // Workgroup address space has read_write access mode by default.
            // In Rust, this maps to &mut T for mutable access.
            quote! { &mut #store_type }.into()
        }
        other => syn::Error::new(
            address_space.span(),
            format!(
                "unsupported address space '{}', only 'function', 'private', and 'workgroup' are \
                 supported",
                other
            ),
        )
        .to_compile_error()
        .into(),
    }
}
