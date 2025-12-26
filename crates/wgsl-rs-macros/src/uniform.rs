//! Provides the `uniform!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::ItemUniform;

/// Converts a SCREAMING_CASE or PascalCase identifier to snake_case.
fn to_snake_case(s: &str) -> String {
    // For SCREAMING_CASE (all uppercase with underscores), just lowercase it
    if s.chars().all(|c| c.is_uppercase() || c == '_') {
        return s.to_lowercase();
    }

    // For PascalCase or camelCase
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result
}

pub fn uniform(input: TokenStream) -> TokenStream {
    let ItemUniform {
        group,
        binding,
        name,
        rust_ty,
        ..
    } = parse_macro_input!(input as ItemUniform);

    let internal_name = format_ident!("__{}", name);
    let name_str = name.to_string();

    // Generate buffer descriptor constant name: FRAME -> FRAME_BUFFER_DESCRIPTOR
    let buffer_descriptor_name = format_ident!("{}_BUFFER_DESCRIPTOR", name);

    // Generate buffer creation function name: FRAME -> create_frame_buffer
    let snake_name = to_snake_case(&name_str);
    let create_buffer_fn_name = format_ident!("create_{}_buffer", snake_name);

    let expanded = quote! {
        static #internal_name: std::sync::LazyLock<UniformVariable<#rust_ty>> =
            std::sync::LazyLock::new(|| UniformVariable {
                group: #group,
                binding: #binding,
                value: Default::default(),
            });
        static #name: &std::sync::LazyLock<UniformVariable<#rust_ty>> = &#internal_name;

        /// Buffer descriptor for the uniform variable.
        pub const #buffer_descriptor_name: wgpu::BufferDescriptor<'static> = wgpu::BufferDescriptor {
            label: Some(#name_str),
            size: std::mem::size_of::<#rust_ty>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                .union(wgpu::BufferUsages::COPY_DST)
                .union(wgpu::BufferUsages::COPY_SRC),
            mapped_at_creation: false,
        };

        /// Creates a buffer for the uniform variable.
        pub fn #create_buffer_fn_name(device: &wgpu::Device) -> wgpu::Buffer {
            device.create_buffer(&#buffer_descriptor_name)
        }
    };

    expanded.into()
}
