//! Provides the `texture!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{ItemTexture, ScalarType, TextureDepthKind, TextureKind, Type};

pub fn texture(input: TokenStream) -> TokenStream {
    let ItemTexture {
        group,
        binding,
        name,
        ty,
        ..
    } = parse_macro_input!(input as ItemTexture);

    // Generate a hidden inner static and a public const reference.
    // This allows users to pass the texture directly (without &) to texture
    // functions, while WGSL sees just the variable name without any reference
    // syntax.
    let inner_name = format_ident!("__{}", name);

    // Generate the Rust-side type and wgpu types based on the texture type
    // TODO(schell): expand the linkage generated
    let expanded = match &ty {
        Type::Texture {
            kind, sampled_type, ..
        } => {
            let rust_type = texture_kind_to_rust_type(*kind);
            let sample_type = scalar_type_to_token(*sampled_type);

            quote! {
                #[doc(hidden)]
                static #inner_name: #rust_type<#sample_type> = #rust_type::new(#group, #binding);
                const #name: &'static #rust_type<#sample_type> = &#inner_name;
            }
        }
        Type::TextureDepth { kind, .. } => {
            let rust_type = texture_depth_kind_to_rust_type(*kind);

            quote! {
                #[doc(hidden)]
                static #inner_name: #rust_type = #rust_type::new(#group, #binding);
                const #name: &'static #rust_type = &#inner_name;
            }
        }
        _ => {
            // This should never happen since ItemTexture validates the type
            quote! {
                compile_error!("texture! macro requires a texture type");
            }
        }
    };

    expanded.into()
}

/// Convert a TextureKind to the corresponding Rust type identifier.
fn texture_kind_to_rust_type(kind: TextureKind) -> proc_macro2::TokenStream {
    match kind {
        TextureKind::Texture1D => quote! { Texture1D },
        TextureKind::Texture2D => quote! { Texture2D },
        TextureKind::Texture2DArray => quote! { Texture2DArray },
        TextureKind::Texture3D => quote! { Texture3D },
        TextureKind::TextureCube => quote! { TextureCube },
        TextureKind::TextureCubeArray => quote! { TextureCubeArray },
        TextureKind::TextureMultisampled2D => quote! { TextureMultisampled2D },
    }
}

/// Convert a TextureDepthKind to the corresponding Rust type identifier.
fn texture_depth_kind_to_rust_type(kind: TextureDepthKind) -> proc_macro2::TokenStream {
    match kind {
        TextureDepthKind::Depth2D => quote! { TextureDepth2D },
        TextureDepthKind::Depth2DArray => quote! { TextureDepth2DArray },
        TextureDepthKind::DepthCube => quote! { TextureDepthCube },
        TextureDepthKind::DepthCubeArray => quote! { TextureDepthCubeArray },
        TextureDepthKind::DepthMultisampled2D => quote! { TextureDepthMultisampled2D },
    }
}

/// Convert a ScalarType to the corresponding Rust type token.
fn scalar_type_to_token(ty: ScalarType) -> proc_macro2::TokenStream {
    match ty {
        ScalarType::F32 => quote! { f32 },
        ScalarType::I32 => quote! { i32 },
        ScalarType::U32 => quote! { u32 },
        ScalarType::Bool => quote! { bool },
    }
}

/// Convert a TextureKind to the corresponding wgpu TextureViewDimension.
#[expect(dead_code)]
fn texture_kind_to_view_dimension(kind: TextureKind) -> proc_macro2::TokenStream {
    match kind {
        TextureKind::Texture1D => quote! { wgpu::TextureViewDimension::D1 },
        TextureKind::Texture2D | TextureKind::TextureMultisampled2D => {
            quote! { wgpu::TextureViewDimension::D2 }
        }
        TextureKind::Texture2DArray => quote! { wgpu::TextureViewDimension::D2Array },
        TextureKind::Texture3D => quote! { wgpu::TextureViewDimension::D3 },
        TextureKind::TextureCube => quote! { wgpu::TextureViewDimension::Cube },
        TextureKind::TextureCubeArray => quote! { wgpu::TextureViewDimension::CubeArray },
    }
}

/// Convert a TextureDepthKind to the corresponding wgpu TextureViewDimension.
#[expect(dead_code)]
fn texture_depth_kind_to_view_dimension(kind: TextureDepthKind) -> proc_macro2::TokenStream {
    match kind {
        TextureDepthKind::Depth2D | TextureDepthKind::DepthMultisampled2D => {
            quote! { wgpu::TextureViewDimension::D2 }
        }
        TextureDepthKind::Depth2DArray => quote! { wgpu::TextureViewDimension::D2Array },
        TextureDepthKind::DepthCube => quote! { wgpu::TextureViewDimension::Cube },
        TextureDepthKind::DepthCubeArray => quote! { wgpu::TextureViewDimension::CubeArray },
    }
}
