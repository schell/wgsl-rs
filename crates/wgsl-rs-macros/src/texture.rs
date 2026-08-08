//! Provides the `texture!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{
    ItemTexture, ScalarType, StorageTextureAccess, TexelFormat, TextureDepthKind, TextureKind,
    TextureStorageKind, Type,
};

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
                pub const #name: &'static #rust_type<#sample_type> = &#inner_name;
            }
        }
        Type::TextureDepth { kind, .. } => {
            let rust_type = texture_depth_kind_to_rust_type(*kind);

            quote! {
                #[doc(hidden)]
                static #inner_name: #rust_type = #rust_type::new(#group, #binding);
                pub const #name: &'static #rust_type = &#inner_name;
            }
        }
        Type::TextureStorage {
            kind,
            format,
            access,
            ..
        } => {
            let rust_type = texture_storage_kind_to_rust_type(*kind);
            let format_type = texel_format_to_rust_type(*format);
            let access_type = storage_access_to_rust_type(*access);

            quote! {
                #[doc(hidden)]
                static #inner_name: #rust_type<#format_type, #access_type> =
                    #rust_type::new(#group, #binding);
                pub const #name: &'static #rust_type<#format_type, #access_type> = &#inner_name;
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

/// Convert a TextureStorageKind to the corresponding Rust type identifier.
fn texture_storage_kind_to_rust_type(kind: TextureStorageKind) -> proc_macro2::TokenStream {
    match kind {
        TextureStorageKind::Storage1D => quote! { TextureStorage1D },
        TextureStorageKind::Storage2D => quote! { TextureStorage2D },
        TextureStorageKind::Storage2DArray => quote! { TextureStorage2DArray },
        TextureStorageKind::Storage3D => quote! { TextureStorage3D },
    }
}

/// Convert a TexelFormat to the corresponding Rust marker type identifier.
fn texel_format_to_rust_type(format: TexelFormat) -> proc_macro2::TokenStream {
    match format {
        TexelFormat::Rgba8unorm => quote! { Rgba8unorm },
        TexelFormat::Rgba8snorm => quote! { Rgba8snorm },
        TexelFormat::Rgba8uint => quote! { Rgba8uint },
        TexelFormat::Rgba8sint => quote! { Rgba8sint },
        TexelFormat::Rgba16uint => quote! { Rgba16uint },
        TexelFormat::Rgba16sint => quote! { Rgba16sint },
        TexelFormat::Rgba16float => quote! { Rgba16float },
        TexelFormat::R32uint => quote! { R32uint },
        TexelFormat::R32sint => quote! { R32sint },
        TexelFormat::R32float => quote! { R32float },
        TexelFormat::Rg32uint => quote! { Rg32uint },
        TexelFormat::Rg32sint => quote! { Rg32sint },
        TexelFormat::Rg32float => quote! { Rg32float },
        TexelFormat::Rgba32uint => quote! { Rgba32uint },
        TexelFormat::Rgba32sint => quote! { Rgba32sint },
        TexelFormat::Rgba32float => quote! { Rgba32float },
        TexelFormat::Bgra8unorm => quote! { Bgra8unorm },
        TexelFormat::Rgba16unorm => quote! { Rgba16unorm },
        TexelFormat::Rgba16snorm => quote! { Rgba16snorm },
        TexelFormat::Rg8unorm => quote! { Rg8unorm },
        TexelFormat::Rg8snorm => quote! { Rg8snorm },
        TexelFormat::Rg8uint => quote! { Rg8uint },
        TexelFormat::Rg8sint => quote! { Rg8sint },
        TexelFormat::Rg16unorm => quote! { Rg16unorm },
        TexelFormat::Rg16snorm => quote! { Rg16snorm },
        TexelFormat::Rg16uint => quote! { Rg16uint },
        TexelFormat::Rg16sint => quote! { Rg16sint },
        TexelFormat::Rg16float => quote! { Rg16float },
        TexelFormat::R8unorm => quote! { R8unorm },
        TexelFormat::R8snorm => quote! { R8snorm },
        TexelFormat::R8uint => quote! { R8uint },
        TexelFormat::R8sint => quote! { R8sint },
        TexelFormat::R16unorm => quote! { R16unorm },
        TexelFormat::R16snorm => quote! { R16snorm },
        TexelFormat::R16uint => quote! { R16uint },
        TexelFormat::R16sint => quote! { R16sint },
        TexelFormat::R16float => quote! { R16float },
        TexelFormat::Rgb10a2unorm => quote! { Rgb10a2unorm },
        TexelFormat::Rgb10a2uint => quote! { Rgb10a2uint },
        TexelFormat::Rg11b10ufloat => quote! { Rg11b10ufloat },
    }
}

/// Convert a StorageTextureAccess to the corresponding Rust marker type.
fn storage_access_to_rust_type(access: StorageTextureAccess) -> proc_macro2::TokenStream {
    match access {
        StorageTextureAccess::Read => quote! { Read },
        StorageTextureAccess::Write => quote! { Write },
        StorageTextureAccess::ReadWrite => quote! { ReadWrite },
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
