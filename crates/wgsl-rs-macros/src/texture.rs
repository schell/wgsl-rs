//! Provides the `texture!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{ItemTexture, ScalarType, TextureDepthKind, TextureKind, Type, to_snake_case};

pub fn texture(input: TokenStream) -> TokenStream {
    let ItemTexture {
        group,
        binding,
        name,
        ty,
        ..
    } = parse_macro_input!(input as ItemTexture);

    let name_str = name.to_string();
    let snake_name = to_snake_case(&name_str);

    // Generate texture view descriptor constant name: DIFFUSE_TEX ->
    // DIFFUSE_TEX_VIEW_DESCRIPTOR
    let view_descriptor_name = format_ident!("{}_VIEW_DESCRIPTOR", name);

    // Generate create view function name: DIFFUSE_TEX -> create_diffuse_tex_view
    let create_view_fn_name = format_ident!("create_{}_view", snake_name);

    // Generate is_multisampled constant name: DIFFUSE_TEX ->
    // DIFFUSE_TEX_IS_MULTISAMPLED
    let is_multisampled_const_name = format_ident!("{}_IS_MULTISAMPLED", name);

    // Generate the Rust-side type and wgpu types based on the texture type
    let expanded = match &ty {
        Type::Texture {
            kind, sampled_type, ..
        } => {
            let rust_type = texture_kind_to_rust_type(*kind);
            let sample_type = scalar_type_to_token(*sampled_type);
            let view_dimension = texture_kind_to_view_dimension(*kind);
            let is_multisampled = matches!(kind, TextureKind::TextureMultisampled2D);

            quote! {
                static #name: #rust_type<#sample_type> = #rust_type::new(#group, #binding);

                /// Texture view descriptor for the texture variable.
                ///
                /// This descriptor defines how the texture should be viewed. By default, it
                /// creates a view that matches the texture's format and dimensionality.
                /// You can modify these settings or create a custom descriptor as needed.
                pub const #view_descriptor_name: TextureViewDescriptor<'static> = TextureViewDescriptor {
                    label: Some(#name_str),
                    format: None,  // Inherit from texture
                    dimension: Some(#view_dimension),
                    usage: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,  // All mip levels
                    base_array_layer: 0,
                    array_layer_count: None,  // All array layers
                };

                /// Creates a texture view for this texture.
                ///
                /// This is a convenience function that creates a texture view using the
                /// pre-defined descriptor constant. You can also create a view with custom
                /// settings by calling `texture.create_view()` directly with your own
                /// descriptor.
                ///
                /// # Arguments
                ///
                /// * `texture` - The GPU texture to create a view for
                ///
                /// # Example
                ///
                /// ```ignore
                /// // Create the texture view with default settings
                /// let view = create_diffuse_tex_view(&texture);
                ///
                /// // Or create with custom settings
                /// let custom_view = texture.create_view(&wgpu::TextureViewDescriptor {
                ///     format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ///     ..DIFFUSE_TEX_VIEW_DESCRIPTOR.into()
                /// });
                /// ```
                pub fn #create_view_fn_name(texture: &wgpu::Texture) -> wgpu::TextureView {
                    texture.create_view(&#view_descriptor_name.into())
                }

                /// Indicates whether this texture is multisampled.
                pub const #is_multisampled_const_name: bool = #is_multisampled;
            }
        }
        Type::TextureDepth { kind, .. } => {
            let rust_type = texture_depth_kind_to_rust_type(*kind);
            let view_dimension = texture_depth_kind_to_view_dimension(*kind);
            let is_multisampled = matches!(kind, TextureDepthKind::DepthMultisampled2D);

            quote! {
                static #name: #rust_type = #rust_type::new(#group, #binding);

                /// Texture view descriptor for the depth texture variable.
                ///
                /// This descriptor defines how the depth texture should be viewed. By default,
                /// it creates a view that matches the texture's format and dimensionality.
                /// You can modify these settings or create a custom descriptor as needed.
                pub const #view_descriptor_name: TextureViewDescriptor<'static> = TextureViewDescriptor {
                    label: Some(#name_str),
                    format: None,  // Inherit from texture
                    dimension: Some(#view_dimension),
                    usage: None,
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: None,  // All mip levels
                    base_array_layer: 0,
                    array_layer_count: None,  // All array layers
                };

                /// Creates a texture view for this depth texture.
                ///
                /// This is a convenience function that creates a texture view using the
                /// pre-defined descriptor constant.
                pub fn #create_view_fn_name(texture: &wgpu::Texture) -> wgpu::TextureView {
                    texture.create_view(&#view_descriptor_name.into())
                }

                /// Indicates whether this depth texture is multisampled.
                pub const #is_multisampled_const_name: bool = #is_multisampled;
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
