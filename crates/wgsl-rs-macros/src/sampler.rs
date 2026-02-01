//! Provides the `sampler!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{to_snake_case, ItemSampler, Type};

pub fn sampler(input: TokenStream) -> TokenStream {
    let ItemSampler {
        group,
        binding,
        name,
        ty,
        ..
    } = parse_macro_input!(input as ItemSampler);

    let name_str = name.to_string();
    let snake_name = to_snake_case(&name_str);

    // Determine if this is a comparison sampler
    let is_comparison = matches!(ty, Type::SamplerComparison { .. });

    // Generate the type based on whether it's a comparison sampler
    let sampler_type = if is_comparison {
        quote! { SamplerComparison }
    } else {
        quote! { Sampler }
    };

    // Generate sampler descriptor constant name: MY_SAMPLER ->
    // MY_SAMPLER_DESCRIPTOR
    let sampler_descriptor_name = format_ident!("{}_DESCRIPTOR", name);

    // Generate sampler creation function name: MY_SAMPLER -> create_my_sampler
    let create_sampler_fn_name = format_ident!("create_{}", snake_name);

    let expanded = quote! {
        static #name: #sampler_type = #sampler_type::new(#group, #binding);

        /// Sampler descriptor for the sampler variable.
        ///
        /// This descriptor defines the properties of the GPU sampler. By default, it
        /// creates a sampler with linear filtering and clamp-to-edge address mode.
        /// You can modify these settings or create a custom descriptor as needed.
        ///
        /// The descriptor includes:
        ///
        /// - **Address Modes**: ClampToEdge for u, v, and w coordinates
        /// - **Mag/Min Filter**: Linear filtering for smooth texture sampling
        /// - **Mipmap Filter**: Linear for smooth transitions between mip levels
        /// - **Compare**: Some(Less) for comparison samplers, None for regular samplers
        pub const #sampler_descriptor_name: wgpu::SamplerDescriptor<'static> = wgpu::SamplerDescriptor {
            label: Some(#name_str),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: if #is_comparison { Some(wgpu::CompareFunction::Less) } else { None },
            anisotropy_clamp: 1,
            border_color: None,
        };

        /// Creates a GPU sampler.
        ///
        /// This is a convenience function that creates a sampler using the pre-defined
        /// descriptor constant. You can also create a sampler with custom settings
        /// by calling `device.create_sampler()` directly with your own descriptor.
        ///
        /// # Example
        ///
        /// ```ignore
        /// // Create the sampler with default settings
        /// let sampler = create_my_sampler(&device);
        ///
        /// // Or create with custom settings
        /// let custom_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        ///     mag_filter: wgpu::FilterMode::Nearest,
        ///     ..MY_SAMPLER_DESCRIPTOR
        /// });
        /// ```
        pub fn #create_sampler_fn_name(device: &wgpu::Device) -> wgpu::Sampler {
            device.create_sampler(&#sampler_descriptor_name)
        }
    };

    expanded.into()
}
