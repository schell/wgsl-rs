//! Provides the `uniform!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{ItemUniform, to_snake_case};

pub fn uniform(input: TokenStream) -> TokenStream {
    let ItemUniform {
        group,
        binding,
        name,
        rust_ty,
        ..
    } = parse_macro_input!(input as ItemUniform);

    let name_str = name.to_string();

    // Generate buffer descriptor constant name: FRAME -> FRAME_BUFFER_DESCRIPTOR
    let buffer_descriptor_name = format_ident!("{}_BUFFER_DESCRIPTOR", name);

    // Generate buffer creation function name: FRAME -> create_frame_buffer
    let snake_name = to_snake_case(&name_str);
    let create_buffer_fn_name = format_ident!("create_{}_buffer", snake_name);

    let expanded = quote! {
        static #name: Uniform<#rust_ty> = Uniform::new(#group, #binding);

        /// Buffer descriptor for the uniform variable.
        ///
        /// This descriptor defines the properties of the GPU buffer that will store
        /// this uniform variable's data. The descriptor includes:
        ///
        /// - **Size**: Automatically calculated from the Rust type's size at compile time
        ///   using `std::mem::size_of::<T>()`. Note that the actual GPU buffer size may
        ///   be subject to device-specific alignment requirements (typically 256 bytes
        ///   for uniform buffers on most hardware).
        ///
        /// - **Usage Flags**: The buffer is created with the following usage flags:
        ///   - `UNIFORM`: Allows the buffer to be bound as a uniform buffer in shaders
        ///   - `COPY_DST`: Allows data to be copied into the buffer (e.g., via `queue.write_buffer()`)
        ///   - `COPY_SRC`: Allows data to be copied from the buffer (useful for readback or buffer-to-buffer copies)
        ///
        /// The buffer created from this descriptor is initially empty and must be
        /// populated with data before use, typically using `queue.write_buffer()`.
        pub const #buffer_descriptor_name: wgpu::BufferDescriptor<'static> = wgpu::BufferDescriptor {
            label: Some(#name_str),
            size: std::mem::size_of::<#rust_ty>() as u64,
            usage: wgpu::BufferUsages::UNIFORM
                .union(wgpu::BufferUsages::COPY_DST)
                .union(wgpu::BufferUsages::COPY_SRC),
            mapped_at_creation: false,
        };

        /// Creates a GPU buffer for the uniform variable.
        ///
        /// This is a convenience function that creates a buffer using the pre-defined
        /// descriptor constant. The returned buffer is empty and must be populated with
        /// data before being used in a shader.
        ///
        /// # Example
        ///
        /// ```ignore
        /// // Create the buffer
        /// let buffer = create_my_uniform_buffer(&device);
        ///
        /// // Populate it with data
        /// let data = MyUniformType { /* ... */ };
        /// queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&data));
        ///
        /// // Or for simple types like u32, f32, etc:
        /// let frame: u32 = 0;
        /// queue.write_buffer(&buffer, 0, &frame.to_ne_bytes());
        /// ```
        pub fn #create_buffer_fn_name(device: &wgpu::Device) -> wgpu::Buffer {
            device.create_buffer(&#buffer_descriptor_name)
        }
    };

    expanded.into()
}
