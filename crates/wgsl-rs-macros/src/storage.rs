//! Provides the `storage!` macro in `wgsl_rs::std`.
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse_macro_input;

use crate::parse::{ItemStorage, StorageAccess, to_snake_case};

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
    let name_str = name.to_string();

    let is_read_write = matches!(access, StorageAccess::ReadWrite);

    // Generate buffer descriptor constant name: INPUT -> INPUT_BUFFER_DESCRIPTOR
    let buffer_descriptor_name = format_ident!("{}_BUFFER_DESCRIPTOR", name);

    // Generate buffer creation function name: INPUT -> create_input_buffer
    let snake_name = to_snake_case(&name_str);
    let create_buffer_fn_name = format_ident!("create_{}_buffer", snake_name);

    let expanded = quote! {
        static #internal_name: std::sync::LazyLock<StorageVariable<#rust_ty>> =
            std::sync::LazyLock::new(|| StorageVariable {
                group: #group,
                binding: #binding,
                read_write: #is_read_write,
                value: Default::default(),
            });
        static #name: &std::sync::LazyLock<StorageVariable<#rust_ty>> = &#internal_name;

        /// Buffer descriptor for the storage variable.
        ///
        /// This descriptor defines the properties of the GPU buffer that will store
        /// this storage variable's data. The descriptor includes:
        ///
        /// - **Size**: Automatically calculated from the Rust type's size at compile time
        ///   using `std::mem::size_of::<T>()`. Note that storage buffers may have
        ///   device-specific alignment requirements (check your device limits for
        ///   `min_storage_buffer_offset_alignment`).
        ///
        /// - **Usage Flags**: The buffer is created with the following usage flags:
        ///   - `STORAGE`: Allows the buffer to be bound as a storage buffer in shaders
        ///   - `COPY_DST`: Allows data to be copied into the buffer (e.g., via `queue.write_buffer()`)
        ///   - `COPY_SRC`: Allows data to be copied from the buffer (useful for readback or buffer-to-buffer copies)
        ///
        /// The buffer created from this descriptor is initially empty and must be
        /// populated with data before use, typically using `queue.write_buffer()`.
        pub const #buffer_descriptor_name: wgpu::BufferDescriptor<'static> = wgpu::BufferDescriptor {
            label: Some(#name_str),
            size: std::mem::size_of::<#rust_ty>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                .union(wgpu::BufferUsages::COPY_DST)
                .union(wgpu::BufferUsages::COPY_SRC),
            mapped_at_creation: false,
        };

        /// Creates a GPU buffer for the storage variable.
        ///
        /// This is a convenience function that creates a buffer using the pre-defined
        /// descriptor constant. The returned buffer is empty and must be populated with
        /// data before being used in a shader.
        ///
        /// # Example
        ///
        /// ```ignore
        /// // Create the buffer
        /// let buffer = create_my_storage_buffer(&device);
        ///
        /// // Populate it with data (for an array of f32)
        /// let data: [f32; 256] = [0.0; 256];
        /// queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
        ///
        /// // For read-write storage buffers, data can be read back after compute:
        /// // ... run compute shader ...
        /// // Then copy or map the buffer to read results
        /// ```
        pub fn #create_buffer_fn_name(device: &wgpu::Device) -> wgpu::Buffer {
            device.create_buffer(&#buffer_descriptor_name)
        }
    };

    expanded.into()
}
