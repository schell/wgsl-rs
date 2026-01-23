//! Provides the `storage!` macro in `wgsl_rs::std`.
//!
//! The `storage!` macro defines the Rust binding as well as the WGSL binding.
//! It also creates linkage.
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

    let name_str = name.to_string();

    let access_mode = if matches!(access, StorageAccess::ReadWrite) {
        quote! { ReadWrite }
    } else {
        quote! { Read }
    };

    // Generate buffer descriptor constant name: INPUT -> INPUT_BUFFER_DESCRIPTOR
    let buffer_descriptor_name = format_ident!("{}_BUFFER_DESCRIPTOR", name);

    // Generate buffer creation function name: INPUT -> create_input_buffer
    let snake_name = to_snake_case(&name_str);
    let create_buffer_fn_name = format_ident!("create_{}_buffer", snake_name);

    let expanded = quote! {
        pub static #name: Storage<#rust_ty, #access_mode> = Storage::new(
            #group,
            #binding,
        );
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

        /// Creates a buffer for the storage variable.
        ///
        /// This function creates an empty GPU buffer using the associated buffer descriptor.
        /// The buffer is created with no initial data and must be populated before use.
        ///
        /// # Populating the Buffer
        /// After creating the buffer, you must write data to it using one of these methods:
        ///
        /// - **Using `queue.write_buffer`** (recommended for most cases):
        ///   ```no_run
        ///   # use wgpu::{Device, Queue, Buffer};
        ///   # fn example(device: &Device, queue: &Queue, buffer: &Buffer) {
        ///   let data: [f32; 256] = [0.0; 256];
        ///   queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
        ///   # }
        ///   ```
        ///
        /// - **Using `encoder.copy_buffer_to_buffer`** for GPU-to-GPU copies:
        ///   ```no_run
        ///   # use wgpu::{CommandEncoder, Buffer};
        ///   # fn example(encoder: &mut CommandEncoder, src: &Buffer, dst: &Buffer) {
        ///   encoder.copy_buffer_to_buffer(src, 0, dst, 0, 1024);
        ///   # }
        ///   ```
        ///
        /// - **Using `mapped_at_creation`** (modify the descriptor if you need this):
        ///   For pre-initialization during buffer creation, set `mapped_at_creation: true`
        ///   in the buffer descriptor.
        ///
        /// # Example
        /// ```no_run
        /// # use wgpu::{Device, Queue};
        /// # fn example(device: &Device, queue: &Queue) {
        /// // Assuming you have a storage variable: storage!(group(0), binding(0), DATA: [f32; 256]);
        /// // This generates: create_data_buffer and DATA_BUFFER_DESCRIPTOR
        ///
        /// // Create the buffer
        /// // let buffer = create_data_buffer(device);
        ///
        /// // Populate with data
        /// let data: [f32; 256] = [1.0; 256];
        /// // queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
        /// # }
        /// ```
        pub fn #create_buffer_fn_name(device: &wgpu::Device) -> wgpu::Buffer {
            device.create_buffer(&#buffer_descriptor_name)
        }
    };

    expanded.into()
}
