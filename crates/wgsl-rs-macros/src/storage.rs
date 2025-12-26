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
        /// This descriptor configures a GPU buffer with the following properties:
        ///
        /// # Buffer Size
        /// The buffer size is calculated at compile time using `std::mem::size_of::<T>()`,
        /// where `T` is the Rust type associated with this storage variable. This ensures
        /// the buffer is appropriately sized for the data type.
        ///
        /// # Usage Flags
        /// The buffer includes three usage flags:
        /// - `STORAGE`: Allows the buffer to be used as a storage buffer in shaders, enabling
        ///   read and/or write operations from compute or fragment shaders.
        /// - `COPY_DST`: Allows data to be copied into the buffer (e.g., via `queue.write_buffer`
        ///   or `encoder.copy_buffer_to_buffer`).
        /// - `COPY_SRC`: Allows data to be copied from the buffer (e.g., for reading results back
        ///   to the CPU or copying to another buffer).
        ///
        /// # Alignment and Size Constraints
        /// Storage buffers must meet specific alignment requirements:
        /// - The buffer size must be a multiple of the GPU's minimum storage buffer alignment
        ///   (typically 16 bytes, but can vary by device).
        /// - Array elements and struct fields have specific alignment requirements defined by
        ///   WGSL's memory layout rules.
        /// - If the Rust type's size doesn't meet alignment requirements, you may need to add
        ///   padding to your type definition.
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
