# Compute Shader

A simple compute shader that demonstrates defining and accessing storage buffers with resource pack fields and the `get!` and `get_mut!` macros, plus the `#[derive(Wgsl)]` macro for user-defined storage types.

## Rust Source

```rust
#[wgsl]
pub mod compute_shader {
    //! A simple compute shader that demonstrates defining and accessing storage
    //! buffers.
    //!
    //! Storage buffers are special on the Rust side and require locking,
    //! so they are accessed with the `get!` and `get_mut!` macros, which
    //! do the heavy lifting for you. These macros are a noop in WGSL and are
    //! stripped during parsing.
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Output {
        pub inner: f32,
    }

    pub struct ComputeShaderPack {
        // Read-only input buffer
        #[storage(group(0), binding(0), read_only)]
        pub input: Storage<[f32; 256], Read>,
        // Read-write output buffer
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<Output, ReadWrite>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &ComputeShaderPack) {
        // Compute the index from global invocation ID
        let idx = global_id.x() as usize;
        // Use the `get!` macro to access the storage
        let input = get!(pack.input)[idx];
        // Use the `get_mut!` macro to access the storage mutably
        get_mut!(pack.output).inner = input;
    }
}
```

## Generated WGSL

Pack fields become mangled WGSL globals (`packlower__field`), and the pack
parameter disappears from the entry point:

```wgsl
@group(0) @binding(0) var<storage, read> computeshaderpack__input: array<f32, 256>;

struct Output {
    inner: f32
}
@group(0) @binding(1) var<storage, read_write> computeshaderpack__output: Output;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = u32(global_id.x);
    let input = computeshaderpack__input[idx];
    computeshaderpack__output.inner = input;
}
```

## Notes

- `#[storage(group(0), binding(0), read_only)] pub input: Storage<[f32; 256], Read>` declares a read-only storage buffer; `read_write` with `Storage<T, ReadWrite>` makes it read-write.
- `get!` and `get_mut!` access storage buffers on the Rust side (performing locking). They are no-ops in WGSL and are stripped during parsing.
- `#[derive(Wgsl)]` makes a struct eligible for use in storage buffers.
- `#[compute]` and `#[workgroup_size(64)]` mark the entry point.