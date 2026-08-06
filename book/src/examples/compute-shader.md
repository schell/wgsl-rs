# Compute Shader

A simple compute shader that demonstrates defining and accessing storage buffers with the `storage!`, `get!`, and `get_mut!` macros, plus the `#[derive(Wgsl)]` macro for user-defined storage types.

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

    // Read-only input buffer
    storage!(group(0), binding(0), INPUT: [f32; 256]);

    #[derive(Wgsl)]
    pub struct Output {
        pub inner: f32,
    }

    // Read-write output buffer
    storage!(group(0), binding(1), read_write, OUTPUT: Output);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Compute the index from global invocation ID
        let idx = global_id.x() as usize;
        // Use the `get!` macro to access the storage
        let input = get!(INPUT)[idx];
        // Use the `get_mut!` macro to access the storage mutably
        get_mut!(OUTPUT).inner = input;
    }
}
```

## Generated WGSL

```wgsl
@group(0) @binding(0) var<storage, read> INPUT: array<f32, 256>;

struct Output {
    inner: f32
}
@group(0) @binding(1) var<storage, read_write> OUTPUT: Output;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = u32(global_id.x);
    let input = INPUT[idx];
    OUTPUT.inner = input;
}
```

## Notes

- `storage!(group(0), binding(0), INPUT: [f32; 256])` declares a read-only storage buffer; `read_write` makes it read-write.
- `get!` and `get_mut!` access storage buffers on the Rust side (performing locking). They are no-ops in WGSL and are stripped during parsing.
- `#[derive(Wgsl)]` makes a struct eligible for use in storage buffers.
- `#[compute]` and `#[workgroup_size(64)]` mark the entry point.