# `storage!`

Declares a storage buffer binding, either read-only or read-write.

## Syntax

```rust
// read-only
storage!(group(N), binding(M), NAME: Type);

// read-write
storage!(group(N), binding(M), read_write, NAME: Type);
```

## What It Generates

Read-only:

```wgsl
@group(N) @binding(M) var<storage, read> NAME: Type;
```

Read-write:

```wgsl
@group(N) @binding(M) var<storage, read_write> NAME: Type;
```

Rust:

```rust
pub static NAME: Storage<Type>;
```

## Access

- `get!(NAME)` reads the buffer.
- `get_mut!(NAME)` writes to the buffer (only valid for `read_write`).

## Example: Compute Shader

```rust
#[wgsl]
pub mod prefix {
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Data {
        pub value: f32,
    }

    storage!(group(0), binding(0), read_write, INPUT: Data);
    storage!(group(0), binding(1), read_write, OUTPUT: Data);

    #[compute]
    #[workgroup_size(64)]
    pub fn cs_main() {
        let mut src = get_mut!(INPUT);
        let mut dst = get_mut!(OUTPUT);
        dst.value = src.value * 2.0;
    }
}
```

## Notes

- Use arrays in `Type` (e.g. `array<f32, N>`) for large buffers.
- `read_write` storage requires the binding to be created with the `read_write` access flag on the host side.
- Slab helpers (`slab_read_array!`, `slab_write_array!`) operate on `storage!` bindings (see [Binding Macros](../binding-macros.md)).