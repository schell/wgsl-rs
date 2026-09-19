# Binding Macros

wgsl-rs provides declarative macros for declaring GPU bindings. Each macro emits both the WGSL binding declaration and a Rust-side static so the same code works on CPU and GPU. To auto-generate `wgpu` bind group layouts and buffer descriptors from these bindings, see [wgpu Linkage](../linkage/overview.md).

## Overview

| Macro | WGSL | Rust static | Access |
| --- | --- | --- | --- |
| [`uniform!`](./binding-macros/uniform.md) | `@group(N) @binding(M) var<uniform> ...` | `Uniform<T>` | `get!(NAME)` |
| [`storage!`](./binding-macros/storage.md) | `@group(N) @binding(M) var<storage, ...> ...` | `Storage<T>` | `get!` / `get_mut!` |
| [`workgroup!`](./binding-macros/workgroup.md) | `var<workgroup> ...` | `Workgroup<T>` | `get!` / `get_mut!` |
| [`texture!`](./binding-macros/texture-sampler.md) | `@group(N) @binding(M) var ...` | hidden `__NAME` + `pub const NAME` | by value |
| [`sampler!`](./binding-macros/texture-sampler.md) | `@group(N) @binding(M) var ...` | hidden `__NAME` + `pub const NAME` | by value |
| [`ptr!`](./binding-macros/ptr.md) | `ptr<address_space, T>` | `&mut T` | `*p` |
| [`discard!`](./binding-macros/discard.md) | `discard;` | thread-local flag | direct call |

## Declaration and Access

Binding macros are used at module scope inside a `#[wgsl]` module. They declare the WGSL binding and the Rust-side static simultaneously:

```rust
#[wgsl]
pub mod shader {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), CAMERA: Camera);

    pub fn view() -> Mat4f {
        get!(CAMERA).view
    }
}
```

## `get!` and `get_mut!`

- `get!(VAR)` reads a `uniform!`, `storage!`, or `workgroup!` binding. It returns a guard that derefs to the value.
- `get!(VAR, T)` reads with an explicit type, used inside generic/template entry points.
- `get_mut!(VAR)` returns a mutable guard for `storage!` and `workgroup!` bindings.

```rust
pub fn add_delta() {
    let mut s = get_mut!(COUNTER);
    s.value += 1;
}
```

## Slab Helpers

For packed slab buffers, use the `slab_copy!` macro. It is bidirectional — pass the slab as the source to read from a storage buffer into a local array, or pass the slab as the destination to write from a local array into a storage buffer:

```rust
slab_copy!(src, src_offset, dest, dest_offset, size)
```

Copies `size` elements from `src[src_offset..]` into `dest[dest_offset..]`. On the GPU this emits a WGSL `for` loop; on the CPU it is a simple element-by-element copy.

```rust
let mut raw = [0u32; 4];
slab_copy!(get!(SLAB), index, raw, 0, 4);
slab_copy!(raw, 0, get_mut!(SLAB), index, 4);
```