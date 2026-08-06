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

For packed slab buffers, use the slab helpers:

- `slab_read_array!(slab, offset, dest, size)` — read `size` elements from `slab` at `offset` into `dest`.
- `slab_write_array!(slab, offset, src, size)` — write `size` elements from `src` into `slab` at `offset`.

Both `slab`, `dest`, `src` refer to declared `storage!` bindings.