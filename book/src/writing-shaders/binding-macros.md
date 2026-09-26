# Binding Macros

wgsl-rs provides declarative macros for declaring GPU bindings. Each macro emits both the WGSL binding declaration and a Rust-side static so the same code works on CPU and GPU. To auto-generate `wgpu` bind group layouts and buffer descriptors from these bindings, see [wgpu Linkage](../linkage/overview.md).

## Overview

| Macro | WGSL | Rust static | Access |
| --- | --- | --- | --- |
| [`uniform!`](./binding-macros/uniform.md) | `@group(N) @binding(M) var<uniform> ...` | `Uniform<T>` | `load!` / `get!(NAME)` |
| [`storage!`](./binding-macros/storage.md) | `@group(N) @binding(M) var<storage, ...> ...` | `Storage<T>` | `load!` / `get!` / `get_mut!` |
| [`workgroup!`](./binding-macros/workgroup.md) | `var<workgroup> ...` | `Workgroup<T>` | `load!` / `get!` / `get_mut!` |
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

## `load!`, `get!` and `get_mut!`

- `load!(VAR)` copies the value of a `uniform!`, `storage!`, or `workgroup!` binding into a local. It returns a plain value that works directly in arithmetic.
- `load!(VAR, T)` loads with an explicit type, used inside generic/template entry points.
- `get!(VAR)` reads a binding as a guard that derefs to the value — use it for borrow-style access (field access, indexing).
- `get!(VAR, T)` reads with an explicit type, used inside generic/template entry points.
- `get_mut!(VAR)` returns a mutable guard for `storage!` and `workgroup!` bindings.

```rust
pub fn add_delta() {
    let mut s = get_mut!(COUNTER);
    s.value += 1;
}

pub fn scale(p: Vec2f) -> Vec2f {
    p / load!(U_RESOLUTION)
}
```

`load!` requires the value type to be `Copy` (all built-in value types are). It is a compile error on textures, samplers, and atomics — atomics must be read with `atomic_load(&get!(COUNTER))`.

## Derefs of Accessors

Module variables are values in WGSL, not pointers. A `*` in front of an accessor is a Rust-side guard artifact, so it is elided in the generated WGSL:

```rust
// Both render as plain value operations in WGSL:
let u = *get!(U);                 // WGSL: U
*get_mut!(DATA) = v;              // WGSL: DATA = v
```

Dereferencing a *local* that holds a value is a compile error — there is no WGSL pointer to deref. Bind the value with `load!` instead:

```rust
let u = load!(U);   // good: a value local
let u = get!(U);    // *u below would not compile
// *u = ...;        // error: derefs a module variable value
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