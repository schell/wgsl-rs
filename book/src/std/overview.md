# The Standard Library

`wgsl_rs::std` is the prelude for every `#[wgsl]` module. It provides WGSL
types, builtin functions, binding macros, entry-point attributes, and
runtime macros that bridge the Rust and WGSL worlds.

## The glob import

Every `#[wgsl]` module begins with a glob import:

```rust
#[wgsl]
pub mod my_shader {
    use wgsl_rs::std::*;
    // ...
}
```

The glob import is **required**. Only `use wgsl_rs::std::*` is recognized by
the transpiler; named imports from `std` are not supported inside `#[wgsl]`
modules.

## What it provides

| Category | Examples | Chapter |
|----------|----------|---------|
| WGSL types | `Vec2f`, `Vec3f`, `Vec4f`, `Mat4f`, `Vec2u`, ... | [Scalars & Vectors](../types/scalars.md), [Vectors & Swizzles](../types/vectors.md), [Matrices](../types/matrices.md) |
| Constructors | `vec2f(...)`, `vec3f(...)`, `vec4f(...)`, `mat4x4f(...)` | [Matrix & Vector Functions](./matrix-vector.md) |
| Numeric builtins | `abs`, `sin`, `cos`, `pow`, `clamp`, `dot`, `cross`, ... | [Numeric Builtins](./numeric.md) |
| Matrix builtins | `determinant`, `transpose` | [Matrix & Vector Functions](./matrix-vector.md) |
| Texture functions | `texture_sample`, `texture_load`, `texture_store`, ... | [Texture & Sampler Functions](./texture-sampler.md) |
| Derivatives | `dpdx`, `dpdy`, `fwidth`, ... | [Derivatives](./derivatives.md) |
| Bitcast | `bitcast_f32`, `bitcast_u32`, `bitcast_vec4i`, ... | [Bitcast](./bitcast.md) |
| Packing | `pack4x8snorm`, `unpack2x16float`, ... | [Packing](./packing.md) |
| Synchronization | `workgroup_barrier`, `storage_barrier`, `workgroup_uniform_load` | [Synchronization](./synchronization.md) |
| Control | `discard!()` | [`discard!()`](./discard.md) |
| Binding macros | `uniform!`, `storage!`, `workgroup!`, `texture!`, `sampler!`, `ptr!` | [Binding Macros](../writing-shaders/binding-macros.md) |
| Entry-point attributes | `#[vertex]`, `#[fragment]`, `#[compute]` | [Vertex / Fragment / Compute](../entry-points/stages.md) |
| Runtime macros | `get!`, `get_mut!`, `discard!`, `slab_read_array!`, `slab_write_array!` | — |
| Marker types | `PhantomData<T>` | [Generic Structs: `PhantomData`](../generics/generic-structs.md#phantomdatat-marker-fields) |

## The `Wgsl` trait

`Wgsl` marks a type usable in a `#[wgsl]` module. Any type passed to a WGSL
function, stored in a `uniform!`/`storage!`, or returned from an entry point
must implement `Wgsl`. The macro and the runtime both rely on this trait to
marshal values between Rust and WGSL.

Related traits:

| Trait | Meaning |
|-------|---------|
| `Wgsl` | A type usable in WGSL modules. |
| `WgslScalar` | A scalar usable in WGSL: `f32`, `i32`, `u32`, `bool`, `f16`. |
| `WgslTextureScalar` | A scalar usable as a texture texel format: `f32`, `i32`, `u32` (not `bool`). |

`WgslTextureScalar` is a stricter subset of `WgslScalar`: only types that have
a corresponding WGSL texture format qualify, so `bool` is excluded.

## CPU and WGSL agreement

Every builtin in `wgsl_rs::std` has a CPU implementation that mirrors WGSL
semantics. When you run a `#[wgsl]` module as ordinary Rust (e.g. under
`cargo test`), the builtins execute on the CPU; when the transpiler emits
WGSL, the same names map to native WGSL builtins. The
[roundtrip tests](../validation/auto-tests.md) verify the two worlds agree.