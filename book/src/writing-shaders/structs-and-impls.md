# Structs, Impls, and Enums

## Structs

Structs are declared with public fields:

```rust
#[wgsl]
pub mod types {
    use wgsl_rs::std::*;

    pub struct Light {
        pub position: Vec3f,
        pub color: Vec3f,
        pub intensity: f32,
    }
}
```

Transpiles to:

```wgsl
struct Light {
  position: vec3<f32>,
  color: vec3<f32>,
  intensity: f32,
}
```

## `#[derive(Wgsl)]`

Structs used in storage or uniform buffers should derive `Wgsl`. This generates the host-side layout and zero-value logic needed for binding setup:

```rust
#[derive(Wgsl)]
pub struct Camera {
    pub view: Mat4f,
    pub proj: Mat4f,
    pub pos: Vec3f,
}
```

## Inherent Impls

Methods in `impl Type` blocks become free WGSL functions named `Type_method`:

```rust
impl Light {
    pub fn direction(self: Light, target: Vec3f) -> Vec3f {
        normalize(target - self.position)
    }
}
```

In Rust you call `Light::direction(light, target)`. In WGSL this becomes:

```wgsl
fn Light_direction(self_1: Light, target: vec3<f32>) -> vec3<f32> {
  return normalize(target - self_1.position);
}
```

## Associated Constants

`const` items inside an `impl` block become associated constants in WGSL:

```rust
impl Light {
    pub const MAX_COUNT: u32 = 64;
}
```

## Trait Impls

Trait definitions are Rust-only (the trait is not emitted to WGSL), but the methods in a trait `impl` are transpiled as if they were inherent methods. This lets you share method syntax between CPU and GPU code:

```rust
pub trait Packed {
    fn pack(self) -> u32;
}

impl Packed for Vec4f {
    pub fn pack(self) -> u32 {
        // bit-packing logic
        0u32
    }
}
```

The `pack` method transpiles to `Vec4f_pack`.

### Non-`pub` items in trait impls

Rust forbids `pub` on any item inside a trait impl (`E0449`), so wgsl-rs does not require `pub` on trait-impl methods or associated constants — only inherent impl blocks require `pub`. This matches Rust's own visibility rules:

```rust
pub trait SlabItem {
    const SLAB_SIZE: usize;
    fn read_at(slab_index: u32) -> Self;
}

impl SlabItem for u32 {
    const SLAB_SIZE: usize = 1;  // no `pub` — correct for a trait impl
    fn read_at(slab_index: u32) -> u32 {
        slab_index
    }
}
```

The associated const is mangled to `u32__1SLAB_SIZE` in WGSL (the `_1` escapes the underscore in `SLAB_SIZE` per the bijective mangling scheme).

### Associated Types in Trait Impls

Trait impls may include associated type definitions (`type Array = ...;`). The concrete type is resolved at monomorphization time and emitted as a WGSL `alias`:

```rust
pub trait SlabItem {
    const SLAB_SIZE: usize;
    type Array: Default;
    fn to_array(data: Self) -> Self::Array;
    fn from_array(arr: Self::Array) -> Self;
}

impl SlabItem for u32 {
    const SLAB_SIZE: usize = 1;
    type Array = [u32; 1];
    fn to_array(data: Self) -> Self::Array {
        [data]
    }
    fn from_array(arr: Self::Array) -> Self {
        arr[0]
    }
}
```

This produces:

```wgsl
alias u32_Array = array<u32, 1>;

fn u32__1to_array(data: u32) -> array<u32, 1> {
    return array(data);
}

fn u32__1from_array(arr: array<u32, 1>) -> u32 {
    return arr[0];
}
```

`Self::Array` in method signatures is resolved to the concrete type (`[u32; 1]` → `array<u32, 1>`) before rendering.

## Enums

`#[repr(u32)]` enums with explicit discriminants transpile to a `u32` alias plus `const` variants:

```rust
#[repr(u32)]
pub enum Mode {
    Add = 0,
    Multiply = 1,
    Screen = 2,
}
```

```wgsl
alias Mode = u32;
const Add: Mode = 0;
const Multiply: Mode = 1;
const Screen: Mode = 2;
```

Enums are used with `match` (see [Control Flow](./control-flow.md)), which transpiles to a WGSL `switch`.