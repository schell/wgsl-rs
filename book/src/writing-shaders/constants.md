# Constants

Constants are declared with `const`. They transpile directly to WGSL `const` declarations.

## Module-Level

```rust
#[wgsl]
pub mod config {
    use wgsl_rs::std::*;

    pub const MAX_LIGHTS: u32 = 64;
    pub const TILE_SIZE: u32 = 16;
    pub const AMBIENT: Vec3f = vec3f(0.1, 0.1, 0.12);
}
```

```wgsl
const MAX_LIGHTS: u32 = 64;
const TILE_SIZE: u32 = 16;
const AMBIENT: vec3<f32> = vec3(0.1, 0.1, 0.12);
```

## Function-Level

`const` declared inside a function becomes a function-scoped WGSL `const`:

```rust
pub fn circle_area(r: f32) -> f32 {
    const PI: f32 = 3.14159265;
    PI * r * r
}
```

## Uses

Constants are commonly used for:

- Array sizes (`const N: u32 = 256; ... var<workgroup> buf: array<f32, N>;`)
- Configuration knobs (light counts, tile sizes)
- Fixed colors or directions

Note: WGSL `const` values are compile-time constants. For values that may vary per dispatch, use a uniform binding (see [Binding Macros](./binding-macros.md)).