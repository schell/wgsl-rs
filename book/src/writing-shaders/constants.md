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

## Initializers Must Be Const-Expressions

WGSL requires a const initializer to be a const-expression. Value
constructors (`vec3f(..)`, `mat2x2f(..)`, struct literals) and `@const`
builtins (`min`, `count_leading_zeros`, ...) are fine; calling a user
function is not, and user functions can never be WGSL const-functions
(WGSL §11.3). Such a call is rejected at compile time:

```rust
pub const fn thing() -> MyThing { MyThing { times: 0 } }

// error: const 'OTHER_THING' initializer calls 'MyThing::thing', but
// WGSL const initializers are const-expressions: only value
// constructors and const builtins are callable, never user functions;
// use the struct-literal form ... or literal values instead
pub const OTHER_THING: MyThing = MyThing::thing();

// Use the struct-literal form instead:
pub const OK_THING: MyThing = MyThing { times: 0 };
```