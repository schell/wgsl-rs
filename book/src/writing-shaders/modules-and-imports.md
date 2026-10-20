# Modules and Imports

Every wgsl-rs shader module is a `#[wgsl] pub mod name { ... }`. The module boundary defines what the macro transpiles and how shaders reference each other.

## Module Structure

```rust
#[wgsl]
pub mod lighting {
    use wgsl_rs::std::*;

    pub fn diffuse(n: Vec3f, l: Vec3f) -> f32 {
        max(dot(n, l), 0.0)
    }
}
```

The generated WGSL is emitted into `WGSL_SOURCE` for that module.

## Glob Imports Only

wgsl-rs supports glob imports exclusively. Named imports are not transpiled. The two valid import forms are:

```rust
use wgsl_rs::std::*;        // standard WGSL types and builtins
use super::other_module::*; // another #[wgsl] module in the same parent
```

`wgsl_rs::std` provides the scalar/vector/matrix types (`Vec2f`, `Vec3f`, `Vec4f`, `vec2f`, `Mat4f`, ...), builtins, and texture types.

## Importing Other wgsl Modules

A module imported via `use super::other_module::*` must itself be a `#[wgsl]` module. This lets you split shaders across files and call functions defined elsewhere:

```rust
#[wgsl]
pub mod math {
    use wgsl_rs::std::*;

    pub fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
        min(max(x, lo), hi)
    }
}

#[wgsl]
pub mod surface {
    use wgsl_rs::std::*;
    use super::math::*;

    pub fn roughness(r: f32) -> f32 {
        clampf(r, 0.0, 1.0)
    }
}
```

## Cross-Module Imports and Deduplication

When a module is imported by multiple sibling modules, wgsl-rs deduplicates the generated functions so each WGSL function appears only once in the final output. You do not need to manage inclusion guards.

## Re-exports and Prelude Modules

A wgsl module can re-export another wgsl module with a glob, giving downstream shaders a single import surface:

```rust
#[wgsl]
pub mod defs {
    use wgsl_rs::std::*;

    pub struct Id<T> {
        pub index: u32,
        pub phantom: PhantomData<T>,
    }

    impl<T> Id<T> {
        pub fn new(i: u32) -> Id<T> {
            Id { index: i, phantom: PhantomData }
        }
    }
}

#[wgsl]
pub mod prelude {
    pub use super::defs::*;
}

#[wgsl]
pub mod shader {
    use super::prelude::*;

    pub fn my_id() -> Id<f32> {
        Id::<f32>::new(0)
    }
}
```

Everything a glob import brings into scope also travels through re-export chains: concrete items and generic templates alike. Instantiating `Id<f32>` through `prelude` assembles the same WGSL as importing `defs` directly. Chains of any depth work, and a defining module reached through several paths is imported only once.

If two different defining modules are both reachable through your imports and both define a template with the same name, instantiation fails with an ambiguity error. Import the defining module directly to pick one.

## Doc Comments

Inner doc comments (`//!`) at the top of a module are preserved in the generated WGSL as comments:

```rust
#[wgsl]
pub mod kernel {
    //! Compute lighting contribution for a single light.
    use wgsl_rs::std::*;
}
```

Outer doc comments (`///`) on items are not emitted into the WGSL; they stay in the Rust docs.