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

## Import Forms

wgsl-rs supports glob imports of a whole module, named imports of a single item, and grouped imports of several at once:

```rust
use wgsl_rs::std::*;                    // standard WGSL types and builtins
use super::other_module::*;             // everything from another #[wgsl] module
use super::other_module::specific_fn;   // one item, by name
use super::other_module::{Const, Struct};   // several items, one statement
```

`wgsl_rs::std` provides the scalar/vector/matrix types (`Vec2f`, `Vec3f`, `Vec4f`, `vec2f`, `Mat4f`, ...), builtins, and texture types.

A named import and a glob of the same module assemble identically: the whole source folds in either way, so every definition the imported item depends on is present. Importing `specific_fn` does not cherry-pick — the source module's other items land in the assembly too.

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

Re-export chains follow `pub`: a downstream module can only resolve names along `pub use` edges. A plain `use` inside a prelude is a private import — its items still assemble, because the prelude's own code needs them, but downstream modules cannot reach through it, exactly as in Rust. `pub(crate)`, `pub(super)` and `pub(in path)` count as private here: a consumer in another crate cannot traverse them in Rust, so the transpiler does not either. Spell `pub use` when a prelude is meant to forward names.

Re-exports work by name too. A named re-export (`pub use path::item;`) exposes exactly one name of the source; a glob re-export (`pub use path::*;`) exposes all of them. Generic items instantiated through the chain resolve by the exported name — this is the shape a prelude is made of:

```rust
#[wgsl]
pub mod prelude {
    pub use super::defs::Id;
}

#[wgsl]
pub mod shader {
    use super::prelude::*; // Id resolves through the named re-export
    // ...
}
```

This also composes whole pipelines in a single module: a fragment module that re-exports a vertex entry by name carries both entries and both stages' bindings in one assembled source — one linkage, one pipeline layout, no runtime merge.

Renaming (`use path::item as other;`) is not transpiled; spell each import under its own name. Grouped imports (`use path::{a, b};`) transpile exactly like the split form, which also makes them the fmt-proof spelling: rustfmt's `imports_granularity` regenerates the grouped form from consecutive same-root imports, so writing groups directly avoids `#[rustfmt::skip]`. A group shares its statement's visibility — `pub use path::{a, b};` re-exports both — so spell separate statements when individual imports need different visibility.

If two different defining modules are both reachable through your imports and both define a template with the same name, instantiation fails with an ambiguity error. Import the defining module directly to pick one. Privately imported colliding definitions do not trigger this — they are not reachable through the hop.

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