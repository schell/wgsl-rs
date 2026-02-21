# Derivative Builtins

**Date**: 2026-02-20
**Branch**: `derivative-builtins`
**Status**: Plan
**Depends on**: Nothing
**Followed by**: [Dispatch Runtime](2026-02-20-dispatch-runtime.md)

## Motivation

WGSL defines 9 derivative builtin functions for fragment shaders that compute
partial derivatives of values across neighboring invocations in a 2x2 quad.
These are essential for anti-aliasing, procedural textures, and LOD calculations.

This plan implements all 9 functions with correct WGSL transpilation and stub
CPU-side behavior. The stubs return zero, which is safe and predictable. A
follow-up plan ([Dispatch Runtime](2026-02-20-dispatch-runtime.md)) will add a
CPU-side quad execution runtime that makes these functions compute real derivatives.

## WGSL Reference

All derivative functions are fragment-shader-only and return `T` where `T` is
`f32`, `vec2<f32>`, `vec3<f32>`, or `vec4<f32>`.

| Function | Description |
|----------|-------------|
| `dpdx(e)` | Partial derivative w.r.t. window X. Same as `dpdxFine` or `dpdxCoarse`. |
| `dpdxCoarse(e)` | Coarse partial derivative w.r.t. X (may use fewer unique positions). |
| `dpdxFine(e)` | Fine partial derivative w.r.t. X (per-pixel granularity). |
| `dpdy(e)` | Partial derivative w.r.t. window Y. Same as `dpdyFine` or `dpdyCoarse`. |
| `dpdyCoarse(e)` | Coarse partial derivative w.r.t. Y. |
| `dpdyFine(e)` | Fine partial derivative w.r.t. Y. |
| `fwidth(e)` | Returns `abs(dpdx(e)) + abs(dpdy(e))`. |
| `fwidthCoarse(e)` | Returns `abs(dpdxCoarse(e)) + abs(dpdyCoarse(e))`. |
| `fwidthFine(e)` | Returns `abs(dpdxFine(e)) + abs(dpdyFine(e))`. |

Spec: https://gpuweb.github.io/gpuweb/wgsl/#derivative-builtin-functions

---

## Implementation

### Step 1. New module: `crates/wgsl-rs/src/std/derivative.rs`

9 traits + 9 free functions following the existing one-trait-per-function pattern
(see `NumericBuiltinAbs` in `numeric.rs` for the canonical example).

**Traits**:

| Trait | Method | WGSL Name |
|-------|--------|-----------|
| `DerivativeBuiltinDpdx` | `fn dpdx(self) -> Self` | `dpdx` |
| `DerivativeBuiltinDpdxCoarse` | `fn dpdx_coarse(self) -> Self` | `dpdxCoarse` |
| `DerivativeBuiltinDpdxFine` | `fn dpdx_fine(self) -> Self` | `dpdxFine` |
| `DerivativeBuiltinDpdy` | `fn dpdy(self) -> Self` | `dpdy` |
| `DerivativeBuiltinDpdyCoarse` | `fn dpdy_coarse(self) -> Self` | `dpdyCoarse` |
| `DerivativeBuiltinDpdyFine` | `fn dpdy_fine(self) -> Self` | `dpdyFine` |
| `DerivativeBuiltinFwidth` | `fn fwidth(self) -> Self` | `fwidth` |
| `DerivativeBuiltinFwidthCoarse` | `fn fwidth_coarse(self) -> Self` | `fwidthCoarse` |
| `DerivativeBuiltinFwidthFine` | `fn fwidth_fine(self) -> Self` | `fwidthFine` |

**Type support**: `f32`, `Vec2f`, `Vec3f`, `Vec4f`

**CPU behavior (stubs)**: Return zero (`0.0` for scalars, zero vector for vectors).
Mark with `// TODO: implement with quad runtime` comments referencing the dispatch
runtime plan.

**Free functions** (Rust API matching WGSL names in snake_case):
```rust
pub fn dpdx<T: DerivativeBuiltinDpdx>(e: T) -> T { <T as DerivativeBuiltinDpdx>::dpdx(e) }
pub fn dpdx_coarse<T: DerivativeBuiltinDpdxCoarse>(e: T) -> T { <T as DerivativeBuiltinDpdxCoarse>::dpdx_coarse(e) }
pub fn dpdx_fine<T: DerivativeBuiltinDpdxFine>(e: T) -> T { <T as DerivativeBuiltinDpdxFine>::dpdx_fine(e) }
pub fn dpdy<T: DerivativeBuiltinDpdy>(e: T) -> T { <T as DerivativeBuiltinDpdy>::dpdy(e) }
pub fn dpdy_coarse<T: DerivativeBuiltinDpdyCoarse>(e: T) -> T { <T as DerivativeBuiltinDpdyCoarse>::dpdy_coarse(e) }
pub fn dpdy_fine<T: DerivativeBuiltinDpdyFine>(e: T) -> T { <T as DerivativeBuiltinDpdyFine>::dpdy_fine(e) }
pub fn fwidth<T: DerivativeBuiltinFwidth>(e: T) -> T { <T as DerivativeBuiltinFwidth>::fwidth(e) }
pub fn fwidth_coarse<T: DerivativeBuiltinFwidthCoarse>(e: T) -> T { <T as DerivativeBuiltinFwidthCoarse>::fwidth_coarse(e) }
pub fn fwidth_fine<T: DerivativeBuiltinFwidthFine>(e: T) -> T { <T as DerivativeBuiltinFwidthFine>::fwidth_fine(e) }
```

**Implementation pattern** (use macros as other builtins do):
```rust
macro_rules! impl_derivative_zero_scalar {
    ($trait:ident, $method:ident) => {
        impl $trait for f32 {
            // TODO: implement with quad runtime
            fn $method(self) -> Self { 0.0 }
        }
    };
}

macro_rules! impl_derivative_zero_vec {
    ($trait:ident, $method:ident, $($ty:ty),+) => {
        $(
            impl $trait for $ty {
                // TODO: implement with quad runtime
                fn $method(self) -> Self { Self::default() }
            }
        )+
    };
}
```

### Step 2. Register in `crates/wgsl-rs/src/std.rs`

Add alongside existing module declarations:
```rust
mod derivative;
pub use derivative::*;
```

### Step 3. Name mappings in `crates/wgsl-rs-macros/src/builtins.rs`

Add 6 entries to `BUILTIN_CASE_NAME_MAP`. The 3 base names (`dpdx`, `dpdy`,
`fwidth`) are identical in Rust and WGSL, so they need no mapping and pass
through unchanged.

```rust
// Derivative builtins
("dpdx_coarse", "dpdxCoarse"),
("dpdx_fine", "dpdxFine"),
("dpdy_coarse", "dpdyCoarse"),
("dpdy_fine", "dpdyFine"),
("fwidth_coarse", "fwidthCoarse"),
("fwidth_fine", "fwidthFine"),
```

### Step 4. Tests

| Location | Test | Verifies |
|----------|------|----------|
| `crates/wgsl-rs/src/std/derivative.rs` | `sanity_dpdx`, `sanity_dpdy`, `sanity_fwidth` (and coarse/fine variants) | Stubs return zero for `f32`, `Vec2f`, `Vec3f`, `Vec4f` |
| `crates/wgsl-rs-macros/src/builtins.rs` | `lookup_derivative_builtins`, `derivative_names_are_reserved` | Name mappings and reservation work |
| `crates/wgsl-rs-macros/src/parse.rs` | `derivative_builtin_translates_to_wgsl` | `dpdx_coarse(x)` transpiles to `dpdxCoarse(x)`, `dpdx(x)` stays `dpdx(x)`, etc. |
| `crates/example/src/examples.rs` | `derivative_example` module | A `#[fragment]` function using all 9 derivative builtins compiles and passes naga validation |

### Step 5. Verify

```bash
cargo fmt && cargo clippy
cargo test
cargo run -p example -- source derivative_example
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `crates/wgsl-rs/src/std.rs` | Add `mod derivative;` + `pub use derivative::*;` |
| `crates/wgsl-rs/src/std/derivative.rs` | **New**: 9 traits, 9 free functions, impls for f32/Vec2f/Vec3f/Vec4f, unit tests |
| `crates/wgsl-rs-macros/src/builtins.rs` | Add 6 name mappings to `BUILTIN_CASE_NAME_MAP` + tests |
| `crates/wgsl-rs-macros/src/parse.rs` | Add transpilation tests for derivative builtins |
| `crates/example/src/examples.rs` | Add `derivative_example` fragment shader module |
