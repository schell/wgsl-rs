# Derivative Builtins

**Date**: 2026-02-20
**Branch**: `feat/derivative-builtins`
**Status**: Implemented
**Depends on**: [Dispatch Runtime](2026-02-20-dispatch-runtime.md) (merged)

## Motivation

WGSL defines 9 derivative builtin functions for fragment shaders that compute
partial derivatives of values across neighboring invocations in a 2x2 quad.
These are essential for anti-aliasing, procedural textures, and LOD calculations.

This plan was originally written before the dispatch runtime was merged and
called for zero-returning stubs. Since the dispatch runtime now provides a
complete `QuadContext` with 2x2 quad synchronization, the implementation uses
**real derivative computation** on the CPU from the start — no stubs needed.

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

### CPU Behavior

Default variants (`dpdx`, `dpdy`, `fwidth`) delegate to their fine counterparts.

When running inside `dispatch_fragments`, derivatives are computed using the
`QuadContext` double-barrier protocol:
1. All 4 quad invocations deposit their value.
2. Barrier — all values visible.
3. Each invocation reads neighbors and computes the derivative.
4. Barrier — slots safe to reuse.

When called outside a dispatch context (no `QuadContext` installed), all
derivative functions return zero as a safe fallback.

### Step 1. New module: `crates/wgsl-rs/src/std/derivative.rs`

9 traits + 9 free functions following the existing one-trait-per-function pattern.

**Traits and type support**: `f32`, `Vec2f`, `Vec3f`, `Vec4f` — see the
implementation for the full list.

Scalar `f32` implementations use `with_quad_context()` to access the
thread-local `QuadContext` installed by `dispatch_fragments`. Vector
implementations use the multi-component methods (`dpdx_fine_components`, etc.).

### Step 2. Register in `crates/wgsl-rs/src/std.rs`

### Step 3. Name mappings in `crates/wgsl-rs-macros/src/builtins.rs`

6 entries added to `BUILTIN_CASE_NAME_MAP`. The 3 base names (`dpdx`, `dpdy`,
`fwidth`) are identical in Rust and WGSL, so they pass through unchanged.

### Step 4. Tests

| Location | Test | Verifies |
|----------|------|----------|
| `crates/wgsl-rs/src/std/derivative.rs` | `dpdx_outside_dispatch_returns_zero`, `dpdx_fine_scalar_in_dispatch`, `fwidth_scalar_both_axes`, etc. | Correct derivative computation inside dispatch, zero fallback outside |
| `crates/wgsl-rs-macros/src/builtins.rs` | `lookup_derivative_builtins`, `derivative_names_are_reserved` | Name mappings and reservation |
| `crates/wgsl-rs-macros/src/parse.rs` | `derivative_builtin_translates_to_wgsl`, `derivative_base_names_pass_through` | WGSL transpilation |
| `crates/example/src/examples.rs` | `derivative_example` module | Naga validation of generated WGSL |
| `crates/gpu-tests/tests/derivative_comparison.rs` | `derivative_gpu_vs_cpu_basic`, `derivative_gpu_vs_cpu_fine_coarse`, `derivative_gpu_all_variants_agree_for_linear` | CPU dispatch runtime matches GPU |

### Step 5. Verify

```bash
cargo fmt && cargo clippy
cargo test
cargo test -p gpu-tests
cargo run -p example -- source derivative_example
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `crates/wgsl-rs/src/std.rs` | Add `mod derivative;` + `pub use derivative::*;` |
| `crates/wgsl-rs/src/std/derivative.rs` | **New**: 9 traits, 9 free functions, impls using `QuadContext`, unit tests |
| `crates/wgsl-rs/src/std/runtime.rs` | Remove `#[allow(dead_code)]` from `with_quad_context` |
| `crates/wgsl-rs-macros/src/builtins.rs` | Add 6 name mappings + tests |
| `crates/wgsl-rs-macros/src/parse.rs` | Add transpilation tests |
| `crates/example/src/examples.rs` | Add `derivative_example` fragment shader module |
| `crates/gpu-tests/` | **New crate**: GPU vs CPU comparison tests |
| `Cargo.toml` | Add `gpu-tests` to workspace members |
