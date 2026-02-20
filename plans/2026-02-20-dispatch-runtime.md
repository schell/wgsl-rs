# CPU-Side Shader Dispatch Runtime + Derivative Builtins

**Date**: 2026-02-20
**Branch**: `derivative-builtins`
**Status**: Plan

## Motivation

WGSL derivative builtins (`dpdx`, `dpdy`, `fwidth` and their coarse/fine variants)
compute screen-space partial derivatives by comparing values across neighboring
fragment invocations in a 2x2 quad. On the GPU this is hardware-level. On the CPU,
we have no equivalent — there is no dispatch infrastructure at all.

The same gap exists for compute shaders: `workgroup_barrier()` is a no-op,
`workgroup_uniform_load()` works only by accident (single-threaded access),
and there is no `dispatch_workgroups()` function to invoke a compute shader
across a grid of invocations with correct builtin values.

This plan addresses both problems with a unified CPU-side shader dispatch runtime.

## Goals

1. Implement all 9 WGSL derivative builtin functions with correct WGSL transpilation
2. Build a CPU-side dispatch runtime (`dispatch_workgroups`, `dispatch_fragments`)
   that mirrors `wgpu`'s API
3. Make synchronization builtins (`workgroup_barrier`, etc.) functional during dispatch
4. Make derivative builtins functional during fragment dispatch

## Non-Goals

- Texture sampling on the CPU (separate concern)
- Full rasterization pipeline (we dispatch fragments, not triangles)
- Performance parity with GPU execution

---

## Phase 1: Derivative Builtins (WGSL Transpilation + Stubs)

Ship the GPU-side support immediately. CPU implementations start as zero-returning
stubs and become functional in Phase 2.

### 1a. New module: `crates/wgsl-rs/src/std/derivative.rs`

9 traits + 9 free functions following the one-trait-per-function pattern.

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

**Phase 1 CPU behavior**: Return zero (`0.0` for scalars, zero vector for vectors).
Add `// TODO: implement with quad runtime (Phase 2)` comments.

**Free functions** (Rust API):
```rust
pub fn dpdx<T: DerivativeBuiltinDpdx>(e: T) -> T { <T as DerivativeBuiltinDpdx>::dpdx(e) }
pub fn dpdx_coarse<T: DerivativeBuiltinDpdxCoarse>(e: T) -> T { ... }
pub fn dpdx_fine<T: DerivativeBuiltinDpdxFine>(e: T) -> T { ... }
pub fn dpdy<T: DerivativeBuiltinDpdy>(e: T) -> T { ... }
pub fn dpdy_coarse<T: DerivativeBuiltinDpdyCoarse>(e: T) -> T { ... }
pub fn dpdy_fine<T: DerivativeBuiltinDpdyFine>(e: T) -> T { ... }
pub fn fwidth<T: DerivativeBuiltinFwidth>(e: T) -> T { ... }
pub fn fwidth_coarse<T: DerivativeBuiltinFwidthCoarse>(e: T) -> T { ... }
pub fn fwidth_fine<T: DerivativeBuiltinFwidthFine>(e: T) -> T { ... }
```

### 1b. Register in `crates/wgsl-rs/src/std.rs`

```rust
mod derivative;
pub use derivative::*;
```

### 1c. Name mappings in `crates/wgsl-rs-macros/src/builtins.rs`

Add 6 entries to `BUILTIN_CASE_NAME_MAP` (the 3 base names `dpdx`, `dpdy`, `fwidth`
match WGSL already and need no entry):

```rust
// Derivative builtins
("dpdx_coarse", "dpdxCoarse"),
("dpdx_fine", "dpdxFine"),
("dpdy_coarse", "dpdyCoarse"),
("dpdy_fine", "dpdyFine"),
("fwidth_coarse", "fwidthCoarse"),
("fwidth_fine", "fwidthFine"),
```

### 1d. Tests

| Test Location | What |
|---------------|------|
| `derivative.rs` `#[cfg(test)]` | Verify stubs return zero for all types |
| `builtins.rs` `#[cfg(test)]` | Verify name lookups and reservation |
| `parse.rs` `#[cfg(test)]` | Verify `dpdx_coarse(x)` transpiles to `dpdxCoarse(x)` |
| `examples.rs` | Fragment shader example using all 9 derivative builtins, validated by naga |

---

## Phase 2: CPU-Side Dispatch Runtime

### 2a. Dependencies

Add to `crates/wgsl-rs/Cargo.toml`:
```toml
[dependencies]
rayon = "1"
```

### 2b. New module: `crates/wgsl-rs/src/std/runtime.rs`

Central dispatch infrastructure. Submodules may be warranted but start flat.

### 2c. Thread-Local Context

The runtime uses thread-local storage to provide context to builtins during dispatch:

```rust
use std::cell::{Cell, RefCell};
use std::sync::Arc;

thread_local! {
    /// Barrier for workgroup synchronization (compute dispatch).
    static WORKGROUP_BARRIER: RefCell<Option<Arc<std::sync::Barrier>>> = RefCell::new(None);

    /// Quad context for derivative computation (fragment dispatch).
    static QUAD_CONTEXT: RefCell<Option<Arc<QuadContext>>> = RefCell::new(None);

    /// This invocation's index within the quad (0-3).
    static QUAD_INDEX: Cell<u8> = Cell::new(0);

    /// Compute builtin values for the current invocation.
    static COMPUTE_BUILTINS: RefCell<Option<ComputeBuiltins>> = RefCell::new(None);
}
```

Builtins check these thread-locals. Outside of dispatch, they gracefully degrade
(barriers are no-ops, derivatives return zero). This preserves backward compatibility
with direct function calls outside a dispatch context.

### 2d. Compute Dispatch

```rust
/// Builtin values provided to each compute shader invocation.
#[derive(Clone, Copy)]
pub struct ComputeBuiltins {
    pub global_invocation_id: Vec3u,
    pub local_invocation_id: Vec3u,
    pub local_invocation_index: u32,
    pub workgroup_id: Vec3u,
    pub num_workgroups: Vec3u,
}

/// Dispatches a compute shader, similar to wgpu's
/// `ComputePass::dispatch_workgroups(x, y, z)`.
///
/// Executes `shader_fn` for every invocation across all workgroups.
/// Within each workgroup, invocations run in parallel on a rayon thread pool.
/// `workgroup_barrier()` calls within `shader_fn` will synchronize correctly.
pub fn dispatch_workgroups<F>(
    workgroup_count: (u32, u32, u32),
    workgroup_size: (u32, u32, u32),
    shader_fn: F,
)
where
    F: Fn(ComputeBuiltins) + Send + Sync,
{
    let (cx, cy, cz) = workgroup_count;
    let (sx, sy, sz) = workgroup_size;
    let invocations_per_workgroup = (sx * sy * sz) as usize;
    let num_workgroups = vec3u(cx, cy, cz);

    // Iterate workgroups sequentially (workgroup variables are scoped per-workgroup)
    for wz in 0..cz {
        for wy in 0..cy {
            for wx in 0..cx {
                let workgroup_id = vec3u(wx, wy, wz);
                let barrier = Arc::new(std::sync::Barrier::new(invocations_per_workgroup));

                // Use rayon to run all invocations within a workgroup in parallel
                rayon::scope(|s| {
                    for lz in 0..sz {
                        for ly in 0..sy {
                            for lx in 0..sx {
                                let barrier = barrier.clone();
                                let local_id = vec3u(lx, ly, lz);
                                let local_index = lx + ly * sx + lz * sx * sy;
                                let global_id = vec3u(
                                    wx * sx + lx,
                                    wy * sy + ly,
                                    wz * sz + lz,
                                );

                                s.spawn(move |_| {
                                    // Set thread-local context
                                    WORKGROUP_BARRIER.with(|b| {
                                        *b.borrow_mut() = Some(barrier);
                                    });
                                    COMPUTE_BUILTINS.with(|b| {
                                        *b.borrow_mut() = Some(ComputeBuiltins {
                                            global_invocation_id: global_id,
                                            local_invocation_id: local_id,
                                            local_invocation_index: local_index,
                                            workgroup_id,
                                            num_workgroups,
                                        });
                                    });

                                    shader_fn(ComputeBuiltins {
                                        global_invocation_id: global_id,
                                        local_invocation_id: local_id,
                                        local_invocation_index: local_index,
                                        workgroup_id,
                                        num_workgroups,
                                    });

                                    // Clean up thread-local context
                                    WORKGROUP_BARRIER.with(|b| { *b.borrow_mut() = None; });
                                    COMPUTE_BUILTINS.with(|b| { *b.borrow_mut() = None; });
                                });
                            }
                        }
                    }
                });

                // Reset workgroup variables between workgroups
                // (TBD: mechanism for this)
            }
        }
    }
}
```

### 2e. Update Synchronization Builtins

Replace no-ops in `synchronization.rs`:

```rust
pub fn workgroup_barrier() {
    crate::std::runtime::WORKGROUP_BARRIER.with(|b| {
        if let Some(barrier) = b.borrow().as_ref() {
            barrier.wait();
        }
        // No-op outside dispatch (backward compatible)
    });
}

pub fn storage_barrier() {
    // Same as workgroup_barrier on CPU — all memory is coherent,
    // we just need the execution synchronization.
    workgroup_barrier();
}

pub fn texture_barrier() {
    workgroup_barrier();
}
```

### 2f. Fragment Quad Dispatch

```rust
/// A 2x2 quad of fragment invocations.
///
/// Quad layout (matching GPU convention):
///
///     [0] (x,y)     [1] (x+1,y)
///     [2] (x,y+1)   [3] (x+1,y+1)
///
struct QuadContext {
    /// Value exchange slots. Each derivative call uses a fresh generation.
    /// Indexed by [quad_member_index], stores the deposited value.
    slots: [Mutex<[f32; 4]>; 4],  // 4 members, each can deposit up to vec4 (4 components)
    /// Barrier for synchronizing quad members at derivative call sites.
    barrier: std::sync::Barrier,
}

impl QuadContext {
    fn new() -> Self {
        QuadContext {
            slots: std::array::from_fn(|_| Mutex::new([0.0; 4])),
            barrier: std::sync::Barrier::new(4),
        }
    }

    /// Deposit a scalar value and compute dpdx (horizontal finite difference).
    fn dpdx_f32(&self, quad_idx: u8, value: f32) -> f32 {
        // 1. Store our value
        *self.slots[quad_idx as usize].lock().unwrap() = [value, 0.0, 0.0, 0.0];
        // 2. Wait for all 4 to deposit
        self.barrier.wait();
        // 3. Compute horizontal difference
        //    Quad members 0,2 are left column; 1,3 are right column
        let left_idx = (quad_idx & !1) as usize;   // 0 or 2
        let right_idx = (quad_idx | 1) as usize;   // 1 or 3
        let left = self.slots[left_idx].lock().unwrap()[0];
        let right = self.slots[right_idx].lock().unwrap()[0];
        // 4. Wait before slots can be reused
        self.barrier.wait();
        right - left
    }

    /// Deposit a scalar value and compute dpdy (vertical finite difference).
    fn dpdy_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize].lock().unwrap() = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let top_idx = (quad_idx & !2) as usize;    // 0 or 1
        let bottom_idx = (quad_idx | 2) as usize;  // 2 or 3
        let top = self.slots[top_idx].lock().unwrap()[0];
        let bottom = self.slots[bottom_idx].lock().unwrap()[0];
        self.barrier.wait();
        bottom - top
    }

    // Vector variants decompose into components and use the same slots.
    // fn dpdx_vec2f(...), fn dpdx_vec3f(...), fn dpdx_vec4f(...), etc.
}

/// Dispatches fragment shader invocations over a 2D grid, processing in 2x2 quads.
///
/// - `width`, `height`: dimensions of the fragment grid
/// - `input_fn`: generates per-fragment input from (x, y) coordinates
/// - `shader_fn`: the fragment shader function
///
/// Returns a width x height grid of outputs. Partial quads at edges use helper
/// invocations (their outputs are discarded).
pub fn dispatch_fragments<I, O, F>(
    width: u32,
    height: u32,
    input_fn: impl Fn(u32, u32) -> I + Send + Sync,
    shader_fn: F,
) -> Vec<Vec<Option<O>>>
where
    I: Send,
    O: Send,
    F: Fn(FragmentBuiltins, I) -> O + Send + Sync,
{
    let padded_w = (width + 1) & !1;   // round up to even
    let padded_h = (height + 1) & !1;

    // Process quads in parallel using rayon
    // Each quad spawns 4 threads via rayon::scope
    // Results collected into the output grid
    // Fragments outside (width, height) are helper invocations — output discarded
    todo!()
}

pub struct FragmentBuiltins {
    /// The fragment's position in window coordinates.
    pub position: Vec4f,
    /// Whether this is a front-facing fragment.
    pub front_facing: bool,
    /// The sample index (0 for non-multisampled).
    pub sample_index: u32,
    /// The sample mask.
    pub sample_mask: u32,
}
```

### 2g. Update Derivative Implementations

Replace zero-stubs with quad-context-aware code:

```rust
impl DerivativeBuiltinDpdx for f32 {
    fn dpdx(self) -> Self {
        QUAD_CONTEXT.with(|ctx| {
            match ctx.borrow().as_ref() {
                Some(qctx) => {
                    let idx = QUAD_INDEX.with(|i| i.get());
                    qctx.dpdx_f32(idx, self)
                }
                None => 0.0  // Outside dispatch: return zero
            }
        })
    }
}
```

Coarse/fine variants:
- `dpdx` delegates to either coarse or fine (implementation-defined; we pick fine)
- `dpdx_coarse`: all 4 quad members get the same derivative (use top-left pair)
- `dpdx_fine`: each row computes its own derivative (per-row difference)

`fwidth` implementations delegate to `dpdx` + `dpdy`:
```rust
impl DerivativeBuiltinFwidth for f32 {
    fn fwidth(self) -> Self {
        dpdx(self).abs() + dpdy(self).abs()
    }
}
```

Note on `fwidth`: it requires two synchronization rounds for the same input value `e`.
Since `e: Copy` for all derivative-supported types, this works — `dpdx(e)` deposits,
syncs, computes; then `dpdy(e)` deposits the same value again, syncs, computes.

---

## Phase 3: Tests and Examples

### Compute Dispatch Tests

```rust
#[test]
fn compute_dispatch_basic() {
    use std::sync::atomic::{AtomicU32, Ordering};
    let counter = AtomicU32::new(0);
    dispatch_workgroups((2, 1, 1), (4, 1, 1), |builtins| {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(counter.load(Ordering::Relaxed), 8); // 2 workgroups * 4 invocations
}

#[test]
fn compute_dispatch_barrier_synchronizes() {
    // Verify that workgroup_barrier() actually forces synchronization:
    // Have invocations write to a shared array, barrier, then read.
}

#[test]
fn compute_builtins_correct() {
    // Verify global_invocation_id, local_invocation_id, etc. are correct
    // for various workgroup counts and sizes.
}
```

### Fragment Derivative Tests

```rust
#[test]
fn dpdx_linear_ramp() {
    // Input: f(x, y) = x (linear ramp in X)
    // Expected: dpdx = 1.0, dpdy = 0.0
    dispatch_fragments(4, 4, |x, y| x as f32, |builtins, value| {
        (dpdx(value), dpdy(value))
    });
    // Verify all fragments report dpdx=1.0, dpdy=0.0
}

#[test]
fn dpdy_linear_ramp() {
    // Input: f(x, y) = y
    // Expected: dpdx = 0.0, dpdy = 1.0
}

#[test]
fn fwidth_diagonal() {
    // Input: f(x, y) = x + y
    // Expected: fwidth = |1| + |1| = 2.0
}
```

### Integration Example

A module in `examples.rs` that uses derivative builtins in a fragment shader,
validated by naga at compile time. The CPU-side test calls the same function
through `dispatch_fragments` and verifies the results.

---

## Open Design Questions

### 1. Quad Context Type Erasure

Derivative calls can operate on `f32`, `Vec2f`, `Vec3f`, or `Vec4f`. The quad
context needs to handle all of them. Options:

- **Fixed slots**: Use `[f32; 4]` per quad member (enough for `Vec4f`). Scalar
  uses slot 0, `Vec2f` uses slots 0-1, etc. Simple but wastes space for scalars.
- **Type-erased slots**: Use `Box<dyn Any>`. Flexible but has allocation overhead.
- **Generic context**: `QuadContext<T>` parameterized by type. Clean but requires
  different context per derivative call site if types differ.

**Recommendation**: Fixed `[f32; 4]` slots. The waste is negligible (16 bytes per
quad member) and it avoids allocation and type erasure complexity.

### 2. Multiple Derivative Calls Per Invocation

A shader may call `dpdx` multiple times on different values. The double-barrier
pattern (deposit -> barrier -> compute -> barrier) handles this naturally — each
derivative call is a self-contained synchronization round. The second barrier
ensures slots are safe to reuse before the next derivative call.

### 3. Helper Invocations at Edges

When the framebuffer dimensions aren't multiples of 2, partial quads at edges
need helper invocations. The dispatch runtime pads to even dimensions and marks
edge invocations as helpers. Helper outputs are discarded.

### 4. Workgroup Variable Lifetime

Currently `Workgroup<T>` uses global statics with `LazyLock<Arc<RwLock<Option<T>>>>`.
In a dispatch runtime, workgroup variables should be scoped per-workgroup (reset
between workgroups). Options:

- **Reset after each workgroup**: After `rayon::scope` completes for a workgroup,
  set all workgroup variables to `None`. Requires knowing which variables exist.
- **Per-workgroup allocation**: Create fresh workgroup variables per workgroup.
  Would require a fundamentally different `Workgroup<T>` design.
- **User-managed**: Document that users must initialize workgroup variables at the
  start of their compute shader. Matches GPU behavior (workgroup vars are undefined
  at workgroup start).

**Recommendation**: User-managed for now. Document that workgroup variables are
in an undefined state at the start of each workgroup, matching WGSL spec behavior.
The global statics persist across workgroups, which is incorrect but acceptable as
a known limitation.

### 5. Interaction with `fwidth` Implementation

`fwidth(e) = abs(dpdx(e)) + abs(dpdy(e))` requires two synchronization rounds
for the same input value `e`. Since `e: Copy` for all derivative-supported types,
this works: `dpdx(e)` deposits, syncs, computes; then `dpdy(e)` deposits the same
value again, syncs, computes. The two rounds are independent.

An optimization would be to deposit once and compute both derivatives in a single
round, but this adds complexity. Defer this optimization.

---

## File Changes Summary

| File | Phase | Change |
|------|-------|--------|
| `crates/wgsl-rs/Cargo.toml` | 2 | Add `rayon = "1"` |
| `crates/wgsl-rs/src/std.rs` | 1+2 | Add `mod derivative; mod runtime;` + `pub use` |
| `crates/wgsl-rs/src/std/derivative.rs` | 1 (stubs), 2 (real) | **New**: 9 traits, 9 free fns, impls for f32/Vec2f/Vec3f/Vec4f |
| `crates/wgsl-rs/src/std/runtime.rs` | 2 | **New**: dispatch_workgroups, dispatch_fragments, QuadContext, thread-locals |
| `crates/wgsl-rs/src/std/synchronization.rs` | 2 | Update barriers to use thread-local context |
| `crates/wgsl-rs-macros/src/builtins.rs` | 1 | Add 6 name mappings + tests |
| `crates/wgsl-rs-macros/src/parse.rs` | 1 | Add transpilation tests |
| `crates/example/src/examples.rs` | 1+3 | Add derivative example module + dispatch tests |
