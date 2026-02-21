# CPU-Side Shader Dispatch Runtime

**Date**: 2026-02-20
**Status**: Plan

## Motivation

The `wgsl-rs` crate maintains two parallel representations: Rust code that runs on
the CPU and transpiled WGSL that runs on the GPU. Currently the CPU side has no
dispatch infrastructure — shader entry points are plain functions with no concept of
workgroups, invocations, or quads. This means:

- `workgroup_barrier()` is a no-op (should synchronize invocations)
- `workgroup_uniform_load()` works only by accident (single-threaded access)
- Derivative builtins (`dpdx`, `dpdy`, `fwidth`) return zero (should compute
  real screen-space partial derivatives across 2x2 quads)
- There is no `dispatch_workgroups()` to run compute shaders across a grid
- Builtin values (`global_invocation_id`, `local_invocation_id`, etc.) have no
  injection mechanism

This plan adds a CPU-side dispatch runtime that mirrors `wgpu`'s API and makes
all of the above functional.

## Goals

1. `dispatch_workgroups()` — compute shader dispatch with correct builtins and barriers
2. `dispatch_fragments()` — fragment shader dispatch in 2x2 quads
3. Make synchronization builtins functional during dispatch
4. Allow derivative builtins to compute real finite differences during fragment dispatch, once they are implemented

## Non-Goals

- Texture sampling on the CPU (separate concern)
- Full rasterization pipeline (we dispatch fragments, not triangles)
- Performance parity with GPU execution
- Vertex shader dispatch (no inter-stage data flow yet)

---

## Step 1. Add `rayon` dependency

In `crates/wgsl-rs/Cargo.toml`:
```toml
[dependencies]
rayon = "1"
```

Rayon provides a work-stealing thread pool that handles variable-size dispatch
workloads well. All parallel iteration and scoped thread spawning uses rayon.

## Step 2. New module: `crates/wgsl-rs/src/std/runtime.rs`

Central dispatch infrastructure. Register in `std.rs`:
```rust
mod runtime;
pub use runtime::*;
```

### Thread-Local Context

The runtime communicates with builtins via thread-local storage. Builtins check
these thread-locals and gracefully degrade outside of dispatch (barriers are
no-ops, derivatives return zero), preserving backward compatibility.

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

## Step 3. Compute Dispatch

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
            }
        }
    }
}
```

## Step 4. Update Synchronization Builtins

Replace the no-ops in `synchronization.rs` with context-aware implementations:

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

## Step 5. Fragment Quad Dispatch

### QuadContext

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
    /// Indexed by [quad_member_index], stores the deposited value as up to
    /// 4 f32 components (enough for vec4<f32>).
    slots: [Mutex<[f32; 4]>; 4],
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
    ///
    /// Uses the double-barrier pattern:
    /// 1. All 4 invocations deposit their value
    /// 2. Barrier — all values visible
    /// 3. Each invocation computes its result from neighbor values
    /// 4. Barrier — safe to reuse slots for the next derivative call
    fn dpdx_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize].lock().unwrap() = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let left_idx = (quad_idx & !1) as usize;   // 0 or 2
        let right_idx = (quad_idx | 1) as usize;   // 1 or 3
        let left = self.slots[left_idx].lock().unwrap()[0];
        let right = self.slots[right_idx].lock().unwrap()[0];
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

    // Vector variants (Vec2f, Vec3f, Vec4f) decompose into components,
    // deposit all components into the [f32; 4] slot, and reassemble after.
    // fn dpdx_vec2f(...), fn dpdx_vec3f(...), fn dpdx_vec4f(...), etc.
}
```

### Fragment Dispatch Function

```rust
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

/// Dispatches fragment shader invocations over a 2D grid, processing in 2x2 quads.
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
    // Set up QuadContext per quad, install in thread-local
    // Results collected into the output grid
    // Fragments outside (width, height) are helper invocations — output discarded
    todo!()
}
```

## Step 6. Update Derivative Implementations

Replace the zero-stubs from the derivative builtins plan with quad-context-aware
implementations:

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

**Coarse/fine semantics**:
- `dpdx` delegates to fine (implementation-defined; we pick fine)
- `dpdx_coarse`: all 4 quad members get the same derivative (use top-left pair)
- `dpdx_fine`: each row computes its own derivative (per-row difference)

**`fwidth` implementation**:
```rust
impl DerivativeBuiltinFwidth for f32 {
    fn fwidth(self) -> Self {
        dpdx(self).abs() + dpdy(self).abs()
    }
}
```

Note: `fwidth(e)` requires two synchronization rounds (one for `dpdx`, one for
`dpdy`). Since `e: Copy`, this works — each round independently deposits and
computes. An optimization to combine into one round is deferred.

## Step 7. Tests

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
    let results = dispatch_fragments(4, 4, |x, _y| x as f32, |_builtins, value| {
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

---

## Open Design Questions

### 1. Quad Context Type Erasure

Derivative calls can operate on `f32`, `Vec2f`, `Vec3f`, or `Vec4f`. The quad
context needs to handle all of them.

**Recommendation**: Fixed `[f32; 4]` slots per quad member (enough for `Vec4f`).
Scalar uses component 0, `Vec2f` uses 0-1, etc. The 16-byte waste per quad member
is negligible and avoids allocation and type erasure complexity.

### 2. Multiple Derivative Calls Per Invocation

A shader may call `dpdx` multiple times on different values. The double-barrier
pattern (deposit -> barrier -> compute -> barrier) handles this naturally — each
derivative call is a self-contained synchronization round. The second barrier
ensures slots are safe to reuse before the next call.

### 3. Helper Invocations at Edges

When framebuffer dimensions aren't multiples of 2, partial quads at edges need
helper invocations. The dispatch runtime pads to even dimensions and marks edge
invocations as helpers. Helper outputs are discarded.

### 4. Workgroup Variable Lifetime

Currently `Workgroup<T>` uses global statics (`LazyLock<Arc<RwLock<Option<T>>>>`).
In a dispatch runtime, workgroup variables should ideally be scoped per-workgroup.

**Recommendation**: User-managed for now. Document that workgroup variables are in
an undefined state at the start of each workgroup, matching WGSL spec behavior.
The global statics persist across workgroups, which is incorrect but acceptable as
a known limitation to revisit later.

### 5. fwidth Double-Barrier Cost

`fwidth(e) = abs(dpdx(e)) + abs(dpdy(e))` requires two barrier rounds. An
optimization would deposit once and compute both derivatives in a single round.
Defer this optimization — correctness first.

---

## File Changes Summary

| File | Change |
|------|--------|
| `crates/wgsl-rs/Cargo.toml` | Add `rayon = "1"` |
| `crates/wgsl-rs/src/std.rs` | Add `mod runtime;` + `pub use runtime::*;` |
| `crates/wgsl-rs/src/std/runtime.rs` | **New**: `dispatch_workgroups`, `dispatch_fragments`, `QuadContext`, `ComputeBuiltins`, `FragmentBuiltins`, thread-locals |
| `crates/wgsl-rs/src/std/synchronization.rs` | Update barrier functions to use thread-local context |
| `crates/wgsl-rs/src/std/derivative.rs` | Replace zero-stubs with quad-context-aware implementations |
