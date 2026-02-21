//! CPU-side shader dispatch runtime.
//!
//! Provides `dispatch_workgroups` and `dispatch_fragments` to execute shader
//! entry points on the CPU with correct builtin values, workgroup barriers,
//! and 2x2 quad contexts for derivative computation.
//!
//! The runtime communicates with builtins via thread-local storage. Builtins
//! check these thread-locals and gracefully degrade outside of dispatch
//! (barriers are no-ops, derivatives return zero), preserving backward
//! compatibility.

use std::cell::{Cell, RefCell};
use std::sync::{Arc, Mutex};

use super::vector::{Vec3u, Vec4f, vec3u, vec4f};

thread_local! {
    /// Barrier for workgroup synchronization (compute dispatch).
    static WORKGROUP_BARRIER: RefCell<Option<Arc<std::sync::Barrier>>> = const { RefCell::new(None) };

    /// Quad context for derivative computation (fragment dispatch).
    static QUAD_CONTEXT: RefCell<Option<Arc<QuadContext>>> = const { RefCell::new(None) };

    /// This invocation's index within the quad (0-3).
    static QUAD_INDEX: Cell<u8> = const { Cell::new(0) };

    /// Compute builtin values for the current invocation.
    static COMPUTE_BUILTINS: RefCell<Option<ComputeBuiltins>> = const { RefCell::new(None) };
}

/// Executes `f` with a reference to the current thread's workgroup barrier,
/// if one is installed.
pub(crate) fn with_workgroup_barrier<R>(f: impl FnOnce(Option<&std::sync::Barrier>) -> R) -> R {
    WORKGROUP_BARRIER.with(|b| {
        let guard = b.borrow();
        f(guard.as_ref().map(|arc| arc.as_ref()))
    })
}

/// Executes `f` with a reference to the current thread's quad context and
/// quad index, if a quad context is installed.
///
/// This is used by derivative builtins (dpdx, dpdy, fwidth) to access the
/// quad synchronization context during fragment dispatch.
#[allow(dead_code)] // Used by derivative builtins once implemented.
pub(crate) fn with_quad_context<R>(f: impl FnOnce(Option<(&QuadContext, u8)>) -> R) -> R {
    QUAD_CONTEXT.with(|ctx| {
        let guard = ctx.borrow();
        match guard.as_ref() {
            Some(qctx) => {
                let idx = QUAD_INDEX.with(|i| i.get());
                f(Some((qctx.as_ref(), idx)))
            }
            None => f(None),
        }
    })
}

/// Builtin values provided to each compute shader invocation.
#[derive(Clone, Copy, Debug)]
pub struct ComputeBuiltins {
    /// The global invocation ID (position in the full dispatch grid).
    pub global_invocation_id: Vec3u,
    /// The local invocation ID (position within the workgroup).
    pub local_invocation_id: Vec3u,
    /// The linearized local invocation index within the workgroup.
    pub local_invocation_index: u32,
    /// The workgroup ID (which workgroup this invocation belongs to).
    pub workgroup_id: Vec3u,
    /// The total number of workgroups dispatched.
    pub num_workgroups: Vec3u,
}

/// Dispatches a compute shader, similar to wgpu's
/// `ComputePass::dispatch_workgroups(x, y, z)`.
///
/// Executes `shader_fn` for every invocation across all workgroups.
/// Within each workgroup, invocations run in parallel on a rayon thread pool.
/// `workgroup_barrier()` calls within `shader_fn` will synchronize correctly.
///
/// Workgroups are processed sequentially. Invocations within each workgroup
/// run in parallel.
pub fn dispatch_workgroups<F>(
    workgroup_count: (u32, u32, u32),
    workgroup_size: (u32, u32, u32),
    shader_fn: F,
) where
    F: Fn(ComputeBuiltins) + Send + Sync,
{
    let (cx, cy, cz) = workgroup_count;
    let (sx, sy, sz) = workgroup_size;
    let invocations_per_workgroup = (sx * sy * sz) as usize;
    let num_workgroups = vec3u(cx, cy, cz);

    for wz in 0..cz {
        for wy in 0..cy {
            for wx in 0..cx {
                let workgroup_id = vec3u(wx, wy, wz);
                let barrier = Arc::new(std::sync::Barrier::new(invocations_per_workgroup));

                rayon::scope(|s| {
                    for lz in 0..sz {
                        for ly in 0..sy {
                            for lx in 0..sx {
                                let barrier = barrier.clone();
                                let local_id = vec3u(lx, ly, lz);
                                let local_index = lx + ly * sx + lz * sx * sy;
                                let global_id = vec3u(wx * sx + lx, wy * sy + ly, wz * sz + lz);

                                let builtins = ComputeBuiltins {
                                    global_invocation_id: global_id,
                                    local_invocation_id: local_id,
                                    local_invocation_index: local_index,
                                    workgroup_id,
                                    num_workgroups,
                                };

                                let shader_fn = &shader_fn;
                                s.spawn(move |_| {
                                    WORKGROUP_BARRIER.with(|b| {
                                        *b.borrow_mut() = Some(barrier);
                                    });
                                    COMPUTE_BUILTINS.with(|b| {
                                        *b.borrow_mut() = Some(builtins);
                                    });

                                    shader_fn(builtins);

                                    WORKGROUP_BARRIER.with(|b| {
                                        *b.borrow_mut() = None;
                                    });
                                    COMPUTE_BUILTINS.with(|b| {
                                        *b.borrow_mut() = None;
                                    });
                                });
                            }
                        }
                    }
                });
            }
        }
    }
}

/// A 2x2 quad of fragment invocations for derivative computation.
///
/// Quad layout (matching GPU convention):
///
/// ```text
///     [0] (x,y)     [1] (x+1,y)
///     [2] (x,y+1)   [3] (x+1,y+1)
/// ```
///
/// Derivatives use the double-barrier pattern:
/// 1. All 4 invocations deposit their value into their slot.
/// 2. Barrier — all values visible.
/// 3. Each invocation reads neighbor values and computes its result.
/// 4. Barrier — slots safe to reuse for the next derivative call.
pub struct QuadContext {
    /// Value exchange slots indexed by quad member (0-3).
    /// Each slot stores up to 4 f32 components (enough for `Vec4f`).
    slots: [Mutex<[f32; 4]>; 4],
    /// Barrier for synchronizing quad members at derivative call sites.
    barrier: std::sync::Barrier,
}

impl QuadContext {
    /// Creates a new quad context for 4 fragment invocations.
    fn new() -> Self {
        QuadContext {
            slots: std::array::from_fn(|_| Mutex::new([0.0; 4])),
            barrier: std::sync::Barrier::new(4),
        }
    }

    /// Deposits a scalar value and computes the fine horizontal partial
    /// derivative (dpdx). Each row computes its own difference.
    pub fn dpdx_fine_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let left_idx = (quad_idx & !1) as usize; // 0 or 2
        let right_idx = (quad_idx | 1) as usize; // 1 or 3
        let left = self.slots[left_idx].lock().expect("quad slot poisoned")[0];
        let right = self.slots[right_idx].lock().expect("quad slot poisoned")[0];
        self.barrier.wait();
        right - left
    }

    /// Deposits a scalar value and computes the coarse horizontal partial
    /// derivative (dpdx). All quad members get the same value, computed from
    /// the top row.
    pub fn dpdx_coarse_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let left = self.slots[0].lock().expect("quad slot poisoned")[0];
        let right = self.slots[1].lock().expect("quad slot poisoned")[0];
        self.barrier.wait();
        right - left
    }

    /// Deposits a scalar value and computes the fine vertical partial
    /// derivative (dpdy). Each column computes its own difference.
    pub fn dpdy_fine_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let top_idx = (quad_idx & !2) as usize; // 0 or 1
        let bottom_idx = (quad_idx | 2) as usize; // 2 or 3
        let top = self.slots[top_idx].lock().expect("quad slot poisoned")[0];
        let bottom = self.slots[bottom_idx].lock().expect("quad slot poisoned")[0];
        self.barrier.wait();
        bottom - top
    }

    /// Deposits a scalar value and computes the coarse vertical partial
    /// derivative (dpdy). All quad members get the same value, computed from
    /// the left column.
    pub fn dpdy_coarse_f32(&self, quad_idx: u8, value: f32) -> f32 {
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = [value, 0.0, 0.0, 0.0];
        self.barrier.wait();
        let top = self.slots[0].lock().expect("quad slot poisoned")[0];
        let bottom = self.slots[2].lock().expect("quad slot poisoned")[0];
        self.barrier.wait();
        bottom - top
    }

    /// Deposits a multi-component value and computes a fine horizontal partial
    /// derivative, returning up to `n` components.
    pub fn dpdx_fine_components(&self, quad_idx: u8, components: &[f32], n: usize) -> [f32; 4] {
        let mut slot = [0.0f32; 4];
        slot[..n].copy_from_slice(&components[..n]);
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = slot;
        self.barrier.wait();
        let left_idx = (quad_idx & !1) as usize;
        let right_idx = (quad_idx | 1) as usize;
        let left = *self.slots[left_idx].lock().expect("quad slot poisoned");
        let right = *self.slots[right_idx].lock().expect("quad slot poisoned");
        self.barrier.wait();
        let mut result = [0.0f32; 4];
        for i in 0..n {
            result[i] = right[i] - left[i];
        }
        result
    }

    /// Deposits a multi-component value and computes a coarse horizontal partial
    /// derivative, returning up to `n` components.
    pub fn dpdx_coarse_components(&self, quad_idx: u8, components: &[f32], n: usize) -> [f32; 4] {
        let mut slot = [0.0f32; 4];
        slot[..n].copy_from_slice(&components[..n]);
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = slot;
        self.barrier.wait();
        let left = *self.slots[0].lock().expect("quad slot poisoned");
        let right = *self.slots[1].lock().expect("quad slot poisoned");
        self.barrier.wait();
        let mut result = [0.0f32; 4];
        for i in 0..n {
            result[i] = right[i] - left[i];
        }
        result
    }

    /// Deposits a multi-component value and computes a fine vertical partial
    /// derivative, returning up to `n` components.
    pub fn dpdy_fine_components(&self, quad_idx: u8, components: &[f32], n: usize) -> [f32; 4] {
        let mut slot = [0.0f32; 4];
        slot[..n].copy_from_slice(&components[..n]);
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = slot;
        self.barrier.wait();
        let top_idx = (quad_idx & !2) as usize;
        let bottom_idx = (quad_idx | 2) as usize;
        let top = *self.slots[top_idx].lock().expect("quad slot poisoned");
        let bottom = *self.slots[bottom_idx].lock().expect("quad slot poisoned");
        self.barrier.wait();
        let mut result = [0.0f32; 4];
        for i in 0..n {
            result[i] = bottom[i] - top[i];
        }
        result
    }

    /// Deposits a multi-component value and computes a coarse vertical partial
    /// derivative, returning up to `n` components.
    pub fn dpdy_coarse_components(&self, quad_idx: u8, components: &[f32], n: usize) -> [f32; 4] {
        let mut slot = [0.0f32; 4];
        slot[..n].copy_from_slice(&components[..n]);
        *self.slots[quad_idx as usize]
            .lock()
            .expect("quad slot poisoned") = slot;
        self.barrier.wait();
        let top = *self.slots[0].lock().expect("quad slot poisoned");
        let bottom = *self.slots[2].lock().expect("quad slot poisoned");
        self.barrier.wait();
        let mut result = [0.0f32; 4];
        for i in 0..n {
            result[i] = bottom[i] - top[i];
        }
        result
    }
}

/// Builtin values provided to each fragment shader invocation.
#[derive(Clone, Copy, Debug)]
pub struct FragmentBuiltins {
    /// The fragment's position in window coordinates.
    /// `xy` contains the pixel center (e.g. `(0.5, 0.5)` for the top-left pixel).
    pub position: Vec4f,
    /// Whether this is a front-facing fragment.
    pub front_facing: bool,
    /// The sample index (0 for non-multisampled).
    pub sample_index: u32,
    /// The sample mask.
    pub sample_mask: u32,
}

/// Dispatches fragment shader invocations over a 2D grid, processing in 2x2
/// quads.
///
/// Returns a `width x height` grid of outputs. Partial quads at edges use
/// helper invocations whose outputs are discarded. Helper invocation inputs
/// are generated by clamping coordinates to the valid range.
///
/// Each quad's 4 invocations run in parallel via `rayon::scope`, with a
/// shared `QuadContext` installed in thread-local storage for derivative
/// computation.
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
    let padded_w = (width + 1) & !1;
    let padded_h = (height + 1) & !1;

    // Pre-allocate output grid. Each cell is behind Arc<Mutex<>> so quad
    // threads can write to their specific position without contention across
    // different quads.
    let grid: Vec<Vec<Arc<Mutex<Option<O>>>>> = (0..height)
        .map(|_| (0..width).map(|_| Arc::new(Mutex::new(None))).collect())
        .collect();

    // Process quads. Quads are independent, so we can parallelize across them
    // using rayon's parallel iterator.
    let quad_coords: Vec<(u32, u32)> = {
        let mut coords = Vec::new();
        let mut qy = 0;
        while qy < padded_h {
            let mut qx = 0;
            while qx < padded_w {
                coords.push((qx, qy));
                qx += 2;
            }
            qy += 2;
        }
        coords
    };

    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let shader_fn = &shader_fn;
        let input_fn = &input_fn;
        let grid = &grid;
        quad_coords.into_par_iter().for_each(|(qx, qy)| {
            let quad_ctx = Arc::new(QuadContext::new());

            rayon::scope(|s| {
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let px = qx + dx;
                        let py = qy + dy;
                        let quad_idx = (dx + dy * 2) as u8;
                        let quad_ctx = quad_ctx.clone();
                        let is_helper = px >= width || py >= height;

                        // Clamp coordinates for helper invocations.
                        let clamped_x = px.min(width.saturating_sub(1));
                        let clamped_y = py.min(height.saturating_sub(1));
                        let input = input_fn(clamped_x, clamped_y);

                        let grid_cell = if !is_helper {
                            Some(grid[py as usize][px as usize].clone())
                        } else {
                            None
                        };

                        s.spawn(move |_| {
                            QUAD_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = Some(quad_ctx);
                            });
                            QUAD_INDEX.with(|i| i.set(quad_idx));

                            let builtins = FragmentBuiltins {
                                position: vec4f(px as f32 + 0.5, py as f32 + 0.5, 0.0, 1.0),
                                front_facing: true,
                                sample_index: 0,
                                sample_mask: 0xFFFFFFFF,
                            };

                            let output = shader_fn(builtins, input);

                            // Store result only for non-helper invocations.
                            if let Some(cell) = grid_cell {
                                *cell.lock().expect("grid cell poisoned") = Some(output);
                            }

                            QUAD_CONTEXT.with(|ctx| {
                                *ctx.borrow_mut() = None;
                            });
                        });
                    }
                }
            });
        });
    }

    // Collect results, unwrapping the Arc<Mutex<>> layer.
    // All parallel work is complete at this point.
    grid.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| {
                    Arc::try_unwrap(cell)
                        .unwrap_or_else(|_| panic!("all quad threads should have completed"))
                        .into_inner()
                        .unwrap_or_else(|_| panic!("grid cell poisoned"))
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn compute_dispatch_basic() {
        let counter = AtomicU32::new(0);
        dispatch_workgroups((2, 1, 1), (4, 1, 1), |_builtins| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn compute_dispatch_3d() {
        let counter = AtomicU32::new(0);
        dispatch_workgroups((2, 3, 2), (2, 2, 2), |_builtins| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        // 2*3*2 workgroups * 2*2*2 invocations = 12 * 8 = 96
        assert_eq!(counter.load(Ordering::Relaxed), 96);
    }

    #[test]
    fn compute_builtins_correct() {
        use std::sync::Mutex;
        let results: Mutex<Vec<ComputeBuiltins>> = Mutex::new(Vec::new());

        dispatch_workgroups((2, 1, 1), (2, 1, 1), |builtins| {
            results.lock().unwrap().push(builtins);
        });

        let mut results = results.into_inner().unwrap();
        results.sort_by_key(|b| {
            let g = super::super::vector::ScalarCompOfVec::vec_to_array(b.global_invocation_id);
            (g[0], g[1], g[2])
        });

        assert_eq!(results.len(), 4);

        // Workgroup 0: local 0,0,0 and 1,0,0
        let b = &results[0];
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.global_invocation_id),
            [0, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.local_invocation_id),
            [0, 0, 0]
        );
        assert_eq!(b.local_invocation_index, 0);
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.workgroup_id),
            [0, 0, 0]
        );

        let b = &results[1];
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.global_invocation_id),
            [1, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.local_invocation_id),
            [1, 0, 0]
        );
        assert_eq!(b.local_invocation_index, 1);
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.workgroup_id),
            [0, 0, 0]
        );

        // Workgroup 1: local 0,0,0 and 1,0,0
        let b = &results[2];
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.global_invocation_id),
            [2, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.local_invocation_id),
            [0, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.workgroup_id),
            [1, 0, 0]
        );

        let b = &results[3];
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.global_invocation_id),
            [3, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.local_invocation_id),
            [1, 0, 0]
        );
        assert_eq!(
            super::super::vector::ScalarCompOfVec::vec_to_array(b.workgroup_id),
            [1, 0, 0]
        );
    }

    #[test]
    fn compute_dispatch_barrier_synchronizes() {
        // Each invocation writes its local index to a shared slot, then
        // after a barrier reads all slots. If the barrier works, all
        // invocations should see all writes.
        use std::sync::Mutex;

        let workgroup_size = 4;
        let shared = Arc::new(Mutex::new(vec![0u32; workgroup_size]));
        let observed = Arc::new(Mutex::new(Vec::<Vec<u32>>::new()));

        dispatch_workgroups((1, 1, 1), (workgroup_size as u32, 1, 1), |builtins| {
            let idx = builtins.local_invocation_index as usize;

            // Write our index.
            shared.lock().unwrap()[idx] = (idx as u32) + 1;

            // Barrier: all writes must be visible after this.
            crate::std::workgroup_barrier();

            // Read all values.
            let snapshot = shared.lock().unwrap().clone();
            observed.lock().unwrap().push(snapshot);
        });

        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), workgroup_size);
        for snapshot in observed.iter() {
            // All invocations should see all writes after the barrier.
            for (i, &val) in snapshot.iter().enumerate() {
                assert_eq!(val, (i as u32) + 1, "barrier failed to synchronize");
            }
        }
    }

    #[test]
    fn fragment_dispatch_basic() {
        let results = dispatch_fragments(4, 4, |x, y| (x, y), |_builtins, input| input);

        assert_eq!(results.len(), 4);
        for (y, row) in results.iter().enumerate() {
            assert_eq!(row.len(), 4);
            for (x, cell) in row.iter().enumerate() {
                let (rx, ry) = cell.expect("fragment should produce output");
                assert_eq!(rx, x as u32);
                assert_eq!(ry, y as u32);
            }
        }
    }

    #[test]
    fn fragment_dispatch_odd_dimensions() {
        // 3x3 grid — partial quads at right and bottom edges.
        let results = dispatch_fragments(3, 3, |x, y| (x, y), |_builtins, input| input);

        assert_eq!(results.len(), 3);
        for (y, row) in results.iter().enumerate() {
            assert_eq!(row.len(), 3);
            for (x, cell) in row.iter().enumerate() {
                let (rx, ry) = cell.expect("fragment should produce output");
                assert_eq!(rx, x as u32);
                assert_eq!(ry, y as u32);
            }
        }
    }

    #[test]
    fn fragment_dispatch_1x1() {
        // Minimal case: single pixel, 3 helper invocations.
        let results = dispatch_fragments(1, 1, |x, y| (x, y), |_builtins, input| input);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);
        let (rx, ry) = results[0][0].expect("fragment should produce output");
        assert_eq!(rx, 0);
        assert_eq!(ry, 0);
    }

    #[test]
    fn fragment_dispatch_position_builtins() {
        // Verify that fragment positions have correct pixel-center offsets.
        let results = dispatch_fragments(2, 2, |_x, _y| (), |builtins, _| builtins.position);

        for (y, row) in results.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                let pos = cell.expect("fragment should produce output");
                let components = super::super::vector::ScalarCompOfVec::vec_to_array(pos);
                assert!(
                    (components[0] - (x as f32 + 0.5)).abs() < f32::EPSILON,
                    "position.x mismatch at ({x}, {y})"
                );
                assert!(
                    (components[1] - (y as f32 + 0.5)).abs() < f32::EPSILON,
                    "position.y mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn quad_context_dpdx_fine() {
        // Directly test QuadContext with 4 threads simulating a quad.
        // Values: [10.0, 30.0, 20.0, 40.0]
        // dpdx_fine: row0 = 30-10 = 20, row1 = 40-20 = 20
        let quad = Arc::new(QuadContext::new());
        let values = [10.0f32, 30.0, 20.0, 40.0];
        let results: Arc<Mutex<[f32; 4]>> = Arc::new(Mutex::new([0.0; 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r = quad.dpdx_fine_f32(idx, values[idx as usize]);
                    results.lock().unwrap()[idx as usize] = r;
                });
            }
        });

        let results = results.lock().unwrap();
        assert_eq!(results[0], 20.0); // 30 - 10
        assert_eq!(results[1], 20.0); // 30 - 10
        assert_eq!(results[2], 20.0); // 40 - 20
        assert_eq!(results[3], 20.0); // 40 - 20
    }

    #[test]
    fn quad_context_dpdy_fine() {
        // Values: [10.0, 30.0, 50.0, 70.0]
        // dpdy_fine: col0 = 50-10 = 40, col1 = 70-30 = 40
        let quad = Arc::new(QuadContext::new());
        let values = [10.0f32, 30.0, 50.0, 70.0];
        let results: Arc<Mutex<[f32; 4]>> = Arc::new(Mutex::new([0.0; 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r = quad.dpdy_fine_f32(idx, values[idx as usize]);
                    results.lock().unwrap()[idx as usize] = r;
                });
            }
        });

        let results = results.lock().unwrap();
        assert_eq!(results[0], 40.0); // 50 - 10
        assert_eq!(results[1], 40.0); // 70 - 30
        assert_eq!(results[2], 40.0); // 50 - 10
        assert_eq!(results[3], 40.0); // 70 - 30
    }

    #[test]
    fn quad_context_dpdx_coarse() {
        // Values: [10.0, 30.0, 20.0, 40.0]
        // dpdx_coarse: always uses top row: 30 - 10 = 20
        let quad = Arc::new(QuadContext::new());
        let values = [10.0f32, 30.0, 20.0, 40.0];
        let results: Arc<Mutex<[f32; 4]>> = Arc::new(Mutex::new([0.0; 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r = quad.dpdx_coarse_f32(idx, values[idx as usize]);
                    results.lock().unwrap()[idx as usize] = r;
                });
            }
        });

        let results = results.lock().unwrap();
        // All members get the same coarse derivative.
        for i in 0..4 {
            assert_eq!(
                results[i], 20.0,
                "coarse dpdx should be 20.0 for all members"
            );
        }
    }

    #[test]
    fn quad_context_dpdy_coarse() {
        // Values: [10.0, 30.0, 50.0, 70.0]
        // dpdy_coarse: always uses left column: 50 - 10 = 40
        let quad = Arc::new(QuadContext::new());
        let values = [10.0f32, 30.0, 50.0, 70.0];
        let results: Arc<Mutex<[f32; 4]>> = Arc::new(Mutex::new([0.0; 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r = quad.dpdy_coarse_f32(idx, values[idx as usize]);
                    results.lock().unwrap()[idx as usize] = r;
                });
            }
        });

        let results = results.lock().unwrap();
        for i in 0..4 {
            assert_eq!(
                results[i], 40.0,
                "coarse dpdy should be 40.0 for all members"
            );
        }
    }

    #[test]
    fn quad_context_multiple_derivative_calls() {
        // Verify the double-barrier pattern allows multiple sequential
        // derivative calls on the same quad.
        let quad = Arc::new(QuadContext::new());
        // First call values, second call values
        let values1 = [1.0f32, 3.0, 2.0, 4.0];
        let values2 = [10.0f32, 10.0, 20.0, 20.0];
        let results: Arc<Mutex<[(f32, f32); 4]>> = Arc::new(Mutex::new([(0.0, 0.0); 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r1 = quad.dpdx_fine_f32(idx, values1[idx as usize]);
                    let r2 = quad.dpdy_fine_f32(idx, values2[idx as usize]);
                    results.lock().unwrap()[idx as usize] = (r1, r2);
                });
            }
        });

        let results = results.lock().unwrap();
        // dpdx of [1,3,2,4]: row0=3-1=2, row1=4-2=2
        // dpdy of [10,10,20,20]: col0=20-10=10, col1=20-10=10
        for i in 0..4 {
            assert_eq!(results[i].0, 2.0, "first derivative call");
            assert_eq!(results[i].1, 10.0, "second derivative call");
        }
    }

    #[test]
    fn quad_context_components_2d() {
        // Test multi-component derivative with 2 components.
        let quad = Arc::new(QuadContext::new());
        // Quad member values (2 components each):
        // [0]: (1.0, 10.0)  [1]: (3.0, 30.0)
        // [2]: (2.0, 20.0)  [3]: (4.0, 40.0)
        let values: [[f32; 2]; 4] = [[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [4.0, 40.0]];
        let results: Arc<Mutex<[[f32; 4]; 4]>> = Arc::new(Mutex::new([[0.0; 4]; 4]));

        rayon::scope(|s| {
            for idx in 0..4u8 {
                let quad = quad.clone();
                let results = results.clone();
                s.spawn(move |_| {
                    let r = quad.dpdx_fine_components(idx, &values[idx as usize], 2);
                    results.lock().unwrap()[idx as usize] = r;
                });
            }
        });

        let results = results.lock().unwrap();
        // dpdx fine: row0: 3-1=2, 30-10=20; row1: 4-2=2, 40-20=20
        for i in 0..4 {
            assert_eq!(results[i][0], 2.0);
            assert_eq!(results[i][1], 20.0);
        }
    }
}
