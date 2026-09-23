//! Roundtrip tests for bool vector types in type position (wgsl-rs#169).
//!
//! `vec2b`/`vec3b`/`vec4b` are not builtin WGSL aliases, so `VecNb`
//! annotations must render as the generic `vecN<bool>` form. This is the
//! issue #169 repro shape: constructors alone always rendered fine —
//! the explicit type annotations are what used to emit the invalid
//! `vecNb` identifiers.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// bool_vec_annotations: explicit `VecNb` type annotations plus
/// constructors — the wgsl-rs#169 repro. `all`/`any` consume the
/// annotated bindings so both worlds agree on the written results.
#[wgsl]
pub mod bool_vec_annotations {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 192]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 9;
        // The wgsl-rs#169 repro shape: bool vectors in explicit type
        // annotations. These used to render `vec2b`/`vec3b`/`vec4b` —
        // unknown identifiers in WGSL — and now render
        // `vec2<bool>`/`vec3<bool>`/`vec4<bool>`.
        let x: Vec2b = vec2b(input[base] != 0u32, input[base + 1] != 0u32);
        let y: Vec3b = vec3b(
            input[base + 2] != 0u32,
            input[base + 3] != 0u32,
            input[base + 4] != 0u32,
        );
        let z: Vec4b = vec4b(
            input[base + 5] != 0u32,
            input[base + 6] != 0u32,
            input[base + 7] != 0u32,
            input[base + 8] != 0u32,
        );

        if all(x) {
            get_mut!(OUTPUT)[idx * 3] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx * 3] = 0u32;
        }
        if any(y) {
            get_mut!(OUTPUT)[idx * 3 + 1] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx * 3 + 1] = 0u32;
        }
        if all(z) {
            get_mut!(OUTPUT)[idx * 3 + 2] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx * 3 + 2] = 0u32;
        }
    }
}

// ============================================================================
// Input Generators
// ============================================================================

fn bool_vec_annotations_inputs() -> [u32; N * 9] {
    let mut inputs = [0u32; N * 9];
    let x_pats = [(0, 0), (1, 1), (1, 0), (0, 1)];
    let y_pats = [
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
    ];
    let z_pats = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 1, 1),
        (1, 0, 1, 0),
    ];
    for i in 0..N {
        let (a, b) = x_pats[i % x_pats.len()];
        inputs[i * 9] = a;
        inputs[i * 9 + 1] = b;
        let (a, b, c) = y_pats[i % y_pats.len()];
        inputs[i * 9 + 2] = a;
        inputs[i * 9 + 3] = b;
        inputs[i * 9 + 4] = c;
        let (a, b, c, d) = z_pats[i % z_pats.len()];
        inputs[i * 9 + 5] = a;
        inputs[i * 9 + 6] = b;
        inputs[i * 9 + 7] = c;
        inputs[i * 9 + 8] = d;
    }
    inputs
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct BoolVectorTypesTest;

impl RoundtripTest for BoolVectorTypesTest {
    fn name(&self) -> &str {
        "bool_vector_types"
    }

    fn description(&self) -> &str {
        "VecNb type annotations (#169)"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        {
            let inputs = bool_vec_annotations_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bool_vec_annotations::WGSL_SOURCE)
                    .unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * 3 * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            bool_vec_annotations::INPUT.set(inputs);
            bool_vec_annotations::OUTPUT.set([0u32; N * 3]);
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    bool_vec_annotations::main(builtins.global_invocation_id);
                },
            );
            let cpu_results: Vec<u32> = bool_vec_annotations::OUTPUT.get().to_vec();

            let labels: Vec<String> = (0..N * 3)
                .map(|i| format!("bool_vec_annotations[{i}]"))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "bool_vec_annotations",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        results
    }
}
