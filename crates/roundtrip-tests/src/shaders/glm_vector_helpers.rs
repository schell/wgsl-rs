//! Roundtrip tests for the glm vector helper functions.
//!
//! Tests: `length_squared2/3/4`, `normalize_or_zero2/3/4`,
//! `angle_between2/3`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 8;

/// glm helpers on 3D vectors, exercising the `Vec3f::ZERO` constant
/// path through `length_squared3`.
#[wgsl]
pub mod glm_vector_ops {
    use wgsl_rs::glm::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 8]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 8]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let a4 = get!(INPUT)[idx];
        let a: Vec3f = a4.xyz();
        let b: Vec3f = a4.yzx();

        get_mut!(OUTPUT)[idx] = vec4f(
            length_squared3(a + Vec3f::ZERO),
            angle_between3(a, b),
            normalize_or_zero3(a).x,
            angle_between2(a4.xy(), a4.yx()),
        );
    }
}

/// glm helpers on 2D and 4D vectors.
#[wgsl]
pub mod glm_vector_ops_24 {
    use wgsl_rs::glm::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 8]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 8]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let a4 = get!(INPUT)[idx];

        get_mut!(OUTPUT)[idx] = vec4f(
            length_squared2(a4.xy()),
            normalize_or_zero2(a4.xy()).x,
            normalize_or_zero4(a4).x,
            length_squared4(a4),
        );
    }
}

/// Generates test inputs covering the zero vector, unit axes, negatives,
/// and mixed magnitudes.
fn glm_inputs() -> [wgsl_rs::std::Vec4f; N] {
    [
        wgsl_rs::std::vec4f(0.0, 0.0, 0.0, 0.0),
        wgsl_rs::std::vec4f(1.0, 0.0, 0.0, 0.0),
        wgsl_rs::std::vec4f(0.0, 1.0, 0.0, 0.0),
        wgsl_rs::std::vec4f(3.0, 4.0, 0.0, 0.0),
        wgsl_rs::std::vec4f(1.0, 1.0, 1.0, 1.0),
        wgsl_rs::std::vec4f(-2.0, 3.0, -6.0, 0.25),
        wgsl_rs::std::vec4f(0.1, -0.7, 2.5, -4.0),
        wgsl_rs::std::vec4f(100.0, 0.5, -0.25, 8.0),
    ]
}

/// Runs one GPU/CPU comparison for a glm shader module. `run_cpu`
/// dispatches the module on the CPU and returns the flattened output
/// floats.
fn compare_module(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgsl_rs::Source,
    run_cpu: impl Fn() -> Vec<f32>,
    labels: &[&str],
    epsilon: f32,
    name: &str,
) -> ComparisonResult {
    let inputs = glm_inputs();
    let flat: Vec<f32> = inputs.iter().flat_map(|v| v.to_array()).collect();
    let input_bytes = bytemuck::cast_slice(&flat);
    let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;

    let mut linkage = wgsl_rs_linkage_wgpu::analyze_wgsl_module(module).unwrap();
    let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
        device,
        queue,
        linkage: &mut linkage,
        entry: "main",
        input_data: input_bytes,
        output_size,
        workgroup_count: (1, 1, 1),
    });
    let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

    let cpu_floats = run_cpu();

    let all_labels: Vec<String> = (0..N)
        .flat_map(|_| labels.iter().map(|l| l.to_string()))
        .collect();
    let label_refs: Vec<&str> = all_labels.iter().map(|s| s.as_str()).collect();
    harness::compare_f32_results(name, gpu_floats, &cpu_floats, &label_refs, epsilon)
}

/// The glm vector helpers roundtrip test.
pub struct GlmVectorHelpersTest;

impl RoundtripTest for GlmVectorHelpersTest {
    fn name(&self) -> &str {
        "glm_vector_helpers"
    }

    fn description(&self) -> &str {
        "glm: length_squared2/3/4, normalize_or_zero2/3/4, angle_between2/3"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let inputs = glm_inputs();
        let zeroed = [Vec4f::default(); N];

        let ops_labels = [
            "length_squared3(a + Vec3f::ZERO)",
            "angle_between3(a, a.yzx())",
            "normalize_or_zero3(a).x",
            "angle_between2(a.xy(), a.yx())",
        ];
        let ops_cpu = || {
            glm_vector_ops::INPUT.set(inputs);
            glm_vector_ops::OUTPUT.set(zeroed);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                glm_vector_ops::main(builtins.global_invocation_id);
            });
            glm_vector_ops::OUTPUT
                .get()
                .iter()
                .flat_map(|v| v.to_array())
                .collect()
        };
        let ops = compare_module(
            device,
            queue,
            &glm_vector_ops::WGSL_SOURCE,
            ops_cpu,
            &ops_labels,
            1e-5,
            "glm_vector_ops",
        );

        let ops24_labels = [
            "length_squared2(a.xy())",
            "normalize_or_zero2(a.xy()).x",
            "normalize_or_zero4(a).x",
            "length_squared4(a)",
        ];
        let ops24_cpu = || {
            glm_vector_ops_24::INPUT.set(inputs);
            glm_vector_ops_24::OUTPUT.set(zeroed);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                glm_vector_ops_24::main(builtins.global_invocation_id);
            });
            glm_vector_ops_24::OUTPUT
                .get()
                .iter()
                .flat_map(|v| v.to_array())
                .collect()
        };
        let ops24 = compare_module(
            device,
            queue,
            &glm_vector_ops_24::WGSL_SOURCE,
            ops24_cpu,
            &ops24_labels,
            1e-5,
            "glm_vector_ops_24",
        );

        vec![ops, ops24]
    }
}
