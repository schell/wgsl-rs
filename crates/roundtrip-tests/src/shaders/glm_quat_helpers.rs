//! Roundtrip tests for the glm quaternion helpers.
//!
//! Tests: `from_axis_angle`, `quat_mul`, `rotate_vec3`, `inverse`,
//! `quat_slerp`, `quat_angle_between`, `Quat::IDENTITY`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 8;

/// glm quaternion ops: compose, rotate, invert, blend.
#[wgsl]
pub mod glm_quat_ops {
    use wgsl_rs::glm::*;

    storage!(group(0), binding(0), INPUT: [f32; 32]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 8]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let angle = get!(INPUT)[base];

        let a = from_axis_angle(Vec3f::Y, angle);
        let b = from_axis_angle(Vec3f::X, 0.5);
        let q = quat_mul(a, b);
        let v = rotate_vec3(q, vec3f(1.0, 2.0, 3.0));
        let back = rotate_vec3(quat_mul(q, inverse(q)), vec3f(1.0, 2.0, 3.0));
        let mid = quat_slerp(Quat::IDENTITY, a, 0.25);

        get_mut!(OUTPUT)[idx] = vec4f(v.x, back.z, mid.y, quat_angle_between(Quat::IDENTITY, a));
    }
}

/// Generates rotation angles for the quat inputs.
fn glm_quat_inputs() -> [f32; 32] {
    let mut values = [0.0f32; 32];
    for (i, chunk) in values.chunks_mut(4).enumerate() {
        let t = i as f32 / 7.0;
        chunk[0] = t * 3.0; // rotation angle
    }
    values
}

/// The glm quaternion helpers roundtrip test.
pub struct GlmQuatHelpersTest;

impl RoundtripTest for GlmQuatHelpersTest {
    fn name(&self) -> &str {
        "glm_quat_helpers"
    }

    fn description(&self) -> &str {
        "glm: from_axis_angle, quat_mul, rotate_vec3, inverse, quat_slerp, quat_angle_between"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = glm_quat_inputs();
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;

        let mut results = Vec::new();

        {
            let epsilon = 1e-5;
            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&glm_quat_ops::WGSL_SOURCE).unwrap();
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

            use wgsl_rs::std::*;
            glm_quat_ops::INPUT.set(inputs);
            glm_quat_ops::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                glm_quat_ops::main(builtins.global_invocation_id);
            });
            let cpu_output = glm_quat_ops::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|_| {
                    vec![
                        "rotate_vec3(quat_mul(a, b), v).x".to_string(),
                        "rotate_vec3(q * inverse(q), v).z".to_string(),
                        "quat_slerp(IDENTITY, a, 0.25).y".to_string(),
                        "quat_angle_between(IDENTITY, a)".to_string(),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "glm_quat_ops",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
