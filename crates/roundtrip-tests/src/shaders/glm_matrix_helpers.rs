//! Roundtrip tests for the glm matrix helper functions.
//!
//! Tests: `translation`, `rotate`, `scale`, `translate`, `transform_point3`,
//! `transform_vector3`, `transform_point2`, `look_at_rh`, `perspective_rh`,
//! `orthographic_rh`, `inverse4x4`, `inverse2x2`, `inverse3x3`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 8;

/// glm matrix ops composing model/view/projection and inverting the
/// result.
#[wgsl]
pub mod glm_matrix_ops {
    use wgsl_rs::glm::*;

    storage!(group(0), binding(0), INPUT: [f32; 32]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 8]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let p = vec4f(
            get!(INPUT)[base],
            get!(INPUT)[base + 1],
            get!(INPUT)[base + 2],
            get!(INPUT)[base + 3],
        );

        // Model: rotate around Y by p.w radians, translate by p.xyz,
        // uniform scale.
        let m = rotation_axis_angle(vec3f(0.0, 1.0, 0.0), p.w);
        let m = translate(m, p.xyz());
        let m = scale(m, vec3f(2.0, 2.0, 2.0));

        let view = look_at_rh(
            vec3f(3.0, 4.0, 5.0),
            vec3f(0.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
        );
        let proj = perspective_rh(1.2, 1.5, 0.1, 100.0);
        let vp = proj * view;

        let moved = transform_point3(m, vec3f(1.0, 2.0, 3.0));
        let clip = vp * vec4f(moved.x, moved.y, moved.z, 1.0);
        let recovered = inverse4x4(vp) * vp;

        get_mut!(OUTPUT)[idx] = vec4f(
            clip.x / clip.w,
            clip.y / clip.w,
            transform_vector3(m, vec3f(0.0, 0.0, 1.0)).z,
            recovered[0u32].x,
        );
    }
}

/// glm 2x2/3x3 inverses, `transform_point2`, and `orthographic_rh`.
#[wgsl]
pub mod glm_matrix_ops_23 {
    use wgsl_rs::glm::*;

    storage!(group(0), binding(0), INPUT: [f32; 32]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 8]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let p = vec4f(
            get!(INPUT)[base],
            get!(INPUT)[base + 1],
            get!(INPUT)[base + 2],
            get!(INPUT)[base + 3],
        );

        let m2 = inverse2x2(mat2x2f(vec2f(4.0, 1.0), vec2f(2.0, 2.0 + p.x)));
        let m3 = inverse3x3(mat3x3f(
            vec3f(1.0, 0.0, 2.0),
            vec3f(0.0, 2.0 + p.y, 0.0),
            vec3f(0.0, 0.0, 4.0),
        ));
        let o = orthographic_rh(p.z - 1.0, p.z + 1.0, -1.0, 1.0, 0.1, 100.0);
        let affine = mat3x3f(
            vec3f(2.0, 0.0, p.x),
            vec3f(0.0, 3.0, p.y),
            vec3f(0.0, 0.0, 1.0),
        );

        get_mut!(OUTPUT)[idx] = vec4f(
            (m2 * vec2f(2.0, 3.0)).x,
            (m3 * vec3f(1.0, 1.0, 1.0)).y,
            (o * vec4f(0.5, 0.25, -5.0, 1.0)).z,
            transform_point2(affine, vec2f(1.0, 1.0)).y,
        );
    }
}

/// Generates test inputs: model positions, rotation angles, and
/// orthographic bounds.
fn glm_matrix_inputs() -> [f32; 32] {
    let mut values = [0.0f32; 32];
    for (i, chunk) in values.chunks_mut(4).enumerate() {
        let t = i as f32 / 7.0;
        chunk[0] = t * 2.0 - 1.0; // translation x / matrix element
        chunk[1] = t * 3.0; // translation y / matrix element
        chunk[2] = 0.5 + t; // translation z / ortho bound
        chunk[3] = t * std::f32::consts::PI; // rotation angle
    }
    values
}

/// The glm matrix helpers roundtrip test.
pub struct GlmMatrixHelpersTest;

impl RoundtripTest for GlmMatrixHelpersTest {
    fn name(&self) -> &str {
        "glm_matrix_helpers"
    }

    fn description(&self) -> &str {
        "glm: translate/rotate/scale, transform_point2/3, transform_vector3, look_at_rh, \
         perspective_rh, orthographic_rh, inverse2x2/3x3/4x4"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = glm_matrix_inputs();
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;

        let mut results = Vec::new();

        {
            let epsilon = 1e-5;
            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&glm_matrix_ops::WGSL_SOURCE).unwrap();
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
            glm_matrix_ops::INPUT.set(inputs);
            glm_matrix_ops::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                glm_matrix_ops::main(builtins.global_invocation_id);
            });
            let cpu_output = glm_matrix_ops::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|_| {
                    vec![
                        "clip.x / clip.w".to_string(),
                        "clip.y / clip.w".to_string(),
                        "transform_vector3(m, z).z".to_string(),
                        "(inverse4x4(vp) * vp)[0].x".to_string(),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "glm_matrix_ops",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        {
            let epsilon = 1e-5;
            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&glm_matrix_ops_23::WGSL_SOURCE).unwrap();
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
            glm_matrix_ops_23::INPUT.set(inputs);
            glm_matrix_ops_23::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                glm_matrix_ops_23::main(builtins.global_invocation_id);
            });
            let cpu_output = glm_matrix_ops_23::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|_| {
                    vec![
                        "(inverse2x2(m2) * v).x".to_string(),
                        "(inverse3x3(m3) * v).y".to_string(),
                        "(orthographic_rh * p).z".to_string(),
                        "transform_point2(affine, v).y".to_string(),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "glm_matrix_ops_23",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
