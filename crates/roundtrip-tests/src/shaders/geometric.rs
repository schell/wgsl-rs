//! Roundtrip tests for geometric and conversion builtin functions.
//!
//! Tests: `dot`, `cross`, `normalize`, `length`, `distance`, `reflect`,
//! `refract`, `face_forward`, `degrees`, `radians`, `sign`, `abs`, `fma`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Scalar conversion functions: degrees, radians, sign, abs, fma.
#[wgsl]
pub mod geo_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(degrees(x), radians(x), sign(x), abs(x));
    }
}

/// FMA and additional scalar ops.
#[wgsl]
pub mod geo_fma {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // fma(x, 2.0, 1.0) = x * 2.0 + 1.0
        // distance(x, 0.0) = abs(x) for scalars
        get_mut!(OUTPUT)[idx] = vec4f(fma(x, 2.0, 1.0), distance(x, 0.0), length(x), 0.0);
    }
}

/// Vector geometric functions: dot, length, distance, normalize.
///
/// Uses pairs of consecutive input values to construct Vec3f test vectors.
#[wgsl]
pub mod geo_vector {
    use wgsl_rs::std::*;

    // 64 floats = 16 Vec4f inputs, producing 16 result Vec4f outputs
    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 16]);

    #[compute]
    #[workgroup_size(16)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let input = get!(INPUT);
        let a = vec3f(input[base], input[base + 1], input[base + 2]);
        let b = vec3f(input[base + 3], input[base + 2], input[base + 1]);

        let d = dot(a, b);
        let len_a = length(a);
        let dist = distance(a, b);
        // Avoid normalizing zero vectors
        let safe_a = vec3f(a.x + 0.001, a.y + 0.001, a.z + 0.001);
        let norm = normalize(safe_a);

        get_mut!(OUTPUT)[idx] = vec4f(d, len_a, dist, length(norm));
    }
}

/// Cross product, reflect, face_forward.
#[wgsl]
pub mod geo_cross_reflect {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    // 8 result vec4s (2 per test case: cross xyz + reflect x, reflect yz +
    // face_forward x + padding)
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 16]);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec3f(input[base], input[base + 1], input[base + 2]);
        let b = vec3f(input[base + 3], input[base + 4], input[base + 5]);
        let n = vec3f(input[base + 6], input[base + 7], 0.0);

        let c = cross(a, b);
        // reflect: e1 - 2 * dot(e2, e1) * e2
        let safe_n = normalize(vec3f(n.x + 0.001, n.y + 0.001, 1.0));
        let r = reflect(a, safe_n);
        let ff = face_forward(a, b, safe_n);

        // Pack results: first vec4 = cross.xyz + reflect.x
        get_mut!(OUTPUT)[idx * 2] = vec4f(c.x, c.y, c.z, r.x);
        // Second vec4 = reflect.yz + face_forward.x + face_forward.y
        get_mut!(OUTPUT)[idx * 2 + 1] = vec4f(r.y, r.z, ff.x, ff.y);
    }
}

/// Generates test inputs for geometric functions.
fn geo_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f32 / (N - 1) as f32;
        *value = t * 6.0 - 3.0; // [-3.0, 3.0]
    }
    values
}

/// The geometric roundtrip test.
pub struct GeometricTest;

impl RoundtripTest for GeometricTest {
    fn name(&self) -> &str {
        "geometric"
    }

    fn description(&self) -> &str {
        "dot, cross, normalize, length, distance, reflect, face_forward, degrees, radians, sign, \
         abs, fma"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = geo_inputs();
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);

        let layout_entries = &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let epsilon = 1e-4;
        let mut results = Vec::new();

        // --- geo_scalar: degrees, radians, sign, abs ---
        {
            let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: geo_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            geo_scalar::INPUT.set(inputs);
            geo_scalar::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                geo_scalar::main(builtins.global_invocation_id);
            });
            let cpu_output = geo_scalar::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("degrees({x:.4})"),
                        format!("radians({x:.4})"),
                        format!("sign({x:.4})"),
                        format!("abs({x:.4})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "geo_scalar",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- geo_fma: fma, distance(scalar), length(scalar) ---
        {
            let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: geo_fma::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            geo_fma::INPUT.set(inputs);
            geo_fma::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                geo_fma::main(builtins.global_invocation_id);
            });
            let cpu_output = geo_fma::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("fma({x:.4}, 2, 1)"),
                        format!("distance({x:.4}, 0)"),
                        format!("length({x:.4})"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "geo_fma",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- geo_vector: dot, length, distance, normalize ---
        {
            const OUTPUT_COUNT: usize = 16;
            let output_size = (OUTPUT_COUNT * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: geo_vector::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            geo_vector::INPUT.set(inputs);
            geo_vector::OUTPUT.set([Vec4f::default(); OUTPUT_COUNT]);
            dispatch_workgroups((1, 1, 1), (OUTPUT_COUNT as u32, 1, 1), |builtins| {
                geo_vector::main(builtins.global_invocation_id);
            });
            let cpu_output = geo_vector::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..OUTPUT_COUNT)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] dot(a,b)"),
                        format!("[{i}] length(a)"),
                        format!("[{i}] distance(a,b)"),
                        format!("[{i}] length(normalize(a))"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "geo_vector",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- geo_cross_reflect: cross, reflect, face_forward ---
        {
            const OUTPUT_COUNT: usize = 16;
            let output_size = (OUTPUT_COUNT * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: geo_cross_reflect::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            geo_cross_reflect::INPUT.set(inputs);
            geo_cross_reflect::OUTPUT.set([Vec4f::default(); OUTPUT_COUNT]);
            dispatch_workgroups((1, 1, 1), (8, 1, 1), |builtins| {
                geo_cross_reflect::main(builtins.global_invocation_id);
            });
            let cpu_output = geo_cross_reflect::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..8)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] cross.x"),
                        format!("[{i}] cross.y"),
                        format!("[{i}] cross.z"),
                        format!("[{i}] reflect.x"),
                        format!("[{i}] reflect.y"),
                        format!("[{i}] reflect.z"),
                        format!("[{i}] face_forward.x"),
                        format!("[{i}] face_forward.y"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "geo_cross_reflect",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
