//! Roundtrip tests for trigonometric builtin functions.
//!
//! Tests: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
//! `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

/// Number of test cases per shader invocation batch.
const N: usize = 64;

/// Basic trig functions: sin, cos, tan, asin.
///
/// Each invocation reads one f32 input and writes four f32 outputs packed as
/// a `Vec4f`.
#[wgsl]
pub mod trig_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(sin(x), cos(x), tan(x), asin(clamp(x, -1.0, 1.0)));
    }
}

/// Inverse trig and atan2: acos, atan, atan2(x, 1.0).
#[wgsl]
pub mod trig_inverse {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let clamped = clamp(x, -1.0, 1.0);
        get_mut!(OUTPUT)[idx] = vec4f(acos(clamped), atan(x), atan2(x, 1.0), 0.0);
    }
}

/// Hyperbolic functions: sinh, cosh, tanh, asinh.
#[wgsl]
pub mod trig_hyperbolic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(sinh(x), cosh(x), tanh(x), asinh(x));
    }
}

/// Inverse hyperbolic: acosh, atanh.
#[wgsl]
pub mod trig_inv_hyperbolic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // acosh requires x >= 1.0; atanh requires |x| < 1.0
        let acosh_in = max(x * x + 1.0, 1.0);
        let atanh_in = clamp(x, -0.99, 0.99);
        get_mut!(OUTPUT)[idx] = vec4f(acosh(acosh_in), atanh(atanh_in), 0.0, 0.0);
    }
}

/// Generates test input values for trig functions.
///
/// Returns 64 values spanning a range useful for trigonometric testing:
/// mostly in `[-pi, pi]` with some outliers.
fn trig_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f32 / (N - 1) as f32;
        // Map to [-4.0, 4.0] — covers > full period for sin/cos
        *value = t * 8.0 - 4.0;
    }
    values
}

/// The trig roundtrip test.
pub struct TrigTest;

impl RoundtripTest for TrigTest {
    fn name(&self) -> &str {
        "trig"
    }

    fn description(&self) -> &str {
        "sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = trig_inputs();
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;

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

        // --- trig_basic: sin, cos, tan, asin ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: trig_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            // CPU side
            use wgsl_rs::std::*;
            trig_basic::INPUT.set(inputs);
            trig_basic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                trig_basic::main(builtins.global_invocation_id);
            });
            let cpu_output = trig_basic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("sin({x:.4})"),
                        format!("cos({x:.4})"),
                        format!("tan({x:.4})"),
                        format!("asin(clamp({x:.4}))"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "trig_basic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- trig_inverse: acos, atan, atan2 ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: trig_inverse::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            trig_inverse::INPUT.set(inputs);
            trig_inverse::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                trig_inverse::main(builtins.global_invocation_id);
            });
            let cpu_output = trig_inverse::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("acos(clamp({x:.4}))"),
                        format!("atan({x:.4})"),
                        format!("atan2({x:.4}, 1.0)"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "trig_inverse",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- trig_hyperbolic: sinh, cosh, tanh, asinh ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: trig_hyperbolic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            trig_hyperbolic::INPUT.set(inputs);
            trig_hyperbolic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                trig_hyperbolic::main(builtins.global_invocation_id);
            });
            let cpu_output = trig_hyperbolic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("sinh({x:.4})"),
                        format!("cosh({x:.4})"),
                        format!("tanh({x:.4})"),
                        format!("asinh({x:.4})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "trig_hyperbolic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- trig_inv_hyperbolic: acosh, atanh ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: trig_inv_hyperbolic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            trig_inv_hyperbolic::INPUT.set(inputs);
            trig_inv_hyperbolic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                trig_inv_hyperbolic::main(builtins.global_invocation_id);
            });
            let cpu_output = trig_inv_hyperbolic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("acosh(max({x:.4}^2+1, 1))"),
                        format!("atanh(clamp({x:.4}))"),
                        format!("(padding)"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "trig_inv_hyperbolic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
