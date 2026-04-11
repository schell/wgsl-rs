//! Roundtrip tests for exponential builtin functions.
//!
//! Tests: `exp`, `exp2`, `log`, `log2`, `pow`, `sqrt`, `inverse_sqrt`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Exponential functions: exp, exp2, log, log2.
#[wgsl]
pub mod exp_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // log/log2 need positive input
        let pos = max(x, 0.001);
        get_mut!(OUTPUT)[idx] = vec4f(exp(x), exp2(x), log(pos), log2(pos));
    }
}

/// Power and root functions: pow, sqrt, inverse_sqrt.
#[wgsl]
pub mod exp_power {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let pos = max(x, 0.001);
        get_mut!(OUTPUT)[idx] = vec4f(pow(pos, 2.0), pow(pos, 0.5), sqrt(pos), inverse_sqrt(pos));
    }
}

/// Generates test inputs for exponential functions.
///
/// Returns 64 values in `[-3.0, 5.0]`, a range where exp/exp2 don't overflow
/// and log inputs (after clamping to positive) cover a useful range.
fn exp_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f32 / (N - 1) as f32;
        *value = t * 8.0 - 3.0;
    }
    values
}

/// The exponential roundtrip test.
pub struct ExponentialTest;

impl RoundtripTest for ExponentialTest {
    fn name(&self) -> &str {
        "exponential"
    }

    fn description(&self) -> &str {
        "exp, exp2, log, log2, pow, sqrt, inverse_sqrt"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = exp_inputs();
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

        let epsilon = 1e-3;
        let mut results = Vec::new();

        // --- exp_basic: exp, exp2, log, log2 ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: exp_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            exp_basic::INPUT.set(inputs);
            exp_basic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                exp_basic::main(builtins.global_invocation_id);
            });
            let cpu_output = exp_basic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("exp({x:.4})"),
                        format!("exp2({x:.4})"),
                        format!("log(max({x:.4}, 0.001))"),
                        format!("log2(max({x:.4}, 0.001))"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "exp_basic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- exp_power: pow, sqrt, inverse_sqrt ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: exp_power::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            exp_power::INPUT.set(inputs);
            exp_power::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                exp_power::main(builtins.global_invocation_id);
            });
            let cpu_output = exp_power::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("pow(max({x:.4},0.001), 2.0)"),
                        format!("pow(max({x:.4},0.001), 0.5)"),
                        format!("sqrt(max({x:.4},0.001))"),
                        format!("inverse_sqrt(max({x:.4},0.001))"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "exp_power",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
