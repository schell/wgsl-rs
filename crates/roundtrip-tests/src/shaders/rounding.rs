//! Roundtrip tests for rounding builtin functions.
//!
//! Tests: `ceil`, `floor`, `round`, `trunc`, `fract`, `saturate`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Rounding functions: ceil, floor, round, trunc.
#[wgsl]
pub mod rounding_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(ceil(x), floor(x), round(x), trunc(x));
    }
}

/// Fractional and saturation: fract, saturate.
#[wgsl]
pub mod rounding_fract {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(fract(x), saturate(x), 0.0, 0.0);
    }
}

/// Generates test inputs for rounding functions.
///
/// Returns 64 values including negative, positive, half-integer, and extreme
/// values to exercise all rounding modes.
fn rounding_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f32 / (N - 1) as f32;
        // Range [-5.0, 5.0] with half-integer values well represented
        *value = t * 10.0 - 5.0;
    }
    // Add some special half-integer cases.
    values[0] = -2.5;
    values[1] = -1.5;
    values[2] = -0.5;
    values[3] = 0.5;
    values[4] = 1.5;
    values[5] = 2.5;
    values[6] = 0.0;
    values[7] = -0.0;
    values
}

/// The rounding roundtrip test.
pub struct RoundingTest;

impl RoundtripTest for RoundingTest {
    fn name(&self) -> &str {
        "rounding"
    }

    fn description(&self) -> &str {
        "ceil, floor, round, trunc, fract, saturate"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = rounding_inputs();
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

        // Rounding functions should produce exact results.
        let epsilon = 0.0;
        let mut results = Vec::new();

        // --- rounding_basic: ceil, floor, round, trunc ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: rounding_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            rounding_basic::INPUT.set(inputs);
            rounding_basic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                rounding_basic::main(builtins.global_invocation_id);
            });
            let cpu_output = rounding_basic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("ceil({x:.4})"),
                        format!("floor({x:.4})"),
                        format!("round({x:.4})"),
                        format!("trunc({x:.4})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "rounding_basic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- rounding_fract: fract, saturate ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: rounding_fract::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            rounding_fract::INPUT.set(inputs);
            rounding_fract::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                rounding_fract::main(builtins.global_invocation_id);
            });
            let cpu_output = rounding_fract::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("fract({x:.4})"),
                        format!("saturate({x:.4})"),
                        format!("(padding)"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // fract and saturate may have tiny floating-point differences.
            let fract_epsilon = 1e-6;
            results.push(harness::compare_f32_results(
                "rounding_fract",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                fract_epsilon,
            ));
        }

        results
    }
}
