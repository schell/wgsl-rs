//! Roundtrip tests for clamping and interpolation builtin functions.
//!
//! Tests: `clamp`, `min`, `max`, `mix`, `smoothstep`, `step`, `select`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Clamping functions: clamp, min, max, step.
#[wgsl]
pub mod clamp_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4f(clamp(x, -1.0, 1.0), min(x, 0.5), max(x, -0.5), step(0.0, x));
    }
}

/// Interpolation functions: mix, smoothstep, select.
#[wgsl]
pub mod clamp_interp {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // mix between 0.0 and 10.0 using x as blend factor (clamped 0..1)
        let t = clamp(x, 0.0, 1.0);
        let mixed = mix(0.0, 10.0, t);
        let smoothed = smoothstep(0.0, 1.0, t);
        let selected = select(1.0, 2.0, x > 0.0);
        get_mut!(OUTPUT)[idx] = vec4f(mixed, smoothed, selected, 0.0);
    }
}

/// Generates test inputs for clamping functions.
fn clamping_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, value) in values.iter_mut().enumerate() {
        let t = i as f32 / (N - 1) as f32;
        *value = t * 4.0 - 2.0; // [-2.0, 2.0]
    }
    values
}

/// The clamping roundtrip test.
pub struct ClampingTest;

impl RoundtripTest for ClampingTest {
    fn name(&self) -> &str {
        "clamping"
    }

    fn description(&self) -> &str {
        "clamp, min, max, mix, smoothstep, step, select"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = clamping_inputs();
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

        let mut results = Vec::new();

        // --- clamp_basic: clamp, min, max, step ---
        {
            let epsilon = 0.0;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: clamp_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            clamp_basic::INPUT.set(inputs);
            clamp_basic::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                clamp_basic::main(builtins.global_invocation_id);
            });
            let cpu_output = clamp_basic::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("clamp({x:.4}, -1, 1)"),
                        format!("min({x:.4}, 0.5)"),
                        format!("max({x:.4}, -0.5)"),
                        format!("step(0, {x:.4})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "clamp_basic",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- clamp_interp: mix, smoothstep, select ---
        {
            let epsilon = 1e-5;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: clamp_interp::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: layout_entries,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            clamp_interp::INPUT.set(inputs);
            clamp_interp::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                clamp_interp::main(builtins.global_invocation_id);
            });
            let cpu_output = clamp_interp::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("mix(0,10,clamp({x:.4}))"),
                        format!("smoothstep(0,1,clamp({x:.4}))"),
                        format!("select(1,2,{x:.4}>0)"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_f32_results(
                "clamp_interp",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        results
    }
}
