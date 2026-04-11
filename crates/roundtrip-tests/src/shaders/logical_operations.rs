//! Roundtrip tests for logical builtin functions.
//!
//! Tests: `all`, `any` on bool vectors
//!
//! Covers Vec2b, Vec3b, Vec4b types.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// all_vec2b: Test all() on Vec2<bool>
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec2b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 128]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 2;
        let vec_input = vec2b(input[base] != 0u32, input[base + 1] != 0u32);

        let result = all(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// all_vec3b: Test all() on Vec3<bool>
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec3b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 192]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 3;
        let vec_input = vec3b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
        );

        let result = all(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// all_vec4b: Test all() on Vec4<bool>
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec4b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 256]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 4;
        let vec_input = vec4b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
            input[base + 3] != 0u32,
        );

        let result = all(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// any_vec2b: Test any() on Vec2<bool>
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec2b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 128]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 2;
        let vec_input = vec2b(input[base] != 0u32, input[base + 1] != 0u32);

        let result = any(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// any_vec3b: Test any() on Vec3<bool>
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec3b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 192]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 3;
        let vec_input = vec3b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
        );

        let result = any(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// any_vec4b: Test any() on Vec4<bool>
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec4b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 256]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 4;
        let vec_input = vec4b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
            input[base + 3] != 0u32,
        );

        let result = any(vec_input);

        if result {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

// ============================================================================
// Input Generators
// ============================================================================

fn all_any_vec2b_inputs() -> [u32; N * 2] {
    let mut inputs = [0u32; N * 2];
    let patterns = [(0, 0), (1, 1), (1, 0), (0, 1)];
    for i in 0..N {
        let (a, b) = patterns[i % 4];
        inputs[i * 2] = a;
        inputs[i * 2 + 1] = b;
    }
    inputs
}

fn all_any_vec3b_inputs() -> [u32; N * 3] {
    let mut inputs = [0u32; N * 3];
    let patterns = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0)];
    for i in 0..N {
        let (a, b, c) = patterns[i % 4];
        inputs[i * 3] = a;
        inputs[i * 3 + 1] = b;
        inputs[i * 3 + 2] = c;
    }
    inputs
}

fn all_any_vec4b_inputs() -> [u32; N * 4] {
    let mut inputs = [0u32; N * 4];
    let patterns = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 0, 0), (1, 1, 0, 0)];
    for i in 0..N {
        let (a, b, c, d) = patterns[i % 4];
        inputs[i * 4] = a;
        inputs[i * 4 + 1] = b;
        inputs[i * 4 + 2] = c;
        inputs[i * 4 + 3] = d;
    }
    inputs
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct LogicalOperationsTest;

impl RoundtripTest for LogicalOperationsTest {
    fn name(&self) -> &str {
        "logical_operations"
    }

    fn description(&self) -> &str {
        "all, any"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // all_vec2b test
        {
            let inputs = all_any_vec2b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: all_vec2b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 2] != 0;
                    let b = inputs[i * 2 + 1] != 0;
                    if a && b { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec2b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec2b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // all_vec3b test
        {
            let inputs = all_any_vec3b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: all_vec3b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 3] != 0;
                    let b = inputs[i * 3 + 1] != 0;
                    let c = inputs[i * 3 + 2] != 0;
                    if a && b && c { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec3b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec3b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // all_vec4b test
        {
            let inputs = all_any_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: all_vec4b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 4] != 0;
                    let b = inputs[i * 4 + 1] != 0;
                    let c = inputs[i * 4 + 2] != 0;
                    let d = inputs[i * 4 + 3] != 0;
                    if a && b && c && d { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec4b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec2b test
        {
            let inputs = all_any_vec2b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: any_vec2b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 2] != 0;
                    let b = inputs[i * 2 + 1] != 0;
                    if a || b { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec2b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec2b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec3b test
        {
            let inputs = all_any_vec3b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: any_vec3b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 3] != 0;
                    let b = inputs[i * 3 + 1] != 0;
                    let c = inputs[i * 3 + 2] != 0;
                    if a || b || c { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec3b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec3b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec4b test
        {
            let inputs = all_any_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: any_vec4b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let a = inputs[i * 4] != 0;
                    let b = inputs[i * 4 + 1] != 0;
                    let c = inputs[i * 4 + 2] != 0;
                    let d = inputs[i * 4 + 3] != 0;
                    if a || b || c || d { 1u32 } else { 0u32 }
                })
                .collect();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec4b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        results
    }
}
