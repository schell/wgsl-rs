//! Roundtrip tests for select builtin function.
//!
//! Tests: `select` on scalars and vectors with scalar and vector bool
//! conditions
//!
//! Covers f32, i32, u32 scalars and Vec2f, Vec4f, Vec4i, Vec4u vectors.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// select_f32_scalar: Test select() on f32 with scalar bool condition
///
/// Returns true_val if condition is true, false_val otherwise
#[wgsl]
pub mod select_f32_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 192]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 3;
        let true_val = bitcast_f32(input[base]);
        let false_val = bitcast_f32(input[base + 1]);
        let condition = input[base + 2] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx] = bitcast_u32(result);
    }
}

/// select_i32_scalar: Test select() on i32 with scalar bool condition
///
/// Returns true_val if condition is true, false_val otherwise
#[wgsl]
pub mod select_i32_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 192]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 3;
        let true_val = input[base] as i32;
        let false_val = input[base + 1] as i32;
        let condition = input[base + 2] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx] = result as u32;
    }
}

/// select_u32_scalar: Test select() on u32 with scalar bool condition
///
/// Returns true_val if condition is true, false_val otherwise
#[wgsl]
pub mod select_u32_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 192]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 3;
        let true_val = input[base];
        let false_val = input[base + 1];
        let condition = input[base + 2] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx] = result;
    }
}

/// select_vec2f_scalar: Test select() on Vec2f with scalar bool condition
///
/// Selects entire vector based on scalar condition
#[wgsl]
pub mod select_vec2f_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 128]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 5;
        let true_val = vec2f(bitcast_f32(input[base]), bitcast_f32(input[base + 1]));
        let false_val = vec2f(bitcast_f32(input[base + 2]), bitcast_f32(input[base + 3]));
        let condition = input[base + 4] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 2] = bitcast_u32(result.x);
        get_mut!(OUTPUT)[idx * 2 + 1] = bitcast_u32(result.y);
    }
}

/// select_vec4f_scalar: Test select() on Vec4f with scalar bool condition
///
/// Selects entire vector based on scalar condition
#[wgsl]
pub mod select_vec4f_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 9;
        let true_val = vec4f(
            bitcast_f32(input[base]),
            bitcast_f32(input[base + 1]),
            bitcast_f32(input[base + 2]),
            bitcast_f32(input[base + 3]),
        );
        let false_val = vec4f(
            bitcast_f32(input[base + 4]),
            bitcast_f32(input[base + 5]),
            bitcast_f32(input[base + 6]),
            bitcast_f32(input[base + 7]),
        );
        let condition = input[base + 8] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = bitcast_u32(result.x);
        get_mut!(OUTPUT)[idx * 4 + 1] = bitcast_u32(result.y);
        get_mut!(OUTPUT)[idx * 4 + 2] = bitcast_u32(result.z);
        get_mut!(OUTPUT)[idx * 4 + 3] = bitcast_u32(result.w);
    }
}

/// select_vec4i_scalar: Test select() on Vec4i with scalar bool condition
///
/// Selects entire vector based on scalar condition
#[wgsl]
pub mod select_vec4i_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 9;
        let true_val = vec4i(
            input[base] as i32,
            input[base + 1] as i32,
            input[base + 2] as i32,
            input[base + 3] as i32,
        );
        let false_val = vec4i(
            input[base + 4] as i32,
            input[base + 5] as i32,
            input[base + 6] as i32,
            input[base + 7] as i32,
        );
        let condition = input[base + 8] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = result.x as u32;
        get_mut!(OUTPUT)[idx * 4 + 1] = result.y as u32;
        get_mut!(OUTPUT)[idx * 4 + 2] = result.z as u32;
        get_mut!(OUTPUT)[idx * 4 + 3] = result.w as u32;
    }
}

/// select_vec4u_scalar: Test select() on Vec4u with scalar bool condition
///
/// Selects entire vector based on scalar condition
#[wgsl]
pub mod select_vec4u_scalar {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 9;
        let true_val = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let false_val = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );
        let condition = input[base + 8] != 0u32;

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = result.x;
        get_mut!(OUTPUT)[idx * 4 + 1] = result.y;
        get_mut!(OUTPUT)[idx * 4 + 2] = result.z;
        get_mut!(OUTPUT)[idx * 4 + 3] = result.w;
    }
}

/// select_vec4f_vec4b: Test select() on Vec4f with Vec4b condition
///
/// Component-wise selection using vector bool condition
#[wgsl]
pub mod select_vec4f_vec4b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 12;
        let true_val = vec4f(
            bitcast_f32(input[base]),
            bitcast_f32(input[base + 1]),
            bitcast_f32(input[base + 2]),
            bitcast_f32(input[base + 3]),
        );
        let false_val = vec4f(
            bitcast_f32(input[base + 4]),
            bitcast_f32(input[base + 5]),
            bitcast_f32(input[base + 6]),
            bitcast_f32(input[base + 7]),
        );
        let condition = vec4b(
            input[base + 8] != 0u32,
            input[base + 9] != 0u32,
            input[base + 10] != 0u32,
            input[base + 11] != 0u32,
        );

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = bitcast_u32(result.x);
        get_mut!(OUTPUT)[idx * 4 + 1] = bitcast_u32(result.y);
        get_mut!(OUTPUT)[idx * 4 + 2] = bitcast_u32(result.z);
        get_mut!(OUTPUT)[idx * 4 + 3] = bitcast_u32(result.w);
    }
}

/// select_vec4i_vec4b: Test select() on Vec4i with Vec4b condition
///
/// Component-wise selection using vector bool condition
#[wgsl]
pub mod select_vec4i_vec4b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 12;
        let true_val = vec4i(
            input[base] as i32,
            input[base + 1] as i32,
            input[base + 2] as i32,
            input[base + 3] as i32,
        );
        let false_val = vec4i(
            input[base + 4] as i32,
            input[base + 5] as i32,
            input[base + 6] as i32,
            input[base + 7] as i32,
        );
        let condition = vec4b(
            input[base + 8] != 0u32,
            input[base + 9] != 0u32,
            input[base + 10] != 0u32,
            input[base + 11] != 0u32,
        );

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = result.x as u32;
        get_mut!(OUTPUT)[idx * 4 + 1] = result.y as u32;
        get_mut!(OUTPUT)[idx * 4 + 2] = result.z as u32;
        get_mut!(OUTPUT)[idx * 4 + 3] = result.w as u32;
    }
}

/// select_vec4u_vec4b: Test select() on Vec4u with Vec4b condition
///
/// Component-wise selection using vector bool condition
#[wgsl]
pub mod select_vec4u_vec4b {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        let base = idx * 12;
        let true_val = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let false_val = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );
        let condition = vec4b(
            input[base + 8] != 0u32,
            input[base + 9] != 0u32,
            input[base + 10] != 0u32,
            input[base + 11] != 0u32,
        );

        let result = select(false_val, true_val, condition);

        get_mut!(OUTPUT)[idx * 4] = result.x;
        get_mut!(OUTPUT)[idx * 4 + 1] = result.y;
        get_mut!(OUTPUT)[idx * 4 + 2] = result.z;
        get_mut!(OUTPUT)[idx * 4 + 3] = result.w;
    }
}

// ============================================================================
// Input Generators
// ============================================================================

fn select_scalar_inputs() -> [u32; N * 3] {
    let mut inputs = [0u32; N * 3];
    for i in 0..N {
        inputs[i * 3] = (i as u32).wrapping_mul(1664525).wrapping_add(22695477); // LCG
        inputs[i * 3 + 1] = (i as u32).wrapping_mul(2891336453).wrapping_add(12345);
        inputs[i * 3 + 2] = if i % 2 == 0 { 0u32 } else { 1u32 };
    }
    inputs
}

fn select_vec2f_scalar_inputs() -> [u32; N * 5] {
    let mut inputs = [0u32; N * 5];
    for i in 0..N {
        inputs[i * 5] = (i as u32).wrapping_mul(1664525);
        inputs[i * 5 + 1] = (i as u32).wrapping_mul(2891336453);
        inputs[i * 5 + 2] = (i as u32).wrapping_mul(1103515245).wrapping_add(12345);
        inputs[i * 5 + 3] = (i as u32).wrapping_mul(134775813);
        inputs[i * 5 + 4] = if i % 2 == 0 { 0u32 } else { 1u32 };
    }
    inputs
}

fn select_vec4_scalar_inputs() -> [u32; N * 9] {
    let mut inputs = [0u32; N * 9];
    for i in 0..N {
        inputs[i * 9] = (i as u32).wrapping_mul(1664525);
        inputs[i * 9 + 1] = (i as u32).wrapping_mul(2891336453);
        inputs[i * 9 + 2] = (i as u32).wrapping_mul(1103515245);
        inputs[i * 9 + 3] = (i as u32).wrapping_mul(134775813);
        inputs[i * 9 + 4] = (i as u32).wrapping_mul(22695477);
        inputs[i * 9 + 5] = (i as u32).wrapping_mul(69069);
        inputs[i * 9 + 6] = (i as u32).wrapping_mul(1812433253);
        inputs[i * 9 + 7] = (i as u32).wrapping_mul(75);
        inputs[i * 9 + 8] = if i % 2 == 0 { 0u32 } else { 1u32 };
    }
    inputs
}

fn select_vec4_vec4b_inputs() -> [u32; N * 12] {
    let mut inputs = [0u32; N * 12];
    for i in 0..N {
        inputs[i * 12] = (i as u32).wrapping_mul(1664525);
        inputs[i * 12 + 1] = (i as u32).wrapping_mul(2891336453);
        inputs[i * 12 + 2] = (i as u32).wrapping_mul(1103515245);
        inputs[i * 12 + 3] = (i as u32).wrapping_mul(134775813);
        inputs[i * 12 + 4] = (i as u32).wrapping_mul(22695477);
        inputs[i * 12 + 5] = (i as u32).wrapping_mul(69069);
        inputs[i * 12 + 6] = (i as u32).wrapping_mul(1812433253);
        inputs[i * 12 + 7] = (i as u32).wrapping_mul(75);
        // Condition bits: different pattern for each component
        inputs[i * 12 + 8] = if i % 4 == 0 { 1u32 } else { 0u32 };
        inputs[i * 12 + 9] = if i % 3 == 0 { 1u32 } else { 0u32 };
        inputs[i * 12 + 10] = if i % 2 == 0 { 1u32 } else { 0u32 };
        inputs[i * 12 + 11] = if i % 5 == 0 { 1u32 } else { 0u32 };
    }
    inputs
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct SelectOperationsTest;

impl RoundtripTest for SelectOperationsTest {
    fn name(&self) -> &str {
        "select_operations"
    }

    fn description(&self) -> &str {
        "select"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // select_f32_scalar test
        {
            let inputs = select_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_f32_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let true_val = f32::from_bits(inputs[i * 3]);
                    let false_val = f32::from_bits(inputs[i * 3 + 1]);
                    let condition = inputs[i * 3 + 2] != 0;
                    if condition { true_val } else { false_val }.to_bits()
                })
                .collect();

            let labels: Vec<String> = (0..N)
                .map(|i| format!("select_f32_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_f32_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_i32_scalar test
        {
            let inputs = select_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_i32_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let true_val = inputs[i * 3] as i32;
                    let false_val = inputs[i * 3 + 1] as i32;
                    let condition = inputs[i * 3 + 2] != 0;
                    (if condition { true_val } else { false_val }) as u32
                })
                .collect();

            let labels: Vec<String> = (0..N)
                .map(|i| format!("select_i32_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_i32_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_u32_scalar test
        {
            let inputs = select_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_u32_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N)
                .map(|i| {
                    let true_val = inputs[i * 3];
                    let false_val = inputs[i * 3 + 1];
                    let condition = inputs[i * 3 + 2] != 0;
                    if condition { true_val } else { false_val }
                })
                .collect();

            let labels: Vec<String> = (0..N)
                .map(|i| format!("select_u32_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_u32_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec2f_scalar test
        {
            let inputs = select_vec2f_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec2f_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 2 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 2)
                .map(|i| {
                    let test_idx = i / 2;
                    let component = i % 2;
                    let base = test_idx * 5;
                    let true_val = [
                        f32::from_bits(inputs[base]),
                        f32::from_bits(inputs[base + 1]),
                    ];
                    let false_val = [
                        f32::from_bits(inputs[base + 2]),
                        f32::from_bits(inputs[base + 3]),
                    ];
                    let condition = inputs[base + 4] != 0;
                    if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }
                    .to_bits()
                })
                .collect();

            let labels: Vec<String> = (0..N * 2)
                .map(|i| format!("select_vec2f_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec2f_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4f_scalar test
        {
            let inputs = select_vec4_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4f_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 9;
                    let true_val = [
                        f32::from_bits(inputs[base]),
                        f32::from_bits(inputs[base + 1]),
                        f32::from_bits(inputs[base + 2]),
                        f32::from_bits(inputs[base + 3]),
                    ];
                    let false_val = [
                        f32::from_bits(inputs[base + 4]),
                        f32::from_bits(inputs[base + 5]),
                        f32::from_bits(inputs[base + 6]),
                        f32::from_bits(inputs[base + 7]),
                    ];
                    let condition = inputs[base + 8] != 0;
                    if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }
                    .to_bits()
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4f_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4f_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4i_scalar test
        {
            let inputs = select_vec4_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4i_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 9;
                    let true_val = [
                        inputs[base] as i32,
                        inputs[base + 1] as i32,
                        inputs[base + 2] as i32,
                        inputs[base + 3] as i32,
                    ];
                    let false_val = [
                        inputs[base + 4] as i32,
                        inputs[base + 5] as i32,
                        inputs[base + 6] as i32,
                        inputs[base + 7] as i32,
                    ];
                    let condition = inputs[base + 8] != 0;
                    (if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }) as u32
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4i_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4i_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4u_scalar test
        {
            let inputs = select_vec4_scalar_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4u_scalar::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 9;
                    let true_val = [
                        inputs[base],
                        inputs[base + 1],
                        inputs[base + 2],
                        inputs[base + 3],
                    ];
                    let false_val = [
                        inputs[base + 4],
                        inputs[base + 5],
                        inputs[base + 6],
                        inputs[base + 7],
                    ];
                    let condition = inputs[base + 8] != 0;
                    if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4u_scalar[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4u_scalar",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4f_vec4b test
        {
            let inputs = select_vec4_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4f_vec4b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 12;
                    let true_val = [
                        f32::from_bits(inputs[base]),
                        f32::from_bits(inputs[base + 1]),
                        f32::from_bits(inputs[base + 2]),
                        f32::from_bits(inputs[base + 3]),
                    ];
                    let false_val = [
                        f32::from_bits(inputs[base + 4]),
                        f32::from_bits(inputs[base + 5]),
                        f32::from_bits(inputs[base + 6]),
                        f32::from_bits(inputs[base + 7]),
                    ];
                    let condition = inputs[base + 8 + component] != 0;
                    if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }
                    .to_bits()
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4f_vec4b[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4f_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4i_vec4b test
        {
            let inputs = select_vec4_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4i_vec4b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 12;
                    let true_val = [
                        inputs[base] as i32,
                        inputs[base + 1] as i32,
                        inputs[base + 2] as i32,
                        inputs[base + 3] as i32,
                    ];
                    let false_val = [
                        inputs[base + 4] as i32,
                        inputs[base + 5] as i32,
                        inputs[base + 6] as i32,
                        inputs[base + 7] as i32,
                    ];
                    let condition = inputs[base + 8 + component] != 0;
                    (if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }) as u32
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4i_vec4b[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4i_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // select_vec4u_vec4b test
        {
            let inputs = select_vec4_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let gpu_output = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: select_vec4u_vec4b::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size: (N * 4 * std::mem::size_of::<u32>()) as u64,
                workgroup_count: (1, 1, 1),
            });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);
            let cpu_results: Vec<u32> = (0..N * 4)
                .map(|i| {
                    let test_idx = i / 4;
                    let component = i % 4;
                    let base = test_idx * 12;
                    let true_val = [
                        inputs[base],
                        inputs[base + 1],
                        inputs[base + 2],
                        inputs[base + 3],
                    ];
                    let false_val = [
                        inputs[base + 4],
                        inputs[base + 5],
                        inputs[base + 6],
                        inputs[base + 7],
                    ];
                    let condition = inputs[base + 8 + component] != 0;
                    if condition {
                        true_val[component]
                    } else {
                        false_val[component]
                    }
                })
                .collect();

            let labels: Vec<String> = (0..N * 4)
                .map(|i| format!("select_vec4u_vec4b[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "select_vec4u_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        results
    }
}
