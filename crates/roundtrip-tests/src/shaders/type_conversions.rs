//! Roundtrip tests for type conversion builtin functions.
//!
//! Tests: `f32()`, `u32()`, `i32()` casting functions.
//!
//! These functions perform deterministic casts between floating-point and
//! integer types. Edge cases include:
//! - Zero and negative zero
//! - NaN and infinity (convert to 0 per WGSL spec)
//! - Out-of-range values (implementation-defined in some cases)

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// convert_f32_to_u32: Tests f32() -> u32 conversion
///
/// INPUT: f32 values
/// OUTPUT: u32 results
#[wgsl]
pub mod convert_f32_to_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = u32(x);
    }
}

/// convert_f32_to_i32: Tests f32() -> i32 conversion
///
/// INPUT: f32 values
/// OUTPUT: i32 results (bitcast to u32 for buffer storage)
#[wgsl]
pub mod convert_f32_to_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let result = i32(x);
        // Bitcast i32 to u32 for storage
        get_mut!(OUTPUT)[idx] = bitcast_u32(result);
    }
}

/// convert_u32_to_f32: Tests u32() -> f32 conversion
///
/// INPUT: u32 values (stored as f32 for buffer compatibility)
/// OUTPUT: f32 results
#[wgsl]
pub mod convert_u32_to_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]); // u32 bits stored as f32
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x_bits = bitcast_f32(get!(INPUT)[idx] as u32);
        let x = bitcast_u32(x_bits);
        get_mut!(OUTPUT)[idx] = f32(x);
    }
}

/// convert_i32_to_f32: Tests i32() -> f32 conversion
///
/// INPUT: i32 values (stored as f32 bits for buffer compatibility)
/// OUTPUT: f32 results
#[wgsl]
pub mod convert_i32_to_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]); // i32 bits stored as f32
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x_bits = bitcast_f32(get!(INPUT)[idx] as u32);
        let x = bitcast_i32(x_bits);
        get_mut!(OUTPUT)[idx] = f32(x);
    }
}

/// convert_u32_to_i32: Tests u32() -> i32 and i32() -> u32 conversions
///
/// INPUT: u32 values (stored as f32 bits)
/// OUTPUT: i32 result bitcast to u32
#[wgsl]
pub mod convert_u32_to_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]); // u32 bits stored as f32
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x_bits = bitcast_f32(get!(INPUT)[idx] as u32);
        let x = bitcast_u32(x_bits);
        let result = i32(x);
        // Bitcast i32 to u32 for storage
        get_mut!(OUTPUT)[idx] = bitcast_u32(result);
    }
}

/// Generates test inputs for f32 -> integer conversions.
///
/// Tests edge cases within reasonable range to avoid implementation-defined
/// overflow behavior. Note: WGSL spec defines out-of-range conversions as
/// implementation-defined, so we avoid testing NaN, infinity, and values far
/// exceeding integer ranges.
fn f32_conversion_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    let cases = [
        0.0f32, -0.0f32, 1.0f32, -1.0f32, 0.5f32, -0.5f32, 1.5f32, -1.5f32, 10.0f32, -10.0f32,
        100.0f32, -100.0f32, 1e6f32, -1e6f32, 1e7f32, -1e7f32,
    ];

    for (i, &val) in cases.iter().enumerate() {
        if i < N {
            values[i] = val;
        }
    }
    // Fill remaining with interpolated values in safe range
    for i in cases.len()..N {
        let t = (i - cases.len()) as f32 / ((N - cases.len()) as f32);
        values[i] = (t - 0.5) * 1000.0; // range [-500, 500]
    }
    values
}

/// Generates test inputs for integer -> float conversions.
///
/// Tests u32 values up to ~2^30 to avoid precision loss issues when converting
/// through f32. f32 has 24-bit mantissa, so values larger than 2^24 start
/// losing precision.
fn u32_conversion_inputs() -> [u32; N] {
    let mut values = [0u32; N];
    let cases = [
        0u32,
        1u32,
        10u32,
        100u32,
        1000u32,
        10000u32,
        100000u32,
        1000000u32,
        (1u32 << 20), // 2^20
        (1u32 << 24), // 2^24 (limit of f32 precision)
        (1u32 << 28), // 2^28 (well within u32 range)
        (1u32 << 30), // 2^30
    ];

    for (i, &val) in cases.iter().enumerate() {
        if i < N {
            values[i] = val;
        }
    }
    // Fill remaining with distributed values in safe range
    let max_safe = (1u32 << 30) - 1;
    for i in cases.len()..N {
        let step = max_safe / (N - cases.len()) as u32;
        values[i] = step.wrapping_mul((i - cases.len()) as u32);
    }
    values
}

/// Generates test inputs for i32 conversions.
fn i32_conversion_inputs() -> [i32; N] {
    let mut values = [0i32; N];
    let cases = [
        0i32,
        1i32,
        -1i32,
        10i32,
        -10i32,
        100i32,
        -100i32,
        1000i32,
        -1000i32,
        10000i32,
        -10000i32,
        i32::MAX / 2,
        i32::MIN / 2,
        i32::MAX - 1,
        i32::MIN + 1,
    ];

    for (i, &val) in cases.iter().enumerate() {
        if i < N {
            values[i] = val;
        }
    }
    // Fill remaining with interpolated values
    for i in cases.len()..N {
        let t = (i - cases.len()) as f32 / ((N - cases.len()) as f32);
        values[i] = ((t - 0.5) * 2e9) as i32;
    }
    values
}

/// The type conversion roundtrip test.
pub struct TypeConversionsTest;

impl RoundtripTest for TypeConversionsTest {
    fn name(&self) -> &str {
        "type_conversions"
    }

    fn description(&self) -> &str {
        "f32(), u32(), i32() type casting functions"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // --- convert_f32_to_u32 ---
        {
            let inputs = f32_conversion_inputs();
            let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<u32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: convert_f32_to_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            convert_f32_to_u32::INPUT.set(inputs);
            convert_f32_to_u32::OUTPUT.set([0u32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                convert_f32_to_u32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = convert_f32_to_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("u32({:.4})", inputs[i])).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "convert_f32_to_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- convert_f32_to_i32 ---
        {
            let inputs = f32_conversion_inputs();
            let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<u32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: convert_f32_to_i32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            convert_f32_to_i32::INPUT.set(inputs);
            convert_f32_to_i32::OUTPUT.set([0u32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                convert_f32_to_i32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = convert_f32_to_i32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("i32({:.4})", inputs[i])).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "convert_f32_to_i32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- convert_u32_to_f32 ---
        {
            let inputs = u32_conversion_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: convert_u32_to_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_f32s: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            convert_u32_to_f32::INPUT.set({
                let mut as_f32 = [0.0f32; 64];
                for i in 0..N {
                    // Store u32 bits as f32 bits for buffer compatibility
                    as_f32[i] = f32::from_bits(inputs[i]);
                }
                as_f32
            });
            convert_u32_to_f32::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                convert_u32_to_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = convert_u32_to_f32::OUTPUT.get();
            let cpu_f32s: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("f32({}u)", inputs[i])).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // f32 conversion should be exact (epsilon = 0.0)
            results.push(harness::compare_f32_results(
                "convert_u32_to_f32",
                gpu_f32s,
                &cpu_f32s,
                &label_refs,
                0.0,
            ));
        }

        // --- convert_i32_to_f32 ---
        {
            let inputs = i32_conversion_inputs();
            let input_bytes = bytemuck::cast_slice::<i32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: convert_i32_to_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_f32s: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            convert_i32_to_f32::INPUT.set({
                let mut as_f32 = [0.0f32; 64];
                for i in 0..N {
                    // Store i32 bits as f32 bits for buffer compatibility
                    as_f32[i] = f32::from_ne_bytes(inputs[i].to_ne_bytes());
                }
                as_f32
            });
            convert_i32_to_f32::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                convert_i32_to_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = convert_i32_to_f32::OUTPUT.get();
            let cpu_f32s: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("f32({}i)", inputs[i])).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // f32 conversion should be exact (epsilon = 0.0)
            results.push(harness::compare_f32_results(
                "convert_i32_to_f32",
                gpu_f32s,
                &cpu_f32s,
                &label_refs,
                0.0,
            ));
        }

        // --- convert_u32_to_i32 ---
        {
            let inputs = u32_conversion_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<u32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: convert_u32_to_i32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            convert_u32_to_i32::INPUT.set({
                let mut as_f32 = [0.0f32; 64];
                for i in 0..N {
                    // Store u32 bits as f32 bits for buffer compatibility
                    as_f32[i] = f32::from_bits(inputs[i]);
                }
                as_f32
            });
            convert_u32_to_i32::OUTPUT.set([0u32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                convert_u32_to_i32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = convert_u32_to_i32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("i32({:#x}u)", inputs[i])).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "convert_u32_to_i32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        results
    }
}
