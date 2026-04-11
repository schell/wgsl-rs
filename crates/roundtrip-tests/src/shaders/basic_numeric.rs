//! Roundtrip tests for basic numeric functions.
//!
//! Tests: `abs`, `sign`, `degrees`, `radians`
//!
//! Covers both scalar and Vec4 types where applicable.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// abs_f32: Absolute value for f32 and Vec4f
///
/// Tests abs() on both scalar f32 and Vec4f
#[wgsl]
pub mod abs_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]); // 64 scalars + 64 vec4s = 320 floats
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 320]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        // Scalar abs
        let scalar_input = input[idx];
        let scalar_result = abs(scalar_input);

        // Vec4 abs
        let vec_base = 64 + idx * 4;
        let vec_input = vec4f(
            input[vec_base],
            input[vec_base + 1],
            input[vec_base + 2],
            input[vec_base + 3],
        );
        let vec_result = abs(vec_input);

        // Write results
        get_mut!(OUTPUT)[idx] = scalar_result;
        let out_base = 64 + idx * 4;
        get_mut!(OUTPUT)[out_base] = vec_result.x;
        get_mut!(OUTPUT)[out_base + 1] = vec_result.y;
        get_mut!(OUTPUT)[out_base + 2] = vec_result.z;
        get_mut!(OUTPUT)[out_base + 3] = vec_result.w;
    }
}

/// abs_i32: Absolute value for i32 and Vec4i
///
/// Tests abs() on both scalar i32 and Vec4i.
/// Note: abs(i32::MIN) wraps to i32::MIN per two's complement arithmetic.
#[wgsl]
pub mod abs_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [i32; 320]); // 64 scalars + 64 vec4s
    storage!(group(0), binding(1), read_write, OUTPUT: [i32; 320]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        // Scalar abs
        let scalar_input = input[idx];
        let scalar_result = abs(scalar_input);

        // Vec4 abs
        let vec_base = 64 + idx * 4;
        let vec_input = vec4i(
            input[vec_base],
            input[vec_base + 1],
            input[vec_base + 2],
            input[vec_base + 3],
        );
        let vec_result = abs(vec_input);

        // Write results
        get_mut!(OUTPUT)[idx] = scalar_result;
        let out_base = 64 + idx * 4;
        get_mut!(OUTPUT)[out_base] = vec_result.x;
        get_mut!(OUTPUT)[out_base + 1] = vec_result.y;
        get_mut!(OUTPUT)[out_base + 2] = vec_result.z;
        get_mut!(OUTPUT)[out_base + 3] = vec_result.w;
    }
}

/// sign_f32: Sign function for f32 and Vec4f
///
/// Returns -1.0 for negative, 0.0 for zero, 1.0 for positive
#[wgsl]
pub mod sign_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 320]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        // Scalar sign
        let scalar_input = input[idx];
        let scalar_result = sign(scalar_input);

        // Vec4 sign
        let vec_base = 64 + idx * 4;
        let vec_input = vec4f(
            input[vec_base],
            input[vec_base + 1],
            input[vec_base + 2],
            input[vec_base + 3],
        );
        let vec_result = sign(vec_input);

        // Write results
        get_mut!(OUTPUT)[idx] = scalar_result;
        let out_base = 64 + idx * 4;
        get_mut!(OUTPUT)[out_base] = vec_result.x;
        get_mut!(OUTPUT)[out_base + 1] = vec_result.y;
        get_mut!(OUTPUT)[out_base + 2] = vec_result.z;
        get_mut!(OUTPUT)[out_base + 3] = vec_result.w;
    }
}

// Note: sign() is only implemented for f32 types in wgsl-rs, not i32.
// WGSL spec does support sign() for integers, but wgsl-rs hasn't implemented it
// yet.

/// degrees_f32: Convert radians to degrees for f32 and Vec4f
///
/// Tests degrees() conversion
#[wgsl]
pub mod degrees_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 320]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        // Scalar degrees
        let scalar_input = input[idx];
        let scalar_result = degrees(scalar_input);

        // Vec4 degrees
        let vec_base = 64 + idx * 4;
        let vec_input = vec4f(
            input[vec_base],
            input[vec_base + 1],
            input[vec_base + 2],
            input[vec_base + 3],
        );
        let vec_result = degrees(vec_input);

        // Write results
        get_mut!(OUTPUT)[idx] = scalar_result;
        let out_base = 64 + idx * 4;
        get_mut!(OUTPUT)[out_base] = vec_result.x;
        get_mut!(OUTPUT)[out_base + 1] = vec_result.y;
        get_mut!(OUTPUT)[out_base + 2] = vec_result.z;
        get_mut!(OUTPUT)[out_base + 3] = vec_result.w;
    }
}

/// radians_f32: Convert degrees to radians for f32 and Vec4f
///
/// Tests radians() conversion
#[wgsl]
pub mod radians_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 320]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);

        // Scalar radians
        let scalar_input = input[idx];
        let scalar_result = radians(scalar_input);

        // Vec4 radians
        let vec_base = 64 + idx * 4;
        let vec_input = vec4f(
            input[vec_base],
            input[vec_base + 1],
            input[vec_base + 2],
            input[vec_base + 3],
        );
        let vec_result = radians(vec_input);

        // Write results
        get_mut!(OUTPUT)[idx] = scalar_result;
        let out_base = 64 + idx * 4;
        get_mut!(OUTPUT)[out_base] = vec_result.x;
        get_mut!(OUTPUT)[out_base + 1] = vec_result.y;
        get_mut!(OUTPUT)[out_base + 2] = vec_result.z;
        get_mut!(OUTPUT)[out_base + 3] = vec_result.w;
    }
}

// ============================================================================
// Input Generators
// ============================================================================

/// Generates test inputs for abs(f32).
/// Includes positive, negative, zero, and edge cases.
fn abs_f32_inputs() -> ([f32; N], [wgsl_rs::std::Vec4f; N]) {
    use wgsl_rs::std::vec4f;

    let mut scalars = [0.0f32; N];
    let mut vectors = [vec4f(0.0, 0.0, 0.0, 0.0); N];

    // Edge cases
    scalars[0] = 0.0;
    scalars[1] = -0.0;
    scalars[2] = 1.0;
    scalars[3] = -1.0;
    scalars[4] = f32::MAX;
    scalars[5] = f32::MIN;
    scalars[6] = f32::EPSILON;
    scalars[7] = -f32::EPSILON;

    vectors[0] = vec4f(0.0, -0.0, 1.0, -1.0);
    vectors[1] = vec4f(f32::MAX, f32::MIN, f32::EPSILON, -f32::EPSILON);
    vectors[2] = vec4f(-100.0, -200.0, -300.0, -400.0);
    vectors[3] = vec4f(100.0, 200.0, 300.0, 400.0);

    // Fill with diverse values
    for i in 8..N {
        let t = i as f32;
        scalars[i] = if i % 2 == 0 { t * 10.0 } else { -t * 10.0 };
        vectors[i] = vec4f(t, -t * 2.0, t * 3.0, -t * 4.0);
    }

    (scalars, vectors)
}

/// Generates test inputs for abs(i32).
/// Includes positive, negative, zero, and edge cases including i32::MIN.
fn abs_i32_inputs() -> ([i32; N], [wgsl_rs::std::Vec4i; N]) {
    use wgsl_rs::std::vec4i;

    let mut scalars = [0i32; N];
    let mut vectors = [vec4i(0, 0, 0, 0); N];

    // Edge cases
    scalars[0] = 0;
    scalars[1] = 1;
    scalars[2] = -1;
    scalars[3] = i32::MAX;
    scalars[4] = i32::MIN; // Special case: abs(i32::MIN) wraps to i32::MIN
    scalars[5] = i32::MIN + 1;
    scalars[6] = 100;
    scalars[7] = -100;

    vectors[0] = vec4i(0, 1, -1, 100);
    vectors[1] = vec4i(i32::MAX, i32::MIN, i32::MIN + 1, -100);
    vectors[2] = vec4i(-1000, -2000, -3000, -4000);
    vectors[3] = vec4i(1000, 2000, 3000, 4000);

    // Fill with diverse values
    for i in 8..N {
        let t = i as i32 * 100;
        scalars[i] = if i % 2 == 0 { t } else { -t };
        vectors[i] = vec4i(t, -t * 2, t * 3, -t * 4);
    }

    (scalars, vectors)
}

/// Generates test inputs for sign(f32).
/// Includes positive, negative, zero, and -0.0.
fn sign_f32_inputs() -> ([f32; N], [wgsl_rs::std::Vec4f; N]) {
    use wgsl_rs::std::vec4f;

    let mut scalars = [0.0f32; N];
    let mut vectors = [vec4f(0.0, 0.0, 0.0, 0.0); N];

    // Edge cases
    scalars[0] = 0.0;
    scalars[1] = -0.0;
    scalars[2] = 1.0;
    scalars[3] = -1.0;
    scalars[4] = 0.5;
    scalars[5] = -0.5;
    scalars[6] = 1000.0;
    scalars[7] = -1000.0;

    vectors[0] = vec4f(0.0, -0.0, 1.0, -1.0);
    vectors[1] = vec4f(0.5, -0.5, 1000.0, -1000.0);
    vectors[2] = vec4f(-100.0, -200.0, -300.0, -400.0);
    vectors[3] = vec4f(100.0, 200.0, 300.0, 400.0);

    // Fill with diverse values - distribute evenly across negative, zero, positive
    for i in 8..N {
        let t = i as f32 * 10.0;
        scalars[i] = match i % 3 {
            0 => -t,
            1 => 0.0,
            _ => t,
        };

        vectors[i] = vec4f(
            if i % 4 == 0 { -t } else { t },
            if i % 4 == 1 { -t } else { t },
            if i % 4 == 2 { 0.0 } else { t },
            if i % 4 == 3 { -t } else { 0.0 },
        );
    }

    (scalars, vectors)
}

// sign_i32_inputs removed - sign() not implemented for i32 in wgsl-rs

/// Generates test inputs for degrees(f32).
/// Common radian values: multiples of π.
fn degrees_inputs() -> ([f32; N], [wgsl_rs::std::Vec4f; N]) {
    use std::f32::consts::PI;
    use wgsl_rs::std::vec4f;

    let mut scalars = [0.0f32; N];
    let mut vectors = [vec4f(0.0, 0.0, 0.0, 0.0); N];

    // Common radian values
    scalars[0] = 0.0;
    scalars[1] = PI / 6.0; // 30 degrees
    scalars[2] = PI / 4.0; // 45 degrees
    scalars[3] = PI / 3.0; // 60 degrees
    scalars[4] = PI / 2.0; // 90 degrees
    scalars[5] = PI; // 180 degrees
    scalars[6] = 2.0 * PI; // 360 degrees
    scalars[7] = -PI; // -180 degrees

    vectors[0] = vec4f(0.0, PI / 6.0, PI / 4.0, PI / 3.0);
    vectors[1] = vec4f(PI / 2.0, PI, 2.0 * PI, -PI);
    vectors[2] = vec4f(-PI / 2.0, -PI / 4.0, -PI / 6.0, 3.0 * PI);
    vectors[3] = vec4f(4.0 * PI, -2.0 * PI, PI / 8.0, -PI / 8.0);

    // Fill with diverse values
    for i in 8..N {
        let t = (i as f32 - 32.0) * PI / 16.0; // Range from -1.5π to +1.5π
        scalars[i] = t;
        vectors[i] = vec4f(t, t * 2.0, -t, t / 2.0);
    }

    (scalars, vectors)
}

/// Generates test inputs for radians(f32).
/// Common degree values.
fn radians_inputs() -> ([f32; N], [wgsl_rs::std::Vec4f; N]) {
    use wgsl_rs::std::vec4f;

    let mut scalars = [0.0f32; N];
    let mut vectors = [vec4f(0.0, 0.0, 0.0, 0.0); N];

    // Common degree values
    scalars[0] = 0.0;
    scalars[1] = 30.0;
    scalars[2] = 45.0;
    scalars[3] = 60.0;
    scalars[4] = 90.0;
    scalars[5] = 180.0;
    scalars[6] = 360.0;
    scalars[7] = -180.0;

    vectors[0] = vec4f(0.0, 30.0, 45.0, 60.0);
    vectors[1] = vec4f(90.0, 180.0, 360.0, -180.0);
    vectors[2] = vec4f(-90.0, -45.0, -30.0, 540.0);
    vectors[3] = vec4f(720.0, -360.0, 22.5, -22.5);

    // Fill with diverse values
    for i in 8..N {
        let t = (i as f32 - 32.0) * 11.25; // Range from -270 to +270 degrees
        scalars[i] = t;
        vectors[i] = vec4f(t, t * 2.0, -t, t / 2.0);
    }

    (scalars, vectors)
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct BasicNumericTest;

impl RoundtripTest for BasicNumericTest {
    fn name(&self) -> &str {
        "basic_numeric"
    }

    fn description(&self) -> &str {
        "Basic numeric functions: abs, sign, degrees, radians"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // --- abs_f32 ---
        {
            let (scalars, vectors) = abs_f32_inputs();
            let mut flattened = [0.0f32; 320];

            // Pack: first 64 scalars, then 64 vec4s
            for i in 0..N {
                flattened[i] = scalars[i];
            }
            for i in 0..N {
                let base = 64 + i * 4;
                flattened[base] = vectors[i].x;
                flattened[base + 1] = vectors[i].y;
                flattened[base + 2] = vectors[i].z;
                flattened[base + 3] = vectors[i].w;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (320 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: abs_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            abs_f32::INPUT.set(flattened);
            abs_f32::OUTPUT.set([0.0f32; 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                abs_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = abs_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..320).map(|i| format!("abs_f32[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "abs_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                0.0, // Exact for abs
            ));
        }

        // --- abs_i32 ---
        {
            let (scalars, vectors) = abs_i32_inputs();
            let mut flattened = [0i32; 320];

            for i in 0..N {
                flattened[i] = scalars[i];
            }
            for i in 0..N {
                let base = 64 + i * 4;
                flattened[base] = vectors[i].x;
                flattened[base + 1] = vectors[i].y;
                flattened[base + 2] = vectors[i].z;
                flattened[base + 3] = vectors[i].w;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (320 * std::mem::size_of::<i32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: abs_i32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_i32s: &[i32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            abs_i32::INPUT.set(flattened);
            abs_i32::OUTPUT.set([0i32; 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                abs_i32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = abs_i32::OUTPUT.get();
            let cpu_i32s: Vec<i32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..320).map(|i| format!("abs_i32[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // Convert to u32 for bit-exact comparison
            let gpu_u32s: Vec<u32> = gpu_i32s.iter().map(|&x| x as u32).collect();
            let cpu_u32s: Vec<u32> = cpu_i32s.iter().map(|&x| x as u32).collect();

            results.push(harness::compare_u32_results(
                "abs_i32",
                &gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- sign_f32 ---
        {
            let (scalars, vectors) = sign_f32_inputs();
            let mut flattened = [0.0f32; 320];

            for i in 0..N {
                flattened[i] = scalars[i];
            }
            for i in 0..N {
                let base = 64 + i * 4;
                flattened[base] = vectors[i].x;
                flattened[base + 1] = vectors[i].y;
                flattened[base + 2] = vectors[i].z;
                flattened[base + 3] = vectors[i].w;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (320 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: sign_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            sign_f32::INPUT.set(flattened);
            sign_f32::OUTPUT.set([0.0f32; 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                sign_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = sign_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..320).map(|i| format!("sign_f32[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "sign_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                0.0, // Exact for sign
            ));
        }

        // sign_i32 test removed - sign() not implemented for i32 in wgsl-rs

        // --- degrees_f32 ---
        {
            let (scalars, vectors) = degrees_inputs();
            let mut flattened = [0.0f32; 320];

            for i in 0..N {
                flattened[i] = scalars[i];
            }
            for i in 0..N {
                let base = 64 + i * 4;
                flattened[base] = vectors[i].x;
                flattened[base + 1] = vectors[i].y;
                flattened[base + 2] = vectors[i].z;
                flattened[base + 3] = vectors[i].w;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (320 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: degrees_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            degrees_f32::INPUT.set(flattened);
            degrees_f32::OUTPUT.set([0.0f32; 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                degrees_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = degrees_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..320).map(|i| format!("degrees_f32[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "degrees_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                1e-4, // Tolerance for float conversion (larger values accumulate more error)
            ));
        }

        // --- radians_f32 ---
        {
            let (scalars, vectors) = radians_inputs();
            let mut flattened = [0.0f32; 320];

            for i in 0..N {
                flattened[i] = scalars[i];
            }
            for i in 0..N {
                let base = 64 + i * 4;
                flattened[base] = vectors[i].x;
                flattened[base + 1] = vectors[i].y;
                flattened[base + 2] = vectors[i].z;
                flattened[base + 3] = vectors[i].w;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (320 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: radians_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            radians_f32::INPUT.set(flattened);
            radians_f32::OUTPUT.set([0.0f32; 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                radians_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = radians_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..320).map(|i| format!("radians_f32[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "radians_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                1e-5, // Small tolerance for float conversion
            ));
        }

        results
    }
}
