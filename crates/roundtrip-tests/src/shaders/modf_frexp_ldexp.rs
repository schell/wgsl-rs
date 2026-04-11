//! Roundtrip tests for modf, frexp, and ldexp builtin functions.
//!
//! Tests: `modf`, `frexp`, `ldexp`.
//!
//! Note: `modf` and `frexp` return structs. We flatten their fields into
//! f32/i32 arrays for storage buffer compatibility.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// modf_basic: Tests modf(x) -> ModfResult{fract, whole}
///
/// We flatten ModfResult<f32> into two consecutive f32 values per input.
/// OUTPUT layout: [fract_0, whole_0, fract_1, whole_1, ...]
#[wgsl]
pub mod modf_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 128]); // 2 per input

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let result = modf(x);
        let out_base = idx * 2;
        get_mut!(OUTPUT)[out_base] = result.fract;
        get_mut!(OUTPUT)[out_base + 1] = result.whole;
    }
}

/// frexp_basic: Tests frexp(x) -> FrexpResult{fract: f32, exp: i32}
///
/// We flatten FrexpResult<f32, i32> into two consecutive f32 values:
/// the fract is stored as-is, and exp is stored as f32 for buffer
/// compatibility. OUTPUT layout: [fract_0, exp_0_as_f32, fract_1, exp_1_as_f32,
/// ...]
#[wgsl]
pub mod frexp_basic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 128]); // 2 per input

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let result = frexp(x);
        let out_base = idx * 2;
        get_mut!(OUTPUT)[out_base] = result.fract;
        get_mut!(OUTPUT)[out_base + 1] = f32(result.exp);
    }
}

/// frexp_ldexp_roundtrip: Tests frexp -> ldexp roundtrip
///
/// INPUT: raw f32 values
/// OUTPUT: ldexp(frexp(x).fract, frexp(x).exp) for each input
/// This tests that ldexp inverts frexp correctly.
#[wgsl]
pub mod frexp_ldexp_roundtrip {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let frexp_result = frexp(x);
        get_mut!(OUTPUT)[idx] = ldexp(frexp_result.fract, frexp_result.exp);
    }
}

/// Generates test inputs for modf/frexp/ldexp.
///
/// Uses diverse edge cases and normal values:
/// - Zero
/// - Negative values
/// - Small/subnormal values (for frexp testing)
/// - Large values
/// - Positive and negative across the range
fn modf_frexp_ldexp_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    let cases = [
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        0.5f32,
        -0.5f32,
        2.5f32,
        -2.5f32,
        10.0f32,
        -10.0f32,
        0.001f32,
        -0.001f32,
        1e-10f32, // very small
        -1e-10f32,
        1e10f32, // very large
        -1e10f32,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    // Fill with cases, then interpolate to fill remaining slots
    for (i, &val) in cases.iter().enumerate() {
        if i < N {
            values[i] = val;
        }
    }
    // Fill remaining with interpolated values in [-100, 100]
    for i in cases.len()..N {
        let t = (i - cases.len()) as f32 / ((N - cases.len()) as f32);
        values[i] = t * 200.0 - 100.0;
    }
    values
}

/// For ldexp roundtrip: frexp(x) -> ldexp(fract, exp) should equal x (for
/// finite normal x)
fn frexp_ldexp_roundtrip_inputs() -> [f32; N] {
    let mut values = [0.0f32; N];
    // Use only finite, non-zero normal values for ldexp roundtrip testing
    for i in 0..N {
        let t = (i as f32) / (N as f32);
        // Generate values in [-100, 100] avoiding zero and very small values
        if i == 0 {
            values[i] = 0.5f32;
        } else if i == 1 {
            values[i] = -0.5f32;
        } else if i < N / 2 {
            values[i] = 0.1 + t * 50.0;
        } else {
            values[i] = -(0.1 + (t - 0.5) * 100.0);
        }
    }
    values
}

/// The modf/frexp/ldexp roundtrip test.
pub struct ModfFrexpLdexpTest;

impl RoundtripTest for ModfFrexpLdexpTest {
    fn name(&self) -> &str {
        "modf_frexp_ldexp"
    }

    fn description(&self) -> &str {
        "modf, frexp, ldexp"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // --- modf_basic ---
        {
            let inputs = modf_frexp_ldexp_inputs();
            let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
            let output_size = (N * 2 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: modf_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            modf_basic::INPUT.set(inputs);
            modf_basic::OUTPUT.set([0.0f32; 128]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                modf_basic::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = modf_basic::OUTPUT.get();
            let cpu_output: Vec<f32> = cpu_output_guard.to_vec();

            let mut labels: Vec<String> = Vec::new();
            for i in 0..N {
                let x = inputs[i];
                labels.push(format!("modf({x:.4}).fract"));
                labels.push(format!("modf({x:.4}).whole"));
            }
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // modf should be exact (epsilon = 0.0)
            results.push(harness::compare_f32_results(
                "modf_basic",
                gpu_floats,
                &cpu_output,
                &label_refs,
                0.0,
            ));
        }

        // --- frexp_basic ---
        {
            let inputs = modf_frexp_ldexp_inputs();
            let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
            let output_size = (N * 2 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: frexp_basic::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            frexp_basic::INPUT.set(inputs);
            frexp_basic::OUTPUT.set([0.0f32; 128]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                frexp_basic::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = frexp_basic::OUTPUT.get();
            let cpu_output: Vec<f32> = cpu_output_guard.to_vec();

            let mut labels: Vec<String> = Vec::new();
            for i in 0..N {
                let x = inputs[i];
                labels.push(format!("frexp({x:.4}).fract"));
                labels.push(format!("frexp({x:.4}).exp"));
            }
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // frexp should be exact (epsilon = 0.0)
            results.push(harness::compare_f32_results(
                "frexp_basic",
                gpu_floats,
                &cpu_output,
                &label_refs,
                0.0,
            ));
        }

        // --- frexp_ldexp_roundtrip ---
        {
            let inputs = frexp_ldexp_roundtrip_inputs();
            let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
            let output_size = (N * std::mem::size_of::<f32>()) as u64;

            // GPU: frexp -> ldexp roundtrip
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: frexp_ldexp_roundtrip::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            // CPU: frexp -> ldexp roundtrip
            use wgsl_rs::std::*;
            frexp_ldexp_roundtrip::INPUT.set(inputs);
            frexp_ldexp_roundtrip::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                frexp_ldexp_roundtrip::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = frexp_ldexp_roundtrip::OUTPUT.get();
            let cpu_output: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N)
                .map(|i| {
                    let x = inputs[i];
                    format!("ldexp(frexp({x:.4}).fract, frexp({x:.4}).exp)")
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // ldexp roundtrip: allow 1e-6 epsilon for accumulated floating-point error
            results.push(harness::compare_f32_results(
                "frexp_ldexp_roundtrip",
                gpu_floats,
                &cpu_output,
                &label_refs,
                1e-6,
            ));
        }

        results
    }
}
