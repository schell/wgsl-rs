//! Roundtrip tests for pack/unpack builtin functions.
//!
//! Tests: `pack4x8snorm`, `pack4x8unorm`, `pack2x16snorm`, `pack2x16unorm`,
//! `pack2x16float`, `unpack4x8snorm`, `unpack4x8unorm`, `unpack2x16snorm`,
//! `unpack2x16unorm`, `unpack2x16float`.
//!
//! Pack/unpack functions involve quantization to lower precision (8-bit or
//! 16-bit), so roundtrip tests compare within the expected precision loss.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// 4x8 snorm/unorm pack-then-unpack roundtrip.
///
/// Packs a Vec4f, then unpacks back to Vec4f. The result should match the
/// input (clamped to the valid range) within quantization precision.
#[wgsl]
pub mod pack_unpack_4x8 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 64]);
    // Output layout: [snorm_unpacked, unorm_unpacked] alternating = 128 Vec4f
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 128]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let v = get!(INPUT)[idx];

        // snorm: pack then unpack
        let packed_snorm = pack4x8snorm(v);
        let unpacked_snorm = unpack4x8snorm(packed_snorm);

        // unorm: clamp to [0,1] first, then pack then unpack
        let clamped = vec4f(
            clamp(v.x, 0.0, 1.0),
            clamp(v.y, 0.0, 1.0),
            clamp(v.z, 0.0, 1.0),
            clamp(v.w, 0.0, 1.0),
        );
        let packed_unorm = pack4x8unorm(clamped);
        let unpacked_unorm = unpack4x8unorm(packed_unorm);

        get_mut!(OUTPUT)[idx * 2] = unpacked_snorm;
        get_mut!(OUTPUT)[idx * 2 + 1] = unpacked_unorm;
    }
}

/// 2x16 snorm/unorm pack-then-unpack roundtrip.
#[wgsl]
pub mod pack_unpack_2x16_norm {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let v = get!(INPUT)[idx];

        // snorm using xy
        let packed_snorm = pack2x16snorm(vec2f(v.x, v.y));
        let unpacked_snorm = unpack2x16snorm(packed_snorm);

        // unorm using zw (clamped to [0,1])
        let packed_unorm = pack2x16unorm(vec2f(clamp(v.z, 0.0, 1.0), clamp(v.w, 0.0, 1.0)));
        let unpacked_unorm = unpack2x16unorm(packed_unorm);

        get_mut!(OUTPUT)[idx] = vec4f(
            unpacked_snorm.x,
            unpacked_snorm.y,
            unpacked_unorm.x,
            unpacked_unorm.y,
        );
    }
}

/// 2x16 float pack-then-unpack roundtrip.
#[wgsl]
pub mod pack_unpack_2x16_float {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let v = get!(INPUT)[idx];

        let packed = pack2x16float(vec2f(v.x, v.y));
        let unpacked = unpack2x16float(packed);

        get_mut!(OUTPUT)[idx] = vec4f(unpacked.x, unpacked.y, 0.0, 0.0);
    }
}

/// Raw packed u32 comparison — verifies the packed integer representations
/// match between GPU and CPU.
#[wgsl]
pub mod pack_raw_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [Vec4f; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let v = get!(INPUT)[idx];
        let p_snorm4 = pack4x8snorm(v);
        let p_unorm4 = pack4x8unorm(v);
        let p_snorm2 = pack2x16snorm(vec2f(v.x, v.y));
        let p_unorm2 = pack2x16unorm(vec2f(v.z, v.w));
        get_mut!(OUTPUT)[idx] = vec4u(p_snorm4, p_unorm4, p_snorm2, p_unorm2);
    }
}

/// Generates test input values for packing functions.
///
/// Returns 64 Vec4f values in `[-1.0, 1.0]` with a good spread of values
/// including the exact endpoints, zero, and many intermediate values.
fn packing_inputs() -> [wgsl_rs::std::Vec4f; N] {
    use wgsl_rs::std::vec4f;
    let mut values = [vec4f(0.0, 0.0, 0.0, 0.0); N];
    // Special values.
    values[0] = vec4f(0.0, 0.0, 0.0, 0.0);
    values[1] = vec4f(1.0, 1.0, 1.0, 1.0);
    values[2] = vec4f(-1.0, -1.0, -1.0, -1.0);
    values[3] = vec4f(0.5, -0.5, 0.25, -0.25);
    values[4] = vec4f(0.0, 1.0, 0.0, 1.0);
    values[5] = vec4f(-1.0, 0.0, 1.0, 0.0);
    // Sweep through the range.
    for (i, value) in values.iter_mut().enumerate().skip(6) {
        let t = i as f32 / (N - 1) as f32;
        let a = t * 2.0 - 1.0; // [-1, 1]
        let b = 1.0 - t * 2.0; // [1, -1]
        let c = t; // [0, 1]
        let d = 1.0 - t; // [1, 0]
        *value = vec4f(a, b, c, d);
    }
    values
}

/// The packing roundtrip test.
pub struct PackingTest;

impl RoundtripTest for PackingTest {
    fn name(&self) -> &str {
        "packing"
    }

    fn description(&self) -> &str {
        "pack4x8snorm, pack4x8unorm, pack2x16snorm, pack2x16unorm, pack2x16float, unpack4x8snorm, \
         unpack4x8unorm, unpack2x16snorm, unpack2x16unorm, unpack2x16float"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = packing_inputs();
        // Flatten Vec4f to f32 for bytemuck (Vec4f is repr(C) [f32;4]).
        let input_floats: Vec<f32> = inputs.iter().flat_map(|v| v.to_array()).collect();
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&input_floats);

        let mut results = Vec::new();

        // --- pack_unpack_4x8: snorm and unorm roundtrip ---
        {
            let output_size = (N * 2 * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: pack_unpack_4x8::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            pack_unpack_4x8::INPUT.set(inputs);
            pack_unpack_4x8::OUTPUT.set([Vec4f::default(); N * 2]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                pack_unpack_4x8::main(builtins.global_invocation_id);
            });
            let cpu_output = pack_unpack_4x8::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] snorm4x8.x"),
                        format!("[{i}] snorm4x8.y"),
                        format!("[{i}] snorm4x8.z"),
                        format!("[{i}] snorm4x8.w"),
                        format!("[{i}] unorm4x8.x"),
                        format!("[{i}] unorm4x8.y"),
                        format!("[{i}] unorm4x8.z"),
                        format!("[{i}] unorm4x8.w"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            // 8-bit precision: ~1/127 for snorm, ~1/255 for unorm.
            // Use the larger tolerance.
            let epsilon = 1.0 / 127.0 + 1e-6;
            results.push(harness::compare_f32_results(
                "pack_unpack_4x8",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- pack_unpack_2x16_norm: snorm and unorm 16-bit roundtrip ---
        {
            let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: pack_unpack_2x16_norm::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            pack_unpack_2x16_norm::INPUT.set(inputs);
            pack_unpack_2x16_norm::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                pack_unpack_2x16_norm::main(builtins.global_invocation_id);
            });
            let cpu_output = pack_unpack_2x16_norm::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] snorm2x16.x"),
                        format!("[{i}] snorm2x16.y"),
                        format!("[{i}] unorm2x16.x"),
                        format!("[{i}] unorm2x16.y"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            // 16-bit precision: ~1/32767 for snorm.
            let epsilon = 1.0 / 32767.0 + 1e-6;
            results.push(harness::compare_f32_results(
                "pack_unpack_2x16_norm",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- pack_unpack_2x16_float: f16 roundtrip ---
        {
            let output_size = (N * 4 * std::mem::size_of::<f32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: pack_unpack_2x16_float::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            pack_unpack_2x16_float::INPUT.set(inputs);
            pack_unpack_2x16_float::OUTPUT.set([Vec4f::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                pack_unpack_2x16_float::main(builtins.global_invocation_id);
            });
            let cpu_output = pack_unpack_2x16_float::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] f16_rt.x"),
                        format!("[{i}] f16_rt.y"),
                        format!("[{i}] (padding)"),
                        format!("[{i}] (padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            // f16 has ~3 decimal digits of precision near 1.0.
            let epsilon = 1e-3;
            results.push(harness::compare_f32_results(
                "pack_unpack_2x16_float",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                epsilon,
            ));
        }

        // --- pack_raw_u32: compare raw packed integers ---
        {
            let output_size = (N * 4 * std::mem::size_of::<u32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: pack_raw_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            pack_raw_u32::INPUT.set(inputs);
            pack_raw_u32::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                pack_raw_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = pack_raw_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] pack4x8snorm"),
                        format!("[{i}] pack4x8unorm"),
                        format!("[{i}] pack2x16snorm"),
                        format!("[{i}] pack2x16unorm"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            // Allow ±1 LSB per packed byte — the WGSL spec permits
            // implementation-defined rounding in pack functions.
            results.push(harness::compare_packed_u32_results(
                "pack_raw_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
                1,
            ));
        }

        results
    }
}
