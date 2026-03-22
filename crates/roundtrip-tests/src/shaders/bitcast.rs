//! Roundtrip tests for bitcast builtin functions.
//!
//! Tests: `bitcast_f32`, `bitcast_u32`, `bitcast_i32`, `bitcast_vec4f`,
//! `bitcast_vec4u`, `bitcast_vec4i`.
//!
//! Bitcast reinterprets bits without changing them, so all comparisons are
//! exact. We verify roundtrip identity: `bitcast_u32(bitcast_f32(x)) == x`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Scalar bitcast roundtrip: u32 -> f32 -> u32 and u32 -> i32 -> u32.
///
/// For each input u32 `x`:
/// - `bitcast_u32(bitcast_f32(x))` should equal `x`
/// - `bitcast_u32(bitcast_i32(x))` should equal `x` (reinterpret as i32 then
///   back)
/// - Also outputs the raw `bitcast_f32(x)` bits for inspection
#[wgsl]
pub mod bitcast_scalar_roundtrip {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // u32 -> f32 -> u32 roundtrip
        let as_f32 = bitcast_f32(x);
        let rt_f32 = bitcast_u32(as_f32);
        // u32 -> i32 -> u32 roundtrip
        let as_i32 = bitcast_i32(x);
        let rt_i32 = bitcast_u32(bitcast_f32(as_i32));
        // i32 -> f32 -> i32 roundtrip (via u32 output)
        let as_f32_from_i32 = bitcast_f32(as_i32);
        let rt_i32_via_f32 = bitcast_u32(as_f32_from_i32);
        get_mut!(OUTPUT)[idx] = vec4u(rt_f32, rt_i32, rt_i32_via_f32, 0);
    }
}

/// Vec4 bitcast roundtrip: Vec4u -> Vec4f -> Vec4u.
#[wgsl]
pub mod bitcast_vec4_roundtrip {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [Vec4u; 16]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 16]);

    #[compute]
    #[workgroup_size(16)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        // Vec4u -> Vec4f -> Vec4u roundtrip
        let as_f = bitcast_vec4f(x);
        let back = bitcast_vec4u(as_f);
        get_mut!(OUTPUT)[idx] = back;
    }
}

/// Generates test input values for bitcast functions.
///
/// Returns 64 u32 values that represent valid (non-NaN) f32 bit patterns,
/// since NaN bits may be canonicalized by the GPU.
fn bitcast_inputs() -> [u32; N] {
    let mut values = [0u32; N];
    // Known f32 bit patterns.
    values[0] = 0x00000000; // +0.0
    values[1] = 0x80000000; // -0.0
    values[2] = 0x3F800000; // 1.0
    values[3] = 0xBF800000; // -1.0
    values[4] = 0x40000000; // 2.0
    values[5] = 0x40490FDB; // pi (~3.14159)
    values[6] = 0x42C80000; // 100.0
    values[7] = 0x7F800000; // +inf
    values[8] = 0xFF800000; // -inf
    values[9] = 0x00000001; // smallest denormal
    values[10] = 0x00800000; // smallest normal
    values[11] = 0x7F7FFFFF; // largest finite f32
    // Simple integer bit patterns (represent denormals or small floats).
    for (i, value) in values.iter_mut().enumerate().take(32).skip(12) {
        *value = (i as u32) * 0x01010101;
    }
    // Larger patterns avoiding NaN range (0x7F800001..0x7FFFFFFF and
    // 0xFF800001..0xFFFFFFFF).
    for (i, value) in values.iter_mut().enumerate().skip(32) {
        let raw = (i as u32).wrapping_mul(0x07654321);
        // Mask off the exponent to avoid NaN: if exponent bits are all 1 and
        // mantissa is nonzero, clear a mantissa bit.
        let exp = raw & 0x7F800000;
        *value = if exp == 0x7F800000 && (raw & 0x007FFFFF) != 0 {
            raw & 0xFF800000 // make it infinity instead of NaN
        } else {
            raw
        };
    }
    values
}

/// The bitcast roundtrip test.
pub struct BitcastTest;

impl RoundtripTest for BitcastTest {
    fn name(&self) -> &str {
        "bitcast"
    }

    fn description(&self) -> &str {
        "bitcast_f32, bitcast_u32, bitcast_i32, bitcast_vec4f, bitcast_vec4u"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = bitcast_inputs();
        let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

        let mut results = Vec::new();

        // --- bitcast_scalar_roundtrip ---
        {
            let output_size = (N * 4 * std::mem::size_of::<u32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: bitcast_scalar_roundtrip::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bitcast_scalar_roundtrip::INPUT.set(inputs);
            bitcast_scalar_roundtrip::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bitcast_scalar_roundtrip::main(builtins.global_invocation_id);
            });
            let cpu_output = bitcast_scalar_roundtrip::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("u32->f32->u32(0x{x:08X})"),
                        format!("u32->i32->f32->u32(0x{x:08X})"),
                        format!("i32->f32->u32(0x{x:08X})"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bitcast_scalar_roundtrip",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bitcast_vec4_roundtrip ---
        {
            const VEC4_COUNT: usize = 16;
            let output_size = (VEC4_COUNT * 4 * std::mem::size_of::<u32>()) as u64;
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: bitcast_vec4_roundtrip::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            // Reinterpret the first 16 groups of 4 u32s as Vec4u.
            let vec4_inputs: [Vec4u; VEC4_COUNT] = std::array::from_fn(|i| {
                vec4u(
                    inputs[i * 4],
                    inputs[i * 4 + 1],
                    inputs[i * 4 + 2],
                    inputs[i * 4 + 3],
                )
            });
            bitcast_vec4_roundtrip::INPUT.set(vec4_inputs);
            bitcast_vec4_roundtrip::OUTPUT.set([Vec4u::default(); VEC4_COUNT]);
            dispatch_workgroups((1, 1, 1), (VEC4_COUNT as u32, 1, 1), |builtins| {
                bitcast_vec4_roundtrip::main(builtins.global_invocation_id);
            });
            let cpu_output = bitcast_vec4_roundtrip::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..VEC4_COUNT)
                .flat_map(|i| {
                    vec![
                        format!("[{i}] vec4u->vec4f->vec4u .x"),
                        format!("[{i}] vec4u->vec4f->vec4u .y"),
                        format!("[{i}] vec4u->vec4f->vec4u .z"),
                        format!("[{i}] vec4u->vec4f->vec4u .w"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bitcast_vec4_roundtrip",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        results
    }
}
