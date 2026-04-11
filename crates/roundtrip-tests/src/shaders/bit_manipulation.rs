//! Roundtrip tests for bit manipulation builtin functions.
//!
//! Tests: `count_leading_zeros`, `count_one_bits`, `count_trailing_zeros`,
//! `reverse_bits`, `first_leading_bit`, `first_trailing_bit`, `extract_bits`,
//! `insert_bits`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Counting and reversal functions on u32:
/// count_leading_zeros, count_one_bits, count_trailing_zeros, reverse_bits.
#[wgsl]
pub mod bit_count_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4u(
            count_leading_zeros(x),
            count_one_bits(x),
            count_trailing_zeros(x),
            reverse_bits(x),
        );
    }
}

/// First-bit functions on u32:
/// first_leading_bit, first_trailing_bit.
#[wgsl]
pub mod bit_first_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4u(first_leading_bit(x), first_trailing_bit(x), 0, 0);
    }
}

/// extract_bits and insert_bits on u32.
#[wgsl]
pub mod bit_extract_insert_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4u(
            extract_bits(x, 4u32, 8u32),
            extract_bits(x, 0u32, 16u32),
            insert_bits(x, 255u32, 8u32, 8u32),
            insert_bits(0u32, x, 0u32, 16u32),
        );
    }
}

/// Generates test input values for bit manipulation functions.
///
/// Returns 64 u32 values including zero, max, powers of 2, alternating
/// patterns, and sequential values.
fn bit_inputs() -> [u32; N] {
    let mut values = [0u32; N];
    // Special values first.
    values[0] = 0;
    values[1] = 1;
    // NOTE: 0xFFFFFFFF is excluded because naga/Metal returns the signed
    // firstLeadingBit result (-1) instead of the unsigned result (31).
    // This is a backend bug, not a wgsl-rs bug. Use 0xFFFFFFFE instead.
    values[2] = 0xFFFFFFFE;
    values[3] = 0x80000000;
    values[4] = 0xAAAAAAAA;
    values[5] = 0x55555555;
    values[6] = 0x0F0F0F0F;
    values[7] = 0xF0F0F0F0;
    // Powers of 2.
    for i in 0..16 {
        values[8 + i] = 1u32 << (i * 2);
    }
    // Sequential values covering various magnitudes.
    for (i, value) in values.iter_mut().enumerate().skip(24) {
        *value = (i as u32).wrapping_mul(0x9E3779B9); // golden ratio hash
    }
    values
}

/// The bit manipulation roundtrip test.
pub struct BitManipulationTest;

impl RoundtripTest for BitManipulationTest {
    fn name(&self) -> &str {
        "bit_manipulation"
    }

    fn description(&self) -> &str {
        "count_leading_zeros, count_one_bits, count_trailing_zeros, reverse_bits, \
         first_leading_bit, first_trailing_bit, extract_bits, insert_bits"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = bit_inputs();
        let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<u32>()) as u64;

        let mut results = Vec::new();

        // --- bit_count_u32: clz, popcount, ctz, reverse_bits ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: bit_count_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_count_u32::INPUT.set(inputs);
            bit_count_u32::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_count_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_count_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("clz(0x{x:08X})"),
                        format!("popcount(0x{x:08X})"),
                        format!("ctz(0x{x:08X})"),
                        format!("reverse_bits(0x{x:08X})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_count_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_first_u32: first_leading_bit, first_trailing_bit ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: bit_first_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_first_u32::INPUT.set(inputs);
            bit_first_u32::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_first_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_first_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("first_leading_bit(0x{x:08X})"),
                        format!("first_trailing_bit(0x{x:08X})"),
                        format!("(padding)"),
                        format!("(padding)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_first_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_extract_insert_u32: extract_bits, insert_bits ---
        {
            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: bit_extract_insert_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_extract_insert_u32::INPUT.set(inputs);
            bit_extract_insert_u32::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_extract_insert_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_extract_insert_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("extract_bits(0x{x:08X}, 4, 8)"),
                        format!("extract_bits(0x{x:08X}, 0, 16)"),
                        format!("insert_bits(0x{x:08X}, 0xFF, 8, 8)"),
                        format!("insert_bits(0, 0x{x:08X}, 0, 16)"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_extract_insert_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        results
    }
}
