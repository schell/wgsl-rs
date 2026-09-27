//! Roundtrip tests for bit manipulation builtin functions.
//!
//! Tests: `count_leading_zeros`, `count_one_bits`, `count_trailing_zeros`,
//! `reverse_bits`, `first_leading_bit`, `first_trailing_bit`, `extract_bits`,
//! `insert_bits`, plus vector bitwise operators `& | ^` and `&= |= ^=`
//! (wgsl-rs#186).

use wgsl_rs::{
    std::{Vec4, Vec4i, Vec4u, vec4i, vec4u},
    wgsl,
};

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

/// Mixed-precedence bitwise/shift comparisons on u32 (wgsl-rs#159): Rust
/// binds `& ^ | << >>` tighter than the comparison operators, WGSL the
/// reverse. Each expression parses as `(x OP k) CMP k` in Rust; a flat
/// WGSL rendering would re-parse as `x OP (k CMP k)` — a type error.
/// The renderer parenthesizes so the two worlds agree.
#[wgsl]
pub mod bit_precedence_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4u(
            select(0u32, 1u32, x & 3 == 3),
            select(0u32, 1u32, x | 1 == 1),
            select(0u32, 1u32, x ^ 2 != 2),
            select(0u32, 1u32, x << 1u32 < 8),
        );
    }
}

/// Vector bitwise binary ops (wgsl-rs#186): componentwise `&`, `|`, `^` on
/// Vec4u, a mixed-precedence chain, and a Vec2u op via swizzles.
///
/// The chain `(a & b) | (a ^ b)` exercises the #159 parenthesization with
/// vector operands: a flat rendering would re-bind under WGSL's precedence
/// rules, the parenthesized one preserves the Rust tree exactly.
#[wgsl]
pub mod bit_vector_binary_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 512]); // 64 vec4u pairs
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 256]); // 4 results * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        // Mixed-precedence chain on vector operands (wgsl-rs#159 shape).
        let mixed = (a & b) | (a ^ b);
        // Vec2u binary op via swizzles.
        let and2 = a.xy() & b.yx();

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base] = a & b;
        get_mut!(OUTPUT)[out_base + 1] = a | b;
        get_mut!(OUTPUT)[out_base + 2] = a ^ b;
        get_mut!(OUTPUT)[out_base + 3] = vec4u(mixed.x(), mixed.y(), and2.x(), and2.y());
    }
}

/// Vector bitwise ops via swizzles plus mixed-precedence vector comparisons
/// (wgsl-rs#186 composing with #159 and #164): Vec3u `|`/`^` via swizzles,
/// and `a & b == b` on Vec4u and Vec2u — which parses as `(a & b) == b` in
/// Rust (bitwise binds tighter than comparison) and renders as
/// `all((a & b) == b)` so WGSL's vecN<bool> result reduces to the bool
/// Rust's PartialEq produces.
#[wgsl]
pub mod bit_vector_swizzle_cmp_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 512]); // 64 vec4u pairs
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 256]); // 4 results * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        // Vec3u ops via swizzles.
        let or3 = a.xyz() | b.zyx();
        let xor3 = a.xyz() ^ b.zyx();
        // Mixed-precedence comparisons on vector operands (the wgsl-rs#159
        // repro shape, riding the wgsl-rs#164 vector `==` lowering).
        let cmp4 = select(0u32, 1u32, a & b == b);
        let cmp2 = select(0u32, 1u32, a.xy() & b.xy() == b.yx());

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base] = vec4u(or3.x(), or3.y(), or3.z(), 0u32);
        get_mut!(OUTPUT)[out_base + 1] = vec4u(xor3.x(), xor3.y(), xor3.z(), 0u32);
        get_mut!(OUTPUT)[out_base + 2] = vec4u(cmp4, 0u32, 0u32, 0u32);
        get_mut!(OUTPUT)[out_base + 3] = vec4u(cmp2, 0u32, 0u32, 0u32);
    }
}

/// Vector bitwise compound assignment (wgsl-rs#186): `&=`, `|=`, `^=` on
/// Vec4u plus `^=` on a swizzled Vec2u. The lowering must emit the WGSL
/// compound operators rather than expanded `x = x op y` forms.
#[wgsl]
pub mod bit_vector_assign_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 512]); // 64 vec4u pairs
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 256]); // 4 results * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        let mut c = a;
        c &= b;
        let mut d = a;
        d |= b;
        let mut e = a;
        e ^= b;
        // Compound assignment on a swizzled Vec2u.
        let mut f = a.xy();
        f ^= b.yx();

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base] = c;
        get_mut!(OUTPUT)[out_base + 1] = d;
        get_mut!(OUTPUT)[out_base + 2] = e;
        get_mut!(OUTPUT)[out_base + 3] = vec4u(f.x(), f.y(), 0u32, 0u32);
    }
}

/// Signed vector bitwise ops (wgsl-rs#186): `&`, `|`, `^` on Vec4i, `&=` on
/// Vec4i, and a Vec3i op via swizzles.
#[wgsl]
pub mod bit_vector_ops_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [i32; 512]); // 64 vec4i pairs
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4i; 256]); // 4 results * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4i(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4i(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        let mut c = a;
        c &= b;
        // Vec3i op via swizzles.
        let or3 = a.xyz() | b.zyx();

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base] = a & b;
        get_mut!(OUTPUT)[out_base + 1] = a | b;
        get_mut!(OUTPUT)[out_base + 2] = a ^ b;
        get_mut!(OUTPUT)[out_base + 3] = vec4i(c.x(), or3.y(), 0, 0);
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

/// Builds the u32 vector operand pair for the vector bitwise roundtrip
/// modules, derived from the scalar bit inputs so the per-lane patterns
/// vary.
fn bit_vector_inputs_u32() -> ([Vec4u; N], [Vec4u; N]) {
    let inputs = bit_inputs();
    let mut a = [Vec4u::default(); N];
    let mut b = [Vec4u::default(); N];
    for i in 0..N {
        let x = inputs[i];
        a[i] = vec4u(x, x ^ 0x55555555, x.rotate_left(8), x.wrapping_mul(3));
        b[i] = vec4u(
            inputs[(i * 7) % N],
            0x0F0F0F0F,
            x.rotate_right(4),
            x ^ 0xFFFF0000,
        );
    }
    (a, b)
}

/// The same operand bit patterns as [`bit_vector_inputs_u32`], reinterpreted
/// as i32. The bitwise results are bit-identical either way, but the CPU
/// side exercises the signed trait impls.
fn bit_vector_inputs_i32() -> ([Vec4i; N], [Vec4i; N]) {
    let (a, b) = bit_vector_inputs_u32();
    let cast = |v: Vec4u| vec4i(v.x() as i32, v.y() as i32, v.z() as i32, v.w() as i32);
    (a.map(cast), b.map(cast))
}

/// Flattens two operand arrays into the interleaved INPUT layout used by
/// the vector bitwise modules: per invocation, A's 4 components then B's.
fn flatten_bit_vector_pairs<T: Copy + Default>(a: &[Vec4<T>; N], b: &[Vec4<T>; N]) -> [T; N * 8] {
    let mut flat = [T::default(); N * 8];
    for i in 0..N {
        flat[i * 8..i * 8 + 4].copy_from_slice(&a[i].to_array());
        flat[i * 8 + 4..i * 8 + 8].copy_from_slice(&b[i].to_array());
    }
    flat
}

/// The bit manipulation roundtrip test.
pub struct BitManipulationTest;

impl RoundtripTest for BitManipulationTest {
    fn name(&self) -> &str {
        "bit_manipulation"
    }

    fn description(&self) -> &str {
        "count_leading_zeros, count_one_bits, count_trailing_zeros, reverse_bits, \
         first_leading_bit, first_trailing_bit, extract_bits, insert_bits, mixed-precedence \
         bitwise comparisons (wgsl-rs#159), vector bitwise ops & | ^ and &= |= ^= (wgsl-rs#186)"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = bit_inputs();
        let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<u32>()) as u64;

        let mut results = Vec::new();

        // --- bit_count_u32: clz, popcount, ctz, reverse_bits ---
        {
            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_count_u32::WGSL_SOURCE).unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
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
            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_first_u32::WGSL_SOURCE).unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
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
                        "(padding)".into(),
                        "(padding)".into(),
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
            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_extract_insert_u32::WGSL_SOURCE)
                    .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
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

        // --- bit_precedence_u32: mixed-precedence comparisons (wgsl-rs#159)
        // ---
        {
            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_precedence_u32::WGSL_SOURCE)
                    .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_precedence_u32::INPUT.set(inputs);
            bit_precedence_u32::OUTPUT.set([Vec4u::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_precedence_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_precedence_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("(0x{x:08X} & 3) == 3"),
                        format!("(0x{x:08X} | 1) == 1"),
                        format!("(0x{x:08X} ^ 2) != 2"),
                        format!("(0x{x:08X} << 1) < 8"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_precedence_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_vector_binary_u32: vector & | ^ (wgsl-rs#186) ---
        {
            let (a_in, b_in) = bit_vector_inputs_u32();
            let flattened = flatten_bit_vector_pairs(&a_in, &b_in);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<u32>()) as u64;

            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_vector_binary_u32::WGSL_SOURCE)
                    .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_vector_binary_u32::INPUT.set(flattened);
            bit_vector_binary_u32::OUTPUT.set([Vec4u::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_vector_binary_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_vector_binary_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            const OPS: [&str; 4] = ["and", "or", "xor", "mixed+and2"];
            let labels: Vec<String> = (0..N)
                .flat_map(|idx| {
                    (0..4).flat_map(move |op| {
                        (0..4).map(move |lane| format!("{} inv{} l{}", OPS[op], idx, lane))
                    })
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_vector_binary_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_vector_swizzle_cmp_u32: swizzles + vector comparisons ---
        {
            let (a_in, b_in) = bit_vector_inputs_u32();
            let flattened = flatten_bit_vector_pairs(&a_in, &b_in);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<u32>()) as u64;

            let mut linkage = wgsl_rs::linkage::wgpu::analyze_wgsl_module(
                &bit_vector_swizzle_cmp_u32::WGSL_SOURCE,
            )
            .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_vector_swizzle_cmp_u32::INPUT.set(flattened);
            bit_vector_swizzle_cmp_u32::OUTPUT.set([Vec4u::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_vector_swizzle_cmp_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_vector_swizzle_cmp_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            const OPS: [&str; 4] = ["or3", "xor3", "eq4", "eq2"];
            let labels: Vec<String> = (0..N)
                .flat_map(|idx| {
                    (0..4).flat_map(move |op| {
                        (0..4).map(move |lane| format!("{} inv{} l{}", OPS[op], idx, lane))
                    })
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_vector_swizzle_cmp_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_vector_assign_u32: vector &= |= ^= (wgsl-rs#186) ---
        {
            let (a_in, b_in) = bit_vector_inputs_u32();
            let flattened = flatten_bit_vector_pairs(&a_in, &b_in);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<u32>()) as u64;

            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_vector_assign_u32::WGSL_SOURCE)
                    .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_vector_assign_u32::INPUT.set(flattened);
            bit_vector_assign_u32::OUTPUT.set([Vec4u::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_vector_assign_u32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_vector_assign_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();

            const OPS: [&str; 4] = ["and_assign", "or_assign", "xor_assign", "xor_assign2"];
            let labels: Vec<String> = (0..N)
                .flat_map(|idx| {
                    (0..4).flat_map(move |op| {
                        (0..4).map(move |lane| format!("{} inv{} l{}", OPS[op], idx, lane))
                    })
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_vector_assign_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- bit_vector_ops_i32: signed vector bitwise ops (wgsl-rs#186) ---
        {
            let (a_in, b_in) = bit_vector_inputs_i32();
            let flattened = flatten_bit_vector_pairs(&a_in, &b_in);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<i32>()) as u64;

            let mut linkage =
                wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bit_vector_ops_i32::WGSL_SOURCE)
                    .unwrap();
            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            bit_vector_ops_i32::INPUT.set(flattened);
            bit_vector_ops_i32::OUTPUT.set([Vec4i::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                bit_vector_ops_i32::main(builtins.global_invocation_id);
            });
            let cpu_output = bit_vector_ops_i32::OUTPUT.get();
            // i32 and u32 bit patterns compare identically; reinterpret so
            // the shared u32 comparison helper applies.
            let cpu_u32s: Vec<u32> = cpu_output
                .iter()
                .flat_map(|v| v.to_array())
                .map(|x: i32| x as u32)
                .collect();

            const OPS: [&str; 4] = ["and4i", "or4i", "xor4i", "assign4i+or3i"];
            let labels: Vec<String> = (0..N)
                .flat_map(|idx| {
                    (0..4).flat_map(move |op| {
                        (0..4).map(move |lane| format!("{} inv{} l{}", OPS[op], idx, lane))
                    })
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "bit_vector_ops_i32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        results
    }
}
