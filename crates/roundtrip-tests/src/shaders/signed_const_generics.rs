//! Roundtrip tests for signed (`i32`) const generic parameters.
//!
//! The motivating use case: internal min/max clamp bounds passed as
//! `const` arguments, including negative values (`clamp_to::<-8, 8>`).
//! Also exercises loop bounds with a negative value (`0..END` with
//! `END < 0` is an empty range in both worlds) and a positive value
//! (non-trivial CPU/GPU agreement), all through const-generic
//! monomorphization with negative-mangled names (`clamp_to_n8_8`).

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// Signed const generics: clamps with negative and positive ranges, and
/// loop bounds with negative (empty) and positive ranges.
#[wgsl]
pub mod signed_const_clamp {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [i32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4i; 64]);

    pub fn clamp_to<const MIN: i32, const MAX: i32>(x: i32) -> i32 {
        max(min(x, MAX), MIN)
    }

    pub fn sum_to<const END: i32>(x: i32) -> i32 {
        let mut total: i32 = x;
        for i in 0..END {
            total += i;
        }
        total
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = vec4i(
            clamp_to::<-8, 8>(x),
            clamp_to::<0, 100>(x),
            sum_to::<-4>(x),
            sum_to::<4>(x),
        );
    }
}

/// Generates test inputs spanning both clamp ranges: [-200, 300].
fn signed_const_inputs() -> [i32; N] {
    let mut values = [0i32; N];
    for (i, value) in values.iter_mut().enumerate() {
        *value = -200 + (i as i32) * 500 / (N as i32 - 1);
    }
    values
}

/// The signed const generics roundtrip test.
pub struct SignedConstGenericsTest;

impl RoundtripTest for SignedConstGenericsTest {
    fn name(&self) -> &str {
        "signed_const_generics"
    }

    fn description(&self) -> &str {
        "i32 const generic clamps (negative + positive) and loop bounds"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = signed_const_inputs();
        let input_bytes = bytemuck::cast_slice::<i32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<i32>()) as u64;

        let mut results = Vec::new();

        {
            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&signed_const_clamp::WGSL_SOURCE)
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
            let gpu_i32s: &[i32] = bytemuck::cast_slice(&gpu_bytes);
            let gpu_u32s: Vec<u32> = gpu_i32s.iter().map(|&x| x as u32).collect();

            use wgsl_rs::std::*;
            signed_const_clamp::INPUT.set(inputs);
            signed_const_clamp::OUTPUT.set([Vec4i::default(); N]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                signed_const_clamp::main(builtins.global_invocation_id);
            });
            let cpu_output = signed_const_clamp::OUTPUT.get();
            let cpu_i32s: Vec<i32> = cpu_output.iter().flat_map(|v| v.to_array()).collect();
            let cpu_u32s: Vec<u32> = cpu_i32s.iter().map(|&x| x as u32).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    let x = inputs[i];
                    vec![
                        format!("clamp_to::<-8, 8>({x})"),
                        format!("clamp_to::<0, 100>({x})"),
                        format!("sum_to::<-4>({x})"),
                        format!("sum_to::<4>({x})"),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            results.push(harness::compare_u32_results(
                "signed_const_clamp",
                &gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        results
    }
}
