//! Roundtrip tests for type-directed integer literal suffix insertion.
//!
//! Tests: bare integer literals that Rust infers as `u32` from context
//! (wgsl-rs#145). Before the suffix pass, these rendered unsuffixed and
//! WGSL read them as `i32` — most visibly through the polymorphic
//! `select` builtin, where naga rejects the module outright.
//!
//! Covers the issue #145 repro shape (`select` inside a `[u32; 1]`
//! array literal), comparison-anchored literals, and same-type builtin
//! groups (`min`).

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

#[wgsl]
pub mod bare_literals {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    /// The issue #145 shape: `0` and `1` are inferred as `u32` from the
    /// array element type. WGSL would read them as `i32` through
    /// `select` unless the suffix pass renders them as `0u` / `1u`.
    pub fn to_array(data: bool) -> [u32; 1] {
        [select(0, 1, data)]
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);
        // Bare `0` against the u32 element — comparison-anchored.
        let data = input[idx] != 0;
        let selected = to_array(data)[0];
        // Bare `63` against the cast-anchored u32 — same-type builtin
        // group.
        let limited = min(idx as u32, 63);
        get_mut!(OUTPUT)[idx] = selected + limited;
    }
}

fn literal_suffix_inputs() -> [u32; N] {
    let mut inputs = [0u32; N];
    for (i, input) in inputs.iter_mut().enumerate() {
        *input = if i % 2 == 0 { 0 } else { 1 };
    }
    inputs
}

pub struct LiteralSuffixTest;

impl RoundtripTest for LiteralSuffixTest {
    fn name(&self) -> &str {
        "literal_suffixes"
    }

    fn description(&self) -> &str {
        "type-directed integer literal suffix insertion (wgsl-rs#145)"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        let inputs = literal_suffix_inputs();
        let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

        let mut linkage =
            wgsl_rs::linkage::wgpu::analyze_wgsl_module(&bare_literals::WGSL_SOURCE).unwrap();
        let gpu_output = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
            device,
            queue,
            linkage: &mut linkage,
            entry: "main",
            input_data: input_bytes,
            output_size: (N * std::mem::size_of::<u32>()) as u64,
            workgroup_count: (1, 1, 1),
        });

        let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

        bare_literals::INPUT.set(inputs);
        bare_literals::OUTPUT.set([0u32; N]);
        dispatch_workgroups(
            (1, 1, 1),
            linkage
                .compute_entry("main")
                .expect("main entry present")
                .workgroup_size,
            |builtins| {
                bare_literals::main(builtins.global_invocation_id);
            },
        );
        let cpu_results: Vec<u32> = bare_literals::OUTPUT.get().to_vec();

        let labels: Vec<String> = (0..N).map(|i| format!("literal_suffixes[{i}]")).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        results.push(harness::compare_u32_results(
            "literal_suffixes",
            gpu_results,
            &cpu_results,
            &label_refs,
        ));

        results
    }
}
