//! Roundtrip test for the `load!` macro and module-variable derefs
//! (wgsl-rs#86, #153).
//!
//! A whole-value copy between storage buffers written with the natural
//! Rust guard spellings: `load!(INPUT)` on the right-hand side and
//! `*get_mut!(OUTPUT) = ...` on the left. The deref on the left is a
//! guard artifact and renders as a plain value assignment in WGSL.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

/// Whole-value copy: `*get_mut!(OUTPUT) = load!(INPUT);`
#[wgsl]
pub mod load_copy {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 4]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 4]);

    #[compute]
    #[workgroup_size(1)]
    pub fn main(#[builtin(global_invocation_id)] _global_id: Vec3u) {
        // `load!` copies the value out of the read guard; the deref on
        // the left writes through the write guard. In WGSL this renders
        // as `OUTPUT = INPUT;` (wgsl-rs#153).
        *get_mut!(OUTPUT) = load!(INPUT);
    }
}

/// The `load!`/deref roundtrip test.
pub struct LoadMacroTest;

impl RoundtripTest for LoadMacroTest {
    fn name(&self) -> &str {
        "load_macro"
    }

    fn description(&self) -> &str {
        "load! value copies and deref assignment"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = [0.25f32, -1.5, 2.75, 42.0];
        let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);
        let output_size = (4 * std::mem::size_of::<f32>()) as u64;

        // A plain copy must be bit-exact.
        let epsilon = 0.0;

        let mut linkage =
            wgsl_rs::linkage::wgpu::analyze_wgsl_module(&load_copy::WGSL_SOURCE).unwrap();
        let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
            device,
            queue,
            linkage: &mut linkage,
            entry: "main",
            input_data: input_bytes,
            output_size,
            workgroup_count: (1, 1, 1),
        });
        let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

        use wgsl_rs::std::*;
        load_copy::INPUT.set(inputs);
        load_copy::OUTPUT.set([0.0f32; 4]);
        dispatch_workgroups((1, 1, 1), (1, 1, 1), |builtins| {
            load_copy::main(builtins.global_invocation_id);
        });
        let cpu_floats: Vec<f32> = load_copy::OUTPUT.get().to_vec();

        let labels: Vec<String> = (0..4).map(|i| format!("out[{i}]")).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        vec![harness::compare_f32_results(
            "load_copy",
            gpu_floats,
            &cpu_floats,
            &label_refs,
            epsilon,
        )]
    }
}
