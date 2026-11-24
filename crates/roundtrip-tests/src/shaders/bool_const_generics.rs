//! Roundtrip tests for `bool` const generic parameters, the compile-time
//! feature-flag use case.
//!
//! The module is a template (its entry point declares `const APPLY_FOG:
//! bool`), so the GPU side instantiates it both ways through the
//! macro-emitted `instantiate::<true>()` / `instantiate::<false>()` and
//! drives the dispatch via `wgsl_rs_linkage_wgpu`:
//!
//! - `analyze_ir_module` builds the linkage from the instantiated module,
//! - the entry point is looked up by its **un-mangled** Rust name (`"main"`);
//!   const-param instantiation does not rename entry points (each instantiation
//!   assembles its own concrete module, so only helper template functions
//!   mangle, e.g. `fog_scale_true`),
//! - the same `main` name is used for the wgpu pipeline entry point.
//!
//! CPU and GPU must agree on every lane for both flag values.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// A feature-flagged compute shader: bool const generics on a helper
/// (mangled both ways), on `select` conditions, on an `if` branch, and on
/// the entry point itself.
#[wgsl(validate_with_instantiation_types(APPLY_TRUE))]
pub mod bool_feature_flag {
    use wgsl_rs::std::*;

    pub const APPLY_TRUE: bool = true;

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 64]);

    /// The feature-flag helper: doubles `x` when the flag is on.
    /// `fog_scale::<true>` monomorphizes to `fog_scale_true` and
    /// `fog_scale::<false>` to `fog_scale_false`.
    pub fn fog_scale<const APPLY_FOG: bool>(x: u32) -> u32 {
        select(x, x * 2u32, APPLY_FOG)
    }

    /// Concrete wrapper pinning the flag on (renders `fog_scale_true`).
    pub fn scale_on(x: u32) -> u32 {
        fog_scale::<true>(x)
    }

    /// Concrete wrapper pinning the flag off (renders `fog_scale_false`).
    pub fn scale_off(x: u32) -> u32 {
        fog_scale::<false>(x)
    }

    /// The motivating shape: a feature-flag entry point. Instantiates via
    /// `instantiate::<true>()` / `instantiate::<false>()`; the entry
    /// point keeps the Rust name `main` in both concrete modules.
    #[compute]
    #[workgroup_size(64)]
    pub fn main<const APPLY_FOG: bool>(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let x = get!(INPUT)[idx];
        let mut lane0 = x;
        if APPLY_FOG {
            lane0 = x * 2u32;
        }
        get_mut!(OUTPUT)[idx] = vec4u(
            lane0,
            scale_on(x),
            scale_off(x),
            select(x, x + 1u32, APPLY_FOG),
        );
    }
}

/// Generates test inputs spanning [0, 990].
fn bool_flag_inputs() -> [u32; N] {
    let mut values = [0u32; N];
    for (i, value) in values.iter_mut().enumerate() {
        *value = (i as u32) * 990 / (N as u32 - 1);
    }
    values
}

/// Runs the CPU world for one flag value, returning the flattened output.
fn cpu_run(flag: bool, inputs: &[u32; N]) -> Vec<u32> {
    use wgsl_rs::std::*;

    bool_feature_flag::INPUT.set(*inputs);
    bool_feature_flag::OUTPUT.set([Vec4u::default(); N]);
    dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
        if flag {
            bool_feature_flag::main::<true>(builtins.global_invocation_id);
        } else {
            bool_feature_flag::main::<false>(builtins.global_invocation_id);
        }
    });
    bool_feature_flag::OUTPUT
        .get()
        .iter()
        .flat_map(|v| v.to_array())
        .collect()
}

/// Labels for the four result lanes, per invocation.
fn lane_labels(inputs: &[u32; N]) -> Vec<String> {
    (0..N)
        .flat_map(|i| {
            let x = inputs[i];
            vec![
                format!("flag_branch({x})"),
                format!("scale_on({x})"),
                format!("scale_off({x})"),
                format!("select_increment({x})"),
            ]
        })
        .collect()
}

/// The bool const generics roundtrip test.
pub struct BoolConstGenericsTest;

impl RoundtripTest for BoolConstGenericsTest {
    fn name(&self) -> &str {
        "bool_const_generics"
    }

    fn description(&self) -> &str {
        "bool const generic feature flags (true + false), linkage entry lookup by un-mangled name"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let inputs = bool_flag_inputs();
        let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);
        let output_size = (N * 4 * std::mem::size_of::<u32>()) as u64;
        let labels = lane_labels(&inputs);
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        // Reuse the module-local flag const for the CPU true-run, the same
        // value the automated instantiation validation test uses.
        let cpu_true = cpu_run(bool_feature_flag::APPLY_TRUE, &inputs);
        let cpu_false = cpu_run(false, &inputs);

        let mut results = Vec::new();

        for (flag, flag_label) in [(true, "true"), (false, "false")] {
            let module = if flag {
                bool_feature_flag::instantiate::<true>()
            } else {
                bool_feature_flag::instantiate::<false>()
            };
            let mut linkage = wgsl_rs_linkage_wgpu::analyze_ir_module(module);

            // The lookup under test: the entry point is addressed by its
            // un-mangled Rust name on the instantiated module, exactly as
            // a user would write it. The harness re-looks it up for the
            // pipeline entry point.
            assert!(
                linkage.compute_entry("main").is_some(),
                "un-mangled entry point lookup failed for {flag_label} instantiation"
            );

            let gpu_bytes = harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                device,
                queue,
                linkage: &mut linkage,
                entry: "main",
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: Vec<u32> = bytemuck::cast_slice(&gpu_bytes).to_vec();

            let cpu = if flag { &cpu_true } else { &cpu_false };
            results.push(harness::compare_u32_results(
                &format!("bool_feature_flag_{flag_label}"),
                &gpu_u32s,
                cpu,
                &label_refs,
            ));
        }

        results
    }
}
