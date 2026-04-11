//! Roundtrip tests for synchronization builtins.
//!
//! Tests: `workgroup_barrier`, `storage_barrier`, `workgroup_uniform_load`

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WG_SIZE: usize = 64;
const WG_COUNT: usize = 2;
const INVOCATIONS: usize = WG_SIZE * WG_COUNT;

#[wgsl]
pub mod workgroup_barrier_sum {
    use wgsl_rs::std::*;

    workgroup!(SHARED: [u32; 64]);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 128]);

    /// Writes local values to workgroup memory, barriers, then sums the group.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        #[builtin(local_invocation_index)] local_idx: u32,
    ) {
        let idx = global_id.x as usize;
        // Each lane writes a unique value into shared workgroup memory.
        get_mut!(SHARED)[local_idx as usize] = local_idx + 1u32;
        // Barrier is required so all lanes see all writes before summing.
        // Passing case: every lane reads full [1..=64], so all outputs are 2080.
        // Failing case: some lanes read stale zeros/partials and produce smaller sums.
        workgroup_barrier();

        let mut sum = 0u32;
        for i in 0..64 {
            sum += get!(SHARED)[i];
        }
        get_mut!(OUTPUT)[idx] = sum;
    }
}

#[wgsl]
pub mod storage_barrier_sum {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 256]);

    /// Writes to storage, barriers, then reads the workgroup chunk and sums it.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        #[builtin(local_invocation_index)] local_idx: u32,
    ) {
        let idx = global_id.x as usize;
        // Each lane writes one value into its workgroup chunk of OUTPUT.
        get_mut!(OUTPUT)[idx] = local_idx + 1u32;
        // storage_barrier should make those storage writes visible to peers.
        // Passing case: all lanes sum the same complete 64-lane chunk (2080).
        // Failing case: lanes observe incomplete writes and sums diverge or shrink.
        storage_barrier();

        let group_base = global_id.x - local_idx;
        let mut sum = 0u32;
        for i in 0u32..64u32 {
            sum += get!(OUTPUT)[(group_base + i) as usize];
        }

        get_mut!(OUTPUT)[128 + idx] = sum;
    }
}

#[wgsl]
pub mod workgroup_uniform_load_scalar {
    use wgsl_rs::std::*;

    workgroup!(GROUP_VALUE: [u32; 1]);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 128]);

    /// Stores one scalar per workgroup, then uniformly loads it for all lanes.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        #[builtin(local_invocation_index)] local_idx: u32,
    ) {
        if local_idx == 0u32 {
            // One lane seeds a workgroup-wide value unique to this workgroup.
            let group_id_x = global_id.x / 64u32;
            get_mut!(GROUP_VALUE)[0] = 1000u32 + group_id_x;
        }

        // Barrier ensures lane 0's write is visible before uniform load.
        // Passing case: all 64 lanes in a workgroup read identical value.
        // Failing case: some lanes read old/default value, causing mixed outputs.
        workgroup_barrier();
        let uniform = workgroup_uniform_load(&GROUP_VALUE);
        get_mut!(OUTPUT)[global_id.x as usize] = uniform[0];
    }
}

/// Runs one u32-based compute shader on the GPU and returns unpacked u32
/// output.
fn run_gpu_u32_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_source: &str,
    input: &[u32],
    output_len: usize,
) -> Vec<u32> {
    let input_bytes = bytemuck::cast_slice(input);
    let output_size = (output_len * std::mem::size_of::<u32>()) as u64;
    let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
        device,
        queue,
        shader_source,
        entry_point: "main",
        bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
        input_data: input_bytes,
        output_size,
        workgroup_count: (WG_COUNT as u32, 1, 1),
    });
    bytemuck::cast_slice(&gpu_bytes).to_vec()
}

/// Pushes one u32 comparison result with generated labels.
fn push_u32_result(results: &mut Vec<ComparisonResult>, name: &str, gpu: &[u32], cpu: &[u32]) {
    let labels: Vec<String> = (0..gpu.len()).map(|i| format!("{name}[{i}]")).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    results.push(harness::compare_u32_results(name, gpu, cpu, &label_refs));
}

pub struct SynchronizationTest;

impl RoundtripTest for SynchronizationTest {
    fn name(&self) -> &str {
        "synchronization"
    }

    fn description(&self) -> &str {
        "workgroup_barrier, storage_barrier, workgroup_uniform_load"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();
        let input = [0u32; 1];

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                workgroup_barrier_sum::linkage::shader_source(),
                &input,
                INVOCATIONS,
            );

            workgroup_barrier_sum::INPUT.set(input);
            workgroup_barrier_sum::SHARED.set([0u32; WG_SIZE]);
            workgroup_barrier_sum::OUTPUT.set([0u32; INVOCATIONS]);
            dispatch_workgroups((WG_COUNT as u32, 1, 1), (WG_SIZE as u32, 1, 1), |b| {
                workgroup_barrier_sum::main(b.global_invocation_id, b.local_invocation_index)
            });
            let cpu = workgroup_barrier_sum::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "workgroup_barrier_sum", &gpu, &cpu);
        }

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                storage_barrier_sum::linkage::shader_source(),
                &input,
                INVOCATIONS * 2,
            );

            storage_barrier_sum::INPUT.set(input);
            storage_barrier_sum::OUTPUT.set([0u32; INVOCATIONS * 2]);
            dispatch_workgroups((WG_COUNT as u32, 1, 1), (WG_SIZE as u32, 1, 1), |b| {
                storage_barrier_sum::main(b.global_invocation_id, b.local_invocation_index)
            });
            let cpu_all = storage_barrier_sum::OUTPUT.get().to_vec();
            let gpu_slice = &gpu[INVOCATIONS..INVOCATIONS * 2];
            let cpu_slice = &cpu_all[INVOCATIONS..INVOCATIONS * 2];

            push_u32_result(&mut results, "storage_barrier_sum", gpu_slice, cpu_slice);
        }

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                workgroup_uniform_load_scalar::linkage::shader_source(),
                &input,
                INVOCATIONS,
            );

            workgroup_uniform_load_scalar::INPUT.set(input);
            workgroup_uniform_load_scalar::GROUP_VALUE.set([0u32; 1]);
            workgroup_uniform_load_scalar::OUTPUT.set([0u32; INVOCATIONS]);
            dispatch_workgroups((WG_COUNT as u32, 1, 1), (WG_SIZE as u32, 1, 1), |b| {
                workgroup_uniform_load_scalar::main(
                    b.global_invocation_id,
                    b.local_invocation_index,
                )
            });
            let cpu = workgroup_uniform_load_scalar::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "workgroup_uniform_load_scalar", &gpu, &cpu);
        }

        results
    }
}
