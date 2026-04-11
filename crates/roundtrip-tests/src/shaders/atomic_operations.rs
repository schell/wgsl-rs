//! Roundtrip tests for atomic builtins.
//!
//! Tests u32 and i32 atomics for scalar operation sequences and contended
//! workgroup increments.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WG_SIZE: usize = 64;

#[wgsl]
pub mod atomic_u32_scalar_ops {
    use wgsl_rs::std::*;

    workgroup!(A: Atomic<u32>);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 16]);

    /// Runs a deterministic sequence of u32 atomic ops from one invocation.
    #[compute]
    #[workgroup_size(1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        if global_id.x != 0u32 {
            return;
        }

        let a = &get!(A);

        atomic_store(a, 10u32);
        get_mut!(OUTPUT)[0] = atomic_load(a);
        get_mut!(OUTPUT)[1] = atomic_add(a, 5u32);
        get_mut!(OUTPUT)[2] = atomic_load(a);
        get_mut!(OUTPUT)[3] = atomic_sub(a, 3u32);
        get_mut!(OUTPUT)[4] = atomic_min(a, 8u32);
        get_mut!(OUTPUT)[5] = atomic_max(a, 20u32);
        get_mut!(OUTPUT)[6] = atomic_and(a, 0x0Fu32);
        get_mut!(OUTPUT)[7] = atomic_or(a, 0x10u32);
        get_mut!(OUTPUT)[8] = atomic_xor(a, 0x03u32);
        get_mut!(OUTPUT)[9] = atomic_exchange(a, 77u32);

        let cas_fail = atomic_compare_exchange_weak(a, 123u32, 99u32);
        get_mut!(OUTPUT)[10] = cas_fail.old_value;
        let mut exchanged = 0u32;
        if cas_fail.exchanged {
            exchanged = 1u32;
        }
        get_mut!(OUTPUT)[11] = exchanged;
        get_mut!(OUTPUT)[12] = atomic_load(a);

        atomic_store(a, 0xFFFF_FFFFu32);
        get_mut!(OUTPUT)[13] = atomic_add(a, 1u32);
        get_mut!(OUTPUT)[14] = atomic_load(a);
        get_mut!(OUTPUT)[15] = 0xA7011Cu32;
    }
}

#[wgsl]
pub mod atomic_i32_scalar_ops {
    use wgsl_rs::std::*;

    workgroup!(A: Atomic<i32>);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 16]);

    /// Runs a deterministic sequence of i32 atomic ops from one invocation.
    #[compute]
    #[workgroup_size(1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        if global_id.x != 0u32 {
            return;
        }

        let a = &get!(A);

        atomic_store_i32(a, -10i32);
        get_mut!(OUTPUT)[0] = atomic_load_i32(a) as u32;
        get_mut!(OUTPUT)[1] = atomic_add_i32(a, 5i32) as u32;
        get_mut!(OUTPUT)[2] = atomic_load_i32(a) as u32;
        get_mut!(OUTPUT)[3] = atomic_sub_i32(a, 3i32) as u32;
        get_mut!(OUTPUT)[4] = atomic_min_i32(a, -20i32) as u32;
        get_mut!(OUTPUT)[5] = atomic_max_i32(a, 7i32) as u32;

        atomic_store_i32(a, 15i32);
        get_mut!(OUTPUT)[6] = atomic_and_i32(a, 10i32) as u32;
        get_mut!(OUTPUT)[7] = atomic_or_i32(a, 5i32) as u32;
        get_mut!(OUTPUT)[8] = atomic_xor_i32(a, 12i32) as u32;
        get_mut!(OUTPUT)[9] = atomic_exchange_i32(a, -33i32) as u32;

        let cas_fail = atomic_compare_exchange_weak_i32(a, 999i32, 44i32);
        get_mut!(OUTPUT)[10] = cas_fail.old_value as u32;
        let mut exchanged = 0u32;
        if cas_fail.exchanged {
            exchanged = 1u32;
        }
        get_mut!(OUTPUT)[11] = exchanged;
        get_mut!(OUTPUT)[12] = atomic_load_i32(a) as u32;

        atomic_store_i32(a, 2_147_483_647i32);
        get_mut!(OUTPUT)[13] = atomic_add_i32(a, 1i32) as u32;
        get_mut!(OUTPUT)[14] = atomic_load_i32(a) as u32;
        get_mut!(OUTPUT)[15] = 0xA7011C32u32;
    }
}

#[wgsl]
pub mod atomic_u32_contended_add {
    use wgsl_rs::std::*;

    workgroup!(COUNTER: Atomic<u32>);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    /// All lanes increment one counter; all should read the same final value.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        #[builtin(local_invocation_index)] local_idx: u32,
    ) {
        let counter = &get!(COUNTER);
        if local_idx == 0u32 {
            atomic_store(counter, 0u32);
        }
        workgroup_barrier();

        atomic_add(counter, 1u32);
        workgroup_barrier();

        get_mut!(OUTPUT)[global_id.x as usize] = atomic_load(counter);
    }
}

#[wgsl]
pub mod atomic_i32_contended_add {
    use wgsl_rs::std::*;

    workgroup!(COUNTER: Atomic<i32>);
    storage!(group(0), binding(0), INPUT: [u32; 1]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    /// All lanes increment one signed counter; all should read 64.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        #[builtin(local_invocation_index)] local_idx: u32,
    ) {
        let counter = &get!(COUNTER);
        if local_idx == 0u32 {
            atomic_store_i32(counter, 0i32);
        }
        workgroup_barrier();

        atomic_add_i32(counter, 1i32);
        workgroup_barrier();

        get_mut!(OUTPUT)[global_id.x as usize] = atomic_load_i32(counter) as u32;
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
    workgroup_count: (u32, u32, u32),
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
        workgroup_count,
    });
    bytemuck::cast_slice(&gpu_bytes).to_vec()
}

/// Pushes one u32 comparison result with generated labels.
fn push_u32_result(results: &mut Vec<ComparisonResult>, name: &str, gpu: &[u32], cpu: &[u32]) {
    let labels: Vec<String> = (0..gpu.len()).map(|i| format!("{name}[{i}]")).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    results.push(harness::compare_u32_results(name, gpu, cpu, &label_refs));
}

pub struct AtomicOperationsTest;

impl RoundtripTest for AtomicOperationsTest {
    fn name(&self) -> &str {
        "atomic_operations"
    }

    fn description(&self) -> &str {
        "u32/i32 atomics: load/store/arithmetic/bitwise/exchange/compare_exchange_weak"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();
        let input = [0u32; 1];

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                atomic_u32_scalar_ops::linkage::shader_source(),
                &input,
                16,
                (1, 1, 1),
            );

            atomic_u32_scalar_ops::INPUT.set(input);
            atomic_u32_scalar_ops::A.set(Atomic::default());
            atomic_u32_scalar_ops::OUTPUT.set([0u32; 16]);
            dispatch_workgroups((1, 1, 1), (1, 1, 1), |b| {
                atomic_u32_scalar_ops::main(b.global_invocation_id)
            });
            let cpu = atomic_u32_scalar_ops::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "atomic_u32_scalar_ops", &gpu, &cpu);
        }

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                atomic_i32_scalar_ops::linkage::shader_source(),
                &input,
                16,
                (1, 1, 1),
            );

            atomic_i32_scalar_ops::INPUT.set(input);
            atomic_i32_scalar_ops::A.set(Atomic::default());
            atomic_i32_scalar_ops::OUTPUT.set([0u32; 16]);
            dispatch_workgroups((1, 1, 1), (1, 1, 1), |b| {
                atomic_i32_scalar_ops::main(b.global_invocation_id)
            });
            let cpu = atomic_i32_scalar_ops::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "atomic_i32_scalar_ops", &gpu, &cpu);
        }

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                atomic_u32_contended_add::linkage::shader_source(),
                &input,
                WG_SIZE,
                (1, 1, 1),
            );

            atomic_u32_contended_add::INPUT.set(input);
            atomic_u32_contended_add::COUNTER.set(Atomic::default());
            atomic_u32_contended_add::OUTPUT.set([0u32; WG_SIZE]);
            dispatch_workgroups((1, 1, 1), (WG_SIZE as u32, 1, 1), |b| {
                atomic_u32_contended_add::main(b.global_invocation_id, b.local_invocation_index)
            });
            let cpu = atomic_u32_contended_add::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "atomic_u32_contended_add", &gpu, &cpu);
        }

        {
            let gpu = run_gpu_u32_shader(
                device,
                queue,
                atomic_i32_contended_add::linkage::shader_source(),
                &input,
                WG_SIZE,
                (1, 1, 1),
            );

            atomic_i32_contended_add::INPUT.set(input);
            atomic_i32_contended_add::COUNTER.set(Atomic::default());
            atomic_i32_contended_add::OUTPUT.set([0u32; WG_SIZE]);
            dispatch_workgroups((1, 1, 1), (WG_SIZE as u32, 1, 1), |b| {
                atomic_i32_contended_add::main(b.global_invocation_id, b.local_invocation_index)
            });
            let cpu = atomic_i32_contended_add::OUTPUT.get().to_vec();

            push_u32_result(&mut results, "atomic_i32_contended_add", &gpu, &cpu);
        }

        results
    }
}
