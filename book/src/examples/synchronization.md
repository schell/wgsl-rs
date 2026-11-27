# Synchronization

Demonstrates synchronization builtin functions for compute shaders: `workgroup_barrier`, `storage_barrier`, and `workgroup_uniform_load`. These coordinate memory visibility and execution ordering across invocations within a workgroup and must only be called from compute entry points in uniform control flow.

## Rust Source

```rust
#[wgsl]
pub mod synchronization_example {
    //! Demonstrates synchronization builtin functions for compute shaders.
    //!
    //! These functions coordinate memory visibility and execution ordering
    //! across invocations within a workgroup. They must only be called from
    //! compute shader entry points in uniform control flow.
    use wgsl_rs::std::*;

    pub struct SynchronizationExamplePack {
        #[workgroup]
        pub scratch: Workgroup<[u32; 64]>,
        #[storage(group(0), binding(0), read_only)]
        pub input: Storage<[u32; 64], Read>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64], ReadWrite>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(
        #[builtin(local_invocation_index)] local_idx: u32,
        pack: &SynchronizationExamplePack,
    ) {
        // Copy input data into workgroup-shared memory.
        get_mut!(pack.scratch)[local_idx as usize] = get!(pack.input)[local_idx as usize];

        // Ensure all workgroup memory writes are visible to every invocation.
        workgroup_barrier();

        // Read from a neighbor's slot (with wrap-around) to demonstrate
        // that the barrier made all writes visible.
        let neighbor_idx: u32 = (local_idx + 1u32) % 64u32;
        let neighbor_val: u32 = get!(pack.scratch)[neighbor_idx as usize];

        // Ensure all storage writes from the workgroup are complete
        // before writing results.
        storage_barrier();

        get_mut!(pack.output)[local_idx as usize] = neighbor_val;
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn uniform_load_example(
        #[builtin(local_invocation_index)] local_idx: u32,
        pack: &SynchronizationExamplePack,
    ) {
        // Each invocation writes its index into shared memory.
        get_mut!(pack.scratch)[local_idx as usize] = local_idx;

        // Ensure all writes are visible before uniform load.
        workgroup_barrier();

        // Uniformly load the first element across the entire workgroup.
        // All invocations receive the same value.
        let first: [u32; 64] = workgroup_uniform_load(&pack.scratch);
        get_mut!(pack.output)[local_idx as usize] = first[0];
    }
}
```

## Generated WGSL

```wgsl
var<workgroup> synchronizationexamplepack__scratch: array<u32, 64>;
@group(0) @binding(0) var<storage, read> synchronizationexamplepack__input: array<u32, 64>;
@group(0) @binding(1) var<storage, read_write> synchronizationexamplepack__output: array<u32, 64>;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) local_idx: u32) {
    synchronizationexamplepack__scratch[u32(local_idx)] = synchronizationexamplepack__input[u32(local_idx)];
    workgroupBarrier();
    let neighbor_idx: u32 = (local_idx + 1u) % 64u;
    let neighbor_val: u32 = synchronizationexamplepack__scratch[u32(neighbor_idx)];
    storageBarrier();
    synchronizationexamplepack__output[u32(local_idx)] = neighbor_val;
}

@compute @workgroup_size(64) fn uniform_load_example(@builtin(local_invocation_index) local_idx: u32) {
    synchronizationexamplepack__scratch[u32(local_idx)] = local_idx;
    workgroupBarrier();
    let first: array<u32, 64> = workgroupUniformLoad(&synchronizationexamplepack__scratch);
    synchronizationexamplepack__output[u32(local_idx)] = first[0];
}
```

## Notes

- `workgroup_barrier()` maps to `workgroupBarrier`; `storage_barrier()` maps to `storageBarrier`.
- `workgroup_uniform_load(&pack.scratch)` maps to `workgroupUniformLoad(&synchronizationexamplepack__scratch)`.
- These functions must only be called from compute entry points in uniform control flow.