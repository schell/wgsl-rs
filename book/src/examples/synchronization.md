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

    workgroup!(SCRATCH: [u32; 64]);

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(local_invocation_index)] local_idx: u32) {
        // Copy input data into workgroup-shared memory.
        get_mut!(SCRATCH)[local_idx as usize] = get!(INPUT)[local_idx as usize];

        // Ensure all workgroup memory writes are visible to every invocation.
        workgroup_barrier();

        // Read from a neighbor's slot (with wrap-around) to demonstrate
        // that the barrier made all writes visible.
        let neighbor_idx: u32 = (local_idx + 1u32) % 64u32;
        let neighbor_val: u32 = get!(SCRATCH)[neighbor_idx as usize];

        // Ensure all storage writes from the workgroup are complete
        // before writing results.
        storage_barrier();

        get_mut!(OUTPUT)[local_idx as usize] = neighbor_val;
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn uniform_load_example(#[builtin(local_invocation_index)] local_idx: u32) {
        // Each invocation writes its index into shared memory.
        get_mut!(SCRATCH)[local_idx as usize] = local_idx;

        // Ensure all writes are visible before uniform load.
        workgroup_barrier();

        // Uniformly load the first element across the entire workgroup.
        // All invocations receive the same value.
        let first: [u32; 64] = workgroup_uniform_load(&SCRATCH);
        get_mut!(OUTPUT)[local_idx as usize] = first[0];
    }
}
```

## Generated WGSL

```wgsl
var<workgroup> SCRATCH: array<u32, 64>;
@group(0) @binding(0) var<storage, read> INPUT: array<u32, 64>;
@group(0) @binding(1) var<storage, read_write> OUTPUT: array<u32, 64>;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) local_idx: u32) {
    SCRATCH[u32(local_idx)] = INPUT[u32(local_idx)];
    workgroupBarrier();
    let neighbor_idx: u32 = (local_idx + 1u) % 64u;
    let neighbor_val: u32 = SCRATCH[u32(neighbor_idx)];
    storageBarrier();
    OUTPUT[u32(local_idx)] = neighbor_val;
}

@compute @workgroup_size(64) fn uniform_load_example(@builtin(local_invocation_index) local_idx: u32) {
    SCRATCH[u32(local_idx)] = local_idx;
    workgroupBarrier();
    let first: array<u32, 64> = workgroupUniformLoad(&SCRATCH);
    OUTPUT[u32(local_idx)] = first[0];
}
```

## Notes

- `workgroup_barrier()` maps to `workgroupBarrier`; `storage_barrier()` maps to `storageBarrier`.
- `workgroup_uniform_load(&SCRATCH)` maps to `workgroupUniformLoad(&SCRATCH)`.
- These functions must only be called from compute entry points in uniform control flow.