# Synchronization

Synchronization builtins order memory accesses between invocations in a
compute workgroup. They are only valid inside compute shaders; calling them
from vertex or fragment stages is a WGSL error.

## Barriers

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `workgroup_barrier()` | `workgroupBarrier` | Sync all invocations in the workgroup at this point. |
| `storage_barrier()` | `storageBarrier` | Memory barrier for `storage!` accesses across the workgroup. |
| `texture_barrier()` | `textureBarrier` | Memory barrier for storage-texture writes. |

All three take no arguments and return unit. On the CPU they are no-ops — a
single-threaded CPU dispatch has nothing to synchronize — but they still
compile and execute, so the same shader can run in both worlds.

## `workgroupUniformLoad`

`workgroupUniformLoad<T>(&var: &T) -> T` reads a workgroup variable such that
all invocations in the workgroup observe the same value. WGSL requires the
loaded address to be uniform across the workgroup.

wgsl-rs models this via the `WorkgroupUniformLoad` trait, implemented for the
types that are safe to load this way.

```rust
#[wgsl]
pub mod sync_example {
    use wgsl_rs::std::*;

    workgroup!(SHARED: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn cs(#[builtin(local_invocation_id)] lid: Vec3u) {
        let idx = lid.x() as usize;
        get_mut!(SHARED)[idx] = lid.x();
        workgroup_barrier();
        let partner = get!(SHARED)[((lid.x() + 1u32) % 64u32) as usize];
        workgroup_barrier();
        let uniform_first = workgroup_uniform_load(&SHARED);
    }
}
```

## When to use which

- **`workgroup_barrier`** — gate control flow: ensure every invocation has
  reached a point before any proceeds.
- **`storage_barrier`** — gate `storage!` reads/writes across invocations.
- **`texture_barrier`** — gate storage-texture writes before subsequent reads.
- **`workgroup_uniform_load`** — broadcast one uniform value to the whole
  workgroup (useful for divergent control flow convergence).