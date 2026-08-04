# Synchronization

Synchronization builtins order memory accesses between invocations in a
compute workgroup. They are only valid inside compute shaders; calling them
from vertex or fragment stages is a WGSL error.

## Barriers

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `workgroupBarrier()` | `workgroupBarrier` | Sync all invocations in the workgroup at this point. |
| `storageBarrier()` | `storageBarrier` | Memory barrier for `storage!` accesses across the workgroup. |
| `textureBarrier()` | `textureBarrier` | Memory barrier for storage-texture writes. |

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

    workgroup!(SHARED, [u32; 64]);

    #[compute]
    pub fn cs(@builtin(workgroup_id) wg: Vec3u,
              @builtin(local_invocation_id) lid: Vec3u) {
        let idx = lid.x();
        SHARED[idx] = idx * 2;
        workgroupBarrier();
        let partner = SHARED[(idx + 1) % 64];
        workgroupBarrier();
        let uniform_first = workgroupUniformLoad(&SHARED[0]);
    }
}
```

## When to use which

- **`workgroupBarrier`** — gate control flow: ensure every invocation has
  reached a point before any proceeds.
- **`storageBarrier`** — gate `storage!` reads/writes across invocations.
- **`textureBarrier`** — gate storage-texture writes before subsequent reads.
- **`workgroupUniformLoad`** — broadcast one uniform value to the whole
  workgroup (useful for divergent control flow convergence).