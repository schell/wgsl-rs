# `workgroup!`

Declares a workgroup-scoped variable shared across invocations in a compute workgroup.

## Syntax

```rust
workgroup!(NAME: Type);
```

## What It Generates

WGSL:

```wgsl
var<workgroup> NAME: Type;
```

Rust:

```rust
pub static NAME: Workgroup<Type>;
```

The Rust-side static is backed by a `LazyLock<RwLock<T>>`, so CPU code can read and write the same value across threads for testing.

## Access

- `get!(NAME)` reads.
- `get_mut!(NAME)` writes.

## Example: Shared Sum

```rust
#[wgsl]
pub mod reduce {
    use wgsl_rs::std::*;

    pub const WG_SIZE: u32 = 64;

    workgroup!(SHARED: array<f32, 64>);

    #[compute]
    #[workgroup_size(64)]
    pub fn cs_main(
        #[builtin(workgroup_id)] wg: Vec3u,
        #[builtin(local_invocation_index)] li: u32,
    ) {
        let mut s = get_mut!(SHARED);
        s[li as usize] = f32(li);

        // barrier omitted for brevity; use workgroupBarrier() via a builtin

        let mut sum: f32 = 0.0;
        for i in 0..WG_SIZE {
            sum += get!(SHARED)[i as usize];
        }

        if li == 0 {
            let mut out = get_mut!(RESULT);
            out.value = sum;
        }
    }
}
```

## Notes

- Workgroup variables are visible to all invocations sharing the same `workgroup_id`.
- Use a workgroup barrier before reading values written by other invocations.
- Lifetime is a single workgroup dispatch; values are not preserved between dispatches.