# Atomics

Demonstrates atomic types and workgroup variables. Atomic types provide thread-safe operations for concurrent access in compute shaders and may only hold `i32` or `u32`. Workgroup variables are shared between all invocations in a workgroup.

## Rust Source

```rust
#[wgsl]
pub mod atomic_example {
    //! Demonstrates atomic types and workgroup variables.
    //!
    //! Atomic types provide thread-safe operations for concurrent access in
    //! compute shaders. They can only hold `i32` or `u32` values.
    //!
    //! Workgroup variables are shared between all invocations in a workgroup
    //! and can only be used in compute shaders.
    use wgsl_rs::std::*;

    // Workgroup variable with atomic counter - shared between all invocations
    workgroup!(COUNTER: Atomic<u32>);

    // Workgroup variable with atomic flags
    workgroup!(FLAGS: Atomic<i32>);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(local_invocation_index)] local_idx: u32) {
        // Each invocation can access the shared atomic counter
        // Note: atomicLoad/atomicStore builtins will be added in a future update
        // For now, this demonstrates the type parsing and code generation
        let _idx = local_idx;
    }
}
```

## Generated WGSL

```wgsl
var<workgroup> COUNTER: atomic<u32>;
var<workgroup> FLAGS: atomic<i32>;

@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) local_idx: u32) {
    let _idx = local_idx;
}
```

## Notes

- `Atomic<T>` maps to WGSL `atomic<T>`; `T` must be `i32` or `u32`.
- `workgroup!(NAME: T)` declares a `var<workgroup>` variable. Workgroup variables are only valid in compute shaders.