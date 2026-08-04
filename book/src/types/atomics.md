# Atomics

`Atomic<T>` is the wgsl-rs wrapper around the WGSL `atomic<T>` type. It is the only mechanism for shared mutable state across invocations within a workgroup or across storage-buffer accesses.

## Allowed Element Types

WGSL atomics are restricted to `i32` and `u32`. wgsl-rs enforces the same restriction:

| wgsl-rs          | WGSL           | CPU backing                 |
| ---------------- | -------------- | --------------------------- |
| `Atomic<i32>`    | `atomic<i32>`  | `std::sync::atomic::AtomicI32` |
| `Atomic<u32>`    | `atomic<u32>`  | `std::sync::atomic::AtomicU32` |

On the CPU side, `Atomic<T>` is backed by the matching `std::sync::atomic` type so the same shader module can run as ordinary Rust in tests and produce consistent results.

## Where Atomics May Appear

Atomics are only valid inside address spaces where multiple invocations can observe each other's writes:

- **workgroup** variables, declared with the `workgroup!` macro.
- **storage** buffers, declared with the `storage!` macro (typically `read_write`).

Function-local atomics are not useful (a single invocation has no contention) and are rejected.

```rust
use wgsl_rs::std::*;

workgroup!(COUNTER: Atomic<u32>);
workgroup!(FLAGS:  Atomic<i32>);

#[compute]
#[workgroup_size(64)]
pub fn main(#[builtin(local_invocation_index)] local_idx: u32) {
    let _idx = local_idx;
}
```

```wgsl
var<workgroup> COUNTER: atomic<u32>;
var<workgroup> FLAGS: atomic<i32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local_idx: u32) {
  let _idx = local_idx;
}
```

## Atomic Operations

The standard WGSL atomic builtins (`atomicLoad`, `atomicStore`, `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExchange`, `atomicCompareExchangeWeak`) are exposed by `wgsl_rs::std` as free functions and transpile to the matching WGSL call. Use them through the `get!` / `get_mut!` accessors that yield a reference to the underlying `Atomic<T>`:

```rust
let current: u32 = atomicLoad(&get!(COUNTER));
atomicAdd(&get_mut!(COUNTER), 1u32);
```

```wgsl
let current: u32 = atomicLoad(&COUNTER);
atomicAdd(&COUNTER, 1u32);
```

Use the `get!` accessor for read-only atomic loads and `get_mut!` for mutating atomic operations, mirroring the storage-buffer conventions in [Binding Macros](../writing-shaders/binding-macros.md).