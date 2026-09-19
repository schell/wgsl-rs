# Generic Functions

Generic free functions let you write a shader helper once and specialize it for several concrete types without preprocessor macros. The macro **monomorphizes** each call-site instantiation into its own concrete WGSL function with a mangled name.

## Defining a Generic Function

A generic function is ordinary Rust with trait bounds. The bounds are required only so Rust can type-check the body; they are stripped from the generated WGSL.

```rust
pub fn double<T: Copy + std::ops::Add<Output = T>>(x: T) -> T {
    x + x
}
```

## Turbofish is Required

Because the transpiler must know which concrete type to monomorphize, **every call to a generic function must use turbofish**:

```rust
pub fn apply_f32(value: f32) -> f32 {
    double::<f32>(value)
}

pub fn apply_i32(value: i32) -> i32 {
    double::<i32>(value)
}
```

Calls without `::<T>` are rejected by the macro even when Rust could infer the type.

## Monomorphization & Name Mangling

Each unique `(function, type-args)` pair produces one concrete WGSL function. The name is mangled as `<fn>_<type>` (with extra type parameters joined):

```wgsl
fn double_f32(x: f32) -> f32 {
  return x + x;
}

fn double_i32(x: i32) -> i32 {
  return x + x;
}
```

Duplicate instantiations across the module (or transitively through other generic functions) are deduplicated — only one copy of each monomorphized function is emitted.

## Transitive Generic Calls

A generic function may call another generic function. The inner turbofish drives its own monomorphization:

```rust
pub fn select_val<T: Copy>(a: T, b: T, cond: bool) -> T {
    if cond { a } else { b }
}

pub fn double_or_keep<T: Copy + std::ops::Add<Output = T>>(x: T, use_double: bool) -> T {
    select_val::<T>(double::<T>(x), x, use_double)
}
```

Calling `double_or_keep::<f32>(...)` pulls in both `double_f32` and `select_val_f32` automatically.

## Multiple Type Parameters

Functions may take more than one type parameter. Each is monomorphized over the full tuple of concrete type arguments:

```rust
pub fn mix<A: Copy, B: Copy>(a: A, b: B) -> A {
    a
}
```

A call `mix::<f32, u32>(x, y)` produces `mix_f32_u32`.

## Const Generic Parameters

Functions can also take `const` generic parameters of type `u32` or `usize` — the only const param types that make sense in WGSL (they're used as array lengths). The const param is substituted with a concrete integer literal at monomorphization time:

```rust
pub fn sum_n<const N: usize>(arr: [u32; N]) -> u32 {
    let mut total: u32 = 0;
    for i in 0..N {
        total += arr[i];
    }
    total
}

pub fn run() -> u32 {
    sum_n::<4>([1, 2, 3, 4])
}
```

The call `sum_n::<4>` produces a WGSL function `sum_n_4` with `N` replaced by the literal `4` throughout (including the array type and loop bound). Const and type params can coexist on the same function; they're monomorphized over the full tuple of arguments.

> Const param references are always bare identifiers (e.g. `N`), per stable Rust's const generics syntax. They're substituted to `Expr::Lit` at monomorphization time — no new IR variant is needed.

## Trait Bounds are Rust-Only

`Copy`, `Clone`, `Add`, `PartialEq`, custom traits — all bounds exist solely for the Rust type checker. They generate no WGSL output. This is the "two worlds" split in action: Rust validates the generic body once on the CPU; WGSL receives fully concrete, monomorphized code for the GPU with no notion of traits or generics.

## A Worked Example

The `generic_functions` example module demonstrates the full pipeline:

```rust
#[wgsl]
pub mod generic_functions {
    pub fn double<T: Copy + std::ops::Add<Output = T>>(x: T) -> T {
        x + x
    }

    pub fn select_val<T: Copy>(a: T, b: T, cond: bool) -> T {
        if cond { a } else { b }
    }

    pub fn double_or_keep<T: Copy + std::ops::Add<Output = T>>(x: T, use_double: bool) -> T {
        select_val::<T>(double::<T>(x), x, use_double)
    }

    pub fn apply_f32(value: f32) -> f32 {
        double_or_keep::<f32>(value, true)
    }

    pub fn apply_i32(value: i32) -> i32 {
        double_or_keep::<i32>(value, false)
    }
}
```

`apply_f32` and `apply_i32` each pull in their own copies of `double_or_keep_*`, `double_*`, and `select_val_*`, with the duplicate instantiations of `double_or_keep` collapsed as needed.