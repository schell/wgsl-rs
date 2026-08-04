# Generic Structs

Generic structs follow the same monomorphization model as generic functions: you write one Rust definition with type parameters, and the macro emits a separate concrete WGSL struct for each `(struct, type-args)` pair used in the module.

## Defining a Generic Struct

```rust
pub struct Pair<T: Copy> {
    pub a: T,
    pub b: T,
}
```

The `Copy` (or other) bound is Rust-only and stripped from the WGSL.

## Usage & Mangling

At every use site, supply the concrete type either as a turbofish on the path or as a type annotation. Each unique instantiation becomes a mangled WGSL struct:

```rust
pub fn use_pair_f32() -> f32 {
    let p = Pair { a: 1.0, b: 2.0 };
    Pair::<f32>::sum(p)
}

pub fn use_pair_i32() -> i32 {
    let p: Pair<i32> = Pair::<i32> { a: 10, b: 20 };
    Pair::<i32>::first(p)
}
```

```wgsl
struct Pair_f32 {
  a: f32,
  b: f32,
}

struct Pair_i32 {
  a: i32,
  b: i32,
}
```

## Generic Impl Blocks

`impl<T> Pair<T>` blocks are monomorphized alongside the struct. Each method becomes a mangled WGSL function named `<Struct>_<type>_<method>`:

```rust
impl<T: Copy + std::ops::Add<Output = T>> Pair<T> {
    pub fn first(p: Pair<T>) -> T {
        p.a
    }

    pub fn sum(p: Pair<T>) -> T {
        p.a + p.b
    }
}
```

For `Pair::<f32>` this yields:

```wgsl
fn Pair_f32_first(p: Pair_f32) -> f32 {
  return p.a;
}

fn Pair_f32_sum(p: Pair_f32) -> f32 {
  return p.a + p.b;
}
```

## Struct Construction

Construct a generic struct by writing the literal form `Pair::<f32> { a, b }` or by relying on a type annotation. The macro emits a positional WGSL constructor call with the mangled name:

```rust
let p = Pair::<f32> { a: 1.0, b: 2.0 };
```

```wgsl
let p = Pair_f32(1.0, 2.0);
```

Fields are emitted in declaration order.

## Known Limitation: Struct Constructor Mangling

There is a known bug in which the bare struct-constructor form `Pair { a, b }` (without a turbofish or annotation that the macro can resolve) is not mangled correctly, producing invalid WGSL. Until this is fixed, the recommended workarounds are:

- Always use the turbofish form `Pair::<T> { ... }` at construction sites, **or**
- Annotate the binding: `let p: Pair<T> = Pair { ... }`.
- For modules that exercise the bug and cannot be restructured, suppress auto-validation with `#[wgsl(skip_validation)]` (see [Disabling Validation](../validation/disabling.md)) so the failing constructor does not break `cargo test`.

The `generic_structs` example currently uses `#[wgsl(skip_validation)]` for this reason:

```rust
#[wgsl(skip_validation)]
pub mod generic_structs {
    pub struct Pair<T: Copy> {
        pub a: T,
        pub b: T,
    }

    impl<T: Copy + std::ops::Add<Output = T>> Pair<T> {
        pub fn first(p: Pair<T>) -> T { p.a }
        pub fn sum(p: Pair<T>) -> T { p.a + p.b }
    }
}
```

## Multiple Type Parameters

A struct may take several type parameters; the mangled name joins all concrete types:

```rust
pub struct Cell<K: Copy, V: Copy> {
    pub key: K,
    pub value: V,
}
```

`Cell::<u32, f32>` produces `Cell_u32_f32`.