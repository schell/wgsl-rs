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

## Using `Self` in Generic Impls

`Self` works everywhere inside a generic impl block — parameter and
return types, associated const types, `Self::method(..)` calls,
`Self { .. }` constructors, and `Self::CONST` paths — and resolves to
the concrete instantiation, whether the struct is monomorphized in the
same module or instantiated through a cross-module import:

```rust
impl<T: Copy> Pair<T> {
    pub fn swap(p: Self) -> Self {
        Self { a: p.b, b: p.a }
    }
}
```

For `Pair::<f32>` the constructor renders the positional form
`Pair_f32(p.b, p.a)` and the signature renders `Pair_f32` in both
positions.

One caveat is a WGSL language rule, not a transpiler one: const
initializers may only use const-expressions, and user functions can
never be WGSL const-functions (WGSL §11.3). Calling one is a compile
error that points at the struct-literal alternative:

```text
error: const 'ZERO' initializer calls 'Pair::zero', but WGSL const
initializers are const-expressions: only value constructors and const
builtins are callable, never user functions; use the struct-literal
form (e.g. `MyType { field: 0 }`) or literal values instead
```

So build the value with a struct literal (or literal values, value
constructors like `vec2f(..)`, and `@const` builtins like `min` — those
are all const-expressions):

```rust
impl<T: Copy> Pair<T> {
    // `Self::zero()` here is a compile error; WGSL forbids calling
    // user functions in const contexts — use the literal form.
    pub const ZERO: Self = Self { a: 0.0, b: 0.0 };
}
```

## Method Calls Require Turbofish

Like generic functions, calls to a generic struct's methods must supply
the concrete type with turbofish. A bare `Pair::sum(p)` renders an
un-mangled name that fails WGSL validation far from the mistake, so the
macro rejects it at compile time:

```text
error: calling generic struct method 'Pair::sum' requires generic
arguments, e.g., Pair::<T>::sum(...)
```

This also applies inside generic function bodies (`Pair::<T>::sum(p)`)
and to associated constants accessed by path: `Self::CONST` works
inside the struct's own impl, but referencing an instantiated struct's
associated const from elsewhere is not yet supported (bare paths are
rejected, and turbofish on const paths does not parse). Cross-module
calls without turbofish cannot be checked at compile time (the
importing module's macro cannot know the imported name is generic);
they are reported when the WGSL source is assembled, with the same
guidance.

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

Note that this limitation is about *constructor expressions* only. Bare
*method calls* (`Pair::sum(p)`) are rejected at compile time (see
[Method Calls Require Turbofish](#method-calls-require-turbofish)), so
they cannot fail silently.

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

## Const Generic Parameters

Structs can also take `const N: usize` or `const N: u32` parameters, which are substituted with concrete integer literals at monomorphization time. This is the natural way to express arrays whose length varies per instantiation:

```rust
pub struct Grid<const N: usize> {
    pub cells: [u32; N],
}

impl<const N: usize> Grid<N> {
    pub fn first(cells: [u32; N]) -> u32 {
        cells[0]
    }
}

pub fn run() -> u32 {
    let g = Grid::<4> { cells: [0, 0, 0, 0] };
    g.cells[0]
}
```

`Grid::<4>` produces a WGSL struct `Grid_4` with `cells: array<u32, 4>`, and `Grid::<4>::first` becomes `Grid_4_first`.

## Generic Trait Impls on Array Types

Generic impl blocks on array self types (`impl<T: Trait> Trait for [T; N]`) are supported. The monomorphizer substitutes the concrete element type and mangles the methods:

```rust
pub trait Zeroable {
    fn zero() -> Self;
}

impl<T: Zeroable> Zeroable for [T; 4] {
    fn zero() -> [T; 4] {
        [T::zero(), T::zero(), T::zero(), T::zero()]
    }
}

impl Zeroable for u32 {
    fn zero() -> u32 {
        0
    }
}

pub fn caller_u32_array() -> [u32; 4] {
    <[u32; 4]>::zero()
}
```

The call with `[u32; 4]` produces a WGSL function `_2array_u32_4_zero` (the `_2` prefix is the bijective mangled encoding of `array_u32_4`). Similarly, `[f32; 4]` produces `_2array_f32_4_zero`.

The scalar form works the same way: `<u32>::zero()` transpiles to `u32_zero`, and inside generic functions `<[T; 4]>::zero()` is rewritten during monomorphization — see the [QSelf call syntax example](../examples/qself-call.md) for the full picture. The `<T as Trait>::method()` disambiguation form is rejected (trait impls are matched by self type only), as is turbofish on a trait path's method segment (`Zeroable::zero::<[u32; 4]>()`) — use the qualified-self form shown above instead.

## `PhantomData<T>` Marker Fields

A generic struct may carry `PhantomData<T>` fields as type-parameter markers (e.g. for slab-id tags or type-level metadata). `PhantomData` is re-exported from `wgsl_rs::std` so the glob import brings it into scope. The proc-macro recognizes `PhantomData<_>` fields specially: they are **retained in the IR** (so extensions can observe which type parameter each phantom slot binds) but **omitted from the rendered WGSL**:

```rust
#[wgsl]
pub mod phantom_example {
    use wgsl_rs::std::*;

    pub struct Id<T> {
        pub index: u32,
        pub phantom: PhantomData<T>,
    }

    pub struct Tagged<T, A> {
        pub x: f32,
        pub t: PhantomData<T>,
        pub a: PhantomData<A>,
    }

    pub fn make_id() -> Id<f32> {
        Id { index: 0u32, phantom: PhantomData }
    }

    pub fn make_tagged() -> Tagged<f32, u32> {
        Tagged { x: 1.0, t: PhantomData, a: PhantomData }
    }
}
```

The rendered WGSL drops phantom fields entirely — `Id_f32` has only `index: u32`, and `Tagged_f32_u32` has only `x: f32`:

```wgsl
struct Id_f32 {
    index: u32
}

struct Tagged_f32_u32 {
    x: f32
}
```

Construction expressions use the bare `PhantomData` value (no turbofish). The macro strips `PhantomData` from the positional constructor call so the rendered arity matches the non-phantom field count: `Id { index: 0u32, phantom: PhantomData }` becomes `Id_f32(0u)`.

> **Why retain phantom fields in the IR?** Extensions consuming the IR via [`WgslExtension::modify_ir`](../extensions/trait.md) must be able to see the full type-parameter binding structure of a generic struct. If phantom fields were skipped at parse time, an extension inspecting `struct Tagged<T, A>` would see `type_params: ["T", "A"]` but only `{ x: f32 }`, with no way to recover which phantom slot bound which parameter. Keeping `Type::Phantom { elem }` in the IR preserves the T↔field provenance.