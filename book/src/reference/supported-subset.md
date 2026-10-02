# Supported Rust Subset

wgsl-rs transpiles a deliberately constrained subset of Rust to WGSL.
The macro is **additive**: it never translates or rewrites user code.
Constructs that cannot map cleanly to WGSL are rejected at **parse time** rather than approximated.

Traits are the one construct that straddles this line: **definitions are
Rust-only** (they produce no WGSL output), while **trait impls generate WGSL
free functions** that monomorphization resolves. The [Traits](#traits) section
below covers the exact forms that work.

## Supported

- Structs (including generic and const-generic structs)
- Enums with `#[repr(u32)]`
- Trait definitions, used as bounds for generics (Rust-only)
- `impl` blocks (free functions in WGSL), including trait impls, generic impls,
  and generic trait impls on fixed-length array types
- Free functions
- `const` items
- `let` and `let mut` bindings
- `if` / `else`, `while`, `loop`, `for`, `match`
- All binary, unary, and compound assignment operators
- Arrays
- Generic functions and generic structs (monomorphized)
- Const generic parameters (`const N: usize` / `const N: u32`) on functions,
  structs, impl blocks, and template entry points
- Direct QSelf calls (`<T>::method()`, e.g. `<[u32; 4]>::zero()`)

## Traits

WGSL has no methods and no vtables, so wgsl-rs treats traits as **compile-time
machinery**: definitions exist so the Rust type checker can validate generic
code on the CPU, impls become ordinary WGSL free functions, and monomorphization
rewrites every call to the concrete function. There is no runtime dispatch.

### What works

- **Trait definitions** type-check as usual Rust but produce **no WGSL
  output**. `Self` may appear in signatures. Default method bodies and
  default associated const values are **rejected at parse time**: only trait
  impls generate WGSL, so every method and const called from shader code
  needs a concrete `impl`. Mark a trait
  [`#[wgsl_ignore]`](../writing-shaders/the-wgsl-macro.md#wgsl_ignore) if it
  is CPU-only and needs defaults.
- **Trait impls** (`impl Addable for f32`) turn each method into a free
  function named `Type_method`, exactly like inherent impl blocks. Items may
  omit `pub` (Rust itself forbids `pub` in trait impls).
- **Complex self types** are mangled: `impl Zeroable for [u32; 4]` emits
  `_2array_u32_4_zero`. Fully-applied generic structs work too
  (`impl Zeroable for Pair<f32>`).
- **Associated consts** in trait impls emit mangled constants
  (`impl SlabItem for u32 { const SLAB_SIZE ... }` emits `u32__1SLAB_SIZE`).
- **Associated type aliases** in impl blocks resolve to concrete WGSL types
  and emit `alias` declarations.
- **Generic trait impls on fixed-length arrays**
  (`impl<T: Zeroable> Zeroable for [T; 4]`) are monomorphized per concrete
  element type.
- **Method calls** resolve by self type only; the trait named in the impl is
  discarded:
  - Inside generic functions, `T::method(args)` rewrites to
    `ConcreteType_method(args)` after monomorphization.
  - At concrete call sites, use the QSelf form: `<[u32; 4]>::zero()`.
  - Associated consts work the same way: `<Tag>::DEFAULT`.

### Forms

| Form | Supported | Notes |
|------|-----------|-------|
| `trait MyTrait { fn m(a: Self) -> Self; }` | ✅ | Rust-only; produces no WGSL |
| `trait MyTrait { fn m() -> u32 { ... } }` | ❌ | Default bodies are rejected at parse time; implement in every impl |
| `trait MyTrait { const N: usize = 4; }` | ❌ | Default const values are rejected; provide in every impl |
| `impl MyTrait for f32 { ... }` | ✅ | Methods become `f32_m` |
| `impl MyTrait for [u32; 4] { ... }` | ✅ | Mangled to `_2array_u32_4_m` |
| `impl MyTrait for Pair<f32> { ... }` | ✅ | Fully-applied struct self type |
| `impl<T: MyTrait> MyTrait for [T; 4]` | ✅ | Monomorphized per element type |
| `impl<T, const N: usize> MyTrait for [T; N]` | ❌ | Const-generic array impls unsupported (#133) |
| `T::method(args)` inside a generic fn | ✅ | Rewritten after monomorphization |
| `<[u32; 4]>::method(args)` | ✅ | Direct QSelf call |
| `<Tag>::DEFAULT` | ✅ | Associated const access |
| `<T as MyTrait>::method(args)` | ❌ | Trait path is discarded; use `<T>::method(args)` |
| `MyTrait::method::<[u32; 4]>()` | ❌ | No turbofish on the method segment; use QSelf |
| `x.method(args)` receiver call | ❌ | Only swizzles (`v.xyz()`, `v.set_xyz(...)`) |
| `dyn MyTrait` | ❌ | No dynamic dispatch in WGSL |

### The specialization pattern

Instead of preprocessor variants (`#ifdef USE_PBR`), define one trait per axis
of variation, implement it per strategy, and write a single generic shader that
monomorphizes per configuration. The [Renderer Specialization](../examples/renderer-specialization.md)
example demonstrates this end to end.

For the underlying mechanics, see [Generic Functions](../generics/generic-functions.md),
[Generic Structs](../generics/generic-structs.md), [Trait Impls](../examples/trait-impls.md),
and [QSelf Call Syntax](../examples/qself-call.md).

## Not Supported

| Feature             | Reason                                                  | Workaround                                   |
|---------------------|---------------------------------------------------------|----------------------------------------------|
| Trait definitions in WGSL | Definitions are Rust-only; impls generate the functions | Use traits freely as bounds; every called method needs a concrete impl |
| Borrowing / refs    | WGSL has no borrow semantics                            | Use the `ptr!` macro for pointer types       |
| Arbitrary imports   | Module mapping is glob-only                             | Use `use crate::module::*;`                  |
| Closures            | No closure capture model in WGSL                        | Write named functions                        |
| `async`             | No async runtime on GPU                                 | n/a                                          |
| Dynamic dispatch    | No vtables in WGSL                                      | Use enums or monomorphization                |

## Feature Table

| Feature                | Supported? | Notes                                              |
|------------------------|------------|----------------------------------------------------|
| Structs                | ✅ | Including generic and const-generic                |
| Enums                  | ✅ | Requires `#[repr(u32)]`                            |
| `impl` blocks          | ✅ | Become free WGSL functions                         |
| Trait definitions      | Rust-only  | No WGSL output; bounds for the Rust type checker    |
| Default method bodies  | ❌ | Rejected at parse time; `#[wgsl_ignore]` for CPU-only traits |
| Trait impls            | ✅ | Methods become mangled WGSL functions              |
| Generic array trait impls | ✅ | `impl<T: Trait> Trait for [T; 4]`; const-generic lengths unsupported (#133) |
| Associated consts in trait impls | ✅ | Mangled `Type_MEMBER`; `pub` not required  |
| Associated type aliases | ✅ | Emit WGSL `alias` declarations                     |
| `T::method()` in generic fns | ✅ | Resolved via monomorphization                |
| QSelf call syntax      | ✅ | `<[u32; 4]>::zero()`; `<T as Trait>::` is rejected  |
| Receiver method calls | ❌ | Swizzles only                                      |
| `dyn Trait`            | ❌ | No dynamic dispatch                                |
| Free functions         | ✅ |                                                    |
| `const` items          | ✅ |                                                    |
| `let` / `let mut`      | ✅ |                                                    |
| `if` / `else`          | ✅ |                                                    |
| `while`                | ✅ |                                                    |
| `loop`                 | ✅ |                                                    |
| `for`                  | ✅ |                                                    |
| `match`                | ✅ | See `non_literal_match_statement_patterns` allow   |
| Binary operators       | ✅ |                                                    |
| Unary operators        | ✅ |                                                    |
| Compound assignments   | ✅ |                                                    |
| Arrays                 | ✅ |                                                    |
| Generic functions      | ✅ | Monomorphized at macro time                        |
| Generic structs        | ✅ | Monomorphized                                      |
| Const generics         | ✅ | `u32`/`usize` only, monomorphized or instantiated   |
| Borrowing / references | ❌ | Use `ptr!` macro                                   |
| Arbitrary imports      | ❌ | Glob only                                          |
| Closures               | ❌ |                                                    |
| `async`                | ❌ |                                                    |
| Dynamic dispatch       | ❌ |                                                    | 
