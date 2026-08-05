# Design Decisions

This chapter curates the key architectural decisions behind wgsl-rs. The full narrative, including rejected alternatives and intermediate experiments, is in the [DEVLOG.md](https://github.com/schell/wgsl-rs/blob/main/DEVLOG.md).

## Core Philosophy

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2025-12-08   | **User code is never translated.** The macro is strictly additive; users write regular Rust and the macro only attaches WGSL metadata. |
| 2025-12-27   | **Rust type system catches all WGSL errors.** Type mismatches surface as Rust compile errors, not runtime shader errors. |
| 2025-12-27   | **Macros for non-Rust WGSL constructs.** `uniform!`, `ptr!`, and similar macros stand in for WGSL constructs that have no Rust analogue. |

## Types & Syntax

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2025-12-08   | **Swizzles are function calls**, not field access, because the macro never alters user code and Rust field access cannot return a different type without translation. |
| 2025-12-27   | **Module imports are glob-only.** `use crate::module::*;` keeps the transpiler's symbol resolution simple and avoids modeling Rust's visibility rules. |
| 2026-01-29   | **Pointer types via `ptr!` macro.** WGSL pointers have no Rust equivalent; a dedicated macro expresses them without inventing reference semantics. |
| 2026-01-31   | **Atomic types and workgroup variables** get first-class IR nodes and macros. |
| 2026-02-11   | **Variadic WGSL builtins** are handled by multi-function name mapping rather than variadic generics. |
| 2026-03-16   | **`discard!()`** is implemented via a thread-local flag rather than a control-flow primitive, preserving the "no code translation" rule. |

## Generics

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2026-04-08   | **Generic functions are monomorphized at macro time.** Each concrete instantiation becomes a distinct WGSL function. |
| 2026-04-17   | **Generic structs are monomorphized.** Same principle, applied to struct definitions. |
| 2026-05-06   | **Generic linkages** use template modules that are instantiated per concrete type set. |

## IR & Extensions

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2026-05-07   | **IR crate for runtime type substitution**, replacing earlier string-placeholder schemes. |
| 2026-05-15   | **`WgslExtension` trait and IR attributes** provide a stable post-transpile hook for downstream code generation. |
| 2026-05-18   | **Bijective name mangling** ensures Rust names map to unique WGSL names and back without collisions. |
| 2026-07-18   | **`ir::Module` is the AST; `wgsl_rs::Source` is the spec.** The IR is the authoritative structure; `Source` is the user-facing handle. |

## Linkage

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2026-05-29   | **`wgsl-rs-layout` is a standalone extension crate**, dogfooding the extension mechanism to compute WGSL memory layout. |
| 2026-06-06   | **Runtime wgpu linkage via IR traversal** reflects bind groups and bindings by walking the IR rather than parsing generated WGSL text. |

## Generics & Monomorphization

| Date         | Decision                                                                                  |
|--------------|-------------------------------------------------------------------------------------------|
| 2026-04-08   | **Generic functions monomorphized at macro time.** Each turbofish call-site produces a mangled, concrete WGSL function. |
| 2026-04-17   | **Generic structs monomorphized** to concrete WGSL structs with mangled names. |
| 2026-08-02   | **Trait impls on complex types** (e.g. `impl Zeroable for [u32; 4]`) transpile to mangled WGSL functions. |
| 2026-08-04   | **Generic trait impls on array types** (`impl<T: Trait> Trait for [T; N]`) supported via monomorphizer widening (#133). |
| 2026-08-04   | **Const generics for `u32`/`usize`** supported on functions, structs, impl blocks, and template entry points (#137). The substitution target is always a bare ident (stable Rust requires bare idents or literals), so no new IR variant is needed. |