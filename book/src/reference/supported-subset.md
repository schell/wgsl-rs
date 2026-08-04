# Supported Rust Subset

wgsl-rs transpiles a deliberately constrained subset of Rust to WGSL. The macro is **additive**: it never translates or rewrites user code. Constructs that cannot map cleanly to WGSL are rejected at compile time rather than approximated.

## Supported

- Structs (including generic structs)
- Enums with `#[repr(u32)]`
- `impl` blocks (free functions in WGSL)
- Free functions
- `const` items
- `let` and `let mut` bindings
- `if` / `else`, `while`, `loop`, `for`, `match`
- All binary, unary, and compound assignment operators
- Arrays
- Generic functions and generic structs

## Not Supported

| Feature             | Reason                                                  | Workaround                                   |
|---------------------|---------------------------------------------------------|----------------------------------------------|
| Trait definitions   | Traits are Rust-only; impls generate WGSL functions     | Use concrete types in function signatures    |
| Trait impls         | No dynamic dispatch in WGSL                             | Write free functions or impl blocks          |
| Borrowing / refs    | WGSL has no borrow semantics                            | Use the `ptr!` macro for pointer types       |
| Arbitrary imports   | Module mapping is glob-only                             | Use `use crate::module::*;`                  |
| Closures            | No closure capture model in WGSL                        | Write named functions                        |
| `async`             | No async runtime on GPU                                 | —                                            |
| Dynamic dispatch    | No vtables in WGSL                                      | Use enums or monomorphization                |

## Feature Table

| Feature                | Supported? | Notes                                              |
|------------------------|------------|----------------------------------------------------|
| Structs                | Yes        | Including generics                                 |
| Enums                  | Yes        | Requires `#[repr(u32)]`                            |
| `impl` blocks          | Yes        | Become free WGSL functions                         |
| Free functions         | Yes        |                                                    |
| `const` items          | Yes        |                                                    |
| `let` / `let mut`      | Yes        |                                                    |
| `if` / `else`          | Yes        |                                                    |
| `while`                | Yes        |                                                    |
| `loop`                 | Yes        |                                                    |
| `for`                  | Yes        |                                                    |
| `match`                | Yes        | See `non_literal_match_statement_patterns` allow   |
| Binary operators       | Yes        |                                                    |
| Unary operators        | Yes        |                                                    |
| Compound assignments   | Yes        |                                                    |
| Arrays                 | Yes        |                                                    |
| Generic functions      | Yes        | Monomorphized at macro time                        |
| Generic structs        | Yes        | Monomorphized                                      |
| Traits                 | No         | Definitions are Rust-only                          |
| Borrowing / references | No         | Use `ptr!` macro                                   |
| Arbitrary imports      | No         | Glob only                                          |
| Closures               | No         |                                                    |
| `async`                | No         |                                                    |
| Dynamic dispatch       | No         |                                                    |