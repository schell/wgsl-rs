# Scalars & Literals

wgsl-rs shares the four WGSL scalar types with Rust directly. The same names mean the same thing in both worlds.

| Type   | Rust     | WGSL     | Notes                          |
| ------ | -------- | -------- | ------------------------------ |
| `f32`  | 32-bit float | `f32`   | IEEE 754 single precision     |
| `i32`  | signed int   | `i32`   | 32-bit two's complement       |
| `u32`  | unsigned int | `u32`   | 32-bit                         |
| `bool` | boolean      | `bool`  | `true` / `false`              |

Because shader code must type-check as ordinary Rust, scalar types are not aliases: they are the literal Rust primitive types. The transpiler maps them onto the matching WGSL keyword.

## Literal Suffixes

Rust literal suffixes carry type information into the generated WGSL. Always suffix literals when the surrounding context does not pin the type (e.g. constants, function arguments to generic constructors).

```rust
const WIDTH: u32 = 1024u32;
const EPS: f32 = 1e-5f32;
let count: i32 = 0i32;
```

Unsuffixed integer literals are accepted by Rust and inferred from context, but explicit suffixes make the transpiler's job unambiguous and the generated WGSL easier to read.

## `as` Casts

Rust's `as` cast operator transpiles to a WGSL conversion expression of the same form.

```rust
let i: i32 = 7;
let u: u32 = i as u32;     // -> u32(i)
let f: f32 = u as f32;     // -> f32(u)
let n: f32 = i as f32;     // -> f32(i)
```

Cross-kind conversions (`i32` <-> `u32` <-> `f32`) all generate the corresponding WGSL `T(x)` conversion. Booleans cannot be cast with `as`; use `select` or a manual comparison instead.