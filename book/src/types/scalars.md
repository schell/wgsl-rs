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

Rust literal suffixes are translated, not copied: `1024u32` renders as `1024u`, `0i32` as `0i`.

```rust
const WIDTH: u32 = 1024u32;   // -> const WIDTH: u32 = 1024u;
let count: i32 = 0i32;         // -> let count: i32 = 0i;
```

Unsuffixed integer literals are fine in typed contexts. Rust infers them from context, and so does the transpiler: it propagates the expected type — return types, variable annotations, array element types, function parameter types, builtin signatures — into the literal and inserts the matching WGSL suffix for you:

```rust
fn to_array(data: bool) -> [u32; 1] {
    [select(0, 1, data)]      // -> array(select(0u, 1u, data))
}

fn clamp_to(x: u32) -> u32 {
    min(x, 4095)              // -> min(x, 4095u)
}
```

Explicit suffixes remain valid and never hurt. They are still useful where no context pins a type — a bare `let` without an annotation defaults to `i32` in both Rust and WGSL.

## `as` Casts

Rust's `as` cast operator transpiles to a WGSL conversion expression of the same form.

```rust
let i: i32 = 7;
let u: u32 = i as u32;     // -> u32(i)
let f: f32 = u as f32;     // -> f32(u)
let n: f32 = i as f32;     // -> f32(i)
```

Cross-kind conversions (`i32` <-> `u32` <-> `f32`) all generate the corresponding WGSL `T(x)` conversion. Booleans cannot be cast with `as`; use `select` or a manual comparison instead.