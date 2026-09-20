# Operators

wgsl-rs transpiles Rust operators to their WGSL equivalents. Most have a 1:1 mapping.

## Arithmetic

| Rust | WGSL | Description |
| --- | --- | --- |
| `a + b` | `a + b` | addition |
| `a - b` | `a - b` | subtraction |
| `a * b` | `a * b` | multiplication |
| `a / b` | `a / b` | division |
| `a % b` | `a % b` | remainder |

## Comparison

| Rust | WGSL |
| --- | --- |
| `a == b` | `a == b` |
| `a != b` | `a != b` |
| `a < b` | `a < b` |
| `a <= b` | `a <= b` |
| `a > b` | `a > b` |
| `a >= b` | `a >= b` |

## Logical

| Rust | WGSL |
| --- | --- |
| `a && b` | `a && b` |
| `a \|\| b` | `a \|\| b` |
| `!a` | `!a` |

## Bitwise

| Rust | WGSL | Description |
| --- | --- | --- |
| `a & b` | `a & b` | bitwise and |
| `a \| b` | `a \| b` | bitwise or |
| `a ^ b` | `a ^ b` | bitwise xor |
| `a << n` | `a << n` | shift left |
| `a >> n` | `a >> n` | shift right |

## Precedence & Parentheses

Rust and WGSL disagree about the relative precedence of the bitwise and
shift operators versus the comparisons. Rust binds `&`, `^`, `|`, `<<`
and `>>` tighter than `==`, `!=`, `<`, `<=`, `>` and `>=`; WGSL binds the
comparisons tighter. In Rust, `x & mask == flags` means
`(x & mask) == flags` — while the same text in WGSL would mean
`x & (mask == flags)`, which does not even type-check.

To keep the two languages from disagreeing, wgsl-rs emits every binary
expression parenthesized, so the generated WGSL always evaluates exactly
like the Rust you wrote:

```rust
pub fn hit(x: u32) -> bool {
    x & 3 == 3
}
```

```wgsl
fn hit(x: u32) -> bool {
    return ((x & 3) == 3);
}
```

Parentheses you write yourself are kept — `!(a & b)` stays
`!(a & b)` — and the generated code may contain parentheses that look
redundant (`(a + b)` where `a + b` would do). That is harmless: it
guarantees your expression structure survives WGSL's precedence rules
exactly as you wrote it in Rust. See
[issue #159](https://github.com/schell/wgsl-rs/issues/159) for the
original report.

## Compound Assignment

`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=` all transpile directly:

```rust
pub fn bump(x: ptr!(function, f32)) {
    *x += 1.0;
    *x *= 2.0;
}
```

> Mutable references in function signatures must use the [`ptr!`](./binding-macros/ptr.md) macro — bare `&mut T` parameters are not supported in `#[wgsl]` modules.

## `select`

`select(false_val, true_val, condition)` maps to the WGSL `select` builtin. Argument order matches WGSL:

```rust
pub fn abs_or(x: f32, sign: bool) -> f32 {
    select(-x, x, sign)
}
```

```wgsl
fn abs_or(x: f32, sign: bool) -> f32 {
  return select(-x, x, sign);
}
```

## Unary

| Rust | WGSL | Description |
| --- | --- | --- |
| `-a` | `-a` | negation |
| `!a` | `!a` | logical/bitwise not |
| `*p` | `*p` | dereference (for `ptr!`) |

## `as` Casts

`as` casts are transpiled when meaningful in WGSL. The common case is `as usize` for array indexing — this is stripped in WGSL, which uses the index directly:

```rust
pub fn at(arr: Vec4f, i: u32) -> f32 {
    arr[i as usize]
}
```

```wgsl
fn at(arr: vec4<f32>, i: u32) -> f32 {
  return arr[i];
}
```