# Control Flow

wgsl-rs supports the usual Rust control-flow constructs. Each transpiles to the corresponding WGSL statement.

## `if` / `else if` / `else`

```rust
pub fn classify(x: f32) -> u32 {
    if x > 0.0 {
        1u32
    } else if x < 0.0 {
        2u32
    } else {
        0u32
    }
}
```

## `while`

```rust
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
```

## `loop` (Infinite Loop)

Rust's bare `loop` transpiles to a WGSL `loop`. Use `break` to exit:

```rust
pub fn first_zero(arr: Vec4f) -> u32 {
    let mut i: u32 = 0;
    loop {
        if i >= 4 { break; }
        if arr[i as usize] == 0.0 { return i; }
        i += 1;
    }
    i
}
```

## `for`

Exclusive range (`0..N`) and inclusive range (`0..=N`) are both supported:

```rust
pub fn sum_to(n: u32) -> u32 {
    let mut s: u32 = 0;
    for i in 0..n {
        s += i;
    }
    s
}

pub fn sum_inclusive(n: u32) -> u32 {
    let mut s: u32 = 0;
    for i in 0..=n {
        s += i;
    }
    s
}
```

Loop bounds must be literals or `const`. For variable bounds, annotate the expression with `#[wgsl_allow(non_literal_loop_bounds)]`:

```rust
pub fn partial_sum(n: u32) -> u32 {
    let mut s: u32 = 0;
    #[wgsl_allow(non_literal_loop_bounds)]
    for i in 0..n {
        s += i;
    }
    s
}
```

> **Why does this warning exist?**
> WGSL `for` loops require explicit, compile-time-known bounds — the spec mandates that loop iteration bounds be literal or const expressions so the shader compiler can reason about termination and resource limits. When `wgsl-rs` transpiles `for i in 0..n`, it emits `for (var i = 0; i < n; i++)`. If `n` is a runtime value, the bound cannot be verified at macro time to be ascending (or even finite), so on stable Rust the macro emits a **compile error** — proc-macro warnings aren't possible on stable. On nightly it emits a warning instead. The `#[wgsl_allow(non_literal_loop_bounds)]` attribute suppresses both: you're telling `wgsl-rs` you've ensured the bound is valid at runtime, taking responsibility the compiler can't.

## `match` (WGSL `switch`)

`match` on an integer-typed value transpiles to a WGSL `switch`:

```rust
#[repr(u32)]
pub enum Op { Add = 0, Sub = 1, Mul = 2 }

pub fn apply(op: Op, a: f32, b: f32) -> f32 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        _ => 0.0,
    }
}
```

Or-patterns and non-literal patterns require `#[wgsl_allow(non_literal_match_statement_patterns)]`:

```rust
pub fn is_zero(op: Op) -> bool {
    #[wgsl_allow(non_literal_match_statement_patterns)]
    match op {
        Op::Add | Op::Sub => false,
        _ => true,
    }
}
```

> **Why does this warning exist?**
> WGSL `switch` case selectors must be literal integer constants — the spec doesn't allow arbitrary expressions or enum variants as case labels. When `wgsl-rs` transpiles a Rust `match`, it maps each arm to a `switch` case. Rust enum variants (`Op::Add`) and const references (`LOW`) are not WGSL literals — they're names that resolve at the Rust or IR level, not at WGSL compile time. On stable, the macro can't emit a warning (proc-macro diagnostics are nightly-only), so it emits a **compile error** instead. The `#[wgsl_allow(non_literal_match_statement_patterns)]` attribute suppresses it: you're asserting the patterns are valid WGSL case selectors once resolved (e.g. `#[repr(u32)]` enum variants become literal `u32` values, consts become literal integers).

## `break`, `continue`, `return`

All three are supported inside loops and functions as in Rust. `break` and `continue` work in `while`, `loop`, and `for`. `return` works in any function and supports early returns (see [Functions](./functions.md)).