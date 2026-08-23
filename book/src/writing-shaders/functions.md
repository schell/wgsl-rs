# Functions

Functions are the basic unit of shader logic. They are written as ordinary Rust functions inside a `#[wgsl]` module.

## Syntax

```rust
pub fn name(arg: T, arg2: U) -> R {
    body
}
```

All functions are `pub fn`. The return type is mandatory unless the function is void (use `-> ()` or omit the arrow for a trailing-statement body).

```rust
#[wgsl]
pub mod funcs {
    use wgsl_rs::std::*;

    pub fn add(a: f32, b: f32) -> f32 {
        a + b
    }

    pub fn noop(x: f32) {
        let _ = x;
    }
}
```

## `let` and `let mut`

- `let x: T = ...` transpiles to a WGSL `let`.
- `let mut x: T = ...` transpiles to a WGSL `var`.

```rust
pub fn accumulate(items: Vec4f) -> f32 {
    let mut sum: f32 = 0.0;
    sum += items.x + items.y + items.z + items.w;
    sum
}
```

## Early Returns and Implicit Return

Early `return` is supported:

```rust
pub fn first_nonzero(v: Vec3f) -> f32 {
    if v.x != 0.0 { return v.x; }
    if v.y != 0.0 { return v.y; }
    v.z
}
```

A trailing expression without a semicolon is the implicit return value, matching Rust.

## `const` Inside Functions

Function-scoped `const` items are supported and transpile to WGSL `const`:

```rust
pub fn area(r: f32) -> f32 {
    const PI: f32 = 3.14159265;
    PI * r * r
}
```

## Function Arguments with IO Annotations

Entry-point and inter-stage functions may carry IO annotations on arguments (`#[location(N)]`, `#[builtin(position)]`, `#[interpolate(flat)]`, etc.). See [Entry Points](../entry-points/stages.md) and [Inter-Stage IO](../entry-points/inter-stage-io.md) for the full list of builtins and annotations.

```rust
#[vertex]
pub fn vs_main(
    #[location(0)] pos: Vec3f,
    #[location(1)] uv: Vec2f,
) -> Vec4f {
    vec4f(pos, 1.0)
}
```

## Pointer Parameters

WGSL functions can take pointer arguments so the callee can mutate the caller's
local or workgroup variable. In `wgsl-rs` you express this with the
[`ptr!`](./binding-macros/ptr.md) macro — bare `&mut T` parameters are **not**
supported inside `#[wgsl]` modules.

```rust
pub fn increment(p: ptr!(function, i32)) {
    *p += 1;
}
```

`ptr!(address_space, T)` expands to `&mut T` in Rust (so the code runs on the
CPU) and transpiles to `ptr<address_space, T>` in WGSL. The supported address
spaces are:

| Address space | WGSL              | Use case                                  |
|---------------|-------------------|-------------------------------------------|
| `function`    | `ptr<function, T>`  | Local variables (`let mut x`)            |
| `private`     | `ptr<private, T>`   | Module-scope private variables           |
| `workgroup`   | `ptr<workgroup, T>` | Workgroup-shared variables (`workgroup!`) |

Dereference with `*p`, and pass a mutable reference with `&mut x`:

```rust
pub fn swap(a: ptr!(function, f32), b: ptr!(function, f32)) {
    let tmp = *a;
    *a = *b;
    *b = tmp;
}

pub fn caller() {
    let mut x: f32 = 1.0;
    let mut y: f32 = 2.0;
    swap(&mut x, &mut y);
}
```

> Both `&x` and `&mut x` transpile to `&x` in WGSL — mutability is determined
> by the access mode in the pointer type, not by the reference syntax. The
> `ptr!` macro always produces a `&mut T` on the Rust side so the CPU path
> can mutate the value.