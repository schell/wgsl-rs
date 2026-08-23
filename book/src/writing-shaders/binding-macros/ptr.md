# `ptr!`

Declares a WGSL pointer parameter. Used in function signatures where the function needs to read or write a variable in a specific address space.

## Syntax

```rust
fn name(p: ptr!(address_space, Type)) { ... }
```

`address_space` is one of `function`, `private`, or `workgroup`.

## What It Generates

Rust:

```rust
fn name(p: &mut Type) { ... }
```

WGSL:

```wgsl
fn name(p: ptr<address_space, Type>) { ... }
```

## Dereference

Read or write through the pointer with `*p`:

```rust
pub fn increment(p: ptr!(function, f32)) {
    *p += 1.0;
}
```

```wgsl
fn increment(p: ptr<function, f32>) {
  *p += 1.0;
}
```

## Example: Swap

```rust
#[wgsl]
pub mod utils {
    use wgsl_rs::std::*;

    pub fn swap(a: ptr!(function, f32), b: ptr!(function, f32)) {
        let t = *a;
        *a = *b;
        *b = t;
    }

    pub fn sort_pair(mut x: f32, mut y: f32) -> Vec2f {
        if x > y {
            swap(&mut x, &mut y);
        }
        vec2f(x, y)
    }
}
```

## Notes

- Use `&mut` at the call site in Rust; the macro translates this to a WGSL pointer of the declared address space.
- `function` is the most common address space for local variables. Use `workgroup` for pointers to `workgroup!` variables and `private` for module-private variables.