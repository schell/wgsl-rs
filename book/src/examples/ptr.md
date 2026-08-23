# Pointer

Demonstrates pointer types in function parameters via the `ptr!` macro. Pointers translate to WGSL `ptr<function, T>` parameters.

## Rust Source

```rust
#[wgsl]
#[allow(dead_code, clippy::manual_swap, clippy::assign_op_pattern)]
pub mod ptr_example {
    //! Demonstrates pointer types in function parameters.
    use wgsl_rs::std::*;

    // Increment a value through a pointer.
    pub fn increment(p: ptr!(function, i32)) {
        *p += 1;
    }

    // Swap two values through pointers.
    // Note: We use manual swap because std::mem::swap is not available in WGSL.
    pub fn swap(a: ptr!(function, f32), b: ptr!(function, f32)) {
        let tmp = *a;
        *a = *b;
        *b = tmp;
    }

    // Double a value in-place through a pointer.
    // Note: We use *p = *p * 2.0 instead of *p *= 2.0 to demonstrate dereference.
    pub fn double_value(p: ptr!(function, f32)) {
        *p = *p * 2.0;
    }

    #[fragment]
    pub fn test_ptr() -> Vec4f {
        let mut x: i32 = 5;
        increment(&mut x);
        // x is now 6

        let mut a: f32 = 1.0;
        let mut b: f32 = 2.0;
        swap(&mut a, &mut b);
        // a is now 2.0, b is now 1.0

        let mut c: f32 = 3.0;
        double_value(&mut c);
        // c is now 6.0

        vec4f(f32(x), a, b, c / 10.0)
    }
}
```

## Generated WGSL

```wgsl
fn increment(p: ptr<function, i32>) {
    *p += 1;
}

fn swap(a: ptr<function, f32>, b: ptr<function, f32>) {
    let tmp = *a;
    *a = *b;
    *b = tmp;
}

fn double_value(p: ptr<function, f32>) {
    *p = *p * 2.0;
}

@fragment fn test_ptr() -> @location(0) vec4f {
    var x: i32 = 5;
    increment(&x);
    var a: f32 = 1.0;
    var b: f32 = 2.0;
    swap(&a, &b);
    var c: f32 = 3.0;
    double_value(&c);
    return vec4f(f32(x), a, b, c / 10.0);
}
```

## Notes

- `ptr!(function, i32)` expands to the WGSL pointer type `ptr<function, i32>`.
- `&mut x` at the call site becomes `&x` in WGSL (WGSL pointers do not distinguish mutability in the reference syntax).