# Generic Functions

Demonstrates generic functions with monomorphization. Trait bounds (`Copy + std::ops::Add`) are required for Rust type-checking but produce no WGSL output. Each concrete call site triggers generation of a specialized WGSL function.

## Rust Source

```rust
#[wgsl]
pub mod generic_functions {
    /// A generic function that doubles a value via addition.
    ///
    /// Trait bounds (`Copy + std::ops::Add`) are required for Rust to
    /// type-check the generic body. They produce no WGSL output.
    pub fn double<T: Copy + std::ops::Add<Output = T>>(x: T) -> T {
        x + x
    }

    /// A generic "select" function: returns `a` if `cond` is true, else `b`.
    pub fn select_val<T: Copy>(a: T, b: T, cond: bool) -> T {
        if cond { a } else { b }
    }

    /// A generic function calling another generic function (transitive
    /// monomorphization). Demonstrates nested turbofish: `double::<T>(x)`.
    pub fn double_or_keep<T: Copy + std::ops::Add<Output = T>>(x: T, use_double: bool) -> T {
        select_val::<T>(double::<T>(x), x, use_double)
    }

    /// Concrete function that calls the generic helpers with `f32`.
    pub fn apply_f32(value: f32) -> f32 {
        double_or_keep::<f32>(value, true)
    }

    /// Concrete function that calls the generic helpers with `i32`.
    pub fn apply_i32(value: i32) -> i32 {
        double_or_keep::<i32>(value, false)
    }
}
```

## Generated WGSL

```wgsl
fn apply_f32(value: f32) -> f32 {
    return _2double_or_keep_f32(value, true);
}

fn apply_i32(value: i32) -> i32 {
    return _2double_or_keep_i32(value, false);
}

fn _2double_or_keep_f32(x: f32, use_double: bool) -> f32 {
    return _1select_val_f32(double_f32(x), x, use_double);
}

fn _2double_or_keep_i32(x: i32, use_double: bool) -> i32 {
    return _1select_val_i32(double_i32(x), x, use_double);
}

fn _1select_val_f32(a: f32, b: f32, cond: bool) -> f32 {
    if cond {
        return a;
    } else {
        return b;
    }
}

fn double_f32(x: f32) -> f32 {
    return x + x;
}

fn _1select_val_i32(a: i32, b: i32, cond: bool) -> i32 {
    if cond {
        return a;
    } else {
        return b;
    }
}

fn double_i32(x: i32) -> i32 {
    return x + x;
}
```

## Notes

- Each generic function is monomorphized per concrete type used, producing specialized WGSL functions suffixed with the type (e.g. `double_f32`, `double_i32`).
- Trait bounds are erased; only the monomorphized bodies appear in WGSL.
- Leading underscores in mangled names (`_1`, `_2`) are used to avoid collisions with user-defined names.