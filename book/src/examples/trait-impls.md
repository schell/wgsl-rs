# Trait Impls

Demonstrates trait definitions and `impl` blocks resolved via monomorphization. The trait definition itself produces no WGSL output; each `impl` method becomes a free function named `Type_method`, and generic calls resolve to the concrete function after monomorphization.

## Rust Source

```rust
#[wgsl]
pub mod trait_impl_example {
    /// A trait for types that support an "add" operation.
    /// This definition is Rust-only — it produces no WGSL output.
    pub trait Addable {
        fn add(a: Self, b: Self) -> Self;
    }

    impl Addable for f32 {
        fn add(a: f32, b: f32) -> f32 {
            a + b
        }
    }

    impl Addable for i32 {
        fn add(a: i32, b: i32) -> i32 {
            a + b
        }
    }

    /// Generic function that sums three values using the trait method.
    /// `T::add(a, b)` resolves to `f32_add(a, b)` or `i32_add(a, b)`
    /// after monomorphization.
    pub fn sum_three<T: Addable>(a: T, b: T, c: T) -> T {
        let ab = T::add(a, b);
        T::add(ab, c)
    }

    /// Concrete caller — triggers monomorphization of `sum_three::<f32>`.
    pub fn sum_f32(a: f32, b: f32, c: f32) -> f32 {
        sum_three::<f32>(a, b, c)
    }

    /// Concrete caller — triggers monomorphization of `sum_three::<i32>`.
    pub fn sum_i32(a: i32, b: i32, c: i32) -> i32 {
        sum_three::<i32>(a, b, c)
    }
}
```

## Generated WGSL

```wgsl
fn f32_add(a: f32, b: f32) -> f32 {
    return a + b;
}

fn i32_add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn sum_f32(a: f32, b: f32, c: f32) -> f32 {
    return _1sum_three_f32(a, b, c);
}

fn sum_i32(a: i32, b: i32, c: i32) -> i32 {
    return _1sum_three_i32(a, b, c);
}

fn _1sum_three_f32(a: f32, b: f32, c: f32) -> f32 {
    let ab = f32_add(a, b);
    return f32_add(ab, c);
}

fn _1sum_three_i32(a: i32, b: i32, c: i32) -> i32 {
    let ab = i32_add(a, b);
    return i32_add(ab, c);
}
```

## Notes

- Trait definitions (`trait Addable { ... }`) produce no WGSL output.
- `impl Addable for f32` methods become free functions `f32_add`, `i32_add`, etc.
- `T::add(a, b)` inside a generic function resolves to the concrete mangled function after monomorphization.