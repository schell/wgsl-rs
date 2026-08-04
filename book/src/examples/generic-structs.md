# Generic Structs

Demonstrates generic structs and `impl` blocks with monomorphization. A `Pair<T>` struct and its methods are specialized for each concrete type used.

> **Note:** This example uses `#[wgsl(skip_validation)]` because of a known monomorphization bug: the `Pair` struct constructor is not mangled correctly when monomorphized. WGSL generation partially works but the constructor call is not produced correctly. There is no validated WGSL output for this example.

## Rust Source

```rust
#[wgsl(skip_validation)] // TODO: monomorphization bug — `Pair` struct constructor not mangled
pub mod generic_structs {
    /// A generic pair of values.
    pub struct Pair<T: Copy> {
        pub a: T,
        pub b: T,
    }

    /// Methods on the generic Pair struct.
    impl<T: Copy + std::ops::Add<Output = T>> Pair<T> {
        /// Extract the first element.
        pub fn first(p: Pair<T>) -> T {
            p.a
        }

        /// Sum both elements.
        pub fn sum(p: Pair<T>) -> T {
            p.a + p.b
        }
    }

    pub fn generic_pair_sum<T: Copy + std::ops::Add<Output = T>>(a: T, b: T) -> T {
        let p = Pair { a, b };
        Pair::sum(p)
    }

    /// Uses `Pair<f32>`.
    pub fn use_pair_f32() -> f32 {
        let p = Pair { a: 1.0, b: 2.0 };
        Pair::<f32>::sum(p)
    }

    /// Uses `Pair<i32>`.
    pub fn use_pair_i32() -> i32 {
        let p: Pair<i32> = Pair::<i32> { a: 10, b: 20 };
        Pair::<i32>::first(p)
    }
}
```

## Notes

- `#[wgsl(skip_validation)]` disables naga validation of the generated WGSL. This is required here due to a known bug where the `Pair` struct constructor is not mangled during monomorphization, so the generated WGSL does not pass validation.
- Generic struct methods (`Pair::sum`, `Pair::first`) are expected to monomorphize to `Pair_f32_sum`, `Pair_i32_first`, etc., but the constructor mangling is currently broken.
- Track the fix for this bug to remove the `skip_validation` attribute.