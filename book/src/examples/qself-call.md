# QSelf Call Syntax

Demonstrates direct `<T>::method()` call syntax on complex (non-ident) self types. Inside a generic function, `T::method()` resolves via monomorphization (see [trait impls](./trait-impls.md)). For *direct* calls where the concrete type is named at the call site, Rust requires the qualified-self form `<T>::method()` when `T` is not a simple identifier — e.g. `<[u32; 4]>::zero()`. The proc-macro mangles the qself type into the same identifier the impl block's self type produces, so the call resolves to the same `Type_method` WGSL function.

## Rust Source

```rust
#[wgsl]
pub mod qself_call_example {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    impl Zeroable for [u32; 4] {
        fn zero() -> [u32; 4] {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    /// Direct QSelf call on a scalar — `<u32>::zero()` resolves to
    /// `u32_zero()`.
    pub fn scalar_zero() -> u32 {
        <u32>::zero()
    }

    /// Direct QSelf call on an array — `<[u32; 4]>::zero()` resolves to
    /// `_2array_u32_4_zero()`.
    pub fn array_zero() -> [u32; 4] {
        <[u32; 4]>::zero()
    }
}
```

## Generated WGSL

```wgsl
fn u32_zero() -> u32 {
    return 0;
}

fn _2array_u32_4_zero() -> array<u32, 4> {
    return array(0u, 0u, 0u, 0u);
}

fn scalar_zero() -> u32 {
    return u32_zero();
}

fn array_zero() -> array<u32, 4> {
    return _2array_u32_4_zero();
}
```

## Notes

- `<u32>::zero()` and `<[u32; 4]>::zero()` resolve to the same mangled functions the impl blocks emit (`u32_zero` and `_2array_u32_4_zero`).
- The `_2` prefix is the bijective mangled encoding of the `array_u32_4` self type.
- Associated constants work too: `<Tag>::DEFAULT` resolves to `Tag_DEFAULT`.
- The `<T as Trait>::method()` disambiguation form is rejected with a helpful error — trait impls are matched by self type only.
- Generic (turbofish) arguments on the method after `>::` are not supported in the QSelf form.