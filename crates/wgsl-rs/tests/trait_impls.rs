//! Tests for trait impl patterns that already work and must keep working.
//!
//! These cover the stable behavior established by #107 (concrete trait impls
//! on scalars and arrays) and the existing generic struct impl monomorphization
//! (`impl<T> Pair<T>`). They guard against regressions when generic array impl
//! support is added (#133).

#![allow(dead_code)]
#![allow(clippy::manual_range_patterns)]

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod concrete_trait_impls {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0u32
        }
    }

    impl Zeroable for f32 {
        fn zero() -> f32 {
            0.0
        }
    }

    impl Zeroable for [u32; 4] {
        fn zero() -> [u32; 4] {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    pub fn go<T: Zeroable>() -> T {
        T::zero()
    }

    pub fn caller_u32() -> u32 {
        go::<u32>()
    }

    pub fn caller_f32() -> f32 {
        go::<f32>()
    }

    pub fn caller_array() -> [u32; 4] {
        go::<[u32; 4]>()
    }
}

#[test]
fn concrete_scalar_trait_impl_transpiles() {
    let src = concrete_trait_impls::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("u32_zero"),
        "expected u32_zero in WGSL, got:\n{src}"
    );
    assert!(
        src.contains("f32_zero"),
        "expected f32_zero in WGSL, got:\n{src}"
    );
}

#[test]
fn concrete_array_trait_impl_transpiles() {
    let src = concrete_trait_impls::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("array_u32_4_zero"),
        "expected array_u32_4_zero in WGSL, got:\n{src}"
    );
}

#[test]
fn concrete_trait_impls_run_on_cpu() {
    assert_eq!(concrete_trait_impls::caller_u32(), 0u32);
    assert_eq!(concrete_trait_impls::caller_f32(), 0.0);
    assert_eq!(
        concrete_trait_impls::caller_array(),
        [0u32, 0u32, 0u32, 0u32]
    );
}

#[wgsl(skip_validation)]
mod generic_struct_trait_impl {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0u32
        }
    }

    impl Zeroable for f32 {
        fn zero() -> f32 {
            0.0
        }
    }

    pub struct Wrapper<T: Copy> {
        pub inner: T,
    }

    impl<T: Copy + Zeroable> Zeroable for Wrapper<T> {
        fn zero() -> Wrapper<T> {
            Wrapper { inner: T::zero() }
        }
    }

    pub fn make_zero<T: Copy + Zeroable>() -> Wrapper<T> {
        Wrapper::zero()
    }

    pub fn make_zero_u32() -> Wrapper<u32> {
        make_zero::<u32>()
    }

    pub fn make_zero_f32() -> Wrapper<f32> {
        make_zero::<f32>()
    }
}

#[test]
fn generic_struct_trait_impl_transpiles() {
    let src = generic_struct_trait_impl::WGSL_SOURCE
        .wgsl_source()
        .unwrap();
    // The generic impl `impl<T: Zeroable> Zeroable for Wrapper<T>` should
    // monomorphize to `Wrapper_u32_zero` and `Wrapper_f32_zero`.
    assert!(
        src.contains("Wrapper_u32_zero"),
        "expected Wrapper_u32_zero in WGSL, got:\n{src}"
    );
    assert!(
        src.contains("Wrapper_f32_zero"),
        "expected Wrapper_f32_zero in WGSL, got:\n{src}"
    );
}

#[test]
fn generic_struct_trait_impl_runs_on_cpu() {
    let w = generic_struct_trait_impl::make_zero_u32();
    assert_eq!(w.inner, 0u32);
    let w = generic_struct_trait_impl::make_zero_f32();
    assert_eq!(w.inner, 0.0);
}

#[wgsl(skip_validation)]
mod generic_fn_trait_dispatch {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0u32
        }
    }

    impl Zeroable for f32 {
        fn zero() -> f32 {
            0.0
        }
    }

    pub fn zero_it<T: Zeroable>() -> T {
        T::zero()
    }

    pub fn caller_u32() -> u32 {
        zero_it::<u32>()
    }

    pub fn caller_f32() -> f32 {
        zero_it::<f32>()
    }
}

#[test]
fn generic_fn_trait_dispatch_transpiles() {
    let src = generic_fn_trait_dispatch::WGSL_SOURCE
        .wgsl_source()
        .unwrap();
    // `T::zero()` dispatches to `u32_zero` and `f32_zero` after monomorphization.
    assert!(
        src.contains("u32_zero"),
        "expected u32_zero in WGSL, got:\n{src}"
    );
    assert!(
        src.contains("f32_zero"),
        "expected f32_zero in WGSL, got:\n{src}"
    );
}

#[test]
fn generic_fn_trait_dispatch_runs_on_cpu() {
    assert_eq!(generic_fn_trait_dispatch::caller_u32(), 0u32);
    assert_eq!(generic_fn_trait_dispatch::caller_f32(), 0.0);
}
