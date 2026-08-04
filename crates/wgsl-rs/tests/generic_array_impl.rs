//! Tests for generic trait impls on array types (#133).
//!
//! These tests verify that `impl<T: Trait> Trait for [T; N]` monomorphizes
//! correctly when a concrete array type is used.

#![allow(dead_code)]

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod generic_array_trait {
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

    impl<T: Zeroable> Zeroable for [T; 4] {
        fn zero() -> [T; 4] {
            [T::zero(), T::zero(), T::zero(), T::zero()]
        }
    }

    pub fn zero_it<T: Zeroable>() -> T {
        T::zero()
    }

    pub fn caller_u32_array() -> [u32; 4] {
        zero_it::<[u32; 4]>()
    }

    pub fn caller_f32_array() -> [f32; 4] {
        zero_it::<[f32; 4]>()
    }
}

#[test]
fn generic_array_impl_transpiles() {
    let src = generic_array_trait::WGSL_SOURCE.wgsl_source().unwrap();
    // `impl<T: Zeroable> Zeroable for [T; 4]` monomorphized with T=u32
    // should produce `_2array_u32_4_zero`.
    assert!(
        src.contains("_2array_u32_4_zero"),
        "expected _2array_u32_4_zero in WGSL, got:\n{src}"
    );
    // And with T=f32, `_2array_f32_4_zero`.
    assert!(
        src.contains("_2array_f32_4_zero"),
        "expected _2array_f32_4_zero in WGSL, got:\n{src}"
    );
}

#[test]
fn generic_array_impl_runs_on_cpu() {
    assert_eq!(
        generic_array_trait::caller_u32_array(),
        [0u32, 0u32, 0u32, 0u32]
    );
    assert_eq!(
        generic_array_trait::caller_f32_array(),
        [0.0, 0.0, 0.0, 0.0]
    );
}
