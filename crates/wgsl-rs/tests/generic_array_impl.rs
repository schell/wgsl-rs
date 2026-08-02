//! Tests for generic trait impls on array types (Gap 1, see #133).
//!
//! These are currently gated behind the `generic_array_impls` cfg flag because
//! the monomorphizer does not yet support generic impl blocks on array self
//! types (`impl<T> Trait for [T; 4]`). The `#[wgsl]` macro runs at compile
//! time, so `#[ignore]` alone is not enough — the module must not compile at
//! all until the fix lands.
//!
//! To enable: run with `RUSTFLAGS="--cfg generic_array_impls" cargo test ...`.
//! Remove the `#![cfg(...)]` gate once the fix lands.

#![cfg(generic_array_impls)]

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
    // should produce `array_u32_4_zero`.
    assert!(
        src.contains("array_u32_4_zero"),
        "expected array_u32_4_zero in WGSL, got:\n{src}"
    );
    // And with T=f32, `array_f32_4_zero`.
    assert!(
        src.contains("array_f32_4_zero"),
        "expected array_f32_4_zero in WGSL, got:\n{src}"
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
