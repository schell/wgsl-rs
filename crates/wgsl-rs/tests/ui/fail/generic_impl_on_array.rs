//! This test currently fails because generic impl blocks on array self types
//! are not yet supported. This WILL pass in the near future when generic
//! array impls are implemented (see #133).

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod generic_array_impl {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0u32
        }
    }

    impl<T: Zeroable> Zeroable for [T; 4] {
        fn zero() -> [T; 4] {
            [T::zero(), T::zero(), T::zero(), T::zero()]
        }
    }
}

fn main() {}