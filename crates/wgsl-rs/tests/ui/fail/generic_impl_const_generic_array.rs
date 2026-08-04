//! This test currently fails because const generic parameters are not yet
//! supported on WGSL impl blocks. This WILL pass in the near future when
//! const generics are added (Gap 2, see #133).

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod const_generic_array_impl {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0u32
        }
    }

    impl<T: Zeroable, const N: usize> Zeroable for [T; N] {
        fn zero() -> [T; N] {
            // This body is not valid for arbitrary N, but the parse error
            // fires before codegen so the body doesn't matter.
            [T::zero()]
        }
    }
}

fn main() {}