//! This test exercises a trait impl on an array with both a type param
//! and a const generic param: `impl<T: Zeroable, const N: usize> Zeroable
//! for [T; N]`. Const generics on impl blocks are now supported, but
//! generic impl blocks on array self types (`[T; N]`) are not yet
//! supported — see issue #133. The parse fails with "generic impl blocks
//! require a struct self type." When generic-impl-on-arrays lands, this
//! test should flip to pass.

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