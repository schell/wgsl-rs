//! Tests for associated constants provided by trait impls.
//!
//! Rust forbids `pub` on any item in a trait impl (`E0449`), so wgsl-rs
//! must accept non-`pub` associated consts in trait impl blocks — the same
//! relaxation already applied to trait-impl methods. These tests guard the
//! `usize` case: `usize` is lowered to `u32` in WGSL.

#![allow(dead_code)]

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod trait_impl_const {
    pub trait SlabItem {
        const SLAB_SIZE: usize;
        fn read_at(slab_index: u32) -> Self;
    }

    impl SlabItem for u32 {
        const SLAB_SIZE: usize = 1;
        fn read_at(slab_index: u32) -> u32 {
            slab_index
        }
    }

    pub fn caller() -> u32 {
        u32::read_at(7u32)
    }
}

#[test]
fn trait_impl_const_transpiles() {
    let src = trait_impl_const::WGSL_SOURCE.wgsl_source().unwrap();
    // `SLAB_SIZE` contains one underscore, so the mangle pass escapes it as
    // `_1SLAB_SIZE` (see `wgsl_rs_ir::mangle`). The full mangled const is
    // `u32__1SLAB_SIZE`.
    assert!(
        src.contains("const u32__1SLAB_SIZE: u32 = 1"),
        "expected 'const u32__1SLAB_SIZE: u32 = 1' in WGSL, got:\n{src}"
    );
    assert!(
        src.contains("fn u32__1read_at"),
        "expected 'fn u32__1read_at' in WGSL, got:\n{src}"
    );
}

#[test]
fn trait_impl_const_runs_on_cpu() {
    assert_eq!(trait_impl_const::caller(), 7u32);
    assert_eq!(<u32 as trait_impl_const::SlabItem>::SLAB_SIZE, 1);
}
