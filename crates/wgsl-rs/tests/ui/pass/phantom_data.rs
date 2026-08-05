//! Passing tests for `PhantomData<T>` marker fields on `#[wgsl]` structs.
//!
//! `PhantomData` is re-exported from `wgsl_rs::std` so it can be brought
//! into scope via glob import. The proc-macro recognizes `PhantomData<_>`
//! fields specially: they are retained in the IR (so extensions can
//! observe which type parameter each phantom slot binds) but omitted
//! from the rendered WGSL. Construction expressions using the bare
//! `PhantomData` value are likewise stripped so the rendered positional
//! constructor has the correct arity.
//!
//! This file exercises:
//! 1. A single-phantom generic struct constructed locally.
//! 2. A multi-phantom generic struct binding two type parameters (the
//!    "which type belongs to which field" extension-visibility case).
//! 3. End-to-end WGSL source generation for both.
use wgsl_rs::wgsl;

// 1. Single-phantom generic struct, constructed locally.
#[wgsl(crate_path = wgsl_rs)]
pub mod single_phantom {
    use wgsl_rs::std::*;

    pub struct Id<T> {
        pub index: u32,
        pub phantom: PhantomData<T>,
    }

    pub fn make() -> Id<f32> {
        Id { index: 0u32, phantom: PhantomData }
    }
}

// 2. Multi-phantom generic struct binding two type parameters. An
//    extension inspecting the IR should still be able to see that
//    field `t` binds `T` and field `a` binds `A`, even though neither
//    survives to the rendered WGSL.
#[wgsl(crate_path = wgsl_rs)]
pub mod multi_phantom {
    use wgsl_rs::std::*;

    pub struct Tagged<T, A> {
        pub x: f32,
        pub t: PhantomData<T>,
        pub a: PhantomData<A>,
    }

    pub fn make() -> Tagged<f32, u32> {
        Tagged { x: 1.0, t: PhantomData, a: PhantomData }
    }
}

fn main() {
    let _ = single_phantom::WGSL_SOURCE.wgsl_source().unwrap();
    let _ = multi_phantom::WGSL_SOURCE.wgsl_source().unwrap();
    let _ = single_phantom::make();
    let _ = multi_phantom::make();
}