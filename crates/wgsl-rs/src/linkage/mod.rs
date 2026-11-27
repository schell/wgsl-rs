//! Linkage machinery for `wgsl-rs`.
//!
//! ## wgpu linkage
//! Runtime wgpu linkage analysis lives outside this crate, in the
//! `wgsl-rs-linkage-wgpu` crate (one feature per supported wgpu
//! major). The runtime analyzer walks
//! the IR produced by a `#[wgsl]` source (post-instantiation for
//! templates) and produces bind group layouts, pipeline state
//! descriptors, and buffer descriptors without any compile-time code
//! generation from the proc-macro. `wgsl-rs` itself stays wgpu-free.
//!
//! ## Source instantiation
//! When a `#[wgsl]` source contains generic resource packs (e.g.
//! `pub struct BinsPack<T> { ... }`) whose fields are accessed through
//! entry-point parameters (`&BinsPack<T>`), the generated `instantiate`
//! function needs to enforce that every entry point agrees on the concrete
//! type of each pack parameter.
//!
//! The [`Type`] trait provides a type-level equality constraint:
//! `T: Type<Is = U>` is satisfied iff `T` and `U` are the same type.
//! This is used in the `where` clause of the generated `instantiate`
//! function to catch conflicting specialisations at compile time.

/// A trivial trait for type equality checking.
///
/// `T: Type<Is = U>` is satisfied iff `T == U`.
///
/// This is used by the generated `instantiate` function's `where` clause
/// to enforce that a linkage variable's type is consistent across all
/// entry points that use it.
pub trait Type {
    /// The associated type, which for the blanket impl is the type itself.
    type Is;
}

impl<T> Type for T {
    type Is = T;
}
