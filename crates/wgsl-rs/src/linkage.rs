//! Type equality trait for generic linkage constraint checking.
//!
//! When a `#[wgsl]` module contains generic linkage variables (e.g.
//! `storage!(group(0), binding(0), BINS: impl std::any::Any)`) that are
//! accessed via `get!(BINS, T)` or `get_mut!(BINS, T)`, the generated
//! `instantiate` function needs to enforce that every entry point agrees on
//! the concrete type of each linkage variable.
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
