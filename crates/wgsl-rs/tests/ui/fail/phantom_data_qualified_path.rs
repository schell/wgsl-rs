//! Failing test: `PhantomData` must be brought into scope via a `use`
//! import (or the `wgsl_rs::std` re-export). A fully-qualified path
//! like `std::marker::PhantomData<T>` is rejected because the
//! proc-macro only recognizes single-segment type names.
//!
//! When `PhantomData<T>` support was added, this case continued to be
//! rejected by the same multi-segment-path rule that guards all other
//! type positions. See `tests/ui/pass/phantom_data.rs` for the
//! supported form.
use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
pub mod qualified_phantom {
    use wgsl_rs::std::*;

    pub struct Id<T> {
        pub index: u32,
        // This should fail: `std::marker::PhantomData` is a fully
        // qualified path. Import `PhantomData` (re-exported from
        // `wgsl_rs::std`) and use the bare name instead.
        pub phantom: std::marker::PhantomData<T>,
    }
}

fn main() {}