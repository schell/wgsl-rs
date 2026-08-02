//! This test ensures that structs must be imported instead of being referenced
//! by path name.
use wgsl_rs::wgsl;

#[wgsl]
pub mod slab {
    use wgsl_rs::std::*;

    /// An identifier that can be used to read or write a type from/into the
    /// slab.
    #[repr(transparent)]
    pub struct Id<T> {
        pub index: u32,
        // This should fail to parse because the struct `PhantomData` is
        // a fully qualified path name.
        pub phantom: std::marker::PhantomData<T>,
    }
}

fn main() {}
