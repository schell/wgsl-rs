//! This test shows that
use wgsl_rs::wgsl;

#[wgsl]
pub mod slab {
    use wgsl_rs::std::*;

    /// An identifier that can be used to read or write a type from/into the
    /// slab.
    #[repr(transparent)]
    pub struct Id<T> {
        pub index: u32,
        pub phantom: std::marker::PhantomData<T>,
    }
}
