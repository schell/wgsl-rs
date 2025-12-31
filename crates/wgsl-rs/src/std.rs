//! The WGSL standard library, for Rust.
//!
//! When using the module-level `wgsl` proc-macro, this module must only be
//! glob-imported.
//!
//! The glob-imported import statement `use wgsl_rs::std::*` pulls in all the
//! WGSL types and functions into Rust that are part of the global scope
//! in WGSL, but don't already exist in Rust's global scope.
//!
//! These types include (but are not limited to) vector types like `Vec2f`,
//! `Vec3f`, etc. and constructors like `vec2`, `vec2f` and `vec3i`, etc.

use std::sync::{Arc, LazyLock, RwLock};

pub use wgsl_rs_macros::{
    builtin, compute, fragment, input, output, storage, uniform, vertex, workgroup_size,
};

mod numeric_builtin_functions;
pub use numeric_builtin_functions::*;

mod vectors;
pub use vectors::*;

mod matrices;
pub use matrices::*;

/// A shader uniform, backed by a storage buffer on the CPU.
pub struct UniformVariable<T> {
    pub group: u32,
    pub binding: u32,
    pub value: Arc<RwLock<Option<T>>>,
}

pub type Uniform<T> = LazyLock<UniformVariable<T>>;

/// A shader storage buffer, backed by a storage buffer on the CPU.
pub struct StorageVariable<T> {
    pub group: u32,
    pub binding: u32,
    pub read_write: bool,
    pub value: Arc<RwLock<Option<T>>>,
}

pub type Storage<T> = LazyLock<StorageVariable<T>>;

/// Used to provide WGSL type conversion functions like `f32(...)`, etc.
pub trait Convert<T> {
    fn convert(self) -> T;
}

impl<A: Clone + Convert<B>, B> Convert<B> for &Uniform<A> {
    fn convert(self) -> B {
        let guard = self.value.read().expect("could not read value");
        let maybe = guard.as_ref();
        let a = maybe.cloned().expect("uniform value has not been set");
        a.convert()
    }
}

macro_rules! impl_convert_as {
    ($from:ty, $to:ty) => {
        impl Convert<$to> for $from {
            fn convert(self) -> $to {
                self as $to
            }
        }
    };
}
impl_convert_as!(f32, u32);
impl_convert_as!(f32, i32);
impl_convert_as!(i32, f32);
impl_convert_as!(i32, u32);
impl_convert_as!(u32, f32);
impl_convert_as!(u32, i32);

/// Returns the input cast to f32.
pub fn f32(t: impl Convert<f32>) -> f32 {
    t.convert()
}

/// Returns the input cast to u32.
pub fn u32(t: impl Convert<u32>) -> u32 {
    t.convert()
}

/// Returns the input cast to i32.
pub fn i32(t: impl Convert<i32>) -> i32 {
    t.convert()
}
