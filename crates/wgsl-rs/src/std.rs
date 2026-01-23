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
    builtin, compute, fragment, input, output, storage, uniform, vertex, wgsl_allow, workgroup_size,
};

mod numeric_builtin_functions;
pub use numeric_builtin_functions::*;

mod vectors;
pub use vectors::*;

mod matrices;
pub use matrices::*;

/// A workgroup variable.
pub struct WorkgroupVariable<T> {
    pub value: Arc<RwLock<Option<T>>>,
}

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

/// A runtime-sized array, backed by a [`Vec`] on CPU.
///
/// In WGSL, this transpiles to `array<T>` (no size parameter).
/// Runtime-sized arrays can only be used in storage buffers,
/// typically raw, or as the last field of a struct.
#[derive(Debug, Clone, Default)]
pub struct RuntimeArray<T> {
    pub data: std::vec::Vec<T>,
}

impl<T> RuntimeArray<T> {
    /// Create a new `RuntimeArray<T>` on the CPU.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    pub fn new() -> Self {
        Self {
            data: std::vec::Vec::new(),
        }
    }

    /// Creates a new `RuntimeArray<T>` with a starting `capacity`.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: std::vec::Vec::with_capacity(capacity),
        }
    }

    /// Returns the length of this `RuntimeArray<T>`.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    /// Use the builtin function [`array_length`] in `#[wgsl]` modules.
    pub fn len(&self) -> u32 {
        self.data.len() as u32
    }

    /// Returns whether this `RuntimeArray<T>` is empty.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Append `value` onto the back of the collection.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Clears the array, removing all values.
    ///
    /// Only available on CPU, this is a parse error in `#[wgsl]` modules.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

// Index by usize (Rust standard)
impl<T> std::ops::Index<usize> for RuntimeArray<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for RuntimeArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// Index by u32 (for WGSL compatibility)
impl<T> std::ops::Index<u32> for RuntimeArray<T> {
    type Output = T;
    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl<T> std::ops::IndexMut<u32> for RuntimeArray<T> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.data[index as usize]
    }
}

/// Trait for types that support the WGSL `arrayLength` builtin function.
///
/// In WGSL, `arrayLength` returns the number of elements in a runtime-sized
/// array. This trait provides the Rust-side equivalent for `RuntimeArray<T>`.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// @group(0) @binding(0) var<storage> data: array<f32>;
///
/// fn example() -> u32 {
///     return arrayLength(&data);
/// }
/// ```
///
/// # Note
///
/// In WGSL, `arrayLength` takes a pointer to a runtime-sized array. On the Rust
/// side, we use a reference instead. The `array_length` free function provides
/// the same interface as the WGSL builtin.
pub trait ArrayLength {
    fn array_length(self) -> u32;
}

impl<T> ArrayLength for &RuntimeArray<T> {
    fn array_length(self) -> u32 {
        self.data.len() as u32
    }
}

/// Returns the number of elements in a runtime-sized array.
///
/// This is the Rust-side equivalent of the WGSL `arrayLength` builtin function.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// let mut particles: RuntimeArray<f32> = RuntimeArray::new();
/// particles.push(1.0);
/// particles.push(2.0);
/// particles.push(3.0);
///
/// assert_eq!(array_length(&particles), 3);
/// ```
///
/// # WGSL Equivalent
///
/// In WGSL, you would write:
///
/// ```wgsl
/// let len: u32 = arrayLength(&runtime_array);
/// ```
pub fn array_length(array: impl ArrayLength) -> u32 {
    array.array_length()
}
