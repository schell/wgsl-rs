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

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

pub use wgsl_rs_macros::{
    builtin, compute, fragment, input, output, ptr, sampler, storage, texture, uniform, vertex,
    wgsl_allow, workgroup, workgroup_size,
};

pub use crate::{get, get_mut};


mod atomic;
mod bitcast;
mod matrix;
mod numeric;
mod packing;
mod texture;
mod vector;

pub use atomic::*;
pub use bitcast::*;
pub use matrix::*;
pub use numeric::*;
pub use packing::*;
pub use texture::*;
pub use vector::*;


/// Shared reference to a uniform, storage or workgroup variable.
pub struct ModuleVarReadGuard<'a, T> {
    inner: RwLockReadGuard<'a, Option<T>>,
}

impl<T> Deref for ModuleVarReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .unwrap_or_else(|| panic!("Accessed an uninitialized module variable"))
    }
}

/// Exclusive reference to a storage or workgroup variable.
pub struct ModuleVarWriteGuard<'a, T> {
    inner: RwLockWriteGuard<'a, Option<T>>,
}

impl<T> Deref for ModuleVarWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .unwrap_or_else(|| panic!("Accessed an uninitialized module variable"))
    }
}

impl<T> DerefMut for ModuleVarWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
            .as_mut()
            .unwrap_or_else(|| panic!("Mutably accessed an uninitialized module variable"))
    }
}

/// Thread-safe module level variable that can be read from or written
/// to from anywhere.
struct ModuleVar<T> {
    inner: LazyLock<Arc<RwLock<Option<T>>>>,
}

impl<T> ModuleVar<T> {
    pub const fn new() -> Self {
        Self {
            inner: LazyLock::new(Default::default),
        }
    }
    /// Returns a reference to the inner `T`.
    ///
    /// ## Panics
    /// - Panics if the underlying lock has been poisoned.
    /// - Dereferencing the returned guard will panic if it has not previously
    ///   been set.
    pub fn read(&self) -> ModuleVarReadGuard<'_, T> {
        let lock = self
            .inner
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a module variable"));
        ModuleVarReadGuard { inner: lock }
    }

    /// Returns a mutable reference to the inner `T`.
    ///
    /// ## Panics
    /// - Panics if the underlying lock has been poisoned.
    /// - Dereferencing the returned guard will panic if it has not previously
    ///   been set.
    pub fn write(&self) -> ModuleVarWriteGuard<'_, T> {
        let lock = self
            .inner
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a module variable"));
        ModuleVarWriteGuard { inner: lock }
    }

    /// Set the inner `T`.
    ///
    /// ## Panics
    /// Panics if the underlying lock on the inner data has been poisoned.
    pub fn set(&self, data: T) {
        *self
            .inner
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a module variable")) =
            Some(data);
    }
}

/// A workgroup variable.
pub struct Workgroup<T> {
    data: ModuleVar<T>,
}

impl<T> Workgroup<T> {
    /// Creates a new workgroup variable.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self {
            data: ModuleVar::new(),
        }
    }

    /// Get a reference to the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get_mut(&self) -> ModuleVarWriteGuard<'_, T> {
        self.data.write()
    }

    /// Set the inner `T`.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: T) {
        self.data.set(data);
    }
}

/// A shader uniform, backed by a `RwLock<T>` on the CPU.
pub struct Uniform<T> {
    pub group: u32,
    pub binding: u32,
    data: ModuleVar<T>,
}

impl<T> Uniform<T> {
    /// Creates a new uniform.
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    /// Returns a reference to the inner `T`.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Set the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Note
    /// Though this method is public, using it in a `#[wgsl]` module triggers
    /// a parse error.
    ///
    /// This is fine because uniform values are read-only from a shader.
    ///
    /// This method still exists to set the value for Rust CPU shader testing.
    pub fn set(&self, data: T) {
        self.data.set(data);
    }
}

/// Marker type for read-only access-mode.
pub struct Read;

/// Marker type for readwrite access-mode.
pub struct ReadWrite;

/// A shader storage buffer, backed by a `T`.
pub struct Storage<T, AM = Read> {
    group: u32,
    binding: u32,
    access_mode: PhantomData<AM>,
    data: ModuleVar<T>,
}

/// Marker trait for read or readwrite storage.
pub trait AccessMode {}

impl AccessMode for Read {}
impl AccessMode for ReadWrite {}

impl<T, AM: AccessMode> Storage<T, AM> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Storage {
            data: ModuleVar::new(),
            group,
            binding,
            access_mode: PhantomData,
        }
    }

    /// Returns a reference to the inner `T`.
    ///
    /// Not available in WGSL. Use [`get!`] instead.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Returns a mutable reference to the inner `T`.
    ///
    /// Not available in WGSL. Use [`get_mut!`] instead.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get_mut(&self) -> ModuleVarWriteGuard<'_, T> {
        self.data.write()
    }

    /// Returns the group index of this storage variable.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Returns the binding index within its group of this storage variable.
    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// Provides access to a storage variable.
///
/// Since storage variables are `static` and implemented with locks,
/// normal borrows aren't possible.
///
/// # Example
/// ```ignore
/// storage!(group(0), binding(0), read_write, OUTPUT: [f32; 256]);
///
/// let value = get!(OUTPUT)[idx];
/// let value = get!(OUTPUT).field;
/// ```
#[macro_export]
macro_rules! get {
    ($var:ident) => {
        $var.get()
    };
}

/// Provides mutable access to a storage variable.
///
/// Since storage variables are `static` and implemented with locks,
/// normal mutable borrows aren't possible.
/// This macro uses interior mutability to enable writes.
///
/// # Example
/// ```ignore
/// storage!(group(0), binding(0), read_write, OUTPUT: [f32; 256]);
///
/// get_mut!(OUTPUT)[idx] = value;
/// get_mut!(OUTPUT).field = value;
/// ```
#[macro_export]
macro_rules! get_mut {
    ($var:ident) => {
        $var.get_mut()
    };
}

/// Used to provide WGSL type conversion functions like `f32(...)`, etc.
pub trait Convert<T> {
    fn convert(self) -> T;
}

impl<A: Clone + Convert<B>, B> Convert<B> for &A {
    fn convert(self) -> B {
        let a = self.clone();
        a.convert()
    }
}

impl<A: Clone + Convert<B>, B> Convert<B> for ModuleVarReadGuard<'_, A> {
    fn convert(self) -> B {
        self.clone().convert()
    }
}

impl<A: Clone + Convert<B>, B> Convert<B> for ModuleVarWriteGuard<'_, A> {
    fn convert(self) -> B {
        self.clone().convert()
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
