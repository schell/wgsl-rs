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
    builtin, compute, fragment, input, output, ptr, storage, uniform, vertex, wgsl_allow,
    workgroup, workgroup_size,
};

mod numeric_builtin_functions;
pub use numeric_builtin_functions::*;

mod vectors;
pub use vectors::*;

mod matrices;
pub use matrices::*;

pub use crate::{get, get_mut};

/// Wrapper for a `RwLockReadGuard<'a, Option<T>>` that dereferences
/// to `&T`.
struct ModuleVarReadGuard<'a, T> {
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

/// Wrapper for a `RwLockWriteGuard<'a, Option<T>>` that dereferences
/// to `&T` and `&mut T`.
struct ModuleVarWriteGuard<'a, T> {
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

    pub fn read(&self) -> impl Deref<Target = T> + '_ {
        let lock = self
            .inner
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a module variable"));
        ModuleVarReadGuard { inner: lock }
    }

    pub fn write(&self) -> impl DerefMut<Target = T> + '_ {
        let lock = self
            .inner
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a module variable"));
        ModuleVarWriteGuard { inner: lock }
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
    pub fn get(&self) -> impl Deref<Target = T> + '_ {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
    pub fn get_mut(&self) -> impl DerefMut<Target = T> + '_ {
        self.data.write()
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

    /// Get a reference to the inner `T`.
    pub fn get(&self) -> impl Deref<Target = T> + '_ {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
    pub fn get_mut(&self) -> impl DerefMut<Target = T> + '_ {
        self.data.write()
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

    /// Get a reference to the inner `T`.
    pub fn get(&self) -> impl Deref<Target = T> + '_ {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
    pub fn get_mut(&self) -> impl DerefMut<Target = T> + '_ {
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

/// Marker trait that denotes what atomic type a certain value takes
/// on the Rust side.
pub trait AtomicScalar {
    type AtomicType: Default;
}

impl AtomicScalar for u32 {
    type AtomicType = std::sync::atomic::AtomicU32;
}

impl AtomicScalar for i32 {
    type AtomicType = std::sync::atomic::AtomicI32;
}

/// An atomic type for thread-safe operations.
///
/// In WGSL, this transpiles to `atomic<T>` where T is either `i32` or `u32`.
/// Atomic types can only be used in workgroup or storage address spaces with
/// `read_write` access mode.
///
/// On the CPU side, this wraps Rust's `std::sync::atomic` types to provide
/// thread-safe operations that match WGSL's atomic semantics.
///
/// # WGSL Atomic Operations
///
/// WGSL provides the following atomic operations (to be added as builtin
/// functions):
/// - `atomicLoad` - Load a value from an atomic
/// - `atomicStore` - Store a value to an atomic
/// - `atomicAdd` - Add and return old value
/// - `atomicSub` - Subtract and return old value
/// - `atomicMax` - Maximum and return old value
/// - `atomicMin` - Minimum and return old value
/// - `atomicAnd` - Bitwise AND and return old value
/// - `atomicOr` - Bitwise OR and return old value
/// - `atomicXor` - Bitwise XOR and return old value
/// - `atomicExchange` - Exchange and return old value
/// - `atomicCompareExchangeWeak` - Compare and exchange
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// // In a storage buffer or workgroup variable
/// workgroup!(COUNTER: Atomic<u32>);
///
/// // Atomic operations will be available as builtin functions
/// ```
#[derive(Debug)]
pub struct Atomic<T: AtomicScalar> {
    #[expect(dead_code, reason = "not used yet")]
    inner: T::AtomicType,
}

impl<T: AtomicScalar> Default for Atomic<T> {
    fn default() -> Self {
        Self {
            inner: <T as AtomicScalar>::AtomicType::default(),
        }
    }
}
