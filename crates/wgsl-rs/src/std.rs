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

    pub fn read(&self) -> ModuleVarReadGuard<'_, T> {
        let lock = self
            .inner
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a module variable"));
        ModuleVarReadGuard { inner: lock }
    }

    pub fn write(&self) -> ModuleVarWriteGuard<'_, T> {
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
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
    pub fn get_mut(&self) -> ModuleVarWriteGuard<'_, T> {
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
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Set the inner `T`.
    ///
    /// ## Note
    /// Though this method is public, using it in a `#[wgsl]` module triggers
    /// a parse error.
    /// This is because uniform values are read-only from a shader.
    /// This method still exists to set the value for Rust CPU shader testing.
    pub fn set(&self) -> ModuleVarWriteGuard<'_, T> {
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
    pub fn get(&self) -> ModuleVarReadGuard<'_, T> {
        self.data.read()
    }

    /// Get a mutable reference to the inner `T`.
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
/// WGSL provides the following atomic operations as builtin functions:
/// - [`atomic_load`] - Load a value from an atomic
/// - [`atomic_store`] - Store a value to an atomic
/// - [`atomic_add`] - Add and return old value
/// - [`atomic_sub`] - Subtract and return old value
/// - [`atomic_max`] - Maximum and return old value
/// - [`atomic_min`] - Minimum and return old value
/// - [`atomic_and`] - Bitwise AND and return old value
/// - [`atomic_or`] - Bitwise OR and return old value
/// - [`atomic_xor`] - Bitwise XOR and return old value
/// - [`atomic_exchange`] - Exchange and return old value
/// - [`atomic_compare_exchange_weak`] - Compare and exchange
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// // In a storage buffer or workgroup variable
/// workgroup!(COUNTER: Atomic<u32>);
///
/// // Use atomic operations
/// let old = atomic_add(&COUNTER, 1);
/// ```
#[derive(Debug)]
pub struct Atomic<T: AtomicScalar> {
    pub(crate) inner: T::AtomicType,
}

impl<T: AtomicScalar> Default for Atomic<T> {
    fn default() -> Self {
        Self {
            inner: <T as AtomicScalar>::AtomicType::default(),
        }
    }
}

/// Result type for [`atomic_compare_exchange_weak`].
///
/// In WGSL, this is `__atomic_compare_exchange_result<T>`.
///
/// Contains the old value that was in the atomic, and a boolean indicating
/// whether the exchange was performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicCompareExchangeResult<T> {
    /// The original value in the atomic variable.
    pub old_value: T,
    /// `true` if the exchange was performed (old value matched the comparand).
    pub exchanged: bool,
}

// ============================================================================
// Atomic Builtin Functions
// ============================================================================
//
// WGSL atomic builtin functions for thread-safe memory operations.
// See: https://gpuweb.github.io/gpuweb/wgsl/#atomic-builtin-functions
//
// All operations use Relaxed memory ordering, which matches WGSL semantics.
// WGSL atomics only operate on `atomic<i32>` and `atomic<u32>` types.

use std::sync::atomic::Ordering;

/// Atomically loads the value pointed to by `p`.
///
/// This is the Rust equivalent of WGSL's `atomicLoad(p: ptr<AS, atomic<T>>) ->
/// T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
///
/// # Returns
/// The value stored in the atomic variable.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 42);
/// assert_eq!(atomic_load(&a), 42);
/// ```
pub fn atomic_load(p: &Atomic<u32>) -> u32 {
    p.inner.load(Ordering::Relaxed)
}

/// Atomically loads the value pointed to by `p` (i32 variant).
///
/// See [`atomic_load`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_load_i32(p: &Atomic<i32>) -> i32 {
    p.inner.load(Ordering::Relaxed)
}

/// Atomically stores the value `v` into the atomic pointed to by `p`.
///
/// This is the Rust equivalent of WGSL's `atomicStore(p: ptr<AS, atomic<T>>, v:
/// T)`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to store
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 42);
/// assert_eq!(atomic_load(&a), 42);
/// ```
pub fn atomic_store(p: &Atomic<u32>, v: u32) {
    p.inner.store(v, Ordering::Relaxed);
}

/// Atomically stores the value `v` into the atomic pointed to by `p` (i32
/// variant).
///
/// See [`atomic_store`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_store_i32(p: &Atomic<i32>, v: i32) {
    p.inner.store(v, Ordering::Relaxed);
}

/// Atomically adds `v` to the value pointed to by `p`, and returns the original
/// value.
///
/// This is the Rust equivalent of WGSL's `atomicAdd(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to add
///
/// # Returns
/// The original value before the addition.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
/// let old = atomic_add(&a, 5);
/// assert_eq!(old, 10);
/// assert_eq!(atomic_load(&a), 15);
/// ```
pub fn atomic_add(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_add(v, Ordering::Relaxed)
}

/// Atomically adds `v` to the value pointed to by `p` (i32 variant).
///
/// See [`atomic_add`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_add_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_add(v, Ordering::Relaxed)
}

/// Atomically subtracts `v` from the value pointed to by `p`, and returns the
/// original value.
///
/// This is the Rust equivalent of WGSL's `atomicSub(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to subtract
///
/// # Returns
/// The original value before the subtraction.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
/// let old = atomic_sub(&a, 3);
/// assert_eq!(old, 10);
/// assert_eq!(atomic_load(&a), 7);
/// ```
pub fn atomic_sub(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_sub(v, Ordering::Relaxed)
}

/// Atomically subtracts `v` from the value pointed to by `p` (i32 variant).
///
/// See [`atomic_sub`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_sub_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_sub(v, Ordering::Relaxed)
}

/// Atomically computes `max(old, v)` and stores the result, returning the
/// original value.
///
/// This is the Rust equivalent of WGSL's `atomicMax(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to compare against
///
/// # Returns
/// The original value before the operation.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
/// let old = atomic_max(&a, 15);
/// assert_eq!(old, 10);
/// assert_eq!(atomic_load(&a), 15);
/// ```
pub fn atomic_max(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_max(v, Ordering::Relaxed)
}

/// Atomically computes `max(old, v)` (i32 variant).
///
/// See [`atomic_max`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_max_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_max(v, Ordering::Relaxed)
}

/// Atomically computes `min(old, v)` and stores the result, returning the
/// original value.
///
/// This is the Rust equivalent of WGSL's `atomicMin(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to compare against
///
/// # Returns
/// The original value before the operation.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
/// let old = atomic_min(&a, 5);
/// assert_eq!(old, 10);
/// assert_eq!(atomic_load(&a), 5);
/// ```
pub fn atomic_min(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_min(v, Ordering::Relaxed)
}

/// Atomically computes `min(old, v)` (i32 variant).
///
/// See [`atomic_min`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_min_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_min(v, Ordering::Relaxed)
}

/// Atomically computes `old & v` and stores the result, returning the original
/// value.
///
/// This is the Rust equivalent of WGSL's `atomicAnd(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to AND with
///
/// # Returns
/// The original value before the operation.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 0b1111);
/// let old = atomic_and(&a, 0b1010);
/// assert_eq!(old, 0b1111);
/// assert_eq!(atomic_load(&a), 0b1010);
/// ```
pub fn atomic_and(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_and(v, Ordering::Relaxed)
}

/// Atomically computes `old & v` (i32 variant).
///
/// See [`atomic_and`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_and_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_and(v, Ordering::Relaxed)
}

/// Atomically computes `old | v` and stores the result, returning the original
/// value.
///
/// This is the Rust equivalent of WGSL's `atomicOr(p: ptr<AS, atomic<T>>, v: T)
/// -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to OR with
///
/// # Returns
/// The original value before the operation.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 0b1010);
/// let old = atomic_or(&a, 0b0101);
/// assert_eq!(old, 0b1010);
/// assert_eq!(atomic_load(&a), 0b1111);
/// ```
pub fn atomic_or(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_or(v, Ordering::Relaxed)
}

/// Atomically computes `old | v` (i32 variant).
///
/// See [`atomic_or`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_or_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_or(v, Ordering::Relaxed)
}

/// Atomically computes `old ^ v` and stores the result, returning the original
/// value.
///
/// This is the Rust equivalent of WGSL's `atomicXor(p: ptr<AS, atomic<T>>, v:
/// T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The value to XOR with
///
/// # Returns
/// The original value before the operation.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 0b1111);
/// let old = atomic_xor(&a, 0b1010);
/// assert_eq!(old, 0b1111);
/// assert_eq!(atomic_load(&a), 0b0101);
/// ```
pub fn atomic_xor(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.fetch_xor(v, Ordering::Relaxed)
}

/// Atomically computes `old ^ v` (i32 variant).
///
/// See [`atomic_xor`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_xor_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.fetch_xor(v, Ordering::Relaxed)
}

/// Atomically exchanges the value in the atomic with `v`, returning the
/// original value.
///
/// This is the Rust equivalent of WGSL's `atomicExchange(p: ptr<AS, atomic<T>>,
/// v: T) -> T`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `v` - The new value to store
///
/// # Returns
/// The original value before the exchange.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
/// let old = atomic_exchange(&a, 20);
/// assert_eq!(old, 10);
/// assert_eq!(atomic_load(&a), 20);
/// ```
pub fn atomic_exchange(p: &Atomic<u32>, v: u32) -> u32 {
    p.inner.swap(v, Ordering::Relaxed)
}

/// Atomically exchanges the value (i32 variant).
///
/// See [`atomic_exchange`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_exchange_i32(p: &Atomic<i32>, v: i32) -> i32 {
    p.inner.swap(v, Ordering::Relaxed)
}

/// Atomically compares the value in the atomic with `cmp`, and if equal,
/// exchanges it with `v`.
///
/// This is the Rust equivalent of WGSL's
/// `atomicCompareExchangeWeak(p: ptr<AS, atomic<T>>, cmp: T, v: T) ->
/// __atomic_compare_exchange_result<T>`.
///
/// # Arguments
/// * `p` - A reference to an atomic variable
/// * `cmp` - The value to compare against
/// * `v` - The new value to store if comparison succeeds
///
/// # Returns
/// An [`AtomicCompareExchangeResult`] containing:
/// - `old_value`: The original value in the atomic
/// - `exchanged`: `true` if the exchange was performed (old value equaled
///   `cmp`)
///
/// # Note
/// This is a "weak" compare-exchange, which means it may spuriously fail even
/// when the comparison would succeed. This matches WGSL semantics and allows
/// for more efficient implementations on some hardware.
///
/// # Example
/// ```
/// use wgsl_rs::std::*;
///
/// let a: Atomic<u32> = Atomic::default();
/// atomic_store(&a, 10);
///
/// // Successful exchange
/// let result = atomic_compare_exchange_weak(&a, 10, 20);
/// assert_eq!(result.old_value, 10);
/// // Note: exchanged may be true or false due to weak semantics
///
/// // Failed exchange (value doesn't match)
/// let result = atomic_compare_exchange_weak(&a, 999, 30);
/// assert_ne!(result.old_value, 999);
/// assert!(!result.exchanged);
/// ```
pub fn atomic_compare_exchange_weak(
    p: &Atomic<u32>,
    cmp: u32,
    v: u32,
) -> AtomicCompareExchangeResult<u32> {
    match p
        .inner
        .compare_exchange_weak(cmp, v, Ordering::Relaxed, Ordering::Relaxed)
    {
        Ok(old_value) => AtomicCompareExchangeResult {
            old_value,
            exchanged: true,
        },
        Err(old_value) => AtomicCompareExchangeResult {
            old_value,
            exchanged: false,
        },
    }
}

/// Atomically compares and exchanges (i32 variant).
///
/// See [`atomic_compare_exchange_weak`] for details.
#[doc(hidden)]
#[inline]
pub fn atomic_compare_exchange_weak_i32(
    p: &Atomic<i32>,
    cmp: i32,
    v: i32,
) -> AtomicCompareExchangeResult<i32> {
    match p
        .inner
        .compare_exchange_weak(cmp, v, Ordering::Relaxed, Ordering::Relaxed)
    {
        Ok(old_value) => AtomicCompareExchangeResult {
            old_value,
            exchanged: true,
        },
        Err(old_value) => AtomicCompareExchangeResult {
            old_value,
            exchanged: false,
        },
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;

    // ========================================================================
    // u32 tests
    // ========================================================================

    #[test]
    fn test_atomic_load_store_u32() {
        let a: Atomic<u32> = Atomic::default();
        assert_eq!(atomic_load(&a), 0);

        atomic_store(&a, 42);
        assert_eq!(atomic_load(&a), 42);

        atomic_store(&a, u32::MAX);
        assert_eq!(atomic_load(&a), u32::MAX);
    }

    #[test]
    fn test_atomic_add_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        let old = atomic_add(&a, 5);
        assert_eq!(old, 10);
        assert_eq!(atomic_load(&a), 15);

        // Test wrapping behavior
        atomic_store(&a, u32::MAX);
        let old = atomic_add(&a, 1);
        assert_eq!(old, u32::MAX);
        assert_eq!(atomic_load(&a), 0); // Wrapped
    }

    #[test]
    fn test_atomic_sub_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        let old = atomic_sub(&a, 3);
        assert_eq!(old, 10);
        assert_eq!(atomic_load(&a), 7);

        // Test wrapping behavior
        atomic_store(&a, 0);
        let old = atomic_sub(&a, 1);
        assert_eq!(old, 0);
        assert_eq!(atomic_load(&a), u32::MAX); // Wrapped
    }

    #[test]
    fn test_atomic_max_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        // Value is larger, should update
        let old = atomic_max(&a, 15);
        assert_eq!(old, 10);
        assert_eq!(atomic_load(&a), 15);

        // Value is smaller, should not update
        let old = atomic_max(&a, 5);
        assert_eq!(old, 15);
        assert_eq!(atomic_load(&a), 15);
    }

    #[test]
    fn test_atomic_min_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        // Value is smaller, should update
        let old = atomic_min(&a, 5);
        assert_eq!(old, 10);
        assert_eq!(atomic_load(&a), 5);

        // Value is larger, should not update
        let old = atomic_min(&a, 15);
        assert_eq!(old, 5);
        assert_eq!(atomic_load(&a), 5);
    }

    #[test]
    fn test_atomic_and_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 0b1111);

        let old = atomic_and(&a, 0b1010);
        assert_eq!(old, 0b1111);
        assert_eq!(atomic_load(&a), 0b1010);
    }

    #[test]
    fn test_atomic_or_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 0b1010);

        let old = atomic_or(&a, 0b0101);
        assert_eq!(old, 0b1010);
        assert_eq!(atomic_load(&a), 0b1111);
    }

    #[test]
    fn test_atomic_xor_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 0b1111);

        let old = atomic_xor(&a, 0b1010);
        assert_eq!(old, 0b1111);
        assert_eq!(atomic_load(&a), 0b0101);
    }

    #[test]
    fn test_atomic_exchange_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        let old = atomic_exchange(&a, 20);
        assert_eq!(old, 10);
        assert_eq!(atomic_load(&a), 20);
    }

    #[test]
    fn test_atomic_compare_exchange_weak_u32() {
        let a: Atomic<u32> = Atomic::default();
        atomic_store(&a, 10);

        // Successful exchange (value matches)
        // Note: We loop because weak CAS may spuriously fail
        loop {
            let result = atomic_compare_exchange_weak(&a, 10, 20);
            if result.exchanged {
                assert_eq!(result.old_value, 10);
                break;
            }
            // Spurious failure, old_value should still be 10
            assert_eq!(result.old_value, 10);
        }
        assert_eq!(atomic_load(&a), 20);

        // Failed exchange (value doesn't match)
        let result = atomic_compare_exchange_weak(&a, 999, 30);
        assert_eq!(result.old_value, 20);
        assert!(!result.exchanged);
        assert_eq!(atomic_load(&a), 20);
    }

    // ========================================================================
    // i32 tests
    // ========================================================================

    #[test]
    fn test_atomic_load_store_i32() {
        let a: Atomic<i32> = Atomic::default();
        assert_eq!(atomic_load_i32(&a), 0);

        atomic_store_i32(&a, 42);
        assert_eq!(atomic_load_i32(&a), 42);

        atomic_store_i32(&a, -42);
        assert_eq!(atomic_load_i32(&a), -42);

        atomic_store_i32(&a, i32::MIN);
        assert_eq!(atomic_load_i32(&a), i32::MIN);

        atomic_store_i32(&a, i32::MAX);
        assert_eq!(atomic_load_i32(&a), i32::MAX);
    }

    #[test]
    fn test_atomic_add_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 10);

        let old = atomic_add_i32(&a, 5);
        assert_eq!(old, 10);
        assert_eq!(atomic_load_i32(&a), 15);

        // Test with negative
        let old = atomic_add_i32(&a, -20);
        assert_eq!(old, 15);
        assert_eq!(atomic_load_i32(&a), -5);
    }

    #[test]
    fn test_atomic_sub_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 10);

        let old = atomic_sub_i32(&a, 3);
        assert_eq!(old, 10);
        assert_eq!(atomic_load_i32(&a), 7);

        // Test with negative
        let old = atomic_sub_i32(&a, -5);
        assert_eq!(old, 7);
        assert_eq!(atomic_load_i32(&a), 12);
    }

    #[test]
    fn test_atomic_max_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, -10);

        // Value is larger, should update
        let old = atomic_max_i32(&a, 5);
        assert_eq!(old, -10);
        assert_eq!(atomic_load_i32(&a), 5);

        // Value is smaller, should not update
        let old = atomic_max_i32(&a, -20);
        assert_eq!(old, 5);
        assert_eq!(atomic_load_i32(&a), 5);
    }

    #[test]
    fn test_atomic_min_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 10);

        // Value is smaller, should update
        let old = atomic_min_i32(&a, -5);
        assert_eq!(old, 10);
        assert_eq!(atomic_load_i32(&a), -5);

        // Value is larger, should not update
        let old = atomic_min_i32(&a, 15);
        assert_eq!(old, -5);
        assert_eq!(atomic_load_i32(&a), -5);
    }

    #[test]
    fn test_atomic_and_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 0b1111);

        let old = atomic_and_i32(&a, 0b1010);
        assert_eq!(old, 0b1111);
        assert_eq!(atomic_load_i32(&a), 0b1010);
    }

    #[test]
    fn test_atomic_or_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 0b1010);

        let old = atomic_or_i32(&a, 0b0101);
        assert_eq!(old, 0b1010);
        assert_eq!(atomic_load_i32(&a), 0b1111);
    }

    #[test]
    fn test_atomic_xor_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 0b1111);

        let old = atomic_xor_i32(&a, 0b1010);
        assert_eq!(old, 0b1111);
        assert_eq!(atomic_load_i32(&a), 0b0101);
    }

    #[test]
    fn test_atomic_exchange_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, 10);

        let old = atomic_exchange_i32(&a, -20);
        assert_eq!(old, 10);
        assert_eq!(atomic_load_i32(&a), -20);
    }

    #[test]
    fn test_atomic_compare_exchange_weak_i32() {
        let a: Atomic<i32> = Atomic::default();
        atomic_store_i32(&a, -10);

        // Successful exchange (value matches)
        loop {
            let result = atomic_compare_exchange_weak_i32(&a, -10, 20);
            if result.exchanged {
                assert_eq!(result.old_value, -10);
                break;
            }
            assert_eq!(result.old_value, -10);
        }
        assert_eq!(atomic_load_i32(&a), 20);

        // Failed exchange (value doesn't match)
        let result = atomic_compare_exchange_weak_i32(&a, 999, 30);
        assert_eq!(result.old_value, 20);
        assert!(!result.exchanged);
        assert_eq!(atomic_load_i32(&a), 20);
    }

    #[test]
    fn test_atomic_compare_exchange_result_struct() {
        // Test the struct itself
        let result_u32 = AtomicCompareExchangeResult {
            old_value: 42u32,
            exchanged: true,
        };
        assert_eq!(result_u32.old_value, 42);
        assert!(result_u32.exchanged);

        let result_i32 = AtomicCompareExchangeResult {
            old_value: -42i32,
            exchanged: false,
        };
        assert_eq!(result_i32.old_value, -42);
        assert!(!result_i32.exchanged);

        // Test equality
        let result1 = AtomicCompareExchangeResult {
            old_value: 10u32,
            exchanged: true,
        };
        let result2 = AtomicCompareExchangeResult {
            old_value: 10u32,
            exchanged: true,
        };
        assert_eq!(result1, result2);
    }
}
