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

use std::sync::RwLock;
use wgsl_rs_ir as ir;

// Re-export `PhantomData` so users can write `use wgsl_rs::std::*;` and refer
// to `PhantomData<T>` as a struct field type. The proc-macro recognizes
// `PhantomData<_>` fields specially: they are retained in the IR (so
// extensions can see which type parameter each phantom slot binds) but
// omitted from the rendered WGSL.
pub use std::marker::PhantomData;

pub use wgsl_rs_macros::{
    Wgsl, builtin, compute, fragment, ptr, vertex, wgsl_allow, wgsl_ignore, workgroup_size,
};

pub use crate::{discard, get, get_mut, load, slab_copy};

mod atomic;
mod bitcast;
mod derivative;
mod matrix;
mod numeric;
mod packing;
#[cfg(feature = "dispatch-runtime")]
pub mod runtime;
mod synchronization;
mod texture;
mod vector;

pub use atomic::*;
pub use bitcast::*;
pub use derivative::*;
pub use matrix::*;
pub use numeric::*;
pub use packing::*;
#[cfg(feature = "dispatch-runtime")]
pub use runtime::*;
pub use synchronization::*;
pub use texture::*;
pub use vector::*;

#[rustfmt::skip] // needed so that rustfmt doesn't merge imports
#[allow(unused_imports)]
#[crate::wgsl(crate_path = crate)]
pub mod builtin_constants {
    use crate::std::builtin_matrix_constants::*;
    use crate::std::builtin_vector_constants::*;
}

pub enum Error {}

/// Marker trait for an owned type that is Send + Sync.
pub trait AnySendSync: std::any::Any + Send + Sync {}
impl<T: std::any::Any + Send + Sync> AnySendSync for T {}

/// Trait for Rust types that can be used in WGSL modules.
pub trait Wgsl: AnySendSync {
    fn to_ir() -> wgsl_rs_ir::Type;
}

/// Trait for Rust types that are WGSL scalar types.
pub trait WgslScalar: Wgsl {
    fn to_scalar_ir() -> wgsl_rs_ir::ScalarType;
}

/// Trait for WGSL scalar types that are valid as sampled texture types.
///
/// WGSL only allows `f32`, `i32`, and `u32` as the sampled type for
/// `texture_2d<T>`, `texture_1d<T>`, etc. `bool` is a valid WGSL scalar
/// but not a valid texture sampled type.
pub trait WgslTextureScalar: WgslScalar {}

macro_rules! impl_scalar_wgsl {
    ($t:ty, $i:ident) => {
        impl WgslScalar for $t {
            fn to_scalar_ir() -> ir::ScalarType {
                ir::ScalarType::$i
            }
        }
        impl Wgsl for $t {
            fn to_ir() -> ir::Type {
                ir::Type::Scalar(<$t>::to_scalar_ir())
            }
        }
    };
}

macro_rules! impl_texture_scalar_wgsl {
    ($t:ty, $i:ident) => {
        impl_scalar_wgsl!($t, $i);
        impl WgslTextureScalar for $t {}
    };
}

impl_texture_scalar_wgsl!(f32, F32);
impl_texture_scalar_wgsl!(u32, U32);
impl_texture_scalar_wgsl!(i32, I32);
impl_scalar_wgsl!(bool, Bool);

impl<T: Wgsl, const N: usize> Wgsl for [T; N] {
    fn to_ir() -> ir::Type {
        ir::Type::Array {
            elem: Box::new(T::to_ir()),
            len: ir::Expr::Lit(ir::Lit::Int {
                digits: N.to_string(),
                suffix: String::new(),
            }),
        }
    }
}

/// A workgroup variable.
/// A workgroup variable, holding its value directly behind a `RwLock` on
/// the CPU.
pub struct Workgroup<T> {
    data: RwLock<T>,
}

impl<T: Wgsl> Workgroup<T> {
    /// Creates a new workgroup variable holding `value`.
    ///
    /// Values are constructed, never defaulted: workgroup variables are
    /// instance-owned, so the caller supplies the seed.
    #[allow(clippy::new_without_default)]
    pub fn new(value: T) -> Self {
        Self {
            data: RwLock::new(value),
        }
    }

    /// Get a guard to the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn get(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.data
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a workgroup variable"))
    }

    /// Get a mutable guard to the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn get_mut(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        self.data
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a workgroup variable"))
    }

    /// Set the inner `T`.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn set(&self, value: T) {
        let mut lock = self
            .data
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a workgroup variable"));
        *lock = value;
    }
}

/// A shader uniform, backed by a `RwLock<T>` on the CPU.
/// A shader uniform, holding its value directly behind a `RwLock` on the CPU.
pub struct Uniform<T> {
    data: RwLock<T>,
}

impl<T: Wgsl> Uniform<T> {
    /// Creates a new uniform holding `value`.
    ///
    /// Values are constructed, never defaulted: uniform values are
    /// instance-owned, so the caller supplies the data.
    #[allow(clippy::new_without_default)]
    pub fn new(value: T) -> Self {
        Self {
            data: RwLock::new(value),
        }
    }

    /// Returns a guard to the inner `T`.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn get(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.data
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a uniform"))
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
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn set(&self, value: T) {
        let mut lock = self
            .data
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a uniform"));
        *lock = value;
    }
}

/// Marker type for read-only access-mode.
pub struct Read;

/// Marker type for write-only access-mode.
pub struct Write;

/// Marker type for readwrite access-mode.
pub struct ReadWrite;

/// A shader storage buffer, holding its value directly behind a `RwLock` on
/// the CPU.
pub struct Storage<T, AM = Read> {
    data: RwLock<T>,
    _access: PhantomData<AM>,
}

/// Marker trait for read or readwrite storage.
pub trait AccessMode {}

impl AccessMode for Read {}
impl AccessMode for ReadWrite {}

impl<T: Wgsl, AM: AccessMode> Storage<T, AM> {
    /// Creates a new storage buffer holding `value`.
    ///
    /// Values are constructed, never defaulted: storage values are
    /// instance-owned, so the caller supplies the data.
    #[allow(clippy::new_without_default)]
    pub fn new(value: T) -> Self {
        Storage {
            data: RwLock::new(value),
            _access: PhantomData,
        }
    }

    /// Returns a guard to the inner `T`.
    ///
    /// Not available in WGSL. Use [`get!`] instead.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn get(&self) -> std::sync::RwLockReadGuard<'_, T> {
        self.data
            .read()
            .unwrap_or_else(|_| panic!("could not acquire a read lock on a storage variable"))
    }

    /// Returns a mutable guard to the inner `T`.
    ///
    /// Not available in WGSL. Use [`get_mut!`] instead.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn get_mut(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        self.data
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a storage variable"))
    }

    /// Set the inner `T`.
    ///
    /// Not available in WGSL. Used for CPU-side shader testing to
    /// populate storage buffers before calling compute entry points.
    ///
    /// ## Panics
    /// Panics if the underlying lock has been poisoned.
    pub fn set(&self, value: T) {
        let mut lock = self
            .data
            .write()
            .unwrap_or_else(|_| panic!("could not acquire a write lock on a storage variable"));
        *lock = value;
    }
}


/// Reads a resource-pack field through its CPU lock guard.
///
/// The target must be a resource-pack parameter's field, `pack.field` or
/// `&pack.field`. In WGSL this transpiles to the mangled global resource
/// name; the access kind (uniform mutability, storage access mode, handle
/// types) is checked against the pack field by the transpiler.
///
/// # Panics
/// Panics if the underlying lock has been poisoned.
///
/// # Example
/// ```ignore
/// #[fragment]
/// pub fn main(pack: &MyPack) -> f32 {
///     let value = get!(pack.input)[0];
///     value + 1.0
/// }
/// ```
#[macro_export]
macro_rules! get {
    (&$pack:ident . $field:ident) => {
        (&$pack.$field).get()
    };
    ($pack:ident . $field:ident) => {
        $pack.$field.get()
    };
}

/// Mutably reads a resource-pack field through its CPU lock guard.
///
/// The target must be a resource-pack parameter's field. Uniform fields are
/// read-only from a shader; `get_mut!` requires a read-write storage or
/// workgroup field (checked against the pack field by the transpiler).
///
/// # Panics
/// Panics if the underlying lock has been poisoned.
///
/// # Example
/// ```ignore
/// #[compute]
/// pub fn main(pack: &mut MyPack) {
///     get_mut!(pack.output)[0] = 1.0;
/// }
/// ```
#[macro_export]
macro_rules! get_mut {
    (&$pack:ident . $field:ident) => {
        (&$pack.$field).get_mut()
    };
    ($pack:ident . $field:ident) => {
        $pack.$field.get_mut()
    };
}

/// Copies a resource-pack field's value out of its CPU lock, by value.
///
/// `get!` returns a lock guard that dereferences to `&T`, which cannot be
/// used directly in arithmetic. `load!` copies the value out of the guard
/// and returns plain `T`, so it works in expressions. In WGSL this
/// transpiles to the bare mangled variable reference (a plain value read).
///
/// The value type must be `Copy`; all built-in WGSL value types are.
/// Atomics and runtime-sized arrays are rejected by the transpiler: read
/// atomics with `atomic_load(&get!(pack.field))` and index into
/// `get!(pack.field)` for runtime arrays.
///
/// # Example
/// ```ignore
/// #[fragment]
/// pub fn main(pack: &MyPack) -> Vec4f {
///     let uv = frag_coord.xy() / load!(pack.u_resolution) * 3.0;
///     vec4f(uv, 0.0, 1.0)
/// }
/// ```
#[macro_export]
macro_rules! load {
    (&$pack:ident . $field:ident) => {{
        let v = *(&$pack.$field).get();
        v
    }};
    ($pack:ident . $field:ident) => {{
        let v = *$pack.$field.get();
        v
    }};
}

/// Copy `$size` elements from `$src` starting at `$src_offset` into `$dest`
/// starting at `$dest_offset`.
///
/// On the GPU side (inside `#[wgsl]` modules), this emits a WGSL `for` loop
/// of the form `dest[dest_offset + i] = src[src_offset + i]`.
///
/// On the CPU side, this is a simple element-by-element copy.
///
/// The macro is bidirectional: pass the slab as `$src` to read from a storage
/// buffer into a local array, or pass the slab as `$dest` to write from a
/// local array into a storage buffer.
///
/// # Example
/// ```ignore
/// let mut raw = [0u32; 4];
/// slab_copy!(get!(SLAB), offset, raw, 0, 4);
/// ```
#[macro_export]
macro_rules! slab_copy {
    ($src:expr, $src_offset:expr, $dest:expr, $dest_offset:expr, $size:expr) => {{
        let src_offset = $src_offset as usize;
        let dest_offset = $dest_offset as usize;
        for i in 0..$size as usize {
            $dest[dest_offset + i] = $src[src_offset + i];
        }
    }};
}

/// Marks the current fragment invocation as discarded.
///
/// In WGSL, this transpiles to `discard;`, which converts the invocation
/// into a helper invocation and prevents the fragment from being processed
/// downstream in the render pipeline.
///
/// On the CPU side (when running via `dispatch_fragments`), this sets a
/// thread-local flag. The shader continues executing (matching WGSL
/// semantics where helper invocations continue for derivative computation),
/// but `dispatch_fragments` suppresses the output.
///
/// # Example
/// ```ignore
/// if depth < 0.001 {
///     discard!();
/// }
/// ```
#[macro_export]
macro_rules! discard {
    () => {
        $crate::std::mark_discarded()
    };
}

/// CPU-side implementation of `discard!()`.
///
/// When the `dispatch-runtime` feature is enabled, this delegates to
/// the runtime's thread-local discard flag. Otherwise, it is a no-op.
#[inline]
pub fn mark_discarded() {
    #[cfg(feature = "dispatch-runtime")]
    {
        runtime::mark_discarded();
    }
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

impl Convert<f32> for f32 {
    fn convert(self) -> f32 {
        self
    }
}

impl Convert<u32> for u32 {
    fn convert(self) -> u32 {
        self
    }
}

impl Convert<i32> for i32 {
    fn convert(self) -> i32 {
        self
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

impl<T: Wgsl> Wgsl for RuntimeArray<T> {
    fn to_ir() -> ir::Type {
        ir::Type::RuntimeArray {
            elem: Box::new(T::to_ir()),
        }
    }
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

impl<T: Wgsl, A: AccessMode> ArrayLength for &Storage<RuntimeArray<T>, A> {
    fn array_length(self) -> u32 {
        self.get().len()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_holders_construct_with_values() {
        let uniform = Uniform::new(1.5_f32);
        let storage = Storage::<[f32; 4], ReadWrite>::new([0.0, 0.0, 0.0, 0.0]);
        let workgroup = Workgroup::new(3.5_f32);

        assert_eq!(*uniform.get(), 1.5_f32);
        assert_eq!(*storage.get(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(*workgroup.get(), 3.5_f32);

        storage.get_mut()[1] = 7.0;
        assert_eq!(*storage.get(), [0.0, 7.0, 0.0, 0.0]);
    }
}
