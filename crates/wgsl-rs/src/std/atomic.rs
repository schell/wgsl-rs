//! # Atomic Builtin Functions
//!
//! WGSL atomic builtin functions for thread-safe memory operations.
//! See: <https://gpuweb.github.io/gpuweb/wgsl/#atomic-builtin-functions>
//!
//! All operations use Relaxed memory ordering, which matches WGSL semantics.
//! WGSL atomics only operate on `atomic<i32>` and `atomic<u32>` types.

use std::sync::atomic::Ordering;

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
