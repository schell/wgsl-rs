//! Synchronization Built-in Functions
//!
//! WGSL synchronization functions for compute shader workgroup coordination.
//! See: <https://gpuweb.github.io/gpuweb/wgsl/#sync-builtin-functions>
//!
//! All synchronization functions execute a control barrier with Acquire/Release
//! memory ordering. They use the `Workgroup` memory scope and execution scope.
//!
//! All synchronization functions must only be used in the compute shader stage
//! and must only be invoked in uniform control flow.
//!
//! ## CPU-Side Behavior
//!
//! On the CPU side, the barrier functions are currently no-ops because there
//! is no parallel dispatch runtime. When a multi-threaded compute dispatch
//! is implemented, these should be revisited to use proper thread barriers.

use super::{
    Workgroup,
    atomic::{Atomic, atomic_load, atomic_load_i32},
};

/// Executes a control barrier synchronization function that affects memory
/// and atomic operations in the storage address space.
///
/// Ensures all storage buffer writes by invocations in the workgroup are
/// visible to all other invocations before any subsequent storage memory
/// or atomic operations execute.
///
/// # Compute Shader Only
/// This function must only be called from compute shader entry points
/// in uniform control flow.
///
/// # CPU-Side Behavior
/// When the `dispatch-runtime` feature is enabled and the calling thread is
/// within a `dispatch_workgroups` invocation, this function synchronizes via
/// a shared barrier. Otherwise, it is a no-op.
pub fn storage_barrier() {
    // On the CPU, all memory is coherent — we just need execution
    // synchronization, which workgroup_barrier provides.
    workgroup_barrier();
}

/// Executes a control barrier synchronization function that affects memory
/// operations in the handle address space.
///
/// Ensures all texture memory operations by invocations in the workgroup
/// are visible to all other invocations before any subsequent handle memory
/// operations execute.
///
/// # Compute Shader Only
/// This function must only be called from compute shader entry points
/// in uniform control flow.
///
/// # CPU-Side Behavior
/// When the `dispatch-runtime` feature is enabled and the calling thread is
/// within a `dispatch_workgroups` invocation, this function synchronizes via
/// a shared barrier. Otherwise, it is a no-op.
pub fn texture_barrier() {
    workgroup_barrier();
}

/// Executes a control barrier synchronization function that affects memory
/// and atomic operations in the workgroup address space.
///
/// Ensures all workgroup memory writes by invocations are visible to all
/// other invocations, and synchronizes execution across the workgroup.
///
/// # Compute Shader Only
/// This function must only be called from compute shader entry points
/// in uniform control flow.
///
/// # CPU-Side Behavior
/// When the `dispatch-runtime` feature is enabled and the calling thread is
/// within a `dispatch_workgroups` invocation, this function synchronizes via
/// a shared barrier. Otherwise, it is a no-op.
pub fn workgroup_barrier() {
    #[cfg(feature = "dispatch-runtime")]
    {
        crate::std::runtime::with_workgroup_barrier(|barrier| {
            if let Some(b) = barrier {
                b.wait();
            }
        });
    }
}

/// Trait for types that support `workgroupUniformLoad`.
///
/// Returns the value from a workgroup variable, ensuring the value is
/// uniform across the workgroup. Also executes a workgroup barrier.
///
/// This trait has two families of implementations:
/// - `Workgroup<T>` where `T: Clone` — returns a clone of the stored value
/// - `Workgroup<Atomic<T>>` — atomically loads and returns the inner scalar
pub trait WorkgroupUniformLoad {
    /// The type returned by the uniform load.
    type Output;

    /// Performs the uniform load, returning the value.
    fn uniform_load(&self) -> Self::Output;
}

impl<T: Clone> WorkgroupUniformLoad for Workgroup<T> {
    type Output = T;

    fn uniform_load(&self) -> T {
        self.get().clone()
    }
}

/// Returns the value pointed to by a workgroup variable, ensuring the
/// value is uniform across the workgroup.
///
/// This is the Rust equivalent of WGSL's
/// `workgroupUniformLoad(p: ptr<workgroup, T>) -> T`.
///
/// The return value is uniform — all invocations in the workgroup observe
/// the same value. Also executes a workgroup barrier.
///
/// # Overloads
///
/// In WGSL, `workgroupUniformLoad` has two overloads:
/// - `fn workgroupUniformLoad(p: ptr<workgroup, T>) -> T` for constructible
///   types
/// - `fn workgroupUniformLoad(p: ptr<workgroup, atomic<T>, read_write>) -> T`
///   for atomic types (returns the inner scalar, not the atomic wrapper)
///
/// Both overloads are handled by the [`WorkgroupUniformLoad`] trait.
///
/// # Compute Shader Only
/// This function must only be called from compute shader entry points
/// in uniform control flow.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// workgroup!(SHARED_VALUE: u32);
///
/// #[compute]
/// #[workgroup_size(64)]
/// fn main(#[builtin(local_invocation_index)] local_idx: u32) {
///     let value: u32 = workgroup_uniform_load(&SHARED_VALUE);
/// }
/// ```
pub fn workgroup_uniform_load<W: WorkgroupUniformLoad>(p: &W) -> W::Output {
    p.uniform_load()
}

/// Specialized implementation for `Workgroup<Atomic<u32>>`.
///
/// Returns the `u32` scalar value via an atomic load, matching the WGSL
/// overload `workgroupUniformLoad(p: ptr<workgroup, atomic<u32>, read_write>)
/// -> u32`.
impl WorkgroupUniformLoad for Workgroup<Atomic<u32>> {
    type Output = u32;

    fn uniform_load(&self) -> u32 {
        atomic_load(&self.get())
    }
}

/// Specialized implementation for `Workgroup<Atomic<i32>>`.
///
/// Returns the `i32` scalar value via an atomic load, matching the WGSL
/// overload `workgroupUniformLoad(p: ptr<workgroup, atomic<i32>, read_write>)
/// -> i32`.
impl WorkgroupUniformLoad for Workgroup<Atomic<i32>> {
    type Output = i32;

    fn uniform_load(&self) -> i32 {
        atomic_load_i32(&self.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_barrier_is_callable() {
        // Barrier functions should be callable without panicking.
        // They are no-ops on the CPU side.
        storage_barrier();
    }

    #[test]
    fn test_texture_barrier_is_callable() {
        texture_barrier();
    }

    #[test]
    fn test_workgroup_barrier_is_callable() {
        workgroup_barrier();
    }

    #[test]
    fn test_workgroup_uniform_load_u32() {
        let wg: Workgroup<u32> = Workgroup::new();
        wg.set(42);
        let value = workgroup_uniform_load(&wg);
        assert_eq!(value, 42u32);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_workgroup_uniform_load_f32() {
        let wg: Workgroup<f32> = Workgroup::new();
        wg.set(3.14);
        let value = workgroup_uniform_load(&wg);
        assert!((value - 3.14).abs() < f32::EPSILON);
    }

    #[test]
    fn test_workgroup_uniform_load_atomic_u32() {
        use crate::std::atomic::atomic_store;

        let wg: Workgroup<Atomic<u32>> = Workgroup::new();
        wg.set(Atomic::default());
        atomic_store(&wg.get(), 99);
        let value = workgroup_uniform_load(&wg);
        assert_eq!(value, 99u32);
    }

    #[test]
    fn test_workgroup_uniform_load_atomic_i32() {
        use crate::std::atomic::atomic_store_i32;

        let wg: Workgroup<Atomic<i32>> = Workgroup::new();
        wg.set(Atomic::default());
        atomic_store_i32(&wg.get(), -42);
        let value = workgroup_uniform_load(&wg);
        assert_eq!(value, -42i32);
    }
}
