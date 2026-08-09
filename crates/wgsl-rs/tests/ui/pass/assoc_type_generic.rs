use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod assoc_type_generic_test {
    use wgsl_rs::std::*;

    pub trait SlabItem {
        const SLAB_SIZE: usize;
        type Array: Default;
        fn to_array(data: Self) -> Self::Array;
        fn from_array(arr: Self::Array) -> Self;
    }

    impl SlabItem for u32 {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];
        fn to_array(data: Self) -> Self::Array {
            [data]
        }
        fn from_array(arr: Self::Array) -> Self {
            arr[0]
        }
    }

    impl SlabItem for f32 {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];
        fn to_array(data: Self) -> Self::Array {
            [bitcast_u32(data)]
        }
        fn from_array(arr: Self::Array) -> Self {
            bitcast_f32(arr[0])
        }
    }

    /// A generic wrapper that delegates `Array` to the inner type's `Array`.
    pub struct Wrapper<T> {
        pub inner: T,
    }

    impl<T: SlabItem> SlabItem for Wrapper<T> {
        const SLAB_SIZE: usize = T::SLAB_SIZE;
        type Array = T::Array;
        fn to_array(data: Self) -> Self::Array {
            T::to_array(data.inner)
        }
        fn from_array(arr: Self::Array) -> Self {
            Wrapper {
                inner: T::from_array(arr),
            }
        }
    }

    pub fn use_wrapper_u32() -> u32 {
        let w = Wrapper::<u32> { inner: 42u32 };
        let arr = Wrapper::<u32>::to_array(w);
        let w2 = Wrapper::<u32>::from_array(arr);
        w2.inner
    }

    pub fn use_wrapper_f32() -> f32 {
        let w = Wrapper::<f32> { inner: 3.14f32 };
        let arr = Wrapper::<f32>::to_array(w);
        let w2 = Wrapper::<f32>::from_array(arr);
        w2.inner
    }
}

fn main() {
    let source = assoc_type_generic_test::WGSL_SOURCE
        .wgsl_source()
        .unwrap();

    // The monomorphized Wrapper_u32 impl should have its Array resolved
    // to [u32; 1] (from T::Array where T = u32).
    assert!(
        source.contains("alias _1Wrapper_u32_Array = array<u32, 1>;"),
        "expected Wrapper_u32_Array alias resolved to array<u32, 1>, got:\n{source}"
    );

    // The to_array method should have the resolved return type.
    assert!(
        source.contains("fn _1Wrapper_u32__1to_array")
            && source.contains("-> array<u32, 1>"),
        "expected Wrapper_u32__1to_array with resolved return type, got:\n{source}"
    );

    // The from_array method should have the resolved param type.
    assert!(
        source.contains("fn _1Wrapper_u32__1from_array")
            && source.contains("arr: array<u32, 1>"),
        "expected Wrapper_u32__1from_array with resolved param type, got:\n{source}"
    );

    // Same for f32 — T::Array resolves to [u32; 1] (from impl SlabItem for f32).
    assert!(
        source.contains("alias _1Wrapper_f32_Array = array<u32, 1>;"),
        "expected Wrapper_f32_Array alias resolved to array<u32, 1>, got:\n{source}"
    );

    // The SLAB_SIZE const should be resolved (T::SLAB_SIZE → u32's SLAB_SIZE).
    // After substitution, T::SLAB_SIZE becomes u32::SLAB_SIZE which renders
    // as u32__1SLAB_SIZE (the mangled associated const name).
    assert!(
        source.contains("const _1Wrapper_u32__1SLAB_SIZE: u32 = u32__1SLAB_SIZE;")
            || source.contains("const _1Wrapper_u32__1SLAB_SIZE: u32 = 1u;"),
        "expected Wrapper_u32 SLAB_SIZE to be resolved from T::SLAB_SIZE, got:\n{source}"
    );
}