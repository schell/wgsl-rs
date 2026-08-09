use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod assoc_type_test {
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

    pub fn use_it() -> u32 {
        let arr = u32::to_array(42u32);
        u32::from_array(arr)
    }
}

fn main() {
    let source = assoc_type_test::WGSL_SOURCE.wgsl_source().unwrap();
    // The associated type alias should be emitted as a WGSL alias.
    assert!(
        source.contains("alias u32_Array = array<u32, 1>;"),
        "expected u32_Array alias in WGSL, got:\n{source}"
    );
    // Self::Array should have been resolved to [u32; 1] (array<u32, 1>).
    assert!(
        source.contains("fn u32__1to_array(data: u32) -> array<u32, 1>"),
        "expected u32__1to_array with resolved return type, got:\n{source}"
    );
    assert!(
        source.contains("fn u32__1from_array(arr: array<u32, 1>) -> u32"),
        "expected u32__1from_array with resolved param type, got:\n{source}"
    );
}
