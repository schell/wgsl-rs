//! Associated type aliases on generic array impl blocks are monomorphized
//! into module-scope WGSL `alias` declarations, mirroring the struct-impl
//! path (wgsl-rs#144).

use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod assoc_type_on_array_test {
    pub trait Slab {
        type Array;
        fn to_array(data: Self) -> Self::Array;
    }

    impl Slab for u32 {
        type Array = [u32; 1];
        fn to_array(data: Self) -> Self::Array {
            [data]
        }
    }

    /// Generic array impl whose alias substitutes `T` and whose method
    /// signatures reference `Self::Array`.
    impl<T: Slab> Slab for [T; 4] {
        type Array = [T; 4];
        fn to_array(data: Self) -> Self::Array {
            data
        }
    }

    pub fn use_it() -> u32 {
        let arr: [u32; 4] = <[u32; 4]>::to_array([1u32, 2u32, 3u32, 4u32]);
        arr[0]
    }
}

fn main() {
    let source = assoc_type_on_array_test::WGSL_SOURCE
        .wgsl_source()
        .unwrap();

    // The core fix: the array impl's associated type alias is emitted as a
    // module-scope WGSL `alias`, with `T` substituted (T = u32). The self
    // type `array_u32_4` has 2 underscores, so composing it with the member
    // name escapes it as `_2array_u32_4`.
    assert!(
        source.contains("alias _2array_u32_4_Array = array<u32, 4>;"),
        "expected array_u32_4_Array alias with substituted element type, got:\n{source}"
    );

    // The scalar impl's alias renders alongside it (existing behavior).
    assert!(
        source.contains("alias u32_Array = array<u32, 1>;"),
        "expected u32_Array alias from the scalar impl, got:\n{source}"
    );

    // The monomorphized method still emits, with the associated type
    // reference `Self::Array` resolved to the concrete array type (same
    // resolution as struct impls).
    assert!(
        source.contains("fn _2array_u32_4__1to_array")
            && source.contains("-> array<u32, 4>"),
        "expected _2array_u32_4__1to_array with resolved return type, got:\n{source}"
    );

    // The qself call site rewrites to the mangled method name (the array
    // literal argument renders as the `array(...)` constructor).
    assert!(
        source.contains("_2array_u32_4__1to_array(array("),
        "expected the <[u32; 4]>::to_array call site to be rewritten, got:\n{source}"
    );

    // The generic impl template itself is removed.
    assert!(
        !source.contains("fn to_array("),
        "the generic impl template should be removed from the WGSL, got:\n{source}"
    );
}