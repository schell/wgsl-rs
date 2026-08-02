use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod complex_trait {

    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    impl Zeroable for [u32; 4] {
        fn zero() -> [u32; 4] {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    pub fn go<T: Zeroable>() -> T {
        T::zero()
    }

    pub fn caller() -> [u32; 4] {
        go::<[u32; 4]>()
    }
}

#[test]
fn array_trait_impl_transpiles() {
    let src = complex_trait::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("u32_zero"),
        "simple trait impl should transpile to u32_zero, got:\n{src}"
    );
    assert!(
        src.contains("_2array_u32_4_zero"),
        "array trait impl should mangle to _2array_u32_4_zero, got:\n{src}"
    );
}

#[test]
fn array_trait_impl_runs_on_cpu() {
    assert_eq!(complex_trait::caller(), [0u32, 0u32, 0u32, 0u32]);
}
