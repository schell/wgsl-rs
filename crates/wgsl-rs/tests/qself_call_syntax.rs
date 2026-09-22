//! Tests for direct `<T>::method()` / `<T>::CONSTANT` call syntax on complex
//! (non-ident) self types — the "QSelf paths" of issue #131.
//!
//! The generic form `T::method()` inside `fn f<T: Trait>()` already worked
//! (it resolves via monomorphization — see `complex_type_trait_impl.rs`).
//! These tests cover the *direct* form where the caller names the concrete
//! complex type itself, e.g. `<[u32; 4]>::zero()`.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod qself_call {
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

    // An inherent impl on a user struct, with an associated constant accessed
    // via QSelf below. (Inherent impls on primitive arrays like `[u32; 4]`
    // are forbidden by Rust, so the constant lives on a struct instead —
    // this still exercises the `Expr::Path` QSelf branch.)
    pub struct Tag {
        pub _x: u32,
    }

    impl Tag {
        pub const DEFAULT: u32 = 7;
    }

    /// Direct QSelf call: `<[u32; 4]>::zero()`.
    pub fn direct_array_zero() -> [u32; 4] {
        <[u32; 4]>::zero()
    }

    /// Direct QSelf call on a scalar, for symmetry with the array case.
    pub fn direct_scalar_zero() -> u32 {
        <u32>::zero()
    }

    /// Direct QSelf access of an associated constant.
    pub fn direct_struct_default() -> u32 {
        <Tag>::DEFAULT
    }
}

#[test]
fn qself_array_call_transpiles() {
    let src = qself_call::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("_2array_u32_4_zero"),
        "direct `<[u32; 4]>::zero()` should mangle to `_2array_u32_4_zero`, got:\n{src}"
    );
    assert!(
        src.contains("u32_zero"),
        "direct `<u32>::zero()` should resolve to `u32_zero`, got:\n{src}"
    );
    assert!(
        src.contains("Tag_DEFAULT"),
        "direct `<Tag>::DEFAULT` should resolve to `Tag_DEFAULT`, got:\n{src}"
    );
}

#[test]
fn qself_array_call_runs_on_cpu() {
    assert_eq!(qself_call::direct_array_zero(), [0u32, 0u32, 0u32, 0u32]);
    assert_eq!(qself_call::direct_scalar_zero(), 0);
    assert_eq!(qself_call::direct_struct_default(), 7);
}

#[wgsl(skip_validation)]
mod generic_qself {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    /// QSelf call on a *type parameter*: `<T>::zero()` inside a generic
    /// function must be rewritten by monomorphization exactly like the
    /// plain `T::zero()` form (substitution keys on the param name "T",
    /// so the qself ty ident must not be mangled/lowercased to "t").
    pub fn via_qself<T: Zeroable>() -> T {
        <T>::zero()
    }

    pub fn caller() -> u32 {
        via_qself::<u32>()
    }
}

#[test]
fn qself_type_param_call_transpiles() {
    let src = generic_qself::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !src.contains("t_zero"),
        "generic `<T>::zero()` must be substituted, not left as a mangled `t_zero()` call, \
         got:\n{src}"
    );
    assert!(
        src.contains("u32_zero"),
        "generic `<T>::zero()` should resolve to `u32_zero` after monomorphization, got:\n{src}"
    );
}

#[test]
fn qself_type_param_call_runs_on_cpu() {
    assert_eq!(generic_qself::caller(), 0);
}

#[wgsl(skip_validation)]
mod compound_qself {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    /// A generic struct whose trait impl is instantiated alongside the
    /// struct (see `instantiate_struct`).
    pub struct Pair<T> {
        pub a: T,
        pub b: T,
    }

    impl<T: Zeroable> Zeroable for Pair<T> {
        fn zero() -> Pair<T> {
            Pair {
                a: T::zero(),
                b: T::zero(),
            }
        }
    }

    impl<T: Zeroable> Zeroable for [T; 4] {
        fn zero() -> [T; 4] {
            [T::zero(), T::zero(), T::zero(), T::zero()]
        }
    }

    /// QSelf call on a *compound* type containing a type parameter:
    /// `<[T; 4]>::zero()` inside a generic function must be rewritten by
    /// monomorphization to the concrete array impl's function
    /// (`array_u32_4_zero` for `T = u32`), not left as the eagerly mangled
    /// generic form (`array_t_4_zero`).
    pub fn via_array_qself<T: Zeroable>() -> [T; 4] {
        <[T; 4]>::zero()
    }

    /// QSelf call on a generic *struct* type containing a type parameter:
    /// `<Pair<T>>::zero()` must resolve to the struct-instantiated method
    /// (`Pair_u32_zero` for `T = u32`).
    pub fn via_struct_qself<T: Zeroable>() -> Pair<T> {
        <Pair<T>>::zero()
    }

    /// QSelf call on a *nested* compound type: `<[[T; 4]; 4]>::zero()`.
    /// The type parameter hides inside an escaped mangled component
    /// (`array__2array_t_4_4` un-mangles to `[array, array_t_4, 4]`), so
    /// substitution must recurse into the nested component rather than
    /// matching only top-level ones.
    pub fn via_nested_array_qself<T: Zeroable>() -> [[T; 4]; 4] {
        <[[T; 4]; 4]>::zero()
    }

    pub fn array_caller() -> [u32; 4] {
        via_array_qself::<u32>()
    }

    pub fn struct_caller() -> Pair<u32> {
        via_struct_qself::<u32>()
    }

    pub fn nested_array_caller() -> [[u32; 4]; 4] {
        via_nested_array_qself::<u32>()
    }
}

#[test]
fn qself_compound_type_call_transpiles() {
    let src = compound_qself::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !src.contains("array_T_4"),
        "compound `<[T; 4]>::zero()` must be substituted, not left as the generic mangled \
         `array_T_4_zero()`, got:\n{src}"
    );
    assert!(
        src.contains("_2array_u32_4_zero"),
        "compound `<[T; 4]>::zero()` should resolve to `array_u32_4_zero` after monomorphization, \
         got:\n{src}"
    );
    assert!(
        !src.contains("Pair_T"),
        "compound `<Pair<T>>::zero()` must be substituted, not left as the generic mangled \
         `Pair_T_zero()`, got:\n{src}"
    );
    assert!(
        src.contains("Pair_u32_zero"),
        "compound `<Pair<T>>::zero()` should resolve to the struct-instantiated `Pair_u32_zero` \
         after monomorphization, got:\n{src}"
    );
    assert!(
        src.contains("array__2array_u32_4_4"),
        "nested `<[[T; 4]; 4]>::zero()` should resolve to the array-impl instantiated \
         `array__2array_u32_4_4_zero` after monomorphization, got:\n{src}"
    );
}

#[test]
fn qself_compound_type_call_runs_on_cpu() {
    assert_eq!(compound_qself::array_caller(), [0u32; 4]);
    let p = compound_qself::struct_caller();
    assert_eq!((p.a, p.b), (0u32, 0u32));
    let nested = compound_qself::nested_array_caller();
    assert_eq!(nested, [[0u32; 4]; 4]);
}

#[wgsl(skip_validation)]
mod qself_case_sensitivity {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    /// A concrete struct whose name is the *lowercase spelling* of a type
    /// parameter used in the same module (`T`). Substitution must match
    /// components exactly, so `t` is never confused with `T`.
    #[allow(non_camel_case_types)] // lowercase is the point of this test
    pub struct t {
        pub x: u32,
    }

    impl Zeroable for t {
        fn zero() -> t {
            t { x: 0 }
        }
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    /// Direct QSelf call on the concrete lowercase struct `t`, from
    /// inside a generic function whose parameter is `T`. The call must
    /// resolve to the concrete impl (`t_zero`), not to the param-
    /// substituted form (`u32_zero`).
    pub fn concrete_lowercase_qself<T: Zeroable>(seed: T) -> t {
        let _keep: T = seed;
        <t>::zero()
    }

    pub fn caller() -> u32 {
        concrete_lowercase_qself::<u32>(0).x
    }
}

#[test]
fn qself_case_sensitive_components_transpiles() {
    let src = qself_case_sensitivity::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("return t_zero();"),
        "concrete `<t>::zero()` must resolve to the concrete `t_zero` impl, got:\n{src}"
    );
}

#[test]
fn qself_case_sensitive_components_runs_on_cpu() {
    assert_eq!(qself_case_sensitivity::caller(), 0);
}

#[allow(non_camel_case_types)] // underscore/lowercase names are the point of this module
#[wgsl(skip_validation)]
mod qself_underscore_idents {
    /// A concrete struct whose name contains an underscore, whose trailing
    /// string component matches the type parameter `bar` in scope below.
    /// The mangled ident `Foo_bar` must never be decomposed and rewritten
    /// as `Foo_u32`.
    pub struct Foo_bar {
        pub x: u32,
    }

    impl Foo_bar {
        pub fn get() -> u32 {
            let f = Foo_bar { x: 7 };
            f.x
        }
    }

    /// QSelf call on the concrete underscore-named struct `Foo_bar`, from
    /// inside a generic function whose parameter is `bar`. The call must
    /// resolve to the concrete impl method (`Foo_bar_get`), not to a
    /// decomposed, param-substituted form.
    pub fn underscore_ident_qself<bar>(seed: bar) -> u32 {
        let _keep: bar = seed;
        <Foo_bar>::get()
    }

    pub fn caller() -> u32 {
        underscore_ident_qself::<u32>(0)
    }
}

#[test]
fn qself_underscore_ident_not_decomposed() {
    let src = qself_underscore_idents::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("Foo_bar_get"),
        "concrete `<Foo_bar>::get()` must resolve to `Foo_bar_get` (rendered as `_1Foo_bar_get`), \
         got:\n{src}"
    );
    assert!(
        !src.contains("Foo_u32"),
        "the underscore in `Foo_bar` must not be mistaken for a mangling separator, got:\n{src}"
    );
}

#[test]
fn qself_underscore_ident_runs_on_cpu() {
    assert_eq!(qself_underscore_idents::caller(), 7);
}

#[wgsl(skip_validation)]
mod qself_const_generic {
    use wgsl_rs::std::*; // brings `wgsl_ignore` into scope

    pub trait Zeroable {
        fn zero() -> Self;
    }

    /// CPU-side resolution for `<[u32; N]>::zero()`. The transpiler skips
    /// this impl (`#[wgsl_ignore]`) because const-generic array impls are
    /// not yet supported on the WGSL side (#133, pinned by the
    /// `generic_impl_const_generic_array` trybuild test), so this module
    /// has no WGSL-side target function. The test below asserts the
    /// QSelf-side *rewrite* only.
    #[wgsl_ignore]
    impl<const N: usize> Zeroable for [u32; N] {
        fn zero() -> [u32; N] {
            [0; N]
        }
    }

    /// Const-generic QSelf: `<[u32; N]>::zero()`. The const parameter N
    /// appears in the array *length* — an `Expr`, not a `TypeParam`, so
    /// type-parameter detection alone never triggers the rewrite (which
    /// is why the substitution is unconditional). With `N = 4` the call
    /// must be rewritten to the concrete spelling `array_u32_4_zero`,
    /// not left as `array_u32_N_zero`.
    pub fn via_qself<const N: usize>() -> [u32; N] {
        <[u32; N]>::zero()
    }

    pub fn caller() -> [u32; 4] {
        via_qself::<4>()
    }
}

#[test]
fn qself_const_generic_call_is_rewritten() {
    let src = qself_const_generic::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !src.contains("array_u32_N"),
        "const-generic `<[u32; N]>::zero()` must have N substituted, not left as \
         `array_u32_N_zero()`, got:\n{src}"
    );
    assert!(
        src.contains("array_u32_4_zero"),
        "const-generic `<[u32; N]>::zero()` should be rewritten to the concrete spelling \
         `array_u32_4_zero` for N = 4, got:\n{src}"
    );
}

#[test]
fn qself_const_generic_call_runs_on_cpu() {
    assert_eq!(qself_const_generic::caller(), [0u32; 4]);
}
