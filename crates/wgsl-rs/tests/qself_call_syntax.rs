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

    pub fn array_caller() -> [u32; 4] {
        via_array_qself::<u32>()
    }

    pub fn struct_caller() -> Pair<u32> {
        via_struct_qself::<u32>()
    }
}

#[test]
fn qself_compound_type_call_transpiles() {
    let src = compound_qself::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !src.contains("array_t_4"),
        "compound `<[T; 4]>::zero()` must be substituted, not left as the generic mangled \
         `array_t_4_zero()`, got:\n{src}"
    );
    assert!(
        src.contains("_2array_u32_4_zero"),
        "compound `<[T; 4]>::zero()` should resolve to `array_u32_4_zero` after monomorphization, \
         got:\n{src}"
    );
    assert!(
        !src.contains("Pair_t"),
        "compound `<Pair<T>>::zero()` must be substituted, not left as the generic mangled \
         `Pair_t_zero()`, got:\n{src}"
    );
    assert!(
        src.contains("Pair_u32_zero"),
        "compound `<Pair<T>>::zero()` should resolve to the struct-instantiated `Pair_u32_zero` \
         after monomorphization, got:\n{src}"
    );
}

#[test]
fn qself_compound_type_call_runs_on_cpu() {
    assert_eq!(compound_qself::array_caller(), [0u32; 4]);
    let p = compound_qself::struct_caller();
    assert_eq!((p.a, p.b), (0u32, 0u32));
}
