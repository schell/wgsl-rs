//! Regression tests for wgsl-rs#160: Rust's `!` on integer-typed operands
//! is bitwise complement, which WGSL spells `~` — `!` is reserved for
//! logical not. The macro must lower integer `!` to `~` while bool `!`
//! stays `!`.
//!
//! These drive the real `#[wgsl]` macro, render to WGSL, and assert on both
//! specific substrings (to pin the exact lowering) and naga validation.
//! `validation` is a default feature, so naga runs in normal `cargo test`.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::let_and_return)]

use wgsl_rs::wgsl;

/// The #160 repro shape: `let x = !0;` (issue used a compute entry point;
/// a plain function renders the same let and validates the same).
///
/// `skip_validation` turns off the macro's auto `__validate_wgsl` test, which
/// is emitted without a `validation` feature gate and breaks
/// `--no-default-features` builds; the explicit `#[cfg(feature =
/// "validation")]` tests below cover validation instead.
#[wgsl(skip_validation)]
mod complement_repro {
    pub fn compute() -> i32 {
        let x = !0;
        x
    }
}

#[wgsl(skip_validation)]
mod complement_lowering {
    pub fn complement_i32() -> i32 {
        !0
    }

    pub fn complement_u32() -> u32 {
        !0u32
    }

    pub fn double_complement() -> i32 {
        !!0
    }

    pub fn complement_of_expr() -> u32 {
        !(1u32 & 2u32)
    }

    pub fn complement_of_neg() -> i32 {
        !-1
    }

    pub fn complement_of_cast(x: i32) -> u32 {
        !(x as u32)
    }

    pub fn not_bool(flag: bool) -> bool {
        !flag
    }

    pub fn not_bool_literal() -> bool {
        !true
    }

    pub fn not_of_comparison(a: u32, b: u32) -> bool {
        !(a == b)
    }
}

// ===== WGSL source assertions =====

#[test]
fn repro_renders_complement_not_not() {
    let src = complement_repro::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("let x = ~0;"),
        "the #160 repro must emit `~0`, got: {src}"
    );
    assert!(
        !src.contains("!0"),
        "the #160 repro must not emit `!0`, got: {src}"
    );
}

#[test]
fn integer_complements_lower_to_tilde() {
    let src = complement_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    for (needle, why) in [
        (
            "return ~0i;",
            "`!0` (i32) lowers to `~` (suffix from the i32 return)",
        ),
        ("return ~0u;", "`!0u32` lowers to `~`"),
        ("return ~~0i;", "`!!0` is complement-of-complement"),
        ("return ~(1u & 2u);", "`!(a & b)` on ints lowers to `~`"),
        ("return ~-1i;", "`!-1` lowers to `~`"),
        ("return ~(u32(x));", "`!(x as u32)` lowers to `~`"),
    ] {
        assert!(
            src.contains(needle),
            "expected `{needle}` ({why}) in: {src}"
        );
    }
    assert!(
        !src.contains("!0"),
        "no integer `!` should survive lowering, got: {src}"
    );
}

#[test]
fn bool_negation_stays_bang() {
    let src = complement_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    for (needle, why) in [
        ("return !flag;", "bool `!` must stay `!`"),
        ("return !true;", "`!true` must stay `!`"),
        ("return !(a == b);", "`!(a == b)` is logical not of a bool"),
    ] {
        assert!(
            src.contains(needle),
            "expected `{needle}` ({why}) in: {src}"
        );
    }
    assert!(
        !src.contains("~flag"),
        "bool operands must not lower to `~`, got: {src}"
    );
}

// ===== naga validation =====
//
// `Source::validate()` only exists with the `validation` feature, and the
// crate supports `default-features = false`; like `shadowing.rs`, the
// validation tests are gated and the substring tests above stay
// unconditional.

#[cfg(feature = "validation")]
#[test]
fn repro_validates() {
    complement_repro::WGSL_SOURCE
        .validate()
        .expect("naga validation");
}

#[cfg(feature = "validation")]
#[test]
fn lowering_validates() {
    complement_lowering::WGSL_SOURCE
        .validate()
        .expect("naga validation");
}

// ===== CPU-side parity =====
//
// The `#[wgsl]` module stays valid Rust; these lock the two-worlds agreement:
// GPU `~` must produce the same values as Rust's integer `!`.

#[test]
fn cpu_complement_values() {
    use complement_lowering as m;
    assert_eq!(m::complement_i32(), !0);
    assert_eq!(m::complement_u32(), !0u32);
    assert_eq!(m::double_complement(), !!0);
    assert_eq!(m::complement_of_expr(), !(1u32 & 2u32));
    assert_eq!(m::complement_of_neg(), !-1);
    assert_eq!(m::complement_of_cast(5), !(5i32 as u32));
    assert!(!m::not_bool(true));
    assert!(!m::not_bool_literal());
    assert!(!m::not_of_comparison(1, 1));
}
