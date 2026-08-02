//! Regression test for wgsl-rs#85.
//!
//! Rust numeric literals with type suffixes like `0.0_f32`, `5_i32`, `10_u32`
//! must not be emitted verbatim into the generated WGSL — WGSL does not
//! recognize Rust-style suffixes. The IR conversion strips the suffix so the
//! rendered WGSL contains only the plain mantissa (e.g. `0.0`), which is
//! valid WGSL `f32`/`i32`/`u32` literal syntax.

#![allow(dead_code)]

use wgsl_rs::wgsl;

#[wgsl]
mod typed_suffixes {
    pub fn suffixed_float() -> f32 {
        let v = 0.0_f32;
        let a = 0.5_f32;
        v + a
    }

    pub fn suffixed_ints() -> u32 {
        let i = 5_i32;
        let u = 10_u32;
        u + (i as u32)
    }

    pub fn plain_literals() -> f32 {
        let v = 0.0;
        let a = 0.5;
        v + a
    }
}

#[test]
fn suffixed_floats_are_stripped_in_wgsl() {
    let src = typed_suffixes::WGSL_SOURCE.wgsl_source().unwrap();
    // The specific suffixed literals used in the test module must not appear
    // verbatim — the suffix should be stripped, leaving only the mantissa.
    assert!(
        !src.contains("0.0_f32"),
        "rendered WGSL should not contain 0.0_f32, got: {src}"
    );
    assert!(
        !src.contains("0.5_f32"),
        "rendered WGSL should not contain 0.5_f32, got: {src}"
    );
    assert!(
        !src.contains("5_i32"),
        "rendered WGSL should not contain 5_i32, got: {src}"
    );
    assert!(
        !src.contains("10_u32"),
        "rendered WGSL should not contain 10_u32, got: {src}"
    );
    // The stripped mantissas/digits should be present.
    assert!(
        src.contains("0.0"),
        "expected stripped 0.0 in WGSL, got: {src}"
    );
    assert!(
        src.contains("0.5"),
        "expected stripped 0.5 in WGSL, got: {src}"
    );
}

#[test]
fn plain_literals_still_render_correctly() {
    let src = typed_suffixes::WGSL_SOURCE.wgsl_source().unwrap();
    // Plain literals should round-trip unchanged.
    assert!(
        src.contains("0.0"),
        "expected plain 0.0 in WGSL, got: {src}"
    );
    assert!(
        src.contains("0.5"),
        "expected plain 0.5 in WGSL, got: {src}"
    );
}
