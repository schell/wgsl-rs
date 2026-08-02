//! Regression test for wgsl-rs#84.
//!
//! `#[fragment]` previously did not strip `#[builtin(...)]` attributes from
//! function parameters, causing a hard Rust compile error:
//! "expected non-macro attribute, found attribute macro `builtin`" when
//! using `#[builtin(position)]` directly on a fragment function parameter.
//! The fix mirrors the stripping logic already present in `#[vertex]` and
//! `#[compute]`.

#![allow(dead_code)]

use wgsl_rs::wgsl;

#[wgsl]
mod fragment_builtin {
    use wgsl_rs::std::*;

    // Direct `#[builtin(...)]` on a fragment parameter — the pattern that
    // failed before the fix. `#[vertex]` and `#[compute]` already supported
    // this; `#[fragment]` now does too.
    #[fragment]
    pub fn frag_main(#[builtin(position)] frag_coord: Vec4f) -> Vec4f {
        vec4f(frag_coord.x, frag_coord.y, frag_coord.z, 1.0)
    }

    // A fragment entry point with no builtins should still work.
    #[fragment]
    pub fn frag_no_builtin() -> Vec4f {
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}

#[test]
fn fragment_with_builtin_param_compiles_and_renders() {
    let src = fragment_builtin::WGSL_SOURCE.wgsl_source().unwrap();
    // The WGSL should contain the `@builtin(position)` decoration on the
    // fragment input, and a valid `@fragment` entry point.
    assert!(
        src.contains("@builtin(position)"),
        "expected @builtin(position) in WGSL, got: {src}"
    );
    assert!(
        src.contains("@fragment"),
        "expected @fragment in WGSL, got: {src}"
    );
    // The Rust-side `builtin` attribute macro must not leak into WGSL.
    assert!(
        !src.contains("#[builtin"),
        "WGSL should not contain #[builtin], got: {src}"
    );
}
