//! Regression test for wgsl-rs#57.
//!
//! Valid `match` patterns that do NOT mix `default` with literals must still
//! transpile correctly. This guards against over-rejection from the parse-time
//! validation added for #57. The macro-generated `__validate_wgsl` test
//! confirms the rendered WGSL is valid.

#![allow(dead_code)]
#![allow(clippy::manual_range_patterns)]
#![allow(unused_assignments)]

use wgsl_rs::wgsl;

#[wgsl]
mod valid_switches {
    pub fn or_literals(x: u32) -> u32 {
        let mut result: u32 = 0u32;
        match x {
            0 | 1 | 2 => {
                result = 1u32;
            }
            _ => {
                result = 0u32;
            }
        }
        result
    }

    pub fn explicit_default(x: u32) -> u32 {
        let mut result: u32 = 0u32;
        match x {
            0 => {
                result = 1u32;
            }
            1 => {
                result = 2u32;
            }
            _ => {
                result = 9u32;
            }
        }
        result
    }
}

#[test]
fn or_literals_render_correctly() {
    let src = valid_switches::WGSL_SOURCE.wgsl_source().unwrap();
    // The or-pattern `0 | 1 | 2` should render as `case 0u, 1u, 2u:`
    // (not mixed with default). The literal suffix pass adds the `u`
    // suffixes because the switch selector is `u32` — WGSL requires
    // case selectors to match the selector's type.
    assert!(
        src.contains("case 0u, 1u, 2u"),
        "expected 'case 0u, 1u, 2u' in WGSL, got: {src}"
    );
    // `default` must NOT be combined with the case selectors.
    assert!(
        !src.contains("case 0u, 1u, 2u, default"),
        "default should not be mixed with case selectors, got: {src}"
    );
    assert!(
        src.contains("default"),
        "expected a default arm, got: {src}"
    );
}
