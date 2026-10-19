//! End-to-end tests for trait impls on std vector/matrix types (wgsl-rs#230).
//!
//! A `Self` in such an impl's signatures must render the WGSL predeclared
//! spelling (`vec2f`), matching the method-name mangling — the written
//! Rust spelling (`Vec2f`) is not a WGSL type name and fails naga
//! validation. Call sites (`Self::method()` inside impl bodies, direct
//! `Vec2f::method(x)` outside) must resolve to the same mangled
//! definitions. The module is validated by the auto-generated
//! `__validate_wgsl` test.

#![allow(dead_code)]

use wgsl_rs::{std::Vec2f, wgsl};

#[wgsl]
mod trait_impl_std_types {
    use wgsl_rs::std::*;

    pub trait Packed {
        fn to_array(data: Self) -> [u32; 2];
        fn from_array(array: [u32; 2]) -> Self;
        /// Round-trips through the array form, calling siblings with
        /// `Self::` syntax.
        fn round_trip(data: Self) -> Self;
    }

    impl Packed for Vec2f {
        fn to_array(data: Self) -> [u32; 2] {
            [bitcast_u32(data.x), bitcast_u32(data.y)]
        }

        fn from_array(array: [u32; 2]) -> Self {
            Vec2f {
                x: bitcast_f32(array[0]),
                y: bitcast_f32(array[1]),
            }
        }

        fn round_trip(data: Self) -> Self {
            Self::from_array(Self::to_array(data))
        }
    }

    /// Matrix self types resolve the same way (`Mat2x3f` → `mat2x3f`).
    pub trait Echo {
        fn echo(m: Self) -> Self;
    }

    impl Echo for Mat2x3f {
        fn echo(m: Self) -> Self {
            m
        }
    }

    /// Direct call sites name the type by its written Rust spelling and
    /// must resolve to the same mangled definitions the impls emit.
    pub fn call_direct(v: Vec2f) -> Vec2f {
        Vec2f::round_trip(v)
    }

    pub fn call_matrix(m: Mat2x3f) -> Mat2x3f {
        Mat2x3f::echo(m)
    }
}

#[test]
fn trait_impl_std_types_renders_resolved_spelling() {
    let src = trait_impl_std_types::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("fn vec2f__1to_array("),
        "expected 'vec2f__1to_array', got:\n{src}"
    );
    assert!(
        src.contains("fn vec2f__1from_array("),
        "expected 'vec2f__1from_array', got:\n{src}"
    );
    assert!(
        src.contains("fn vec2f__1round_trip("),
        "expected 'vec2f__1round_trip', got:\n{src}"
    );
    assert!(
        src.contains("fn mat2x3f_echo("),
        "expected 'mat2x3f_echo', got:\n{src}"
    );
    assert!(
        !src.contains("Vec2f") && !src.contains("Mat2x3f"),
        "the written Rust spellings must not reach WGSL, got:\n{src}"
    );
}

#[test]
fn trait_impl_std_types_round_trips_on_cpu() {
    let v = Vec2f { x: 1.5, y: -2.5 };
    let rt = trait_impl_std_types::Packed::round_trip(v);
    assert_eq!(rt.x, 1.5);
    assert_eq!(rt.y, -2.5);

    let direct = trait_impl_std_types::call_direct(v);
    assert_eq!(direct.x, 1.5);
    assert_eq!(direct.y, -2.5);
}
