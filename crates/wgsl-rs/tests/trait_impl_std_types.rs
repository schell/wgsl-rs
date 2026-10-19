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

use wgsl_rs::{
    std::{Mat2x3f, Vec2b, Vec2f},
    wgsl,
};

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

/// Assoc-type projections on std self types (`Self::Array` inside
/// `impl ... for Vec2b`) resolve to the impl's concrete associated type,
/// so the literal-suffix pass can anchor unsuffixed `select` literals
/// through the return type — the crabslab `Vec4b::to_array` shape
/// (wgsl-rs#230 follow-up). Validated by the auto-generated
/// `__validate_wgsl` test.
#[wgsl]
mod bool_vec_assoc {
    use wgsl_rs::std::*;

    pub trait PackedBool {
        type Array;
        fn to_array(data: Self) -> Self::Array;
    }

    impl PackedBool for Vec2b {
        type Array = [u32; 2];

        fn to_array(data: Self) -> Self::Array {
            [select(0, 1, data.x), select(0, 1, data.y)]
        }
    }
}

#[test]
fn bool_vec_assoc_resolves_and_suffixes() {
    let src = bool_vec_assoc::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("fn vec2b__1to_array(data: vec2<bool>) -> array<u32, 2>"),
        "expected the Self::Array return to resolve to array<u32, 2>, got:\n{src}"
    );
    assert!(
        src.contains("select(0u, 1u, data.x)"),
        "expected the suffix pass to anchor the select literals to u32, got:\n{src}"
    );
}

#[test]
fn bool_vec_assoc_runs_on_cpu() {
    let out = bool_vec_assoc::PackedBool::to_array(Vec2b { x: false, y: true });
    assert_eq!(out, [0u32, 1u32]);
}

/// Matrix self types resolve their assoc-type projections the same way
/// (`Mat2x3f` → `mat2x3f`); validated by the auto-generated
/// `__validate_wgsl` test. The CPU matrix types expose no component
/// access yet, so the body is trivial — the test pins the resolution and
/// the rendered signature, not bit-packing.
#[wgsl]
mod matrix_assoc {
    use wgsl_rs::std::*;

    pub trait PackedMat {
        type Array;
        fn to_array(data: Self) -> Self::Array;
    }

    impl PackedMat for Mat2x3f {
        type Array = [u32; 6];

        fn to_array(_data: Self) -> Self::Array {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }
}

#[test]
fn matrix_assoc_renders_resolved_spelling() {
    let src = matrix_assoc::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("fn mat2x3f__1to_array(_data: mat2x3f) -> array<u32, 6>"),
        "expected the matrix Self::Array to resolve to array<u32, 6>, got:\n{src}"
    );
}

#[test]
fn matrix_assoc_runs_on_cpu() {
    let out = matrix_assoc::PackedMat::to_array(Mat2x3f::default());
    assert_eq!(out, [0u32; 6]);
}
