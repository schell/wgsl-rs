//! Tests for the non-default `serde` cargo feature (GitHub issue #147).
//!
//! These are integration tests on purpose: `tests/` compiles as a
//! separate crate that consumes `wgsl-rs` exactly like a downstream user
//! would. If these pass, any downstream crate can derive
//! `Serialize`/`Deserialize` on its own structs containing
//! `wgsl_rs::std` math types — the orphan rule no longer blocks the
//! issue's use case.
//!
//! Run with: `cargo test -p wgsl-rs --features serde`

#![cfg(feature = "serde")]

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{from_str, to_string, to_value};
use wgsl_rs::std::{
    Mat2x2f, Mat2x3f, Mat2x4f, Mat3x2f, Mat3x3f, Mat3x4f, Mat4x2f, Mat4x3f, Mat4x4f, Vec2, Vec2f,
    Vec3f, Vec3i, Vec4b, Vec4f, mat2x2f, mat2x3f, mat2x4f, mat3x2f, mat3x3f, mat3x4f, mat4x2f,
    mat4x3f, mat4x4f, vec2f, vec3f, vec4f,
};

/// Serializes and deserializes a value through JSON, returning the copy.
fn roundtrip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    from_str(&to_string(value).expect("serialize")).expect("deserialize")
}

/// A downstream user's struct holding `std` math types — the exact
/// scenario from issue #147 that the orphan rule blocked.
#[derive(Serialize, Deserialize)]
struct Camera {
    position: Vec3f,
    view: Mat4x4f,
    scale: f32,
}

/// Vectors serialize as field-named objects.
#[test]
fn vector_golden_shape() {
    let v = Vec2f { x: 1.0, y: 2.0 };
    assert_eq!(to_string(&v).expect("serialize"), r#"{"x":1.0,"y":2.0}"#);
}

/// Matrices serialize their column arrays.
#[test]
fn matrix_golden_shape() {
    let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
    assert_eq!(
        to_string(&m).expect("serialize"),
        r#"{"columns":[{"x":1.0,"y":2.0},{"x":3.0,"y":4.0}]}"#
    );
}

/// Every vector alias round-trips through JSON with values intact.
#[test]
fn vector_aliases_roundtrip() {
    assert_eq!(
        roundtrip(&Vec2f { x: 1.5, y: -2.5 }),
        Vec2f { x: 1.5, y: -2.5 }
    );
    assert_eq!(
        roundtrip(&Vec3f {
            x: 1.0,
            y: 2.0,
            z: 3.0
        }),
        Vec3f {
            x: 1.0,
            y: 2.0,
            z: 3.0
        }
    );
    assert_eq!(
        roundtrip(&Vec4f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0
        }),
        Vec4f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0
        }
    );
    assert_eq!(
        roundtrip(&Vec3i { x: -1, y: 2, z: 3 }),
        Vec3i { x: -1, y: 2, z: 3 }
    );
    assert_eq!(
        roundtrip(&Vec4b {
            x: true,
            y: false,
            z: true,
            w: false
        }),
        Vec4b {
            x: true,
            y: false,
            z: true,
            w: false
        }
    );
}

/// `f16` component vectors round-trip via the forwarded `half/serde`
/// feature.
#[test]
fn half_f16_vectors_roundtrip() {
    let v = Vec2 {
        x: half::f16::from_f32(1.5),
        y: half::f16::from_f32(-2.5),
    };
    let back: Vec2<half::f16> = roundtrip(&v);
    assert_eq!(back.x.to_f32(), 1.5);
    assert_eq!(back.y.to_f32(), -2.5);
}

/// All nine matrix shapes round-trip. Matrices don't derive `PartialEq`,
/// so their JSON trees are compared instead.
#[test]
fn matrices_roundtrip() {
    macro_rules! roundtrip_matrix {
        ($value:expr, $ty:ty) => {{
            let m: $ty = $value;
            let back: $ty = roundtrip(&m);
            assert_eq!(
                to_value(&m).expect("serialize"),
                to_value(&back).expect("serialize")
            );
        }};
    }

    roundtrip_matrix!(mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0)), Mat2x2f);
    roundtrip_matrix!(mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0)), Mat2x3f);
    roundtrip_matrix!(
        mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0)),
        Mat2x4f
    );
    roundtrip_matrix!(
        mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0)),
        Mat3x2f
    );
    roundtrip_matrix!(
        mat3x3f(
            vec3f(1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0)
        ),
        Mat3x3f
    );
    roundtrip_matrix!(
        mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0)
        ),
        Mat3x4f
    );
    roundtrip_matrix!(
        mat4x2f(
            vec2f(1.0, 2.0),
            vec2f(3.0, 4.0),
            vec2f(5.0, 6.0),
            vec2f(7.0, 8.0)
        ),
        Mat4x2f
    );
    roundtrip_matrix!(
        mat4x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
            vec3f(10.0, 11.0, 12.0)
        ),
        Mat4x3f
    );
    roundtrip_matrix!(
        mat4x4f(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0)
        ),
        Mat4x4f
    );
}

/// The issue's actual use case: a user struct containing `std` math
/// types derives serde impls of its own.
#[test]
fn user_struct_with_std_types_roundtrips() {
    let camera = Camera {
        position: Vec3f {
            x: 0.0,
            y: 1.0,
            z: 2.0,
        },
        view: Mat4x4f::IDENTITY,
        scale: 0.5,
    };
    let back: Camera = roundtrip(&camera);
    assert_eq!(back.position, camera.position);
    assert_eq!(back.scale, camera.scale);
    assert_eq!(
        to_value(back.view).expect("serialize"),
        to_value(camera.view).expect("serialize")
    );
}
