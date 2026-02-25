//! Provides `bitcast` function for Rust and WGSL.

use crate::std::{Vec2f, Vec2i, Vec2u, Vec3f, Vec3i, Vec3u, Vec4f, Vec4i, Vec4u};

/// Trait for WGSL `bitcast` operations that reinterpret bits from one type as
/// another.
///
/// Unlike [`Convert`], which performs value-preserving type conversion (e.g.
/// `f32(1u)` yields `1.0`), `Bitcast` reinterprets the raw bit pattern of the
/// source value as the target type (e.g. `bitcast<f32>(0x3F800000u)` yields
/// `1.0`).
///
/// See [WGSL spec section 17.2.1](https://gpuweb.github.io/gpuweb/wgsl/#bitcast-builtin).
pub trait Bitcast<T> {
    /// Reinterpret the bits of `self` as type `T`.
    fn bitcast(self) -> T;
}

/// Reinterpret the bits of `e` as `f32`.
///
/// Corresponds to WGSL `bitcast<f32>(e)`.
pub fn bitcast_f32(e: impl Bitcast<f32>) -> f32 {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `u32`.
///
/// Corresponds to WGSL `bitcast<u32>(e)`.
pub fn bitcast_u32(e: impl Bitcast<u32>) -> u32 {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `i32`.
///
/// Corresponds to WGSL `bitcast<i32>(e)`.
pub fn bitcast_i32(e: impl Bitcast<i32>) -> i32 {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec2f`.
///
/// Corresponds to WGSL `bitcast<vec2<f32>>(e)`.
pub fn bitcast_vec2f(e: impl Bitcast<Vec2f>) -> Vec2f {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec2u`.
///
/// Corresponds to WGSL `bitcast<vec2<u32>>(e)`.
pub fn bitcast_vec2u(e: impl Bitcast<Vec2u>) -> Vec2u {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec2i`.
///
/// Corresponds to WGSL `bitcast<vec2<i32>>(e)`.
pub fn bitcast_vec2i(e: impl Bitcast<Vec2i>) -> Vec2i {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec3f`.
///
/// Corresponds to WGSL `bitcast<vec3<f32>>(e)`.
pub fn bitcast_vec3f(e: impl Bitcast<Vec3f>) -> Vec3f {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec3u`.
///
/// Corresponds to WGSL `bitcast<vec3<u32>>(e)`.
pub fn bitcast_vec3u(e: impl Bitcast<Vec3u>) -> Vec3u {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec3i`.
///
/// Corresponds to WGSL `bitcast<vec3<i32>>(e)`.
pub fn bitcast_vec3i(e: impl Bitcast<Vec3i>) -> Vec3i {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec4f`.
///
/// Corresponds to WGSL `bitcast<vec4<f32>>(e)`.
pub fn bitcast_vec4f(e: impl Bitcast<Vec4f>) -> Vec4f {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec4u`.
///
/// Corresponds to WGSL `bitcast<vec4<u32>>(e)`.
pub fn bitcast_vec4u(e: impl Bitcast<Vec4u>) -> Vec4u {
    e.bitcast()
}

/// Reinterpret the bits of `e` as `Vec4i`.
///
/// Corresponds to WGSL `bitcast<vec4<i32>>(e)`.
pub fn bitcast_vec4i(e: impl Bitcast<Vec4i>) -> Vec4i {
    e.bitcast()
}

mod bitcast_impls {
    use crate::std::*;

    // Identity impls: bitcast<T>(e: T) -> T is always self.
    impl Bitcast<f32> for f32 {
        fn bitcast(self) -> f32 {
            self
        }
    }
    impl Bitcast<u32> for u32 {
        fn bitcast(self) -> u32 {
            self
        }
    }
    impl Bitcast<i32> for i32 {
        fn bitcast(self) -> i32 {
            self
        }
    }

    // Scalar cross-type bitcasts.
    impl Bitcast<f32> for u32 {
        fn bitcast(self) -> f32 {
            f32::from_bits(self)
        }
    }
    impl Bitcast<f32> for i32 {
        fn bitcast(self) -> f32 {
            f32::from_ne_bytes(self.to_ne_bytes())
        }
    }
    impl Bitcast<u32> for f32 {
        fn bitcast(self) -> u32 {
            self.to_bits()
        }
    }
    impl Bitcast<u32> for i32 {
        fn bitcast(self) -> u32 {
            u32::from_ne_bytes(self.to_ne_bytes())
        }
    }
    impl Bitcast<i32> for f32 {
        fn bitcast(self) -> i32 {
            i32::from_ne_bytes(self.to_ne_bytes())
        }
    }
    impl Bitcast<i32> for u32 {
        fn bitcast(self) -> i32 {
            i32::from_ne_bytes(self.to_ne_bytes())
        }
    }

    // Vector bitcasts: component-wise reinterpretation of bits.
    //
    // Each vector bitcast converts the source vector to an array of its scalar
    // type, bitcasts each component, and constructs the target vector from the
    // resulting array.
    macro_rules! impl_vec_bitcast {
        ($n:literal, $src_vec:ty, $dst_vec:ty, $src_scalar:ty, $dst_scalar:ty) => {
            impl Bitcast<$dst_vec> for $src_vec {
                fn bitcast(self) -> $dst_vec {
                    let src: [$src_scalar; $n] = self.to_array();
                    let dst: [$dst_scalar; $n] = src.map(|s| Bitcast::<$dst_scalar>::bitcast(s));
                    <$dst_vec>::from_array(dst)
                }
            }
        };
    }

    // Identity vector bitcasts (same type in, same type out).
    macro_rules! impl_vec_bitcast_identity {
        ($vec:ty) => {
            impl Bitcast<$vec> for $vec {
                fn bitcast(self) -> $vec {
                    self
                }
            }
        };
    }

    // Vec2 identity
    impl_vec_bitcast_identity!(Vec2f);
    impl_vec_bitcast_identity!(Vec2u);
    impl_vec_bitcast_identity!(Vec2i);

    // Vec3 identity
    impl_vec_bitcast_identity!(Vec3f);
    impl_vec_bitcast_identity!(Vec3u);
    impl_vec_bitcast_identity!(Vec3i);

    // Vec4 identity
    impl_vec_bitcast_identity!(Vec4f);
    impl_vec_bitcast_identity!(Vec4u);
    impl_vec_bitcast_identity!(Vec4i);

    // Vec2 cross-type
    impl_vec_bitcast!(2, Vec2u, Vec2f, u32, f32);
    impl_vec_bitcast!(2, Vec2i, Vec2f, i32, f32);
    impl_vec_bitcast!(2, Vec2f, Vec2u, f32, u32);
    impl_vec_bitcast!(2, Vec2i, Vec2u, i32, u32);
    impl_vec_bitcast!(2, Vec2f, Vec2i, f32, i32);
    impl_vec_bitcast!(2, Vec2u, Vec2i, u32, i32);

    // Vec3 cross-type
    impl_vec_bitcast!(3, Vec3u, Vec3f, u32, f32);
    impl_vec_bitcast!(3, Vec3i, Vec3f, i32, f32);
    impl_vec_bitcast!(3, Vec3f, Vec3u, f32, u32);
    impl_vec_bitcast!(3, Vec3i, Vec3u, i32, u32);
    impl_vec_bitcast!(3, Vec3f, Vec3i, f32, i32);
    impl_vec_bitcast!(3, Vec3u, Vec3i, u32, i32);

    // Vec4 cross-type
    impl_vec_bitcast!(4, Vec4u, Vec4f, u32, f32);
    impl_vec_bitcast!(4, Vec4i, Vec4f, i32, f32);
    impl_vec_bitcast!(4, Vec4f, Vec4u, f32, u32);
    impl_vec_bitcast!(4, Vec4i, Vec4u, i32, u32);
    impl_vec_bitcast!(4, Vec4f, Vec4i, f32, i32);
    impl_vec_bitcast!(4, Vec4u, Vec4i, u32, i32);
}

#[cfg(test)]
mod bitcast_tests {
    use crate::std::*;

    // Scalar identity bitcasts.
    #[test]
    fn bitcast_f32_identity() {
        assert_eq!(bitcast_f32(1.0f32), 1.0f32);
        assert_eq!(bitcast_f32(0.0f32), 0.0f32);
        assert_eq!(bitcast_f32(-1.0f32), -1.0f32);
    }

    #[test]
    fn bitcast_u32_identity() {
        assert_eq!(bitcast_u32(42u32), 42u32);
        assert_eq!(bitcast_u32(0u32), 0u32);
    }

    #[test]
    fn bitcast_i32_identity() {
        assert_eq!(bitcast_i32(42i32), 42i32);
        assert_eq!(bitcast_i32(-1i32), -1i32);
    }

    // Scalar cross-type bitcasts.
    #[test]
    fn bitcast_u32_to_f32() {
        // IEEE 754: 0x3F800000 == 1.0f32
        assert_eq!(bitcast_f32(0x3F800000u32), 1.0f32);
        // 0x00000000 == 0.0f32
        assert_eq!(bitcast_f32(0u32), 0.0f32);
    }

    #[test]
    fn bitcast_f32_to_u32() {
        assert_eq!(bitcast_u32(1.0f32), 0x3F800000u32);
        assert_eq!(bitcast_u32(0.0f32), 0u32);
    }

    #[test]
    fn bitcast_i32_to_f32() {
        // 0x3F800000 as i32 == 1065353216
        assert_eq!(bitcast_f32(0x3F800000i32), 1.0f32);
    }

    #[test]
    fn bitcast_f32_to_i32() {
        assert_eq!(bitcast_i32(1.0f32), 0x3F800000i32);
    }

    #[test]
    fn bitcast_u32_to_i32() {
        assert_eq!(bitcast_i32(0xFFFFFFFFu32), -1i32);
        assert_eq!(bitcast_i32(0u32), 0i32);
    }

    #[test]
    fn bitcast_i32_to_u32() {
        assert_eq!(bitcast_u32(-1i32), 0xFFFFFFFFu32);
        assert_eq!(bitcast_u32(0i32), 0u32);
    }

    // Round-trip property: bitcast back and forth should be lossless.
    #[test]
    fn bitcast_f32_u32_roundtrip() {
        let original = std::f32::consts::PI;
        let through_u32: u32 = bitcast_u32(original);
        let back: f32 = bitcast_f32(through_u32);
        assert_eq!(original, back);
    }

    #[test]
    fn bitcast_f32_i32_roundtrip() {
        let original = -std::f32::consts::E;
        let through_i32: i32 = bitcast_i32(original);
        let back: f32 = bitcast_f32(through_i32);
        assert_eq!(original, back);
    }

    // Vector bitcasts.
    #[test]
    fn bitcast_vec2u_to_vec2f() {
        let v = vec2u(0x3F800000, 0x40000000); // 1.0, 2.0 in IEEE 754
        let result: Vec2f = bitcast_vec2f(v);
        assert_eq!(result, vec2f(1.0, 2.0));
    }

    #[test]
    fn bitcast_vec2f_to_vec2u() {
        let v = vec2f(1.0, 2.0);
        let result: Vec2u = bitcast_vec2u(v);
        assert_eq!(result, vec2u(0x3F800000, 0x40000000));
    }

    #[test]
    fn bitcast_vec3i_to_vec3f() {
        let v = vec3i(0x3F800000, 0x40000000, 0x40400000); // 1.0, 2.0, 3.0
        let result: Vec3f = bitcast_vec3f(v);
        assert_eq!(result, vec3f(1.0, 2.0, 3.0));
    }

    #[test]
    fn bitcast_vec4f_to_vec4u() {
        let v = vec4f(1.0, 2.0, 3.0, 4.0);
        let result: Vec4u = bitcast_vec4u(v);
        assert_eq!(
            result,
            vec4u(0x3F800000, 0x40000000, 0x40400000, 0x40800000)
        );
    }

    #[test]
    fn bitcast_vec4u_to_vec4i() {
        let v = vec4u(0xFFFFFFFF, 0, 1, 0x80000000);
        let result: Vec4i = bitcast_vec4i(v);
        assert_eq!(result, vec4i(-1, 0, 1, i32::MIN));
    }

    // Vector identity.
    #[test]
    fn bitcast_vec3f_identity() {
        let v = vec3f(1.0, 2.0, 3.0);
        assert_eq!(bitcast_vec3f(v), v);
    }

    // Vector round-trip.
    #[test]
    fn bitcast_vec4f_u32_roundtrip() {
        let original = vec4f(1.0, -2.0, 3.1, 0.0);
        let through_u: Vec4u = bitcast_vec4u(original);
        let back: Vec4f = bitcast_vec4f(through_u);
        assert_eq!(original, back);
    }
}
