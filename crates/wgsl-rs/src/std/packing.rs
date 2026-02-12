//! WGSL data packing and unpacking builtin functions.
//!
//! These functions convert between vector types and packed integer
//! representations, useful for vertex attribute compression and storage
//! optimization.
//!
//! See [WGSL spec §17.9](https://gpuweb.github.io/gpuweb/wgsl/#pack-builtin-functions)
//! and [§17.10](https://gpuweb.github.io/gpuweb/wgsl/#unpack-builtin-functions).

use crate::std::{Vec2f, Vec4f, vec2f, vec4f};

/// Packs four normalized signed floats into a `u32`.
///
/// Each component of `e` is clamped to [-1, 1], then converted to an 8-bit
/// signed integer via `floor(0.5 + 127 * clamp(e[i]))`. The four bytes are
/// packed into a `u32` with component `i` occupying bits `8*i` through
/// `8*i + 7`.
pub fn pack4x8snorm(e: Vec4f) -> u32 {
    fn to_snorm8(v: f32) -> u8 {
        let clamped = v.clamp(-1.0, 1.0);
        let scaled = (0.5 + 127.0 * clamped).floor() as i8;
        scaled as u8
    }
    let b0 = to_snorm8(e.x()) as u32;
    let b1 = to_snorm8(e.y()) as u32;
    let b2 = to_snorm8(e.z()) as u32;
    let b3 = to_snorm8(e.w()) as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Packs four normalized unsigned floats into a `u32`.
///
/// Each component of `e` is clamped to [0, 1], then converted to an 8-bit
/// unsigned integer via `floor(0.5 + 255 * clamp(e[i]))`. The four bytes are
/// packed into a `u32` with component `i` occupying bits `8*i` through
/// `8*i + 7`.
pub fn pack4x8unorm(e: Vec4f) -> u32 {
    fn to_unorm8(v: f32) -> u8 {
        let clamped = v.clamp(0.0, 1.0);
        (0.5 + 255.0 * clamped).floor() as u8
    }
    let b0 = to_unorm8(e.x()) as u32;
    let b1 = to_unorm8(e.y()) as u32;
    let b2 = to_unorm8(e.z()) as u32;
    let b3 = to_unorm8(e.w()) as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Packs two normalized signed floats into a `u32`.
///
/// Each component of `e` is clamped to [-1, 1], then converted to a 16-bit
/// signed integer via `floor(0.5 + 32767 * clamp(e[i]))`. The two shorts are
/// packed into a `u32` with component `i` occupying bits `16*i` through
/// `16*i + 15`.
pub fn pack2x16snorm(e: Vec2f) -> u32 {
    fn to_snorm16(v: f32) -> u16 {
        let clamped = v.clamp(-1.0, 1.0);
        let scaled = (0.5 + 32767.0 * clamped).floor() as i16;
        scaled as u16
    }
    let lo = to_snorm16(e.x()) as u32;
    let hi = to_snorm16(e.y()) as u32;
    lo | (hi << 16)
}

/// Packs two normalized unsigned floats into a `u32`.
///
/// Each component of `e` is clamped to [0, 1], then converted to a 16-bit
/// unsigned integer via `floor(0.5 + 65535 * clamp(e[i]))`. The two shorts are
/// packed into a `u32` with component `i` occupying bits `16*i` through
/// `16*i + 15`.
pub fn pack2x16unorm(e: Vec2f) -> u32 {
    fn to_unorm16(v: f32) -> u16 {
        let clamped = v.clamp(0.0, 1.0);
        (0.5 + 65535.0 * clamped).floor() as u16
    }
    let lo = to_unorm16(e.x()) as u32;
    let hi = to_unorm16(e.y()) as u32;
    lo | (hi << 16)
}

/// Packs two `f32` values into a `u32` as IEEE-754 binary16 (half-precision).
///
/// Each component of `e` is converted to an `f16`, then the two 16-bit values
/// are packed into a `u32` with component `i` occupying bits `16*i` through
/// `16*i + 15`.
pub fn pack2x16float(e: Vec2f) -> u32 {
    let lo = half::f16::from_f32(e.x()).to_bits() as u32;
    let hi = half::f16::from_f32(e.y()).to_bits() as u32;
    lo | (hi << 16)
}

/// Unpacks a `u32` into four normalized signed floats.
///
/// Extracts four 8-bit signed integers from `e`, where component `i` is taken
/// from bits `8*i` through `8*i + 7` as twos-complement. Each is converted
/// to `f32` via `max(v / 127.0, -1.0)`.
pub fn unpack4x8snorm(e: u32) -> Vec4f {
    fn from_snorm8(bits: u8) -> f32 {
        let v = bits as i8;
        (v as f32 / 127.0).max(-1.0)
    }
    vec4f(
        from_snorm8(e as u8),
        from_snorm8((e >> 8) as u8),
        from_snorm8((e >> 16) as u8),
        from_snorm8((e >> 24) as u8),
    )
}

/// Unpacks a `u32` into four normalized unsigned floats.
///
/// Extracts four 8-bit unsigned integers from `e`, where component `i` is taken
/// from bits `8*i` through `8*i + 7`. Each is converted to `f32` via
/// `v / 255.0`.
pub fn unpack4x8unorm(e: u32) -> Vec4f {
    fn from_unorm8(bits: u8) -> f32 {
        bits as f32 / 255.0
    }
    vec4f(
        from_unorm8(e as u8),
        from_unorm8((e >> 8) as u8),
        from_unorm8((e >> 16) as u8),
        from_unorm8((e >> 24) as u8),
    )
}

/// Unpacks a `u32` into two normalized signed floats.
///
/// Extracts two 16-bit signed integers from `e`, where component `i` is taken
/// from bits `16*i` through `16*i + 15` as twos-complement. Each is converted
/// to `f32` via `max(v / 32767.0, -1.0)`.
pub fn unpack2x16snorm(e: u32) -> Vec2f {
    fn from_snorm16(bits: u16) -> f32 {
        let v = bits as i16;
        (v as f32 / 32767.0).max(-1.0)
    }
    vec2f(from_snorm16(e as u16), from_snorm16((e >> 16) as u16))
}

/// Unpacks a `u32` into two normalized unsigned floats.
///
/// Extracts two 16-bit unsigned integers from `e`, where component `i` is taken
/// from bits `16*i` through `16*i + 15`. Each is converted to `f32` via
/// `v / 65535.0`.
pub fn unpack2x16unorm(e: u32) -> Vec2f {
    fn from_unorm16(bits: u16) -> f32 {
        bits as f32 / 65535.0
    }
    vec2f(from_unorm16(e as u16), from_unorm16((e >> 16) as u16))
}

/// Unpacks a `u32` into two `f32` values from IEEE-754 binary16
/// (half-precision).
///
/// Extracts two 16-bit values from `e`, where component `i` is taken from bits
/// `16*i` through `16*i + 15`, and converts each from `f16` to `f32`.
pub fn unpack2x16float(e: u32) -> Vec2f {
    let lo = half::f16::from_bits(e as u16).to_f32();
    let hi = half::f16::from_bits((e >> 16) as u16).to_f32();
    vec2f(lo, hi)
}

#[cfg(test)]
mod test {
    use super::*;

    // pack4x8snorm / unpack4x8snorm

    #[test]
    fn pack4x8snorm_known_values() {
        // (0, 0, 0, 0) => all bytes zero
        assert_eq!(pack4x8snorm(vec4f(0.0, 0.0, 0.0, 0.0)), 0);

        // (1, 1, 1, 1) => 0x7F7F7F7F (127 in each byte)
        assert_eq!(pack4x8snorm(vec4f(1.0, 1.0, 1.0, 1.0)), 0x7F7F7F7F);

        // (-1, -1, -1, -1) => 0x81818181 (-127 as i8 = 0x81 in each byte)
        assert_eq!(pack4x8snorm(vec4f(-1.0, -1.0, -1.0, -1.0)), 0x81818181);
    }

    #[test]
    fn pack4x8snorm_clamps_out_of_range() {
        // Values beyond [-1, 1] should be clamped
        assert_eq!(
            pack4x8snorm(vec4f(2.0, -2.0, 0.0, 0.0)),
            pack4x8snorm(vec4f(1.0, -1.0, 0.0, 0.0))
        );
    }

    #[test]
    fn pack4x8snorm_roundtrip() {
        let original = vec4f(0.5, -0.5, 0.25, -0.75);
        let packed = pack4x8snorm(original);
        let unpacked = unpack4x8snorm(packed);
        // 8-bit precision: tolerance of ~1/127
        assert!((unpacked.x() - 0.5).abs() < 0.01);
        assert!((unpacked.y() - (-0.5)).abs() < 0.01);
        assert!((unpacked.z() - 0.25).abs() < 0.01);
        assert!((unpacked.w() - (-0.75)).abs() < 0.01);
    }

    #[test]
    fn unpack4x8snorm_known_values() {
        // All zeros => (0, 0, 0, 0)
        let v = unpack4x8snorm(0);
        assert_eq!(v, vec4f(0.0, 0.0, 0.0, 0.0));

        // 0x7F = 127 => 127/127 = 1.0
        let v = unpack4x8snorm(0x7F7F7F7F);
        assert!((v.x() - 1.0).abs() < 1e-6);
        assert!((v.y() - 1.0).abs() < 1e-6);

        // 0x80 as i8 = -128 => max(-128/127, -1.0) = -1.0
        let v = unpack4x8snorm(0x80808080);
        assert!((v.x() - (-1.0)).abs() < 1e-6);
    }

    // pack4x8unorm / unpack4x8unorm

    #[test]
    fn pack4x8unorm_known_values() {
        // (0, 0, 0, 0) => all bytes zero
        assert_eq!(pack4x8unorm(vec4f(0.0, 0.0, 0.0, 0.0)), 0);

        // (1, 1, 1, 1) => 0xFFFFFFFF (255 in each byte)
        assert_eq!(pack4x8unorm(vec4f(1.0, 1.0, 1.0, 1.0)), 0xFFFFFFFF);
    }

    #[test]
    fn pack4x8unorm_clamps_out_of_range() {
        assert_eq!(
            pack4x8unorm(vec4f(2.0, -1.0, 0.0, 0.0)),
            pack4x8unorm(vec4f(1.0, 0.0, 0.0, 0.0))
        );
    }

    #[test]
    fn pack4x8unorm_roundtrip() {
        let original = vec4f(0.0, 0.25, 0.5, 1.0);
        let packed = pack4x8unorm(original);
        let unpacked = unpack4x8unorm(packed);
        // 8-bit precision: tolerance of ~1/255
        assert!((unpacked.x() - 0.0).abs() < 0.005);
        assert!((unpacked.y() - 0.25).abs() < 0.005);
        assert!((unpacked.z() - 0.5).abs() < 0.005);
        assert!((unpacked.w() - 1.0).abs() < 0.005);
    }

    #[test]
    fn unpack4x8unorm_known_values() {
        let v = unpack4x8unorm(0);
        assert_eq!(v, vec4f(0.0, 0.0, 0.0, 0.0));

        let v = unpack4x8unorm(0xFFFFFFFF);
        assert!((v.x() - 1.0).abs() < 1e-6);
        assert!((v.y() - 1.0).abs() < 1e-6);
        assert!((v.z() - 1.0).abs() < 1e-6);
        assert!((v.w() - 1.0).abs() < 1e-6);
    }

    // pack2x16snorm / unpack2x16snorm

    #[test]
    fn pack2x16snorm_known_values() {
        assert_eq!(pack2x16snorm(vec2f(0.0, 0.0)), 0);

        // 1.0 => 32767 = 0x7FFF
        assert_eq!(pack2x16snorm(vec2f(1.0, 1.0)), 0x7FFF7FFF);

        // -1.0 => -32767 as i16 = 0x8001
        assert_eq!(pack2x16snorm(vec2f(-1.0, -1.0)), 0x80018001);
    }

    #[test]
    fn pack2x16snorm_clamps_out_of_range() {
        assert_eq!(
            pack2x16snorm(vec2f(5.0, -5.0)),
            pack2x16snorm(vec2f(1.0, -1.0))
        );
    }

    #[test]
    fn pack2x16snorm_roundtrip() {
        let original = vec2f(0.5, -0.75);
        let packed = pack2x16snorm(original);
        let unpacked = unpack2x16snorm(packed);
        // 16-bit precision: tolerance of ~1/32767
        assert!((unpacked.x() - 0.5).abs() < 0.0001);
        assert!((unpacked.y() - (-0.75)).abs() < 0.0001);
    }

    #[test]
    fn unpack2x16snorm_known_values() {
        let v = unpack2x16snorm(0);
        assert_eq!(v, vec2f(0.0, 0.0));

        let v = unpack2x16snorm(0x7FFF7FFF);
        assert!((v.x() - 1.0).abs() < 1e-6);
        assert!((v.y() - 1.0).abs() < 1e-6);

        // 0x8000 as i16 = -32768 => max(-32768/32767, -1.0) = -1.0
        let v = unpack2x16snorm(0x80008000);
        assert!((v.x() - (-1.0)).abs() < 1e-6);
        assert!((v.y() - (-1.0)).abs() < 1e-6);
    }

    // pack2x16unorm / unpack2x16unorm

    #[test]
    fn pack2x16unorm_known_values() {
        assert_eq!(pack2x16unorm(vec2f(0.0, 0.0)), 0);

        // 1.0 => 65535 = 0xFFFF
        assert_eq!(pack2x16unorm(vec2f(1.0, 1.0)), 0xFFFFFFFF);
    }

    #[test]
    fn pack2x16unorm_clamps_out_of_range() {
        assert_eq!(
            pack2x16unorm(vec2f(2.0, -1.0)),
            pack2x16unorm(vec2f(1.0, 0.0))
        );
    }

    #[test]
    fn pack2x16unorm_roundtrip() {
        let original = vec2f(0.25, 0.75);
        let packed = pack2x16unorm(original);
        let unpacked = unpack2x16unorm(packed);
        // 16-bit precision: tolerance of ~1/65535
        assert!((unpacked.x() - 0.25).abs() < 0.0001);
        assert!((unpacked.y() - 0.75).abs() < 0.0001);
    }

    #[test]
    fn unpack2x16unorm_known_values() {
        let v = unpack2x16unorm(0);
        assert_eq!(v, vec2f(0.0, 0.0));

        let v = unpack2x16unorm(0xFFFFFFFF);
        assert!((v.x() - 1.0).abs() < 1e-6);
        assert!((v.y() - 1.0).abs() < 1e-6);
    }

    // pack2x16float / unpack2x16float

    #[test]
    fn pack2x16float_known_values() {
        // f16 for 1.0 = 0x3C00, for 0.0 = 0x0000
        let packed = pack2x16float(vec2f(0.0, 1.0));
        assert_eq!(packed & 0xFFFF, 0x0000); // lo = 0.0
        assert_eq!(packed >> 16, 0x3C00); // hi = 1.0
    }

    #[test]
    fn pack2x16float_roundtrip() {
        let original = vec2f(1.5, -0.25);
        let packed = pack2x16float(original);
        let unpacked = unpack2x16float(packed);
        // f16 has ~3 decimal digits of precision
        assert!((unpacked.x() - 1.5).abs() < 0.001);
        assert!((unpacked.y() - (-0.25)).abs() < 0.001);
    }

    #[test]
    fn unpack2x16float_known_values() {
        // 0x3C00 = f16 1.0, 0xC000 = f16 -2.0
        let packed = 0x3C00 | (0xC000 << 16);
        let v = unpack2x16float(packed);
        assert!((v.x() - 1.0).abs() < 1e-6);
        assert!((v.y() - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn pack2x16float_roundtrip_special() {
        // Test zero
        let v = vec2f(0.0, 0.0);
        assert_eq!(unpack2x16float(pack2x16float(v)), v);

        // Test negative zero
        let packed = pack2x16float(vec2f(-0.0, -0.0));
        let unpacked = unpack2x16float(packed);
        assert!(unpacked.x().is_sign_negative() || unpacked.x() == 0.0);
    }
}
