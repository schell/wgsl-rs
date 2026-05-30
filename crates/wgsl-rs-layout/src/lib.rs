//! WGSL memory layout computation for Rust types.
//!
//! This crate provides traits and a derive macro for computing the size,
//! alignment, and field offsets of types per the WGSL specification
//! §14.4.1 ("Alignment and Size"). It is intended for tools that marshal
//! data between CPU and GPU memory.
//!
//! # Quick Start
//!
//! ```ignore
//! use wgsl_rs_layout::{Layout, WgslLayout, FieldLayout};
//! use wgsl_rs::{Mat4x4f, Vec4f};
//!
//! #[derive(Layout)]
//! struct Uniforms {
//!     model: Mat4x4f,
//!     color: Vec4f,
//!     time: f32,
//! }
//!
//! assert_eq!(Uniforms::SIZE, 96);
//! assert_eq!(Uniforms::ALIGN, 16);
//!
//! for f in Uniforms::FIELDS {
//!     println!(
//!         "{}: offset={}, size={}, align={}, pad_before={}",
//!         f.name, f.offset, f.size, f.alignment, f.pad_before,
//!     );
//! }
//! ```
//!
//! # The `pad_before` Field
//!
//! [`FieldLayout::pad_before`] is the number of dead bytes between the
//! end of the previous field and the start of the current field. It is
//! always 0 for the first field. **You must account for these padding
//! bytes when marshalling data into GPU buffers** — write zero bytes to
//! fill gaps before writing the next field's data.

pub use wgsl_rs_layout_macros::Layout;

// Allow the crate's own tests to resolve `::wgsl_rs_layout` paths emitted
// by the derive macro (external users resolve it via the crate dependency).
extern crate self as wgsl_rs_layout;

mod field;
mod types;

pub use field::FieldLayout;

/// Error returned by [`WgslLayout::write_layout_bytes`] when the destination
/// buffer is too small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The buffer is too small for the type's size.
    BufferTooSmall {
        /// The number of bytes needed.
        needed: usize,
        /// The number of bytes available.
        actual: usize,
    },
}

/// A type with a known WGSL memory layout.
///
/// Each type that can appear as a field in a host-shareable WGSL struct
/// must implement this trait. The associated constants encode the size
/// and alignment per the WGSL specification §14.4.1.
///
/// Built-in scalar, vector, matrix, array, and atomic types have
/// predefined impls. User-defined structs receive impls via
/// `#[derive(Layout)]`.
pub trait WgslLayout: Sized {
    /// The size in bytes of this type per WGSL layout rules.
    const SIZE: usize;
    /// The alignment in bytes of this type per WGSL layout rules.
    /// Must be a power of two.
    const ALIGN: usize;

    /// Write this value's bytes into `buf` at offset 0, using WGSL layout
    /// rules (little-endian scalars, alignment-padded fields).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `buf.len() < Self::SIZE`.
    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error>;

    /// Read a value of this type from `buf` at offset 0, using WGSL layout
    /// rules (little-endian scalars, alignment-padded fields).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `buf.len() < Self::SIZE`.
    fn read_layout_bytes(buf: &[u8]) -> Result<Self, Error>;
}

/// A struct whose field layout has been computed per WGSL rules.
///
/// Implemented automatically by `#[derive(Layout)]`. Provides field-level
/// offset, size, alignment, and inter-field padding information.
pub trait Layout: WgslLayout {
    /// Per-field layout information in declaration order.
    const FIELDS: &'static [FieldLayout];
}

/// Write `t`'s bytes into `bytes` at offset 0 per WGSL layout rules.
///
/// Convenience wrapper around [`WgslLayout::write_layout_bytes`].
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] if `bytes.len() < T::SIZE`.
pub fn write_layout_bytes<T: WgslLayout>(bytes: &mut [u8], t: &T) -> Result<(), Error> {
    t.write_layout_bytes(bytes)
}

/// Read a value of type `T` from `bytes` at offset 0 per WGSL layout rules.
///
/// Convenience wrapper around [`WgslLayout::read_layout_bytes`].
///
/// # Errors
///
/// Returns [`Error::BufferTooSmall`] if `bytes.len() < T::SIZE`.
pub fn read_layout_bytes<T: WgslLayout>(bytes: &[u8]) -> Result<T, Error> {
    T::read_layout_bytes(bytes)
}

/// Round `val` up to the next multiple of `align`.
///
/// `align` must be a power of two.
#[doc(hidden)]
pub const fn round_up(val: usize, align: usize) -> usize {
    (val + (align - 1)) & !(align - 1)
}

/// Zero the first `len` bytes of `buf`.
///
/// Internal helper used by `#[derive(Layout)]` to zero padding bytes before
/// writing field data.
#[doc(hidden)]
pub fn zero_buffer(buf: &mut [u8], len: usize) {
    buf[..len].fill(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_layout_bytes, write_layout_bytes};

    // ===== Scalar tests =====

    #[test]
    fn scalar_layouts() {
        assert_eq!(f32::SIZE, 4);
        assert_eq!(f32::ALIGN, 4);
        assert_eq!(i32::SIZE, 4);
        assert_eq!(i32::ALIGN, 4);
        assert_eq!(u32::SIZE, 4);
        assert_eq!(u32::ALIGN, 4);
        assert_eq!(bool::SIZE, 4);
        assert_eq!(bool::ALIGN, 4);
    }

    // ===== Vector tests =====

    #[test]
    fn vector_layouts() {
        assert_eq!(<wgsl_rs::std::Vec2f>::SIZE, 8);
        assert_eq!(<wgsl_rs::std::Vec2f>::ALIGN, 8);
        assert_eq!(<wgsl_rs::std::Vec3f>::SIZE, 12);
        assert_eq!(<wgsl_rs::std::Vec3f>::ALIGN, 16);
        assert_eq!(<wgsl_rs::std::Vec4f>::SIZE, 16);
        assert_eq!(<wgsl_rs::std::Vec4f>::ALIGN, 16);
    }

    // ===== Matrix tests =====

    #[test]
    fn matrix_layouts() {
        // matCxR: Align = AlignOf(vecR), Size = SizeOf(array<vecR, C>)
        // mat2x2f: Align = AlignOf(vec2f) = 8, Size = SizeOf(array<vec2f, 2>) = 2*8 =
        // 16
        assert_eq!(<wgsl_rs::std::Mat2x2f>::SIZE, 16);
        assert_eq!(<wgsl_rs::std::Mat2x2f>::ALIGN, 8);
        // mat2x3f: Align = AlignOf(vec3f) = 16, Size = 2*roundUp(16,12) = 2*16 = 32
        assert_eq!(<wgsl_rs::std::Mat2x3f>::SIZE, 32);
        assert_eq!(<wgsl_rs::std::Mat2x3f>::ALIGN, 16);
        // mat3x3f: Align = 16, Size = 3*16 = 48
        assert_eq!(<wgsl_rs::std::Mat3x3f>::SIZE, 48);
        assert_eq!(<wgsl_rs::std::Mat3x3f>::ALIGN, 16);
        // mat4x4f: Align = 16, Size = 4*16 = 64
        assert_eq!(<wgsl_rs::std::Mat4x4f>::SIZE, 64);
        assert_eq!(<wgsl_rs::std::Mat4x4f>::ALIGN, 16);
        // mat3x2f: Align = AlignOf(vec2f) = 8, Size = 3*8 = 24
        assert_eq!(<wgsl_rs::std::Mat3x2f>::SIZE, 24);
        assert_eq!(<wgsl_rs::std::Mat3x2f>::ALIGN, 8);
    }

    // ===== Array tests =====

    #[test]
    fn array_layouts() {
        // [f32; 3]: align 4, size = 3 * roundUp(4, 4) = 12
        assert_eq!(<[f32; 3]>::SIZE, 12);
        assert_eq!(<[f32; 3]>::ALIGN, 4);
        // [Vec3f; 2]: align 16, size = 2 * roundUp(16, 12) = 2*16 = 32
        assert_eq!(<[wgsl_rs::std::Vec3f; 2]>::SIZE, 32);
        assert_eq!(<[wgsl_rs::std::Vec3f; 2]>::ALIGN, 16);
    }

    #[test]
    fn runtime_array_layout() {
        assert_eq!(<wgsl_rs::std::RuntimeArray<f32>>::SIZE, 0);
        assert_eq!(<wgsl_rs::std::RuntimeArray<f32>>::ALIGN, 4);
    }

    // ===== Atomic tests =====

    #[test]
    fn atomic_layouts() {
        assert_eq!(<wgsl_rs::std::Atomic<u32>>::SIZE, 4);
        assert_eq!(<wgsl_rs::std::Atomic<u32>>::ALIGN, 4);
        assert_eq!(<wgsl_rs::std::Atomic<i32>>::SIZE, 4);
        assert_eq!(<wgsl_rs::std::Atomic<i32>>::ALIGN, 4);
    }

    // ===== round_up tests =====

    #[test]
    fn round_up_valid() {
        assert_eq!(round_up(0, 4), 0);
        assert_eq!(round_up(1, 4), 4);
        assert_eq!(round_up(4, 4), 4);
        assert_eq!(round_up(5, 4), 8);
        assert_eq!(round_up(12, 16), 16);
        assert_eq!(round_up(16, 16), 16);
        assert_eq!(round_up(17, 16), 32);
        assert_eq!(round_up(80, 4), 80);
        assert_eq!(round_up(84, 16), 96);
    }

    // ===== Derived Layout tests =====

    #[derive(Layout)]
    struct Empty {}

    #[test]
    fn empty_struct() {
        assert_eq!(<Empty>::SIZE, 0);
        assert_eq!(<Empty>::ALIGN, 1);
        assert_eq!(<Empty>::FIELDS.len(), 0);
    }

    #[derive(Layout)]
    struct SingleScalar {
        value: f32,
    }

    #[test]
    fn single_scalar_struct() {
        assert_eq!(<SingleScalar>::SIZE, 4);
        assert_eq!(<SingleScalar>::ALIGN, 4);
        assert_eq!(<SingleScalar>::FIELDS.len(), 1);
        assert_eq!(<SingleScalar>::FIELDS[0].name, "value");
        assert_eq!(<SingleScalar>::FIELDS[0].offset, 0);
        assert_eq!(<SingleScalar>::FIELDS[0].size, 4);
        assert_eq!(<SingleScalar>::FIELDS[0].alignment, 4);
        assert_eq!(<SingleScalar>::FIELDS[0].pad_before, 0);
    }

    #[derive(Layout, Debug, PartialEq)]
    struct TightLayout {
        velocity: wgsl_rs::std::Vec3f,
        acceleration: wgsl_rs::std::Vec3f,
        frame_count: u32,
    }

    #[test]
    fn tight_layout_offsets() {
        // velocity: offset 0, size 12, align 16
        assert_eq!(<TightLayout>::FIELDS[0].name, "velocity");
        assert_eq!(<TightLayout>::FIELDS[0].offset, 0);
        assert_eq!(<TightLayout>::FIELDS[0].size, 12);
        assert_eq!(<TightLayout>::FIELDS[0].alignment, 16);
        assert_eq!(<TightLayout>::FIELDS[0].pad_before, 0);

        // acceleration: roundUp(0+12, 16) = 16, size 12
        assert_eq!(<TightLayout>::FIELDS[1].name, "acceleration");
        assert_eq!(<TightLayout>::FIELDS[1].offset, 16);
        assert_eq!(<TightLayout>::FIELDS[1].size, 12);
        assert_eq!(<TightLayout>::FIELDS[1].alignment, 16);
        assert_eq!(<TightLayout>::FIELDS[1].pad_before, 4);

        // frame_count: roundUp(16+12, 4) = 28, u32=4
        assert_eq!(<TightLayout>::FIELDS[2].name, "frame_count");
        assert_eq!(<TightLayout>::FIELDS[2].offset, 28);
        assert_eq!(<TightLayout>::FIELDS[2].size, 4);
        assert_eq!(<TightLayout>::FIELDS[2].alignment, 4);
        assert_eq!(<TightLayout>::FIELDS[2].pad_before, 0);

        // struct size = roundUp(max(16,16,4)=16, 28+4=32) = 32
        assert_eq!(<TightLayout>::SIZE, 32);
        assert_eq!(<TightLayout>::ALIGN, 16);
    }

    #[derive(Layout)]
    struct ComplexEx4 {
        velocity: wgsl_rs::std::Vec3f,
        size: f32,
        direction: [wgsl_rs::std::Vec3f; 1],
        scale: f32,
        friction: f32,
    }

    #[test]
    fn complex_ex4() {
        // velocity: offset 0,  size 12, align 16
        // size:     roundUp(12, 4) = 12, size 4, align 4, pad_before 0
        // direction: roundUp(16, 16) = 16, [Vec3f;1] = 1*roundUp(16,12)=16, align 16
        // scale:    roundUp(32, 4) = 32, size 4, align 4, pad_before 0
        // friction: roundUp(36, 4) = 36, size 4, align 4, pad_before 0
        // SIZE = roundUp(16, 36+4=40) = 48
        assert_eq!(<ComplexEx4>::SIZE, 48);
        assert_eq!(<ComplexEx4>::ALIGN, 16);

        assert_eq!(<ComplexEx4>::FIELDS[0].name, "velocity");
        assert_eq!(<ComplexEx4>::FIELDS[0].offset, 0);
        assert_eq!(<ComplexEx4>::FIELDS[0].size, 12);
        assert_eq!(<ComplexEx4>::FIELDS[0].alignment, 16);
        assert_eq!(<ComplexEx4>::FIELDS[0].pad_before, 0);

        assert_eq!(<ComplexEx4>::FIELDS[1].name, "size");
        assert_eq!(<ComplexEx4>::FIELDS[1].offset, 12);
        assert_eq!(<ComplexEx4>::FIELDS[1].size, 4);
        assert_eq!(<ComplexEx4>::FIELDS[1].alignment, 4);
        assert_eq!(<ComplexEx4>::FIELDS[1].pad_before, 0);

        // direction: [Vec3f;1] → align 16, size 16. roundUp(12+4=16, 16) = 16
        assert_eq!(<ComplexEx4>::FIELDS[2].name, "direction");
        assert_eq!(<ComplexEx4>::FIELDS[2].offset, 16);
        assert_eq!(<ComplexEx4>::FIELDS[2].size, 16);
        assert_eq!(<ComplexEx4>::FIELDS[2].alignment, 16);
        assert_eq!(<ComplexEx4>::FIELDS[2].pad_before, 0);

        assert_eq!(<ComplexEx4>::FIELDS[3].name, "scale");
        assert_eq!(<ComplexEx4>::FIELDS[3].offset, 32);
        assert_eq!(<ComplexEx4>::FIELDS[3].size, 4);
        assert_eq!(<ComplexEx4>::FIELDS[3].alignment, 4);
        assert_eq!(<ComplexEx4>::FIELDS[3].pad_before, 0);

        assert_eq!(<ComplexEx4>::FIELDS[4].name, "friction");
        assert_eq!(<ComplexEx4>::FIELDS[4].offset, 36);
        assert_eq!(<ComplexEx4>::FIELDS[4].size, 4);
        assert_eq!(<ComplexEx4>::FIELDS[4].alignment, 4);
        assert_eq!(<ComplexEx4>::FIELDS[4].pad_before, 0);
    }

    #[derive(Layout, Debug, PartialEq)]
    pub struct NestedPadded {
        orientation: wgsl_rs::std::Vec3f,
        size: f32,
        direction: [wgsl_rs::std::Vec3f; 1],
        scale: f32,
        info: Vec3Struct,
        friction: f32,
    }

    #[derive(Layout, Debug, PartialEq)]
    struct Vec3Struct {
        velocity: wgsl_rs::std::Vec3f,
    }

    #[test]
    fn nested_struct_with_align_gaps() {
        // Vec3Struct: ALIGN = 16, SIZE = roundUp(16, 0+12) = 16
        assert_eq!(<Vec3Struct>::SIZE, 16);
        assert_eq!(<Vec3Struct>::ALIGN, 16);

        // orientation: offset 0, size 12, align 16
        // size:        roundUp(12, 4) = 12, size 4, align 4
        // direction:   roundUp(16, 16) = 16, [Vec3f;1]=16, align 16
        // scale:       roundUp(32, 4) = 32, size 4, align 4
        // info:        roundUp(36, 16) = 48, size 16, align 16 → pad_before = 12
        // friction:    roundUp(64, 4) = 64, size 4, align 4
        // SIZE = roundUp(16, 64+4=68) = 80

        assert_eq!(<NestedPadded>::SIZE, 80);
        assert_eq!(<NestedPadded>::ALIGN, 16);

        assert_eq!(<NestedPadded>::FIELDS[0].name, "orientation");
        assert_eq!(<NestedPadded>::FIELDS[0].offset, 0);
        assert_eq!(<NestedPadded>::FIELDS[0].size, 12);
        assert_eq!(<NestedPadded>::FIELDS[0].alignment, 16);
        assert_eq!(<NestedPadded>::FIELDS[0].pad_before, 0);

        assert_eq!(<NestedPadded>::FIELDS[1].name, "size");
        assert_eq!(<NestedPadded>::FIELDS[1].offset, 12);
        assert_eq!(<NestedPadded>::FIELDS[1].size, 4);
        assert_eq!(<NestedPadded>::FIELDS[1].alignment, 4);
        assert_eq!(<NestedPadded>::FIELDS[1].pad_before, 0);

        assert_eq!(<NestedPadded>::FIELDS[2].name, "direction");
        assert_eq!(<NestedPadded>::FIELDS[2].offset, 16);
        assert_eq!(<NestedPadded>::FIELDS[2].size, 16);
        assert_eq!(<NestedPadded>::FIELDS[2].alignment, 16);
        assert_eq!(<NestedPadded>::FIELDS[2].pad_before, 0);

        assert_eq!(<NestedPadded>::FIELDS[3].name, "scale");
        assert_eq!(<NestedPadded>::FIELDS[3].offset, 32);
        assert_eq!(<NestedPadded>::FIELDS[3].size, 4);
        assert_eq!(<NestedPadded>::FIELDS[3].alignment, 4);
        assert_eq!(<NestedPadded>::FIELDS[3].pad_before, 0);

        // info: roundUp(32+4=36, 16) = 48, size 16, pad_before = 48 - 36 = 12
        assert_eq!(<NestedPadded>::FIELDS[4].name, "info");
        assert_eq!(<NestedPadded>::FIELDS[4].offset, 48);
        assert_eq!(<NestedPadded>::FIELDS[4].size, 16);
        assert_eq!(<NestedPadded>::FIELDS[4].alignment, 16);
        assert_eq!(<NestedPadded>::FIELDS[4].pad_before, 12);

        assert_eq!(<NestedPadded>::FIELDS[5].name, "friction");
        assert_eq!(<NestedPadded>::FIELDS[5].offset, 64);
        assert_eq!(<NestedPadded>::FIELDS[5].size, 4);
        assert_eq!(<NestedPadded>::FIELDS[5].alignment, 4);
        assert_eq!(<NestedPadded>::FIELDS[5].pad_before, 0);
    }

    #[derive(Layout)]
    struct GenericPair<T: PartialEq> {
        a: T,
        b: T,
    }

    #[test]
    fn generic_struct() {
        let layout_pair_f32 = GenericPair::<f32>::FIELDS;
        assert_eq!(GenericPair::<f32>::SIZE, 8);
        assert_eq!(GenericPair::<f32>::ALIGN, 4);

        assert_eq!(layout_pair_f32[0].name, "a");
        assert_eq!(layout_pair_f32[0].offset, 0);
        assert_eq!(layout_pair_f32[0].size, 4);
        assert_eq!(layout_pair_f32[0].alignment, 4);
        assert_eq!(layout_pair_f32[0].pad_before, 0);

        assert_eq!(layout_pair_f32[1].name, "b");
        assert_eq!(layout_pair_f32[1].offset, 4);
        assert_eq!(layout_pair_f32[1].size, 4);
        assert_eq!(layout_pair_f32[1].alignment, 4);
        assert_eq!(layout_pair_f32[1].pad_before, 0);

        // Generic with Vec3f (align 16, size 12)
        let layout_pair_vec3 = GenericPair::<wgsl_rs::std::Vec3f>::FIELDS;
        assert_eq!(GenericPair::<wgsl_rs::std::Vec3f>::SIZE, 32);
        assert_eq!(GenericPair::<wgsl_rs::std::Vec3f>::ALIGN, 16);

        assert_eq!(layout_pair_vec3[0].offset, 0);
        assert_eq!(layout_pair_vec3[0].size, 12);
        assert_eq!(layout_pair_vec3[0].pad_before, 0);

        // b: roundUp(12, 16) = 16
        assert_eq!(layout_pair_vec3[1].offset, 16);
        assert_eq!(layout_pair_vec3[1].size, 12);
        assert_eq!(layout_pair_vec3[1].pad_before, 4);
    }

    // Verify pad_before with mixed alignments
    #[derive(Layout)]
    struct MixedAlign {
        a: f32,
        b: wgsl_rs::std::Mat4x4f,
        c: f32,
    }

    #[test]
    fn mixed_align_pad_before() {
        // a: offset 0, size 4, align 4
        // b: roundUp(4, 16) = 16, size 64, align 16, pad_before = 12
        // c: roundUp(80, 4) = 80, size 4, align 4, pad_before = 0
        // SIZE = roundUp(16, 84) = 96
        assert_eq!(<MixedAlign>::SIZE, 96);
        assert_eq!(<MixedAlign>::ALIGN, 16);

        assert_eq!(<MixedAlign>::FIELDS[0].pad_before, 0);
        assert_eq!(<MixedAlign>::FIELDS[1].offset, 16);
        assert_eq!(<MixedAlign>::FIELDS[1].pad_before, 12);
        assert_eq!(<MixedAlign>::FIELDS[2].pad_before, 0);
    }

    // ===== write_layout_bytes tests =====

    fn read_f32_le(buf: &[u8]) -> f32 {
        f32::from_le_bytes(buf[..4].try_into().unwrap())
    }

    fn read_u32_le(buf: &[u8]) -> u32 {
        u32::from_le_bytes(buf[..4].try_into().unwrap())
    }

    #[test]
    fn write_scalars() {
        let mut buf = [0u8; 4];
        let v = 1.25f32;
        write_layout_bytes(&mut buf, &v).unwrap();
        assert_eq!(read_f32_le(&buf), 1.25);

        let v = -42i32;
        write_layout_bytes(&mut buf, &v).unwrap();
        assert_eq!(i32::from_le_bytes(buf[..4].try_into().unwrap()), -42);

        let v = true;
        write_layout_bytes(&mut buf, &v).unwrap();
        assert_eq!(read_u32_le(&buf), 1);

        let v = false;
        write_layout_bytes(&mut buf, &v).unwrap();
        assert_eq!(read_u32_le(&buf), 0);
    }

    #[test]
    fn write_buffer_too_small() {
        let mut buf = [0u8; 3];
        let err = write_layout_bytes(&mut buf, &42.0f32).unwrap_err();
        assert_eq!(
            err,
            Error::BufferTooSmall {
                needed: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn write_vec3f() {
        let v = wgsl_rs::std::vec3f(1.0, 2.0, 3.0);
        let mut buf = vec![0u8; 12];
        write_layout_bytes(&mut buf, &v).unwrap();
        assert_eq!(read_f32_le(&buf[0..4]), 1.0);
        assert_eq!(read_f32_le(&buf[4..8]), 2.0);
        assert_eq!(read_f32_le(&buf[8..12]), 3.0);
    }

    #[test]
    fn write_single_scalar_struct() {
        let s = SingleScalar { value: 3.5 };
        let mut buf = vec![0u8; <SingleScalar>::SIZE];
        write_layout_bytes(&mut buf, &s).unwrap();
        assert_eq!(read_f32_le(&buf), 3.5);
    }

    #[test]
    fn write_tight_layout_padding() {
        let s = TightLayout {
            velocity: wgsl_rs::std::vec3f(1.0, 0.0, -1.0),
            acceleration: wgsl_rs::std::vec3f(0.0, -9.8, 0.0),
            frame_count: 42,
        };
        // SIZE = 32, with padding at bytes 12-15 and 28-31
        let mut buf = vec![0u8; 32];
        write_layout_bytes(&mut buf, &s).unwrap();

        // velocity at offset 0-11
        assert_eq!(read_f32_le(&buf[0..4]), 1.0);
        assert_eq!(read_f32_le(&buf[4..8]), 0.0);
        assert_eq!(read_f32_le(&buf[8..12]), -1.0);
        // padding at 12-15 should be zero
        assert_eq!(&buf[12..16], &[0u8; 4]);
        // acceleration at offset 16-27
        assert_eq!(read_f32_le(&buf[16..20]), 0.0);
        assert_eq!(read_f32_le(&buf[20..24]), -9.8);
        assert_eq!(read_f32_le(&buf[24..28]), 0.0);
        // frame_count at offset 28-31 (offset 28 is NOT padding — it's field data)
        assert_eq!(read_u32_le(&buf[28..32]), 42);
    }

    #[test]
    fn write_nested_struct() {
        let info = Vec3Struct {
            velocity: wgsl_rs::std::vec3f(1.0, 2.0, 3.0),
        };
        let s = NestedPadded {
            orientation: wgsl_rs::std::vec3f(0.0, 1.0, 0.0),
            size: 2.5,
            direction: [wgsl_rs::std::vec3f(4.0, 5.0, 6.0)],
            scale: 0.1,
            info,
            friction: 0.05,
        };
        let mut buf = vec![0u8; <NestedPadded>::SIZE];
        write_layout_bytes(&mut buf, &s).unwrap();

        // orientation at 0-11
        assert_eq!(read_f32_le(&buf[0..4]), 0.0);
        assert_eq!(read_f32_le(&buf[4..8]), 1.0);
        assert_eq!(read_f32_le(&buf[8..12]), 0.0);
        // size at 12-15
        assert_eq!(read_f32_le(&buf[12..16]), 2.5);
        // direction at 16-31
        assert_eq!(read_f32_le(&buf[16..20]), 4.0);
        assert_eq!(read_f32_le(&buf[20..24]), 5.0);
        assert_eq!(read_f32_le(&buf[24..28]), 6.0);
        // padding at 28-31 zero
        assert_eq!(&buf[28..32], &[0u8; 4]);
        // scale at 32-35
        assert_eq!(read_f32_le(&buf[32..36]), 0.1);
        // padding at 36-47 zero (12 bytes before info)
        assert_eq!(&buf[36..48], &[0u8; 12]);
        // info at 48-63
        assert_eq!(read_f32_le(&buf[48..52]), 1.0);
        assert_eq!(read_f32_le(&buf[52..56]), 2.0);
        assert_eq!(read_f32_le(&buf[56..60]), 3.0);
        // padding at 60-63 zero
        assert_eq!(&buf[60..64], &[0u8; 4]);
        // friction at 64-67
        assert_eq!(read_f32_le(&buf[64..68]), 0.05);
    }

    #[test]
    fn write_generic_struct() {
        let p = GenericPair::<f32> { a: 1.0, b: 2.0 };
        let mut buf = vec![0u8; 8];
        write_layout_bytes(&mut buf, &p).unwrap();
        assert_eq!(read_f32_le(&buf[0..4]), 1.0);
        assert_eq!(read_f32_le(&buf[4..8]), 2.0);
    }

    #[test]
    fn write_zero_size_empty_struct() {
        let s = Empty {};
        let mut buf = [];
        write_layout_bytes(&mut buf, &s).unwrap();
    }

    #[test]
    fn write_mixed_align_padding() {
        let s = MixedAlign {
            a: 1.0,
            b: wgsl_rs::std::Mat4x4f::default(),
            c: 2.0,
        };
        let mut buf = vec![0xAAu8; 96]; // fill with non-zero to verify zeroing
        write_layout_bytes(&mut buf, &s).unwrap();

        // a at 0-3
        assert_eq!(read_f32_le(&buf[0..4]), 1.0);
        // padding at 4-15 should be zero
        assert_eq!(&buf[4..16], &[0u8; 12]);
        // b at 16-79 (mat4x4f)
        assert_eq!(read_f32_le(&buf[16..20]), 0.0); // default Mat4x4f is identity
        // c at 80-83
        assert_eq!(read_f32_le(&buf[80..84]), 2.0);
        // trailing padding at 84-95 zero
        assert_eq!(&buf[84..96], &[0u8; 12]);
    }

    // ===== read_layout_bytes tests =====

    #[test]
    fn read_scalars() {
        let buf = 1.25f32.to_le_bytes();
        let v: f32 = read_layout_bytes(&buf).unwrap();
        assert_eq!(v, 1.25);

        let buf = (-42i32).to_le_bytes();
        let v: i32 = read_layout_bytes(&buf).unwrap();
        assert_eq!(v, -42);

        let buf = 42u32.to_le_bytes();
        let v: u32 = read_layout_bytes(&buf).unwrap();
        assert_eq!(v, 42);

        let buf = 0u32.to_le_bytes();
        let v: bool = read_layout_bytes(&buf).unwrap();
        assert!(!v);

        let buf = 1u32.to_le_bytes();
        let v: bool = read_layout_bytes(&buf).unwrap();
        assert!(v);
    }

    #[test]
    fn read_buffer_too_small() {
        let buf = [0u8; 3];
        let err = read_layout_bytes::<f32>(&buf).unwrap_err();
        assert_eq!(
            err,
            Error::BufferTooSmall {
                needed: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn read_vec3f() {
        let original = wgsl_rs::std::vec3f(1.0, 2.0, 3.0);
        let mut buf = vec![0u8; 12];
        write_layout_bytes(&mut buf, &original).unwrap();
        let restored: wgsl_rs::std::Vec3f = read_layout_bytes(&buf).unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn read_write_roundtrip_struct() {
        let s = TightLayout {
            velocity: wgsl_rs::std::vec3f(1.0, 0.0, -1.0),
            acceleration: wgsl_rs::std::vec3f(0.0, -9.8, 0.0),
            frame_count: 42,
        };
        let mut buf = vec![0u8; <TightLayout>::SIZE];
        write_layout_bytes(&mut buf, &s).unwrap();
        let restored: TightLayout = read_layout_bytes(&buf).unwrap();
        assert_eq!(restored, s);
    }

    #[test]
    fn read_write_roundtrip_nested() {
        let info = Vec3Struct {
            velocity: wgsl_rs::std::vec3f(1.0, 2.0, 3.0),
        };
        let s = NestedPadded {
            orientation: wgsl_rs::std::vec3f(0.0, 1.0, 0.0),
            size: 2.5,
            direction: [wgsl_rs::std::vec3f(4.0, 5.0, 6.0)],
            scale: 0.1,
            info,
            friction: 0.05,
        };
        let mut buf = vec![0u8; <NestedPadded>::SIZE];
        write_layout_bytes(&mut buf, &s).unwrap();
        let restored: NestedPadded = read_layout_bytes(&buf).unwrap();
        assert_eq!(restored, s);
    }

    #[test]
    fn read_write_roundtrip_generic() {
        let p = GenericPair::<f32> { a: 1.0, b: 2.0 };
        let mut buf = vec![0u8; <GenericPair<f32>>::SIZE];
        write_layout_bytes(&mut buf, &p).unwrap();
        let restored: GenericPair<f32> = read_layout_bytes(&buf).unwrap();
        assert_eq!(restored.a, p.a);
        assert_eq!(restored.b, p.b);
    }

    #[test]
    fn read_zero_size_empty_struct() {
        let s: Empty = read_layout_bytes(&[]).unwrap();
        // An empty struct has no fields to compare, so we just verify it returns Ok.
        let _ = s;
    }

    #[test]
    fn read_write_mixed_align_roundtrip() {
        let s = MixedAlign {
            a: 1.0,
            b: wgsl_rs::std::Mat4x4f::default(),
            c: 2.0,
        };
        let mut buf = vec![0u8; <MixedAlign>::SIZE];
        write_layout_bytes(&mut buf, &s).unwrap();
        let restored: MixedAlign = read_layout_bytes(&buf).unwrap();
        assert_eq!(restored.a, s.a);
        // Mat4x4f only implements Default (not Debug/PartialEq easily), so check
        // columns
        assert_eq!(restored.b[0u32], s.b[0u32]);
        assert_eq!(restored.b[1u32], s.b[1u32]);
        assert_eq!(restored.b[2u32], s.b[2u32]);
        assert_eq!(restored.b[3u32], s.b[3u32]);
        assert_eq!(restored.c, s.c);
    }
}
