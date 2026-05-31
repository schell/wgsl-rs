/// Layout information for a single field in a WGSL struct, computed per
/// the WGSL specification §14.4.1 ("Alignment and Size").
///
/// WGSL struct fields are laid out sequentially with alignment-required
/// gaps between fields. The GPU does not reorder fields or pack them
/// tightly — every field starts at a byte offset that is a multiple of
/// its alignment requirement.
///
/// # Understanding `pad_after`
///
/// `pad_after` is the number of **padding bytes after this field's data
/// before the next field begins** (or before the end of the struct for
/// the last field). It is the gap created when the next field's alignment
/// requires a larger offset than where this field's data naturally ends.
///
/// When writing data sequentially, you write this field's bytes, then
/// write `pad_after` zero bytes to fill the gap before the next field.
/// The last field's `pad_after` covers trailing padding to the struct's
/// total size.
///
/// ## Concrete Example
///
/// ```rust
/// use wgsl_rs::std::Vec3f;
/// use wgsl_rs_layout::{Layout, WgslLayout};
///
/// #[derive(wgsl_rs_layout::Layout)]
/// struct Ex4 {
///     velocity: Vec3f,  // offset 0,  size 12, align 16, pad_after = 0
///     size: f32,        // offset 12, size 4,  align 4,  pad_after = 0
///     direction: Vec3f, // offset 16, size 12, align 16, pad_after = 0
///     scale: f32,       // offset 28, size 4,  align 4,  pad_after = 0
/// }
///
/// // Verify the layout:
/// assert_eq!(Ex4::FIELDS[0].offset, 0);
/// assert_eq!(Ex4::FIELDS[0].pad_after, 0); // size fits at 12 (4-aligned)
/// assert_eq!(Ex4::FIELDS[1].offset, 12);
/// assert_eq!(Ex4::FIELDS[1].pad_after, 0); // 12+4=16 is 16-aligned for direction
/// assert_eq!(Ex4::FIELDS[2].offset, 16);
/// assert_eq!(Ex4::FIELDS[2].pad_after, 0); // 16+12=28 is 4-aligned for scale
/// assert_eq!(Ex4::FIELDS[3].offset, 28);
/// assert_eq!(Ex4::FIELDS[3].pad_after, 0); // 28+4=32, struct size is 32
/// assert_eq!(<Ex4 as WgslLayout>::SIZE, 32);
/// assert_eq!(<Ex4 as WgslLayout>::ALIGN, 16);
/// ```
///
/// Note: this example uses `Vec3f` rather than `[Vec3f; 1]` to keep the
/// focus on inter-field padding. Arrays have their own internal stride
/// padding (each element occupies `roundUp(align, size)` bytes), visible
/// via the array's `SIZE`, and that internal padding is handled by the
/// array's own write/read impls, not by `pad_after` on the parent field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    /// The field name as declared in the Rust struct.
    pub name: &'static str,
    /// Byte offset from the start of the struct.
    pub offset: usize,
    /// Byte size of the field's data per WGSL layout rules.
    pub size: usize,
    /// Alignment requirement of the field's type.
    pub alignment: usize,
    /// Bytes of padding AFTER this field (gap to the next field, or to the
    /// struct end for the last field).
    pub pad_after: usize,
}
