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
/// ```ignore
/// struct Ex4 {
///     velocity: Vec3f,     // offset 0,  size 12, align 16, pad_after = 0
///     size: f32,           // offset 12, size 4,  align 4,  pad_after = 0
///     direction: Vec3f,    // offset 16, size 12, align 16, pad_after = 4
///     scale: f32,          // offset 32, size 4,  align 4,  pad_after = 12
///     info: Ex4a,          // offset 48, size 16, align 16, pad_after = 0
///     friction: f32,       // offset 64, size 4,  align 4,  pad_after = 12
/// }
/// ```
///
/// - `velocity` has `pad_after = 0`: `size`'s alignment (4) fits at offset 12
///   (12 + 0 = 12 which is already 4-aligned).
/// - `direction` has `pad_after = 4`: the next field `scale` has alignment 4,
///   but `direction` ends at offset 28. To reach the next 16-byte-aligned
///   position (32), 4 bytes of padding are needed. So you write `direction`'s
///   data at offset 16..28, then 4 zero bytes at 28..32 before `scale`.
/// - `scale` has `pad_after = 12`: `info` has align 16, so `scale`'s end at
///   offset 36 needs 12 zero bytes to reach offset 48.
/// - `info` has `pad_after = 0`: `friction`'s alignment (4) fits at offset 64
///   (48 + 16 = 64, already 4-aligned).
/// - `friction` has `pad_after = 12`: the struct ends at offset 68, but its
///   alignment is 16, so the total size rounds up to 80 — 12 trailing zero
///   bytes.
///
/// Note: this example uses `Vec3f` for `direction` rather than `[Vec3f; 1]`
/// to keep the focus on inter-field padding. Arrays have their own internal
/// stride padding (each element occupies `roundUp(align, size)` bytes),
/// visible via the array's `SIZE`, and that internal padding is handled by
/// the array's own write/read impls, not by `pad_after` on the parent field.
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
