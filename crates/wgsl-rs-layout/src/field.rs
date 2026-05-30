/// Layout information for a single field in a WGSL struct, computed per
/// the WGSL specification §14.4.1 ("Alignment and Size").
///
/// WGSL struct fields are laid out sequentially with alignment-required
/// gaps between fields. The GPU does not reorder fields or pack them
/// tightly — every field starts at a byte offset that is a multiple of
/// its alignment requirement.
///
/// # Understanding `pad_before`
///
/// `pad_before` is the number of **padding bytes between the end of the
/// previous field and the start of this field**. It is always 0 for the
/// first field in a struct. When `pad_before` is non-zero, this means
/// the previous field's end did not satisfy this field's alignment
/// requirement, and the GPU will insert dead bytes there.
///
/// **When marshalling data to/from GPU buffers, you must account for
/// these padding bytes.** They are NOT part of any field's data. If you
/// are writing raw bytes into a buffer, you need to insert `pad_before`
/// zero bytes before writing this field's data.
///
/// ## Concrete Example
///
/// ```ignore
/// struct Ex4 {
///     velocity: Vec3f,      // offset 0,  size 12, align 16
///     size: f32,            // offset 12, size 4,  align 4,  pad_before = 0
///     direction: [Vec3f; 1],// offset 16, size 12, align 16, pad_before = 0
///     scale: f32,           // offset 32, size 4,  align 4,  pad_before = 4
///     info: Ex4a,           // offset 48, size 12, align 16, pad_before = 12
///     friction: f32,        // offset 64, size 4,  align 4,  pad_before = 4
/// }
/// ```
///
/// Notice how `scale` has `pad_before = 4` — this is the gap between the
/// end of `direction` (offset 16 + size 12 = 28) and `scale`'s required
/// 16-byte-aligned start (offset 32). When writing `scale` to a buffer,
/// you must write 4 zero bytes at offset 28, then `scale`'s data at
/// offset 32.
///
/// Another large gap appears before `info`: the `Ex4a` struct has align
/// 16 (from its `Vec3f` field), so `info` starts at offset 48, leaving
/// 12 bytes of padding between `scale` (ends at 32 + 4 = 36) and `info`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    /// The field name as declared in the Rust struct.
    pub name: &'static str,
    /// Byte offset from the start of the struct.
    pub offset: usize,
    /// Byte size of the field's data (not including padding).
    pub size: usize,
    /// Alignment requirement of the field's type.
    pub alignment: usize,
    /// Bytes of padding BEFORE this field (gap from previous field's end).
    /// Always 0 for the first field.
    pub pad_before: usize,
}
