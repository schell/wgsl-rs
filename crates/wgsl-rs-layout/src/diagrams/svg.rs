//! SVG byte-diagram renderer.
//!
//! Output structure:
//!
//! - A header strip with the struct name and a `size: N bytes · align: N bytes`
//!   meta line (or `size: <offset> + LEN × <stride> bytes` for runtime-array
//!   structs).
//! - One `<bytes_per_row>`-byte "row" per chunk of struct size. The row width
//!   is supplied by the caller (the orchestrator derives it from `T::ALIGN`).
//! - Each row has a label band and a byte band.
//! - The label band shows the field name in a wide colored rect; padding is
//!   shown as `-pad-` on a light-grey background.
//! - The byte band shows colored cells with the absolute byte offset rendered
//!   inside each one. Word boundaries (every 4 bytes) are signalled by an
//!   alternating background tint overlay on every other word in the byte band
//!   only.

use super::{DiagramConfig, RenderField};
use std::fmt::Write;

/// Width of a word in bytes. Used to compute the alternating
/// background tint on every other word.
const WORD_BYTES: usize = 4;

/// Heuristic average glyph width, in pixels, for the 16px-bold
/// struct-name text. Used to estimate header width.
const TITLE_CHAR_PX: f32 = 9.0;
/// Heuristic average glyph width, in pixels, for the 11px meta
/// line. Used to estimate header width.
const META_CHAR_PX: f32 = 6.0;
/// Horizontal gap between the struct name and the meta text when
/// inline.
const INLINE_GAP_PX: f32 = 8.4; // 0.6em at 14px monospace

/// Render a diagram for the given fields and total size.
///
/// `bytes_per_row` is the row width in bytes; the diagram wraps
/// after every `bytes_per_row` bytes.
///
/// The header (struct name + meta line) is rendered inline on a
/// single line when it fits within the diagram's body width, and
/// wraps to two lines otherwise. The diagram's viewBox always
/// accommodates the header so the meta line is fully visible.
pub(super) fn render(
    fields: &[RenderField],
    size: usize,
    bytes_per_row: usize,
    _align: usize,
    type_name: &str,
    meta: &str,
    config: &DiagramConfig,
) -> String {
    let cell = config.cell_size;
    let label_height = cell;
    let byte_height = cell;
    let row_gap = cell / 4.0;
    let pad_left = 0.0;
    let pad_right = cell / 3.0;

    // Estimate the inline header width: name + gap + meta.
    let short_name = short_type_name(type_name);
    let inline_header_estimate =
        short_name.len() as f32 * TITLE_CHAR_PX + INLINE_GAP_PX + meta.len() as f32 * META_CHAR_PX;

    // Body width is the dominant width when the header fits
    // inline; otherwise the header estimate is the dominant
    // constraint and the diagram widens to fit it.
    let body_width = (bytes_per_row as f32) * cell + pad_left + pad_right;
    let total_width = body_width.max(inline_header_estimate + pad_right);

    // Title (struct name) and meta line. If they fit on one line,
    // share a single <text> element via <tspan> children. Otherwise
    // the title sits on line 1 and the meta on line 2.
    let wrap_header = inline_header_estimate + pad_right > body_width;
    let title_line_height = if wrap_header { cell * 0.95 } else { cell * 1.1 };
    let meta_line_height = if wrap_header { cell * 0.7 } else { 0.0 };
    let header_height = title_line_height + meta_line_height;

    let n_rows = size.div_ceil(bytes_per_row).max(1);
    let body_height = (label_height + byte_height + row_gap) * n_rows as f32 + row_gap;
    let total_height = header_height + body_height;

    let mut out = String::new();
    write!(
        &mut out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {h:.0}" class="wgsl-rs-byte-diagram" data-struct="{name}">"#,
        w = total_width,
        h = total_height,
        name = escape_attr(type_name),
    )
    .unwrap();

    // Inline style scoped to this class. Light-mode only.
    write!(
        &mut out,
        r#"<style>
.wgsl-rs-byte-diagram {{ font-family: monospace; }}
.wgsl-rs-byte-diagram text {{ dominant-baseline: middle; }}
.wgsl-rs-byte-diagram .struct-name {{ font-weight: 700; font-size: 16px; fill: #111; }}
.wgsl-rs-byte-diagram .meta {{ font-size: 11px; fill: #666; }}
.wgsl-rs-byte-diagram .field-name {{ font-size: 11px; fill: #111; }}
.wgsl-rs-byte-diagram .pad-text {{ font-size: 11px; fill: #888; }}
.wgsl-rs-byte-diagram .byte-index {{ font-size: 9px; fill: rgba(0,0,0,0.55); }}
.wgsl-rs-byte-diagram .byte-cell {{ stroke: rgba(0,0,0,0.2); stroke-width: 1; }}
.wgsl-rs-byte-diagram .label-cell {{ stroke: rgba(0,0,0,0.2); stroke-width: 1; }}
.wgsl-rs-byte-diagram .pad-cell {{ stroke: rgba(0,0,0,0.1); stroke-width: 1; fill: #f7f7f7; }}
.wgsl-rs-byte-diagram .pad-label-cell {{ stroke: rgba(0,0,0,0.1); stroke-width: 1; fill: #f7f7f7; }}
/* Alternating background tint for every other 4-byte word, applied
   only to the byte band. The tint is a translucent overlay so it
   darkens the underlying field/pad color without overriding it. */
.wgsl-rs-byte-diagram .word-tint {{ fill: rgba(0, 0, 0, 0.08); stroke: none; pointer-events: none; }}
</style>"#
    )
    .unwrap();

    // Header. Inline when the estimate fits, wrapped to two lines
    // otherwise.
    if wrap_header {
        let title_y = title_line_height * 0.7;
        let meta_y = title_line_height + meta_line_height * 0.7;
        write!(
            &mut out,
            r#"<text class="struct-name" x="0" y="{y:.1}">{name}</text>"#,
            y = title_y,
            name = escape_text(short_name),
        )
        .unwrap();
        write!(
            &mut out,
            r#"<text class="meta" x="0" y="{y:.1}">{meta}</text>"#,
            y = meta_y,
            meta = escape_text(meta),
        )
        .unwrap();
    } else {
        write!(
            &mut out,
            r#"<text class="struct-name" x="0" y="{y:.1}"><tspan class="struct-name">{name}</tspan><tspan class="meta" dx="0.6em">{meta}</tspan></text>"#,
            y = title_line_height * 0.7,
            name = escape_text(short_name),
            meta = escape_text(meta),
        )
        .unwrap();
    }

    // Build a byte map: index -> (kind, owner_field_idx).
    let mut byte_map: Vec<(ByteKind, usize)> = vec![(ByteKind::Pad, 0); size];
    for (i, f) in fields.iter().enumerate() {
        for b in f.offset..(f.offset + f.size) {
            if b < byte_map.len() {
                byte_map[b] = (ByteKind::Data, i);
            }
        }
    }

    let bytes_x = pad_left;
    let body_top = header_height + row_gap;

    for row in 0..n_rows {
        let row_offset = row * bytes_per_row;
        let row_y = body_top + row as f32 * (label_height + byte_height + row_gap);
        let label_y = row_y;
        let byte_y = label_y + label_height;

        // Label band: one rect per run of consecutive same-field bytes.
        let runs = build_runs(&byte_map, row_offset, bytes_per_row, size);
        for run in &runs {
            let x = bytes_x + run.start as f32 * cell;
            let w = run.len as f32 * cell;
            match run.kind {
                ByteKind::Data => {
                    let field = &fields[run.owner];
                    let hue = hue_for(&field.name);
                    let label_color = lch_to_hex(hue, 90, 60);
                    write!(
                        &mut out,
                        r#"<rect class="label-cell" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" fill="{fill}"/>"#,
                        x = x,
                        y = label_y,
                        w = w,
                        h = label_height,
                        fill = label_color,
                    )
                    .unwrap();
                    write!(
                        &mut out,
                        r#"<text class="field-name" x="{tx:.1}" y="{ty:.1}" text-anchor="start">.{name}</text>"#,
                        tx = x,
                        ty = label_y + label_height / 2.0,
                        name = escape_text(&field.name),
                    )
                    .unwrap();
                }
                ByteKind::Pad => {
                    if w > 0.0 {
                        write!(
                            &mut out,
                            r#"<rect class="pad-label-cell" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}"/>"#,
                            x = x,
                            y = label_y,
                            w = w,
                            h = label_height,
                        )
                        .unwrap();
                        if w >= 36.0 {
                            write!(
                                &mut out,
                                r#"<text class="pad-text" x="{tx:.1}" y="{ty:.1}" text-anchor="start">-pad-</text>"#,
                                tx = x,
                                ty = label_y + label_height / 2.0,
                            )
                            .unwrap();
                        }
                    }
                }
            }
        }

        // Pre-compute per-field byte colors so we don't re-hash and
        // re-convert every cell. The cached `String` is cheap to
        // clone; for very large layouts this turns an O(cells) cost
        // into O(fields).
        let mut field_byte_colors: Vec<Option<String>> = vec![None; fields.len()];
        for (i, f) in fields.iter().enumerate() {
            let hue = hue_for(&f.name);
            field_byte_colors[i] = Some(lch_to_hex(hue, 65, 50));
        }

        // Byte band: unlabelled colored squares with the absolute
        // byte offset rendered inside each cell.
        for b in 0..bytes_per_row {
            let abs = row_offset + b;
            if abs >= size {
                break;
            }
            let x = bytes_x + b as f32 * cell;
            let (kind, owner) = byte_map[abs];
            match kind {
                ByteKind::Data => {
                    let field = &fields[owner];
                    let byte_color = field_byte_colors[owner]
                        .as_deref()
                        .expect("color pre-computed for every field");
                    let is_first_in_field = abs == field.offset;
                    let is_last_in_field = abs == field.offset + field.size - 1;
                    let class = byte_cell_class(is_first_in_field, is_last_in_field);
                    write!(
                        &mut out,
                        r#"<rect class="{class}" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" fill="{fill}"/>"#,
                        x = x,
                        y = byte_y,
                        w = cell,
                        h = byte_height,
                        class = class,
                        fill = byte_color,
                    )
                    .unwrap();
                }
                ByteKind::Pad => {
                    write!(
                        &mut out,
                        r#"<rect class="pad-cell" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}"/>"#,
                        x = x,
                        y = byte_y,
                        w = cell,
                        h = byte_height,
                    )
                    .unwrap();
                }
            }
        }

        // Word-end tint overlay: paint a translucent dark rect over
        // every other 4-byte word in the byte band. Emitted AFTER
        // the byte-band cells so the tint sits on top of the opaque
        // field colors and is visible to the user.
        for w in 0..bytes_per_row / WORD_BYTES {
            let word_start = w * WORD_BYTES;
            if row_offset + word_start >= size {
                break;
            }
            if !w.is_multiple_of(2) {
                continue;
            }
            let x = bytes_x + word_start as f32 * cell;
            let last_in_row =
                ((row_offset + word_start + WORD_BYTES).min(size)) - (row_offset + word_start);
            let w_px = last_in_row as f32 * cell;
            write!(
                &mut out,
                r#"<rect class="word-tint" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}"/>"#,
                x = x,
                y = byte_y,
                w = w_px,
                h = byte_height,
            )
            .unwrap();
        }

        // Byte-index text labels: emitted last so they sit on top of
        // both the byte cells and the word-tint overlay.
        for b in 0..bytes_per_row {
            let abs = row_offset + b;
            if abs >= size {
                break;
            }
            let x = bytes_x + b as f32 * cell;
            let tx = x + cell / 2.0;
            let ty = byte_y + byte_height / 2.0;
            write!(
                &mut out,
                r#"<text class="byte-index" x="{tx:.1}" y="{ty:.1}" text-anchor="middle">{n}</text>"#,
                tx = tx,
                ty = ty,
                n = abs,
            )
            .unwrap();
        }
    }

    write!(&mut out, "</svg>").unwrap();
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteKind {
    Data,
    Pad,
}

#[derive(Debug)]
struct Run {
    start: usize,
    len: usize,
    kind: ByteKind,
    owner: usize,
}

/// Group consecutive bytes of the same kind into runs.
fn build_runs(
    byte_map: &[(ByteKind, usize)],
    row_start: usize,
    row_len: usize,
    size: usize,
) -> Vec<Run> {
    let mut runs = Vec::new();
    let row_end = (row_start + row_len).min(size);
    let mut i = row_start;
    while i < row_end {
        let (kind, owner) = byte_map[i];
        let mut j = i + 1;
        while j < row_end && byte_map[j].0 == kind && byte_map[j].1 == owner {
            j += 1;
        }
        runs.push(Run {
            start: i - row_start,
            len: j - i,
            kind,
            owner,
        });
        i = j;
    }
    runs
}

/// Build a class string for a byte-band cell. Word boundaries are
/// signalled by an overlay rect, so the cell itself carries no
/// `word-end` class.
fn byte_cell_class(is_first: bool, is_last: bool) -> &'static str {
    match (is_first, is_last) {
        (true, true) => "byte-cell elem-start elem-end",
        (true, false) => "byte-cell elem-start",
        (false, true) => "byte-cell elem-end",
        (false, false) => "byte-cell",
    }
}

/// Take the last `::`-separated segment of a fully-qualified type
/// name so the header shows `Uniforms` rather than
/// `my_crate::foo::Uniforms`.
fn short_type_name(name: &str) -> &str {
    name.rsplit("::").next().unwrap_or(name)
}

/// Stable hash of a string mapped to a hue in `[0, 360)`. Uses FNV-1a.
fn hue_for(name: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash % 360
}

/// Convert a hue + a (lightness, chroma) pair to an `#rrggbb` hex
/// string. Approximates the LCH color we'd get from `lch(L C H)` in
/// CSS but emits a format that all SVG renderers understand.
fn lch_to_hex(hue: u32, lightness: u32, chroma: u32) -> String {
    let l = lightness.min(100) as f32 / 100.0;
    let s = (chroma.min(150) as f32 * 0.6).min(100.0) / 100.0;
    let h = hue as f32;
    hsl_to_hex(h, s, l)
}

fn hsl_to_hex(h: f32, s: f32, l: f32) -> String {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    let r = ((r1 + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = ((g1 + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = ((b1 + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(all(test, feature = "doc-diagrams"))]
mod tests {
    use super::*;

    fn meta_for(size: usize, align: usize) -> String {
        format!("size: {size} bytes \u{00b7} align: {align} bytes")
    }

    fn fields_one_scalar() -> Vec<RenderField> {
        vec![RenderField {
            name: "value".into(),
            offset: 0,
            size: 4,
            pad_after: 0,
        }]
    }

    fn fields_three() -> Vec<RenderField> {
        vec![
            RenderField {
                name: "a".into(),
                offset: 0,
                size: 4,
                pad_after: 12,
            },
            RenderField {
                name: "b".into(),
                offset: 16,
                size: 64,
                pad_after: 0,
            },
            RenderField {
                name: "c".into(),
                offset: 80,
                size: 4,
                pad_after: 12,
            },
        ]
    }

    #[test]
    fn renders_simple_struct() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_one_scalar(), 4, 4, 4, "S", &meta_for(4, 4), &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("wgsl-rs-byte-diagram"));
        assert!(svg.contains("data-struct=\"S\""));
        assert!(svg.contains(".value"));
    }

    #[test]
    fn renders_three_field_struct() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(svg.contains(".a"));
        assert!(svg.contains(".b"));
        assert!(svg.contains(".c"));
        assert!(svg.contains("-pad-"));
    }

    #[test]
    fn header_shows_struct_name() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(svg.contains("class=\"struct-name\""));
        assert!(svg.contains(">S<"));
    }

    #[test]
    fn header_shows_size_and_align() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(svg.contains("size: 96 bytes"));
        assert!(svg.contains("align: 16 bytes"));
    }

    #[test]
    fn name_and_meta_on_same_line() {
        // The header is a single <text class="struct-name"> element
        // containing two <tspan> children. The name and the meta
        // share the same baseline.
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        // Exactly one <text> opens with the struct-name class.
        let open = svg.matches(r#"<text class="struct-name""#).count();
        assert_eq!(
            open, 1,
            "expected one <text class=\"struct-name\">, got {open}"
        );
        // The header text element contains both the name and the meta.
        let header_start = svg.find(r#"<text class="struct-name""#).unwrap();
        let header_end = svg[header_start..].find("</text>").unwrap() + header_start;
        let header = &svg[header_start..=header_end];
        assert!(header.contains("S"), "name missing from header text");
        assert!(
            header.contains("size: 96 bytes"),
            "meta missing from header text"
        );
        assert!(
            header.contains(r#"<tspan class="meta""#),
            "meta should be a tspan"
        );
        // No separate <text class="meta"> element.
        let meta_text_count = svg.matches(r#"<text class="meta""#).count();
        assert_eq!(meta_text_count, 0, "meta should not be a separate <text>");
    }

    #[test]
    fn no_column_header_strip() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        // The "offset" / "layout" column-header strip is removed; the
        // per-cell byte indices make the row offset redundant.
        assert!(!svg.contains("col-header"));
        assert!(!svg.contains(">offset<"));
        assert!(!svg.contains(">layout<"));
    }

    #[test]
    fn no_offset_column() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        // The leftmost column of large offset numbers is gone.
        assert!(!svg.contains(r#"class="offset""#));
    }

    #[test]
    fn word_tint_overlay_appears() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(
            svg.contains("word-tint"),
            "expected word-tint overlay class for word boundaries: {svg}"
        );
    }

    #[test]
    fn word_tint_only_on_byte_band() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        // 6 rows × 4 words/row = 24 words; half get tinted = 12.
        let count = svg.matches(r#"class="word-tint""#).count();
        assert_eq!(count, 12, "expected 12 word-tint overlays, got {count}");
    }

    #[test]
    fn byte_indices_rendered_in_every_cell() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        // 96 bytes -> 96 byte-index text elements, one per cell.
        let count = svg.matches(r#"class="byte-index""#).count();
        assert_eq!(count, 96, "expected 96 byte-index labels, got {count}");
    }

    #[test]
    fn byte_indices_are_absolute() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(svg.contains(">16<"), "expected absolute offset 16 in svg");
        assert!(svg.contains(">80<"), "expected absolute offset 80 in svg");
        assert!(svg.contains(">95<"), "expected absolute offset 95 in svg");
    }

    #[test]
    fn pad_label_cell_uses_light_grey_fill() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(
            svg.contains("class=\"pad-label-cell\""),
            "expected pad-label-cell rects in label band: {svg}"
        );
    }

    #[test]
    fn no_per_byte_scalar_label() {
        let cfg = DiagramConfig::default();
        let svg = render(&fields_three(), 96, 16, 16, "S", &meta_for(96, 16), &cfg);
        assert!(!svg.contains("f32-"));
        assert!(!svg.contains("f16-"));
        assert!(!svg.contains("i32-"));
        assert!(!svg.contains("u32-"));
    }

    #[test]
    fn short_type_name_strips_module_path() {
        assert_eq!(short_type_name("my_crate::foo::Uniforms"), "Uniforms");
        assert_eq!(short_type_name("Uniforms"), "Uniforms");
        assert_eq!(short_type_name(""), "");
    }

    #[test]
    fn hue_is_stable() {
        assert_eq!(hue_for("view"), hue_for("view"));
    }
}
