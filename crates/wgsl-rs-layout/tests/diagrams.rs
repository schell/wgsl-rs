//! End-to-end tests for the `doc-diagrams` feature.

#![cfg(feature = "doc-diagrams")]

use wgsl_rs::std::{Mat4x4f, Vec3f, Vec4f};
use wgsl_rs_layout::{
    Layout, WgslLayout,
    diagrams::{DiagramConfig, DiagramError, generate_svg},
};

#[derive(Layout)]
struct Uniforms {
    view: Mat4x4f,
    color: Vec4f,
    time: f32,
}

#[derive(Layout)]
struct Tight {
    velocity: Vec3f,
    acceleration: Vec3f,
    frame_count: u32,
}

#[test]
fn generates_diagram_for_concrete_struct() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();

    assert!(svg.contains("wgsl-rs-byte-diagram"));
    assert!(svg.contains(".view"));
    assert!(svg.contains(".color"));
    assert!(svg.contains(".time"));
    assert!(svg.contains("-pad-"));
    // 96 bytes total; one byte-index label per cell.
    let count = svg.matches(r#"class="byte-index""#).count();
    assert_eq!(count, 96);
}

#[test]
fn custom_cell_size_changes_layout() {
    // Verifies that the cell_size knob still affects the rendered
    // SVG (it was the only user-configurable knob in
    // `DiagramConfig`).
    let cfg_small = DiagramConfig::builder().cell_size(10.0).build().unwrap();
    let cfg_big = DiagramConfig::builder().cell_size(40.0).build().unwrap();
    let small = generate_svg::<Tight>(&cfg_small).unwrap();
    let big = generate_svg::<Tight>(&cfg_big).unwrap();
    assert_ne!(small, big);
    assert!(small.contains("width=\"10"));
    assert!(big.contains("width=\"40"));
}

#[test]
fn respects_config_cell_size() {
    let cfg_a = DiagramConfig::builder().cell_size(20.0).build().unwrap();
    let cfg_b = DiagramConfig::builder().cell_size(40.0).build().unwrap();
    let a = generate_svg::<Uniforms>(&cfg_a).unwrap();
    let b = generate_svg::<Uniforms>(&cfg_b).unwrap();
    assert_ne!(a, b);
    assert!(a.contains("width=\"20"));
    assert!(b.contains("width=\"40"));
}

#[test]
fn diagram_config_builder_rejects_invalid_cell_size() {
    // Zero.
    let err = DiagramConfig::builder().cell_size(0.0).build().unwrap_err();
    assert!(matches!(err, DiagramError::InvalidConfig { .. }));
    // Negative.
    let err = DiagramConfig::builder()
        .cell_size(-1.0)
        .build()
        .unwrap_err();
    assert!(matches!(err, DiagramError::InvalidConfig { .. }));
    // NaN.
    let err = DiagramConfig::builder()
        .cell_size(f32::NAN)
        .build()
        .unwrap_err();
    assert!(matches!(err, DiagramError::InvalidConfig { .. }));
    // Infinity.
    let err = DiagramConfig::builder()
        .cell_size(f32::INFINITY)
        .build()
        .unwrap_err();
    assert!(matches!(err, DiagramError::InvalidConfig { .. }));
    // Valid default still works.
    let _cfg = DiagramConfig::builder().cell_size(22.0).build().unwrap();
}

#[test]
fn row_width_scales_with_alignment() {
    // Render an align=4-only struct and an align=16 struct. The
    // align=16 diagram should be wider than the align=4 diagram
    // because each row holds more bytes. Both diagrams may widen
    // beyond the body width to fit the header text; we just check
    // that the align=16 case is wider.
    #[derive(Layout)]
    #[allow(dead_code)]
    struct Align4 {
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    }

    let cfg = DiagramConfig::default();
    let svg4 = generate_svg::<Align4>(&cfg).unwrap();
    let svg16 = generate_svg::<Uniforms>(&cfg).unwrap();

    // Both diagrams should render without error and have a viewBox.
    assert!(svg4.contains("viewBox="));
    assert!(svg16.contains("viewBox="));

    // The align=16 diagram is wider than align=4 because each row
    // holds more bytes.
    let vb4 = viewbox_width(&svg4);
    let vb16 = viewbox_width(&svg16);
    assert!(
        vb16 > vb4,
        "align=16 diagram ({vb16}px) should be wider than align=4 ({vb4}px)"
    );
}

fn viewbox_width(svg: &str) -> f32 {
    let start = svg.find("viewBox=\"").unwrap() + "viewBox=\"".len();
    let end = svg[start..].find('"').unwrap() + start;
    let parts: Vec<&str> = svg[start..end].split_whitespace().collect();
    let w: f32 = parts[2].parse().unwrap();
    w
}

#[test]
fn fields_appear_in_declaration_order() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    let view_pos = svg.find(".view").expect(".view in svg");
    let color_pos = svg.find(".color").expect(".color in svg");
    let time_pos = svg.find(".time").expect(".time in svg");
    assert!(view_pos < color_pos);
    assert!(color_pos < time_pos);
}

#[test]
fn diagram_data_struct_attribute_contains_type_name() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    let attr_start = svg.find("data-struct=\"").unwrap() + "data-struct=\"".len();
    let attr_end = svg[attr_start..].find('"').unwrap() + attr_start;
    let attr = &svg[attr_start..attr_end];
    assert!(attr.contains("Uniforms"), "data-struct value was {attr:?}");
}

#[test]
fn header_renders_struct_name() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    assert!(svg.contains("class=\"struct-name\""));
    let name_start = svg.find("class=\"struct-name\"").unwrap();
    let after = &svg[name_start..];
    let close = after.find("</text>").unwrap();
    assert!(after[..close].contains("Uniforms"));
}

#[test]
fn header_renders_size_and_align() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    assert!(svg.contains("size: 96 bytes"));
    assert!(svg.contains("align: 16 bytes"));
}

#[test]
fn header_name_and_meta_on_same_line() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    // Exactly one <text class="struct-name"> element.
    let open_count = svg.matches(r#"<text class="struct-name""#).count();
    assert_eq!(
        open_count, 1,
        "expected one <text class=\"struct-name\">, got {open_count}"
    );
    // Both name and meta live inside that single <text> element.
    let header_start = svg.find(r#"<text class="struct-name""#).unwrap();
    let header_end = svg[header_start..].find("</text>").unwrap() + header_start;
    let header = &svg[header_start..=header_end];
    assert!(header.contains("Uniforms"));
    assert!(header.contains("size: 96 bytes"));
    assert!(header.contains(r#"<tspan class="meta""#));
    // No separate <text class="meta"> element.
    let meta_text_count = svg.matches(r#"<text class="meta""#).count();
    assert_eq!(meta_text_count, 0);
}

#[test]
fn narrow_row_wraps_header_to_two_lines() {
    // For an align=4 struct (4-byte rows), the body is only 88px
    // wide but the header text is ~250px. The header should wrap
    // to two lines: one <text class="struct-name"> for the title
    // and one <text class="meta"> for the meta line.
    #[derive(Layout)]
    #[allow(dead_code)]
    struct SmallAlign4 {
        a: f32,
        b: f32,
        c: f32,
    }
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<SmallAlign4>(&cfg).unwrap();

    // Two separate <text> elements: the title and the meta.
    let title_count = svg.matches(r#"<text class="struct-name""#).count();
    let meta_count = svg.matches(r#"<text class="meta""#).count();
    assert_eq!(title_count, 1, "expected one <text class=\"struct-name\">");
    assert_eq!(meta_count, 1, "expected one <text class=\"meta\">");

    // The header is wider than the body would be alone.
    let vb = viewbox_width(&svg) as f32;
    let body_only = 4.0 * 22.0 + 0.0 + 22.0 / 3.0; // 4-byte row + padding
    assert!(
        vb > body_only,
        "viewBox {vb} should be wider than body {body_only} after header wrap"
    );

    // Both texts contain the expected text.
    assert!(svg.contains(">SmallAlign4<"));
    assert!(svg.contains("size: 12 bytes"));
    assert!(svg.contains("align: 4 bytes"));
}

#[test]
fn no_column_header_strip() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    // The "offset" / "layout" column-header strip is removed; the
    // per-cell byte indices make the row offset redundant.
    assert!(!svg.contains("col-header"));
    assert!(!svg.contains(">offset<"));
    assert!(!svg.contains(">layout<"));
}

#[test]
fn no_offset_column() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    // The leftmost column of large offset numbers is gone.
    assert!(!svg.contains(r#"class="offset""#));
}

#[test]
fn word_tint_overlay_present() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    let count = svg.matches(r#"class="word-tint""#).count();
    assert!(count > 0, "expected at least one word-tint overlay: {svg}");
}

#[test]
fn word_tint_only_on_byte_band() {
    // 6 rows × 4 words/row = 24 words; half get tinted = 12.
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    let count = svg.matches(r#"class="word-tint""#).count();
    assert_eq!(count, 12, "expected 12 word-tint overlays, got {count}");
}

#[test]
fn byte_indices_are_absolute() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    // Row 1 starts at absolute offset 16, row 5 at 80.
    assert!(svg.contains(">16<"), "expected absolute offset 16 in svg");
    assert!(svg.contains(">80<"), "expected absolute offset 80 in svg");
    assert!(svg.contains(">95<"), "expected absolute offset 95 in svg");
}

#[test]
fn pad_label_cell_uses_light_grey_fill() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    assert!(
        svg.contains("class=\"pad-label-cell\""),
        "expected pad-label-cell rects in label band: {svg}"
    );
}

#[test]
fn no_per_byte_scalar_label() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<Uniforms>(&cfg).unwrap();
    assert!(!svg.contains("f32-"));
    assert!(!svg.contains("f16-"));
    assert!(!svg.contains("i32-"));
    assert!(!svg.contains("u32-"));
}

// ===== Runtime-array tests =====

use wgsl_rs::std::RuntimeArray;

#[derive(Layout)]
struct Ex4ForRuntime {
    velocity: wgsl_rs::std::Vec3f,
}

#[derive(Layout)]
#[allow(dead_code)]
struct WithRuntimeArray {
    descriptor: wgsl_rs::std::Vec4f,
    count: u32,
    data: RuntimeArray<Ex4ForRuntime>,
}

#[test]
fn runtime_array_meta_uses_len_placeholder() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<WithRuntimeArray>(&cfg).unwrap();
    // The meta line should indicate that the size depends on LEN.
    assert!(
        svg.contains("LEN"),
        "expected LEN placeholder in meta line: {svg}"
    );
    assert!(
        svg.contains("align: 16 bytes"),
        "expected align in meta line: {svg}"
    );
}

#[test]
fn runtime_array_shows_one_example() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<WithRuntimeArray>(&cfg).unwrap();
    // Exactly one example element is rendered, labeled `data[0]`.
    assert!(svg.contains("data[0]"), "missing data[0] in svg");
    assert!(!svg.contains("data[1]"), "data[1] should not be rendered");
    assert!(!svg.contains("data[2]"), "data[2] should not be rendered");
    assert!(!svg.contains("..."), "ellipsis should not be rendered");
}

#[test]
fn runtime_array_prefix_fields_appear() {
    let cfg = DiagramConfig::default();
    let svg = generate_svg::<WithRuntimeArray>(&cfg).unwrap();
    assert!(svg.contains(".descriptor"));
    assert!(svg.contains(".count"));
}
