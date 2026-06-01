//! Byte-layout diagram generation for `cargo doc`.
//!
//! Enable with the `doc-diagrams` cargo feature to use this module.
//!
//! # Overview
//!
//! Given any type `T` that implements both [`Layout`](crate::Layout)
//! and [`WgslLayout`](crate::WgslLayout) (i.e. any type annotated
//! with `#[derive(Layout)]` and every built-in WGSL type),
//! [`generate_svg`] produces a self-contained SVG byte-layout
//! diagram. The visual style mirrors
//! <https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html>:
//! one label band + one byte band, repeated for every row, with
//! per-field HSL coloring.
//!
//! The row width is derived from the type: each row spans
//! `T::ALIGN` bytes. A struct with `align = 4` renders in 4-byte
//! rows, a struct with `align = 16` renders in 16-byte rows,
//! matching the WGSL layout's natural word size.
//!
//! All layout information is sourced directly from the trait
//! constants — `<T as WgslLayout>::SIZE` / `::ALIGN` and
//! `<T as Layout>::FIELDS` — so the rendered diagram is always
//! consistent with what the `#[derive(Layout)]` macro emits.
//!
//! # Example
//!
//! ```no_run
//! use wgsl_rs_layout::{
//!     Layout,
//!     diagrams::{DiagramConfig, generate_svg},
//! };
//!
//! #[derive(Layout)]
//! struct Uniforms {/* ... */}
//!
//! fn generate_one() -> Result<(), Box<dyn std::error::Error>> {
//!     let svg = generate_svg::<Uniforms>(&DiagramConfig::default())?;
//!     std::fs::write("target/doc-resources/wgsldiagrams/Uniforms.svg", svg)?;
//!     Ok(())
//! }
//! ```
//!
//! # Wiring into `cargo doc`
//!
//! 1. In your crate, add the `doc-diagrams` feature to `wgsl-rs-layout`.
//!
//! 2. Write a small driver (a binary, a `build.rs`, an `xtask` step, or just a
//!    hand-run command) that calls [`generate_svg::<MyType>()`](generate_svg)
//!    for every type you want to diagram and writes the SVGs into a directory
//!    `rustdoc` can find via `--resource-files`.
//!
//! 3. In `.cargo/config.toml`:
//!
//!    ```toml
//!    [build]
//!    rustdocflags = ["--resource-files", "target/doc-resources/"]
//!    ```
//!
//! 4. In your doc comments, reference the generated SVG by relative path:
//!
//!    ```ignore
//!    /// Camera state.
//!    ///
//!    /// ![byte layout](wgsldiagrams/Uniforms.svg)
//!    pub struct Uniforms { ... }
//!    ```

mod svg;

use snafu::prelude::*;

use crate::{Layout, WgslLayout};

/// Tuning knobs for [`generate_svg`].
///
/// Build a config via [`DiagramConfig::builder`] so values are
/// validated up front. The [`DiagramConfig::default`] constructor
/// returns a known-good config.
#[derive(Debug, Clone)]
pub struct DiagramConfig {
    /// Width and height of a single byte cell, in pixels.
    cell_size: f32,
}

impl Default for DiagramConfig {
    fn default() -> Self {
        Self::builder()
            .cell_size(22.0)
            .build()
            .expect("default DiagramConfig is valid")
    }
}

/// Builder for [`DiagramConfig`].
///
/// Validates inputs on [`DiagramConfigBuilder::build`]: `cell_size`
/// must be positive and finite.
#[derive(Debug, Clone)]
pub struct DiagramConfigBuilder {
    cell_size: Option<f32>,
}

impl DiagramConfigBuilder {
    /// Set the cell size, in pixels. Must be a positive, finite
    /// `f32` (e.g. `22.0`). Returns the builder for chaining.
    pub fn cell_size(mut self, value: f32) -> Self {
        self.cell_size = Some(value);
        self
    }

    /// Build a [`DiagramConfig`], validating all inputs.
    pub fn build(self) -> Result<DiagramConfig, DiagramError> {
        let cell_size = self.cell_size.unwrap_or(22.0);
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return InvalidConfigSnafu {
                reason: format!("cell_size must be a positive finite f32, got {cell_size}"),
            }
            .fail();
        }
        Ok(DiagramConfig { cell_size })
    }
}

impl DiagramConfig {
    /// Start a new builder with no fields set; defaults are applied
    /// on build.
    pub fn builder() -> DiagramConfigBuilder {
        DiagramConfigBuilder { cell_size: None }
    }

    /// Width and height of a single byte cell, in pixels.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }
}

/// Errors that abort diagram generation.
#[derive(Debug, Snafu)]
pub enum DiagramError {
    /// The diagram exceeds the renderer's maximum supported size.
    #[snafu(display("layout size {size} bytes exceeds the renderer max ({max} bytes)"))]
    SizeTooLarge {
        /// The size in bytes of the type being diagrammed.
        size: usize,
        /// The renderer maximum.
        max: usize,
    },
    /// A [`DiagramConfig`] value was invalid.
    #[snafu(display("invalid DiagramConfig: {reason}"))]
    InvalidConfig {
        /// Human-readable description of the validation failure.
        reason: String,
    },
}

/// Maximum byte size the renderer supports. Above this, diagrams
/// become unreadable, so [`generate_svg`] returns
/// [`DiagramError::SizeTooLarge`].
const MAX_BYTES: usize = 16 * 1024;

/// Number of example array elements rendered for a runtime array.
/// Chosen as 1 to keep the diagram compact: the meta line already
/// conveys that the actual size depends on `LEN`.
const RUNTIME_ARRAY_PREVIEW_COUNT: usize = 1;

/// Number of bytes per row in the rendered diagram. Equal to
/// `T::ALIGN`, so the diagram's natural word size scales with
/// the type's alignment.
const ROW_BYTES_PER_TYPE_ALIGN: usize = 1;

/// One field's layout, internally borrowed by the renderer.
struct RenderField {
    /// Field name as written in the source struct.
    name: String,
    /// Byte offset from the start of the struct.
    offset: usize,
    /// Byte size of the field's data.
    size: usize,
    /// Bytes of padding after this field. Currently unused by the
    /// renderer (padding is recomputed from offsets), but kept for
    /// future use and to mirror [`crate::FieldLayout`].
    #[allow(dead_code)]
    pad_after: usize,
}

/// Generate an SVG byte-layout diagram for `T`.
///
/// Reads `<T as WgslLayout>::SIZE` / `::ALIGN` and
/// `<T as Layout>::FIELDS` to build the diagram; no source parsing.
///
/// The row width is `T::ALIGN` bytes, so a 4-byte-aligned struct
/// renders in 4-byte rows and a 16-byte-aligned struct renders in
/// 16-byte rows.
///
/// If `T` has a runtime array field, the diagram shows the prefix
/// fields normally followed by a single example element of the array
/// (labeled `<name>[0]`). The actual total size is reported in the
/// meta line as `<prefix_offset> + LEN × <stride> bytes`.
///
/// # Errors
///
/// Returns [`DiagramError::SizeTooLarge`] if the rendered diagram's
/// total byte count (including the runtime-array preview element)
/// would exceed [`MAX_BYTES`].
pub fn generate_svg<T: WgslLayout + Layout>(
    config: &DiagramConfig,
) -> Result<String, DiagramError> {
    let prefix_fields: Vec<RenderField> = T::FIELDS
        .iter()
        .map(|f| RenderField {
            name: f.name.to_string(),
            offset: f.offset,
            size: f.size,
            pad_after: f.pad_after,
        })
        .collect();

    let mut fields = prefix_fields;
    let mut display_size = T::SIZE;
    let bytes_per_row = T::ALIGN * ROW_BYTES_PER_TYPE_ALIGN;
    let meta = if let Some(ra) = T::RUNTIME_ARRAY_FIELD {
        let stride =
            T::RUNTIME_ARRAY_STRIDE.expect("RUNTIME_ARRAY_FIELD implies RUNTIME_ARRAY_STRIDE");
        let name = ra.name;
        let start = ra.offset;
        for i in 0..RUNTIME_ARRAY_PREVIEW_COUNT {
            fields.push(RenderField {
                name: format!("{name}[{i}]"),
                offset: start + i * stride,
                size: stride,
                pad_after: 0,
            });
        }
        display_size = start + RUNTIME_ARRAY_PREVIEW_COUNT * stride;
        format!(
            "size: {prefix} + LEN \u{00d7} {stride} bytes \u{00b7} align: {align} bytes",
            prefix = start,
            stride = stride,
            align = T::ALIGN
        )
    } else {
        format!(
            "size: {size} bytes \u{00b7} align: {align} bytes",
            size = T::SIZE,
            align = T::ALIGN
        )
    };

    snafu::ensure!(
        display_size <= MAX_BYTES,
        SizeTooLargeSnafu {
            size: display_size,
            max: MAX_BYTES,
        }
    );

    Ok(svg::render(
        &fields,
        display_size,
        bytes_per_row,
        T::ALIGN,
        std::any::type_name::<T>(),
        &meta,
        config,
    ))
}
