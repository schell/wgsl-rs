# SVG Byte-Layout Diagrams

Behind the `doc-diagrams` cargo feature, `wgsl-rs-layout` can generate self-contained SVG diagrams visualizing a type's byte layout.

## Enabling the Feature

```toml
[dependencies]
wgsl-rs-layout = { version = "0.1", features = ["doc-diagrams"] }
```

## `generate_svg`

```rust
use wgsl_rs_layout::diagram::{generate_svg, DiagramConfig};

let svg: String = generate_svg::<Particle>(&DiagramConfig::default());
std::fs::write("particle_layout.svg", svg).unwrap();
```

The returned string is a complete SVG document — no external assets, CSS, or fonts required.

## Style

The diagram style mirrors [webgpufundamentals.org](https://webgpufundamentals.org): each field is a labeled box sized to its `size`, padding cells are shaded, and rows are laid out left-to-right.

**Row width** is `T::ALIGN` bytes. This keeps each row exactly one alignment unit wide, so padding to alignment is visually obvious as a partial row.

All dimensions are sourced from the `WgslLayout` and `Layout` trait constants — the diagrams are a pure rendering of the same data exposed by `FIELDS`.

## Wiring into `cargo doc`

To embed diagrams in rustdoc, generate the SVG files and place them in a directory passed via `--resource-files`:

```sh
cargo doc --resource-files --resources-path ./doc-resources
```

Reference the image from doc-comments using relative paths:

```rust
/// # Layout
///
/// ![Particle layout](particle_layout.svg)
pub struct Particle { /* ... */ }
```