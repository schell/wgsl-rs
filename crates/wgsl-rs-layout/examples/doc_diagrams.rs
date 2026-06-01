//! End-to-end demo: generate SVG diagrams for a few fixture types
//! and write them to a directory `rustdoc` can pick up via
//! `--resource-files`.
//!
//! Run with:
//!
//! ```text
//! cargo run -p wgsl-rs-layout --features doc-diagrams --example doc_diagrams -- \
//!     path/to/output_dir
//! ```
//!
//! Then `cargo doc` with
//! `RUSTDOCFLAGS="--resource-files path/to/output_dir"`.

#[cfg(feature = "doc-diagrams")]
use std::path::PathBuf;

#[cfg(feature = "doc-diagrams")]
use wgsl_rs::std::{Mat4x4f, RuntimeArray, Vec3f, Vec3u, Vec4f};
#[cfg(feature = "doc-diagrams")]
use wgsl_rs_layout::diagrams::{DiagramConfig, generate_svg};
#[cfg(feature = "doc-diagrams")]
use wgsl_rs_layout::{Layout, WgslLayout};

/// A typical uniform buffer.
#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
struct Uniforms {
    view: Mat4x4f,
    color: Vec4f,
    time: f32,
}

/// A tight struct where field alignment drives the layout.
#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
struct Tight {
    velocity: Vec3f,
    frame_count: u32,
    acceleration: Vec3f,
    delta: f32,
}


#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
/// An example from
/// <https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html>.
struct Ex4a {
    velocity: Vec3f,
}

#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
/// An example from
/// <https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html>.
struct Ex4 {
    orientation: Vec3f,
    size: f32,
    direction: [Vec3u; 1],
    scale: f32,
    info: Ex4a,
    friction: f32,
}

/// A small struct for showcasing per-byte offset labels.
#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
struct Small {
    a: f32,
    b: u32,
    c: f32,
}

/// A struct ending in a runtime array.
#[cfg(feature = "doc-diagrams")]
#[derive(Layout)]
#[allow(dead_code)]
struct MyStorageType {
    descriptor: Vec4f,
    count: u32,
    data: RuntimeArray<Ex4>,
}

#[cfg(feature = "doc-diagrams")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .ok_or("missing output directory argument")?
        .parse()?;
    let svg_dir = out_dir.join("wgsldiagrams");
    std::fs::create_dir_all(&svg_dir)?;

    let cfg = DiagramConfig::default();

    for (name, svg) in [
        ("Uniforms", generate_svg::<Uniforms>(&cfg)?),
        ("Tight", generate_svg::<Tight>(&cfg)?),
        ("Ex4", generate_svg::<Ex4>(&cfg)?),
        ("Small", generate_svg::<Small>(&cfg)?),
        ("MyStorageType", generate_svg::<MyStorageType>(&cfg)?),
    ] {
        let dest = svg_dir.join(format!("{name}.svg"));
        std::fs::write(&dest, svg)?;
        eprintln!("wrote {}", dest.display());
    }
    Ok(())
}

#[cfg(not(feature = "doc-diagrams"))]
fn main() {
    eprintln!(
        "this example requires the `doc-diagrams` feature:\n  cargo run -p wgsl-rs-layout \
         --features doc-diagrams --example doc_diagrams -- \\\n  <output_dir>"
    );
    std::process::exit(1);
}
