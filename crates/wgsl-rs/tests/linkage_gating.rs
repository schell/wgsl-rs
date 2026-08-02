//! Regression test for wgsl-rs#72.
//!
//! Ensures that wgpu linkage is not produced when the `linkage-wgpu` feature
//! is disabled. The macro should emit only the CPU-side runtime API
//! (`Uniform::new`, `Storage::new`, `get!`, `get_mut!`) and no `wgpu::*`,
//! `linkage::wgpu::*`, `WgpuLinkage`, or bind-group references should appear
//! in the compiled output.
//!
//! This test is gated on `not(feature = "linkage-wgpu")` so it only runs in
//! the default/no-feature build. The companion `linkage_wgpu.rs` integration
//! test covers the feature-enabled path.

#![cfg(not(feature = "linkage-wgpu"))]
#![allow(dead_code)]

use wgsl_rs::wgsl;

// A module that exercises the `uniform!` and `storage!` macros — the paths
// that previously emitted compile-time wgpu linkage before PR #124.
#[wgsl]
mod linkage_gating {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: u32);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 16]);

    #[vertex]
    pub fn vs_main(#[builtin(vertex_index)] _vertex_index: u32) -> Vec4f {
        let pos = vec2f(0.0, 0.0);
        vec4f(pos.x, pos.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn fs_main() -> Vec4f {
        let frame = get!(FRAME);
        let mut output = get_mut!(OUTPUT);
        // Use both linkage variables to keep them live in both worlds.
        // The arithmetic is valid in Rust and in WGSL.
        output[0] = f32(frame) / 16.0;
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}

#[test]
fn module_renders_without_linkage_wgpu_feature() {
    // With `linkage-wgpu` off, the `#[wgsl]` module must still expand and
    // render to valid WGSL — the CPU-side runtime API (`uniform!`/`storage!`/
    // `get!`/`get_mut!`) is available without the feature, and no wgpu
    // linkage codegen should be triggered. This is a smoke test that the
    // module compiles and renders; the next test asserts the rendered
    // source contains no wgpu/linkage artifacts.
    let _ = linkage_gating::WGSL_SOURCE.wgsl_source().unwrap();
}

#[test]
fn source_renders_without_wgpu_references() {
    // The rendered WGSL must not contain any `wgpu` linkage artifacts —
    // only the shader source itself.
    let src = linkage_gating::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !src.contains("wgpu"),
        "rendered WGSL should not mention wgpu, got: {src}"
    );
    assert!(
        !src.contains("linkage"),
        "rendered WGSL should not mention linkage, got: {src}"
    );
    assert!(
        !src.contains("bind_group"),
        "rendered WGSL should not mention bind_group, got: {src}"
    );
}
