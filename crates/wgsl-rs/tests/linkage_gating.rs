//! Regression test for wgsl-rs#72.
//!
//! Ensures that a `#[wgsl]` module's rendered WGSL carries no wgpu
//! linkage artifacts. Since wgpu linkage moved out of this crate (into
//! the `wgsl-rs-linkage-wgpu` crates), `wgsl-rs` no longer even has a
//! `wgpu` dependency; this guards the CPU-side runtime API (packs,
//! `get!`, `get_mut!`) and the rendered source against any `wgpu`,
//! `linkage`, or bind-group references creeping back in.

#![allow(dead_code)]

use wgsl_rs::wgsl;

// A module that exercises resource packs and the access macros — the
// paths that previously emitted compile-time wgpu linkage before PR #124.
#[wgsl]
mod linkage_gating {
    use wgsl_rs::std::*;

    pub struct GatingPack {
        #[uniform(group(0), binding(0))]
        pub frame: Uniform<u32>,

        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[f32; 16]>,
    }

    #[vertex]
    pub fn vs_main(#[builtin(vertex_index)] _vertex_index: u32) -> Vec4f {
        let pos = vec2f(0.0, 0.0);
        vec4f(pos.x, pos.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn fs_main(pack: &GatingPack) -> Vec4f {
        let frame = load!(pack.frame);
        let mut output = get_mut!(pack.output);
        // Use both pack fields to keep them live in both worlds.
        // The arithmetic is valid in Rust and in WGSL.
        output[0] = f32(frame) / 16.0;
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}

#[test]
fn module_renders_without_linkage_crates() {
    // The `#[wgsl]` module must still expand and render to valid WGSL —
    // the CPU-side runtime API (packs, `get!`, `get_mut!`) is available
    // without any linkage crate, and no wgpu linkage codegen should be
    // triggered. This is a smoke test that the module compiles and
    // renders; the next test asserts the rendered source contains no
    // wgpu/linkage artifacts.
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
