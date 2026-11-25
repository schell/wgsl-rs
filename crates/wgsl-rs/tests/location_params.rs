//! Regression tests for IO attributes on entry-point function parameters.
//!
//! Covers wgsl-rs#259: location attributes must be stripped from Rust-world
//! emission, and valid combinations such as location plus interpolation must
//! survive parsing into the WGSL IR.

#![allow(dead_code)]

use wgsl_rs::{
    std::{vec2f, vec3f, vec4f},
    wgsl,
};

#[wgsl]
mod fragment_location {
    use wgsl_rs::std::*;

    #[fragment]
    pub fn frag_main(#[location(0)] pos: Vec3f) -> Vec4f {
        vec4f(pos.x, pos.y, pos.z, 1.0)
    }
}

#[wgsl]
mod vertex_locations {
    use wgsl_rs::std::*;

    #[vertex]
    pub fn vtx_main(
        #[location(0)] position: Vec3f,
        #[location(1)] uv: Vec2f,
        #[builtin(vertex_index)] vertex_index: u32,
    ) -> Vec4f {
        vec4f(
            position.x + uv.x,
            position.y + uv.y,
            position.z + f32(vertex_index),
            1.0,
        )
    }
}

#[wgsl]
mod combined_location_interpolation {
    use wgsl_rs::std::*;

    #[fragment]
    pub fn frag_main(
        #[location(0)]
        #[interpolate(flat)]
        material: u32,
    ) -> Vec4f {
        vec4f(f32(material), 0.0, 0.0, 1.0)
    }
}

#[test]
fn fragment_location_param_compiles_renders_and_runs_on_cpu() {
    let output = fragment_location::frag_main(vec3f(0.25, 0.5, 0.75));
    assert_eq!(output, vec4f(0.25, 0.5, 0.75, 1.0));

    let src = fragment_location::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("@location(0) pos: vec3f"),
        "expected a direct fragment location parameter, got: {src}"
    );
}

#[test]
fn vertex_location_params_render_alongside_a_builtin() {
    let output = vertex_locations::vtx_main(vec3f(1.0, 2.0, 3.0), vec2f(4.0, 5.0), 6);
    assert_eq!(output, vec4f(5.0, 7.0, 9.0, 1.0));

    let src = vertex_locations::WGSL_SOURCE.wgsl_source().unwrap();
    for expected in [
        "@location(0) position: vec3f",
        "@location(1) uv: vec2f",
        "@builtin(vertex_index) vertex_index: u32",
    ] {
        assert!(
            src.contains(expected),
            "expected `{expected}` in WGSL, got: {src}"
        );
    }
}

#[test]
fn location_and_interpolation_combine_on_one_param() {
    let output = combined_location_interpolation::frag_main(7);
    assert_eq!(output, vec4f(7.0, 0.0, 0.0, 1.0));

    let src = combined_location_interpolation::WGSL_SOURCE
        .wgsl_source()
        .unwrap();
    assert!(
        src.contains("@location(0) @interpolate(flat) material: u32"),
        "expected both IO attributes on the parameter, got: {src}"
    );
}
