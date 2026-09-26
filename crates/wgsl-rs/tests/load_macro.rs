//! Tests for the `load!` macro (wgsl-rs#86, read-side of wgsl-rs#153).
//!
//! `load!` copies a module variable's value into a local on the Rust side and
//! transpiles to the bare variable name in WGSL — no deref, no identity
//! constructors.

use wgsl_rs::{
    std::{Vec2f, Vec3f},
    wgsl,
};

#[wgsl]
mod load_macro {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U_TIME: f32);
    uniform!(group(0), binding(1), U_RESOLUTION: Vec2f);
    storage!(group(0), binding(3), read_write, DATA: Vec3f);

    /// `load!` results work directly in arithmetic (issue #86's friction:
    /// `get!` returns a guard, `load!` returns a value).
    pub fn scaled_st(st: Vec2f) -> Vec2f {
        st / load!(U_RESOLUTION) * 3.0
    }

    /// Scalar uniform reads without the `f32(get!(...))` identity wrapper.
    pub fn offset_time() -> f32 {
        load!(U_TIME) + 1.0
    }

    /// Whole-value storage read (read_write access).
    pub fn read_data() -> Vec3f {
        load!(DATA)
    }
}

// `FRAME: impl Convert<f32>` makes this a template module; the
// `validate_with_instantiation_types` attribute auto-generates a WGSL
// validation test for the f32 instantiation. Compiling this module also
// exercises `load!(VAR, T)` constraint collection in the builder.
#[wgsl(validate_with_instantiation_types(f32))]
mod load_generic {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(2), FRAME: impl Convert<f32>);

    /// Generic form: `load!(VAR, T)` mirrors `get!(VAR, T)`. `T` must be
    /// `Copy` — `load!` copies the value out of the guard.
    pub fn frame_as<T: Wgsl + Convert<f32> + Copy>() -> f32 {
        f32(load!(FRAME, T))
    }
}

#[test]
fn load_transpiles_to_bare_variable_names() {
    let src = load_macro::WGSL_SOURCE.wgsl_source().unwrap();

    // `load!(VAR)` renders as the bare variable reference.
    assert!(src.contains("U_RESOLUTION"), "missing U_RESOLUTION:\n{src}");
    assert!(src.contains("U_TIME"), "missing U_TIME:\n{src}");
    assert!(src.contains("DATA"), "missing DATA:\n{src}");

    // No deref or identity constructors may leak into the WGSL.
    assert!(
        !src.contains("*U_") && !src.contains("*DATA"),
        "deref leaked into WGSL:\n{src}"
    );
    assert!(
        !src.contains("f32(U_TIME)") && !src.contains("vec2f(U_RESOLUTION.x, U_RESOLUTION.y)"),
        "identity constructor leaked into WGSL:\n{src}"
    );
}

#[test]
fn load_has_value_semantics_on_the_cpu() {
    load_macro::U_TIME.set(2.0_f32);
    load_macro::U_RESOLUTION.set(Vec2f::vec2(4.0, 2.0));
    load_macro::DATA.set(Vec3f::vec3(1.0, 2.0, 3.0));

    assert_eq!(load_macro::offset_time(), 3.0_f32);

    let st = Vec2f::vec2(8.0, 2.0);
    let scaled = load_macro::scaled_st(st);
    assert_eq!(scaled.x(), 6.0_f32);
    assert_eq!(scaled.y(), 3.0_f32);

    assert_eq!(load_macro::read_data(), Vec3f::vec3(1.0, 2.0, 3.0));
}

#[test]
fn load_generic_form_has_value_semantics_on_the_cpu() {
    load_generic::FRAME.set_typed(5.0_f32);
    assert_eq!(load_generic::frame_as::<f32>(), 5.0_f32);
}
