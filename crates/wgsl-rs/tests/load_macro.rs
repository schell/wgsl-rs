//! Tests for the `load!` macro (wgsl-rs#86, read-side of wgsl-rs#153).
//!
//! `load!` copies a pack field's value into a local on the Rust side and
//! transpiles to the bare mangled variable in WGSL — no deref, no identity
//! constructors.

use wgsl_rs::{
    std::{Storage, Uniform, Vec2f, Vec3f},
    wgsl,
};

#[wgsl]
mod load_macro {
    use wgsl_rs::std::*;

    pub struct LoadPack {
        #[uniform(group(0), binding(0))]
        pub u_time: Uniform<f32>,
        #[uniform(group(0), binding(1))]
        pub u_resolution: Uniform<Vec2f>,
        #[storage(group(0), binding(3), read_write)]
        pub data: Storage<Vec3f>,
    }

    /// `load!` results work directly in arithmetic (issue #86's friction:
    /// `get!` returns a guard, `load!` returns a value).
    pub fn scaled_st(pack: &LoadPack, st: Vec2f) -> Vec2f {
        st / load!(pack.u_resolution) * 3.0
    }

    /// Scalar uniform reads without the `f32(get!(...))` identity wrapper.
    pub fn offset_time(pack: &LoadPack) -> f32 {
        load!(pack.u_time) + 1.0
    }

    /// Whole-value storage read (read_write access).
    pub fn read_data(pack: &LoadPack) -> Vec3f {
        load!(pack.data)
    }
}

// A generic pack field makes this a template module; the
// `validate_with_instantiation_types` attribute auto-generates a WGSL
// validation test for the f32 instantiation. Compiling this module also
// exercises the type-generic pack slots in the builder.
#[wgsl(validate_with_instantiation_types(f32))]
mod load_generic {
    use wgsl_rs::std::*;

    pub struct FramePack<T> {
        #[uniform(group(0), binding(2))]
        pub frame: Uniform<T>,
    }

    /// Generic read: the field type is concrete per instantiation, so no
    /// `, T` form exists. `T` must be `Copy` — `load!` copies the value
    /// out of the guard.
    pub fn frame_as<T: Wgsl + Convert<f32> + Copy>(pack: &FramePack<T>) -> f32 {
        f32(load!(pack.frame))
    }
}

#[test]
fn load_transpiles_to_bare_variable_names() {
    let src = load_macro::WGSL_SOURCE.wgsl_source().unwrap();

    // `load!(pack.field)` renders as the bare mangled variable reference.
    assert!(
        src.contains("loadpack__u_resolution"),
        "missing loadpack__u_resolution:\n{src}"
    );
    assert!(
        src.contains("loadpack__u_time"),
        "missing loadpack__u_time:\n{src}"
    );
    assert!(
        src.contains("loadpack__data"),
        "missing loadpack__data:\n{src}"
    );

    // No deref or identity constructors may leak into the WGSL.
    assert!(
        !src.contains("*loadpack__") && !src.contains("*loadpack__data"),
        "deref leaked into WGSL:\n{src}"
    );
    assert!(
        !src.contains("f32(loadpack__u_time)")
            && !src.contains("vec2f(loadpack__u_resolution.x, loadpack__u_resolution.y)"),
        "identity constructor leaked into WGSL:\n{src}"
    );
}

#[test]
fn load_has_value_semantics_on_the_cpu() {
    let pack = load_macro::LoadPack {
        u_time: Uniform::new(2.0_f32),
        u_resolution: Uniform::new(Vec2f::vec2(4.0, 2.0)),
        data: Storage::new(Vec3f::vec3(1.0, 2.0, 3.0)),
    };

    assert_eq!(load_macro::offset_time(&pack), 3.0_f32);

    let st = Vec2f::vec2(8.0, 2.0);
    let scaled = load_macro::scaled_st(&pack, st);
    assert_eq!(scaled.x(), 6.0_f32);
    assert_eq!(scaled.y(), 3.0_f32);

    assert_eq!(load_macro::read_data(&pack), Vec3f::vec3(1.0, 2.0, 3.0));
}

#[test]
fn load_generic_form_has_value_semantics_on_the_cpu() {
    let pack = load_generic::FramePack::<f32> {
        frame: Uniform::new(5.0_f32),
    };
    assert_eq!(load_generic::frame_as::<f32>(&pack), 5.0_f32);
}
