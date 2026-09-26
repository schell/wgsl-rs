//! Tests for module-variable deref handling (wgsl-rs#153).
//!
//! A `*` in front of a linkage accessor is a Rust-side guard artifact —
//! module variables are value references in WGSL, not pointers. The deref
//! is elided and the bare variable name is emitted, on both sides of an
//! assignment. Compound assignment, parenthesized operands, and field
//! writes through the guard all elide the same way.

use wgsl_rs::{std::Vec3f, wgsl};

// The explicit deref in `set_pos_x` is the point of the test (it exercises
// the field-write elision path), so silence clippy's auto-deref suggestion.
#[wgsl]
#[allow(clippy::explicit_auto_deref)]
mod deref_elision {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec3f);
    storage!(group(0), binding(1), read_write, DATA: Vec3f);
    storage!(group(0), binding(2), read_write, SCALE: f32);

    #[derive(Wgsl)]
    pub struct Pos {
        pub x: f32,
    }

    storage!(group(0), binding(3), read_write, POS: Pos);

    /// The wgsl-rs#153 repro: a deref of the read guard renders the bare
    /// variable name, not `*U`.
    pub fn read_u() -> Vec3f {
        *get!(U)
    }

    /// Parenthesized operands take the same elision path.
    pub fn paren_read() -> Vec3f {
        *(get!(U))
    }

    /// The #153 comment-thread repro: a deref on each side of an
    /// assignment renders as a plain value assignment.
    pub fn write_data() {
        *get_mut!(DATA) = load!(U);
    }

    /// Compound assignment through the guard also elides the deref:
    /// WGSL `SCALE += v`.
    pub fn bump_scale(v: f32) {
        *get_mut!(SCALE) += v;
    }

    /// Field writes through the guard: WGSL `POS.x = v`.
    pub fn set_pos_x(v: f32) {
        (*get_mut!(POS)).x = v;
    }

    /// Regression: a pointer parameter shadowed by a branch-local value
    /// binding must not poison the parameter's deref. The deref below
    /// targets the `ptr!` parameter — valid WGSL — even though `u` is
    /// bound from `load!` inside the branch.
    pub fn shadow_param(c: bool, u: ptr!(function, f32)) -> f32 {
        if c {
            let u = load!(U);
            let _unused = u;
        }
        *u
    }
}

#[test]
fn deref_of_linkage_access_renders_bare_names() {
    let src = deref_elision::WGSL_SOURCE.wgsl_source().unwrap();

    assert!(!src.contains("*U"), "deref leaked into WGSL:\n{src}");
    assert!(!src.contains("*DATA"), "deref leaked into WGSL:\n{src}");
    assert!(
        src.contains("DATA = U;"),
        "missing value assignment:\n{src}"
    );
    assert!(
        src.contains("SCALE += v;"),
        "missing compound assignment:\n{src}"
    );
    assert!(src.contains("(POS).x = v;"), "missing field write:\n{src}");
    assert!(
        src.contains("*u"),
        "the pointer-parameter deref must be preserved:\n{src}"
    );
}

#[test]
fn deref_guards_agree_on_the_cpu() {
    deref_elision::U.set(Vec3f::vec3(1.0, 2.0, 3.0));
    deref_elision::DATA.set(Vec3f::vec3(0.0, 0.0, 0.0));
    deref_elision::SCALE.set(1.0_f32);
    deref_elision::POS.set(deref_elision::Pos { x: 0.0 });

    // `*get!(U)` copies the value out of the guard; parens elide the same.
    assert_eq!(deref_elision::read_u(), Vec3f::vec3(1.0, 2.0, 3.0));
    assert_eq!(deref_elision::paren_read(), Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get_mut!(DATA) = load!(U)` writes through the write guard.
    deref_elision::write_data();
    assert_eq!(*deref_elision::DATA.get(), Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get_mut!(SCALE) += v` compounds through the write guard.
    deref_elision::bump_scale(0.5);
    assert_eq!(*deref_elision::SCALE.get(), 1.5_f32);

    // `(*get_mut!(POS)).x = v` writes the field through the guard.
    deref_elision::set_pos_x(7.0);
    assert_eq!(deref_elision::POS.get().x, 7.0_f32);

    // The pointer parameter deref works regardless of the shadowing
    // branch.
    let mut p = 5.0_f32;
    assert_eq!(deref_elision::shadow_param(false, &mut p), 5.0_f32);
    assert_eq!(deref_elision::shadow_param(true, &mut p), 5.0_f32);
}
