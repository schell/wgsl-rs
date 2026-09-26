//! Tests for module-variable deref handling (wgsl-rs#153).
//!
//! A `*` in front of a linkage accessor is a Rust-side guard artifact —
//! module variables are value references in WGSL, not pointers. The deref
//! is elided and the bare variable name is emitted, on both sides of an
//! assignment.

use wgsl_rs::{std::Vec3f, wgsl};

#[wgsl]
mod deref_elision {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec3f);
    storage!(group(0), binding(1), read_write, DATA: Vec3f);

    /// The wgsl-rs#153 repro: a deref of the read guard renders the bare
    /// variable name, not `*U`.
    pub fn read_u() -> Vec3f {
        *get!(U)
    }

    /// The #153 comment-thread repro: a deref on each side of an
    /// assignment renders as a plain value assignment.
    pub fn write_data() {
        *get_mut!(DATA) = load!(U);
    }

    /// Compound assignment through the guard also elides the deref.
    pub fn bump_data(v: Vec3f) {
        *get_mut!(DATA) = v;
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
}

#[test]
fn deref_guards_agree_on_the_cpu() {
    deref_elision::U.set(Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get!(U)` copies the value out of the guard.
    assert_eq!(deref_elision::read_u(), Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get_mut!(DATA) = load!(U)` writes through the write guard.
    deref_elision::DATA.set(Vec3f::vec3(0.0, 0.0, 0.0));
    deref_elision::write_data();
    assert_eq!(*deref_elision::DATA.get(), Vec3f::vec3(1.0, 2.0, 3.0));

    deref_elision::bump_data(Vec3f::vec3(4.0, 5.0, 6.0));
    assert_eq!(*deref_elision::DATA.get(), Vec3f::vec3(4.0, 5.0, 6.0));
}
