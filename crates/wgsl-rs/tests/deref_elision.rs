//! Tests for pack-field deref handling (wgsl-rs#153).
//!
//! A `*` in front of an access-macro target is a Rust-side guard artifact —
//! pack fields are value references in WGSL, not pointers. The deref
//! is elided and the bare mangled name is emitted, on both sides of an
//! assignment. Compound assignment, parenthesized operands, and field
//! writes through the guard all elide the same way.

use wgsl_rs::{std::Vec3f, wgsl};

// The explicit deref in `set_pos_x` is the point of the test (it exercises
// the field-write elision path), so silence clippy's auto-deref suggestion.
#[wgsl]
#[allow(clippy::explicit_auto_deref)]
mod deref_elision {
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Pos {
        pub x: f32,
    }

    pub struct DerefPack {
        #[uniform(group(0), binding(0))]
        pub u: Uniform<Vec3f>,
        #[storage(group(0), binding(1), read_write)]
        pub data: Storage<Vec3f>,
        #[storage(group(0), binding(2), read_write)]
        pub scale: Storage<f32>,
        #[storage(group(0), binding(3), read_write)]
        pub pos: Storage<Pos>,
    }

    /// The wgsl-rs#153 repro: a deref of the read guard renders the bare
    /// variable name, not `*u`.
    pub fn read_u(pack: &DerefPack) -> Vec3f {
        *get!(pack.u)
    }

    /// Parenthesized operands take the same elision path.
    pub fn paren_read(pack: &DerefPack) -> Vec3f {
        *(get!(pack.u))
    }

    /// The #153 comment-thread repro: a deref on each side of an
    /// assignment renders as a plain value assignment.
    pub fn write_data(pack: &DerefPack) {
        *get_mut!(pack.data) = load!(pack.u);
    }

    /// Compound assignment through the guard also elides the deref:
    /// WGSL `scale += v`.
    pub fn bump_scale(pack: &DerefPack, v: f32) {
        *get_mut!(pack.scale) += v;
    }

    /// Field writes through the guard: WGSL `pos.x = v`.
    pub fn set_pos_x(pack: &DerefPack, v: f32) {
        (*get_mut!(pack.pos)).x = v;
    }

    /// Regression: a pointer parameter shadowed by a branch-local value
    /// binding must not poison the parameter's deref. The deref below
    /// targets the `ptr!` parameter — valid WGSL — even though `u` is
    /// bound from `load!` inside the branch.
    pub fn shadow_param(c: bool, u: ptr!(function, f32), pack: &DerefPack) -> f32 {
        if c {
            let u = load!(pack.u);
            let _unused = u;
        }
        *u
    }
}

#[test]
fn deref_of_pack_field_access_renders_bare_names() {
    let src = deref_elision::WGSL_SOURCE.wgsl_source().unwrap();

    assert!(
        !src.contains("*derefpack__u"),
        "deref leaked into WGSL:\n{src}"
    );
    assert!(
        !src.contains("*derefpack__data"),
        "deref leaked into WGSL:\n{src}"
    );
    assert!(
        src.contains("derefpack__data = derefpack__u;"),
        "missing value assignment:\n{src}"
    );
    assert!(
        src.contains("derefpack__scale += v;"),
        "missing compound assignment:\n{src}"
    );
    assert!(
        src.contains("(derefpack__pos).x = v;"),
        "missing field write:\n{src}"
    );
    assert!(
        src.contains("*u"),
        "the pointer-parameter deref must be preserved:\n{src}"
    );
}

#[test]
fn deref_guards_agree_on_the_cpu() {
    use wgsl_rs::std::{Storage, Uniform};

    let pack = deref_elision::DerefPack {
        u: Uniform::new(Vec3f::vec3(1.0, 2.0, 3.0)),
        data: Storage::new(Vec3f::vec3(0.0, 0.0, 0.0)),
        scale: Storage::new(1.0_f32),
        pos: Storage::new(deref_elision::Pos { x: 0.0 }),
    };

    // `*get!(u)` copies the value out of the guard; parens elide the same.
    assert_eq!(deref_elision::read_u(&pack), Vec3f::vec3(1.0, 2.0, 3.0));
    assert_eq!(deref_elision::paren_read(&pack), Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get_mut!(data) = load!(u)` writes through the write guard.
    deref_elision::write_data(&pack);
    assert_eq!(*pack.data.get(), Vec3f::vec3(1.0, 2.0, 3.0));

    // `*get_mut!(scale) += v` compounds through the write guard.
    deref_elision::bump_scale(&pack, 0.5);
    assert_eq!(*pack.scale.get(), 1.5_f32);

    // `(*get_mut!(pos)).x = v` writes the field through the guard.
    deref_elision::set_pos_x(&pack, 7.0);
    assert_eq!(pack.pos.get().x, 7.0_f32);

    // The pointer parameter deref works regardless of the shadowing
    // branch.
    let mut p = 5.0_f32;
    assert_eq!(deref_elision::shadow_param(false, &mut p, &pack), 5.0_f32);
    assert_eq!(deref_elision::shadow_param(true, &mut p, &pack), 5.0_f32);
}
