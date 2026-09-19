//! Regression tests for wgsl-rs#164: vector `==` must lower to
//! `all(lhs == rhs)` and `!=` to `!(all(lhs == rhs))`.
//!
//! Rust's `PartialEq` on vectors yields `bool` (elementwise-all), but WGSL
//! `==` on vectors yields `vecN<bool>` — naga rejects the generated shader
//! wherever a `bool` is required (bindings, returns, conditions).
//!
//! These drive the real `#[wgsl]` macro, render to WGSL, and assert on both
//! specific substrings (to pin the exact lowering) and naga validation.
//! Like `unary_complement.rs`, `skip_validation` keeps the macro's
//! auto-emitted `__validate_wgsl` test off the `--no-default-features`
//! path; the explicit `#[cfg(feature = "validation")]` tests below cover
//! validation instead.

#![allow(dead_code)]
#![allow(unused_variables)]

use wgsl_rs::wgsl;

/// The #164 repro shape: a bool-typed let binding initialized from a
/// vector comparison. Before this fix it rendered `vecN<bool>` into a
/// `bool` binding, which naga rejected; it must now render the wrapped
/// `all(...)`. The `bool` annotation also keeps clippy's
/// `let_and_return` from firing (it skips annotated bindings).
#[wgsl(skip_validation)]
mod eq_repro {
    use wgsl_rs::std::*;

    pub fn compute() -> bool {
        let x: bool = vec2f(0.0, 0.0) == vec2f(0.0, 0.0);
        x
    }
}

#[wgsl(skip_validation)]
mod eq_lowering {
    use wgsl_rs::std::*;

    /// Constructors on both sides.
    pub fn constructors_eq() -> bool {
        vec3f(1.0, 2.0, 3.0) == vec3f(1.0, 2.0, 3.0)
    }

    /// `!=` lowers to `!(all(lhs == rhs))` — matching `PartialEq::ne`,
    /// which is `!(self == other)`, not `all(lhs != rhs)`.
    pub fn constructors_ne() -> bool {
        vec4f(1.0, 2.0, 3.0, 4.0) != vec4f(4.0, 3.0, 2.0, 1.0)
    }

    /// Locals typed by their initializer.
    pub fn locals_eq(a: Vec2f, b: Vec2f) -> bool {
        let c = a;
        let d = b;
        c == d
    }

    /// Locals with explicit type annotations.
    pub fn annotated_eq(a: Vec3f, b: Vec3f) -> bool {
        let e: Vec3f = a;
        let f: Vec3f = b;
        e == f
    }

    /// Multi-component swizzles are vectors.
    pub fn swizzle_eq(a: Vec3f, b: Vec3f) -> bool {
        a.xzy() == b.zyx()
    }

    // NOTE: bool-vector params (`Vec2b` / `Vec2<bool>`) currently render as
    // the unknown WGSL identifier `vec2b`, so a `Vec2b == Vec2b` case cannot
    // validate yet. That type-render gap is separate from this fix.

    /// Monomorphized generic: the wrap must also apply inside the
    /// instantiated copy of `generic_eq`, where `T` has become `Vec2f`.
    pub fn generic_eq<T: PartialEq>(a: T, b: T) -> bool {
        a == b
    }

    pub fn mono_eq(a: Vec2f, b: Vec2f) -> bool {
        generic_eq::<Vec2f>(a, b)
    }

    /// `cmp_eq` is the componentwise escape hatch: it renders as the raw
    /// WGSL `==` operator, yielding a vecN<bool> mask. Feeding it to
    /// `all` exercises the whole path without needing a bool-vector
    /// return type in the signature (the `Vec2b` alias currently renders
    /// as an unknown identifier — a separate, pre-existing gap).
    pub fn mask_all_eq(a: Vec2f, b: Vec2f) -> bool {
        all(cmp_eq(a, b))
    }

    /// `cmp_ne` renders as the raw WGSL `!=` operator.
    pub fn mask_any_ne(a: Vec3f, b: Vec3f) -> bool {
        any(cmp_ne(a, b))
    }

    /// Scalars must NOT be wrapped.
    pub fn scalar_eq_stays(a: f32, b: f32) -> bool {
        a == b
    }

    /// Single-component swizzles are scalars and must NOT be wrapped.
    pub fn swizzle_scalar_ne_stays(a: Vec2f, b: Vec2f) -> bool {
        a.x != b.y
    }
}

/// Review-gap cases (PR #176): comparisons whose operands come from the
/// mask builtins, vector-preserving builtins, impl methods, or which are
/// vulnerable to nested-module symbol shadowing, must all still infer as
/// vector-typed and get the `all(...)` wrap.
#[wgsl(skip_validation)]
mod infer_gaps {
    use wgsl_rs::std::*;

    /// Masks compared with `==` are vector comparisons too.
    pub fn mask_vs_mask(a: Vec2f, b: Vec2f, c: Vec2f, d: Vec2f) -> bool {
        cmp_eq(a, b) == cmp_eq(c, d)
    }

    /// Mask locals via let-inference.
    pub fn mask_local_eq(a: Vec2f, b: Vec2f, c: Vec2f, d: Vec2f) -> bool {
        let m = cmp_eq(a, b);
        let n = cmp_eq(c, d);
        m == n
    }

    /// Vector-preserving builtins keep comparisons vector-typed.
    pub fn normalize_eq(a: Vec3f, b: Vec3f) -> bool {
        normalize(a) == normalize(b)
    }

    /// Impl methods with vector returns.
    pub struct Helpers {
        _v: Vec2f,
    }

    impl Helpers {
        pub fn make(v: Vec2f) -> Vec2f {
            v
        }
    }

    pub fn method_call_eq(a: Vec2f, b: Vec2f) -> bool {
        Helpers::make(a) == Helpers::make(b)
    }

    // Linkage values are inferable from the module's `storage!`
    // declarations (a plain comment: macro invocations cannot carry doc
    // comments without an unused-doc-comment warning).
    storage!(group(0), binding(0), INPUT: [Vec2f; 2]);

    pub fn linkage_eq() -> bool {
        let input = get!(INPUT);
        input[0] == input[1]
    }

    /// Associated consts of std vector types are inferable from the type
    /// alias shape.
    pub fn zero_one_eq() -> bool {
        Vec2f::ZERO == Vec2f::ONE
    }

    /// Associated consts of user impls are inferable from the impl's
    /// declared type.
    impl Helpers {
        pub const ORIGIN: Vec2f = vec2f(0.0, 0.0);
    }

    pub fn origin_eq(a: Vec2f) -> bool {
        Helpers::ORIGIN == a
    }

    /// The extended vector-preserving builtin list, spot-checked with
    /// `step`.
    pub fn step_eq(a: Vec3f, b: Vec3f, c: Vec3f, d: Vec3f) -> bool {
        step(a, b) == step(c, d)
    }

    /// A nested module's `make` must not shadow the enclosing module's
    /// `make` during inference (nested mods emit no WGSL, but their
    /// symbols used to leak into the enclosing module's table).
    pub fn make() -> Vec2f {
        vec2f(0.0, 0.0)
    }

    pub fn shadowed_eq() -> bool {
        make() == make()
    }

    mod inner {
        pub fn make() -> f32 {
            0.0
        }
    }
}

// ===== WGSL source assertions =====

#[test]
fn repro_wraps_constructors_in_all() {
    let src = eq_repro::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("let x: bool = all((vec2f(0.0, 0.0) == vec2f(0.0, 0.0)));"),
        "the #164 repro must emit `all(...)` (binary op parenthesized per wgsl-rs#159), got: {src}"
    );
}

#[test]
fn vector_eq_ne_lower_to_all() {
    let src = eq_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    for (needle, why) in [
        (
            "return all((vec3f(1.0, 2.0, 3.0) == vec3f(1.0, 2.0, 3.0)));",
            "constructor == wraps in all()",
        ),
        (
            "return !all((vec4f(1.0, 2.0, 3.0, 4.0) == vec4f(4.0, 3.0, 2.0, 1.0)));",
            "vector != lowers to !(all(lhs == rhs)); binary op parenthesized per wgsl-rs#159",
        ),
        ("return all((c == d));", "locals compared via =="),
        ("return all((e == f));", "annotated locals compared via =="),
        (
            "return all((a.xzy == b.zyx));",
            "multi-component swizzles are vectors",
        ),
        (
            "return all((a == b));",
            "cmp_eq lowers to the raw == operator; the monomorphized generic_eq body renders the \
             same way",
        ),
        (
            "return any((a != b));",
            "cmp_ne lowers to the raw != operator",
        ),
        (
            "return _1generic_eq_vec2f(a, b);",
            "mono call site is rewritten",
        ),
    ] {
        assert!(
            src.contains(needle),
            "expected `{needle}` ({why}) in: {src}"
        );
    }
    // The mono instance `_1generic_eq_vec2f` must wrap its body.
    assert!(
        src.contains("generic_eq") && src.contains("return all((a == b));"),
        "monomorphized generic_eq must wrap `a == b` in all(), got: {src}"
    );
    // No raw vector == survived unwrapped at a return position.
    assert!(
        !src.contains("return c == d;")
            && !src.contains("return e == f;")
            && !src.contains("return a.xzy() == b.zyx();"),
        "no vector ==/!= should survive lowering, got: {src}"
    );
}

#[test]
fn scalar_comparisons_stay_raw() {
    let src = eq_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("return (a == b);"),
        "scalar == must stay free of the all() wrap, got: {src}"
    );
    assert!(
        src.contains("return (a.x != b.y);"),
        "single-component swizzle != must stay free of the all() wrap, got: {src}"
    );
}

#[test]
fn infer_gaps_lower_to_all() {
    let src = infer_gaps::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    for (needle, why) in [
        ("return all(((a == b) == (c == d)));", "mask == mask wraps"),
        (
            "let m = (a == b);",
            "mask local binding renders the raw mask",
        ),
        ("return all((m == n));", "mask locals compare via =="),
        (
            "return all((normalize(a) == normalize(b)));",
            "vector-preserving builtins infer",
        ),
        (
            "return all((Helpers_make(a) == Helpers_make(b)));",
            "impl method calls infer",
        ),
        (
            "return all((make() == make()));",
            "nested module does not shadow outer symbols",
        ),
        (
            "return all((input[0] == input[1]));",
            "storage linkage values infer",
        ),
        (
            "return all((vec2f_ZERO == vec2f_ONE));",
            "std vector associated consts infer",
        ),
        (
            "return all((Helpers_ORIGIN == a));",
            "user impl associated consts infer",
        ),
        (
            "return all((step(a, b) == step(c, d)));",
            "step is a vector-preserving builtin",
        ),
    ] {
        assert!(
            src.contains(needle),
            "expected `{needle}` ({why}) in: {src}"
        );
    }
}

// ===== naga validation =====

#[cfg(feature = "validation")]
#[test]
fn repro_validates() {
    eq_repro::WGSL_SOURCE.validate().expect("naga validation");
}

#[cfg(feature = "validation")]
#[test]
fn lowering_validates() {
    eq_lowering::WGSL_SOURCE
        .validate()
        .expect("naga validation");
}

#[cfg(feature = "validation")]
#[test]
fn infer_gaps_validates() {
    infer_gaps::WGSL_SOURCE.validate().expect("naga validation");
}

// ===== CPU-side parity =====
//
// The `#[wgsl]` module stays valid Rust; these lock the two-worlds
// agreement: GPU `all(lhs == rhs)` must produce the same values as
// Rust's vector `PartialEq`.

#[test]
fn cpu_vector_eq_values() {
    use eq_lowering as m;
    use wgsl_rs::std::*;

    let v = vec2f(1.0, 2.0);
    let w = vec2f(3.0, 4.0);
    assert!(m::locals_eq(v, v));
    assert!(!m::locals_eq(v, w));
    assert!(m::annotated_eq(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
    assert!(!m::annotated_eq(vec3f(1.0, 2.0, 3.0), vec3f(3.0, 2.0, 1.0)));
    assert!(m::constructors_eq());
    assert!(m::constructors_ne());
    assert!(m::swizzle_eq(vec3f(1.0, 2.0, 3.0), vec3f(2.0, 3.0, 1.0)));
    assert!(!m::swizzle_eq(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
    assert!(m::mono_eq(v, v));
    assert!(!m::mono_eq(v, w));
    assert!(m::scalar_eq_stays(2.0, 2.0));
    assert!(!m::scalar_eq_stays(1.0, 2.0));
    assert!(m::swizzle_scalar_ne_stays(vec2f(1.0, 2.0), vec2f(3.0, 4.0)));
    assert!(m::mask_all_eq(v, v));
    assert!(!m::mask_all_eq(v, w));
    assert!(m::mask_any_ne(vec3f(1.0, 2.0, 3.0), vec3f(3.0, 2.0, 1.0)));
    use infer_gaps as g;
    assert!(g::mask_vs_mask(v, w, v, w));
    assert!(!g::mask_vs_mask(v, w, v, v));
    assert!(g::mask_local_eq(v, v, v, v));
    assert!(!g::mask_local_eq(v, w, v, v));
    assert!(g::normalize_eq(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
    assert!(!g::normalize_eq(vec3f(1.0, 2.0, 3.0), vec3f(3.0, 2.0, 1.0)));
    assert!(g::method_call_eq(v, v));
    assert!(!g::method_call_eq(v, w));
    assert!(g::shadowed_eq());
    g::INPUT.set([vec2f(1.0, 2.0), vec2f(1.0, 2.0)]);
    assert!(g::linkage_eq());
    g::INPUT.set([vec2f(1.0, 2.0), vec2f(3.0, 4.0)]);
    assert!(!g::linkage_eq());
    assert!(!g::zero_one_eq());
    assert!(g::origin_eq(vec2f(0.0, 0.0)));
    assert!(!g::origin_eq(vec2f(1.0, 1.0)));
    assert!(g::step_eq(
        vec3f(0.0, 0.0, 0.0),
        vec3f(1.0, 1.0, 1.0),
        vec3f(0.0, 0.0, 0.0),
        vec3f(1.0, 1.0, 1.0)
    ));
    assert!(!m::mask_any_ne(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
}
