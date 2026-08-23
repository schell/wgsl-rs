//! Regression tests for wgsl-rs#149 (same-scope variable shadowing produces
//! invalid WGSL) and the related findings from the deshadow-pass review.
//!
//! These drive the real `#[wgsl]` macro, render to WGSL, and assert on both
//! specific substrings (to pin the exact rename behavior) and naga validation
//! (to catch the `redefinition of x` error #149 reported). `validation` is a
//! default feature, so naga runs in normal `cargo test`.
//!
//! Cases covered:
//! - #1: pending renames leaking into nested scopes that rebind the same name
//!   (silent miscompilation; naga accepts it, so substring asserts are the
//!   primary catch).
//! - #2: shadowing inside `impl` method bodies (the deshadow pass only walks
//!   `Item::Fn`, skipping `Item::Impl`/`ImplItem::Fn`; naga catches this with a
//!   `redefinition` error).
//! - #3 (type-position rename gap) was considered but cannot be triggered
//!   through valid Rust: two same-scope `const` items with the same name is
//!   E0428, and array lengths in WGSL require const expressions so `let`
//!   shadowing of a `const` can't produce a type-position reference to the
//!   shadowed binding. The gap is real in the IR but unreachable from the macro
//!   surface.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use wgsl_rs::wgsl;

// ===== #1: nested-scope rename leak (for loop) =====

#[wgsl]
mod shadow_nested_for {
    /// `i` is shadowed, then a `for` loop rebinds `i` as its loop variable.
    /// The loop body must reference the loop counter, NOT the outer
    /// shadowed `i_1`. The trailing expression references the outer `i_1`.
    pub fn leak() -> u32 {
        let i = 0u32;
        let i = i + 1u32;
        let mut sum: u32 = 0u32;
        for i in 0u32..10u32 {
            sum += i;
        }
        sum + i
    }
}

#[test]
fn nested_for_loop_body_references_loop_var_not_outer_shadow() {
    let src = shadow_nested_for::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    // Outer shadow renamed.
    assert!(
        src.contains("let i_1 = i + 1u;"),
        "outer shadowed `i` should be renamed to `i_1`, got: {src}"
    );
    // Loop variable keeps its name (new binding, no shadowing).
    assert!(
        src.contains("for (var i = 0u; i < 10u; i++)"),
        "loop var should keep name `i`, got: {src}"
    );
    // CRITICAL: the loop body must reference `i` (the loop counter), not
    // `i_1` (the outer shadowed binding). This assertion fails today due
    // to the flat rename leaking into the nested scope.
    assert!(
        src.contains("sum += i;"),
        "loop body should reference loop var `i`, not `i_1` (rename leak), got: {src}"
    );
    assert!(
        !src.contains("sum += i_1;"),
        "loop body must NOT contain `sum += i_1;` (rename leak), got: {src}"
    );
    // Trailing expression references the outer shadowed binding.
    assert!(
        src.contains("return sum + i_1;"),
        "trailing expr should reference `i_1`, got: {src}"
    );
}

// ===== #1: nested-scope rename leak (plain block) =====

#[wgsl]
mod shadow_nested_block {
    /// Outer `x` is shadowed, then a nested block rebinds `x`. The
    /// assignment inside the block must reference the inner `x`, not the
    /// outer `x_1`.
    pub fn leak() -> f32 {
        let x = 1.0;
        let x = 2.0;
        let mut y = 0.0;
        {
            let x = 3.0;
            y = x;
        }
        y + x
    }
}

#[test]
fn nested_block_body_references_inner_binding_not_outer_shadow() {
    let src = shadow_nested_block::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    // Outer shadow renamed.
    assert!(
        src.contains("let x_1 = 2.0;"),
        "outer shadowed `x` should be renamed to `x_1`, got: {src}"
    );
    // Inner block binding keeps its name (new scope).
    assert!(
        src.contains("let x = 3.0;"),
        "inner block `x` should keep its name (new scope), got: {src}"
    );
    // CRITICAL: the assignment must reference the inner `x`, not `x_1`.
    assert!(
        src.contains("y = x;"),
        "inner block should reference inner `x`, not `x_1` (rename leak), got: {src}"
    );
    assert!(
        !src.contains("y = x_1;"),
        "inner block must NOT contain `y = x_1;` (rename leak), got: {src}"
    );
    // Trailing expression references the outer shadowed binding.
    assert!(
        src.contains("return y + x_1;"),
        "trailing expr should reference `x_1`, got: {src}"
    );
}

// ===== #2: impl method with same-scope shadowing =====

#[wgsl]
mod shadow_impl_method {
    pub struct Light {
        pub intensity: f32,
    }

    impl Light {
        /// A method body with same-scope shadowing. The deshadow pass
        /// currently skips `Item::Impl`/`ImplItem::Fn`, so this emits
        /// `redefinition of x` (the exact #149 symptom).
        pub fn attenuate(intensity: f32, distance: f32) -> f32 {
            let x = 1.0;
            let x = 2.0;
            intensity / (distance * distance) * x
        }
    }

    pub fn caller(d: f32) -> f32 {
        Light::attenuate(5.0, d)
    }
}

#[test]
fn impl_method_shadowing_is_deshadowed() {
    let src = shadow_impl_method::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    // The method body's shadow must be renamed. This fails today: impl
    // methods are skipped by the deshadow pass.
    assert!(
        src.contains("let x_1 = 2.0;"),
        "impl method shadow should be renamed to `x_1`, got: {src}"
    );
    assert!(
        !src.contains("let x = 1.0;\n    let x = 2.0;"),
        "impl method must not emit two `let x` in the same scope, got: {src}"
    );
}

// ===== Naga validation for all modules =====
//
// `validation` is a default feature (Cargo.toml default = ["validation", ...]).
// These run in normal `cargo test`. Each `validate()` call parses the rendered
// WGSL with naga and runs full semantic validation.
//
// - #1 (for / block): naga accepts the miscompiled output, so these would pass
//   today even with the bug. The substring asserts above are the real catch;
//   validation here guards against future regressions after the fix.
// - #2 (impl method): naga rejects with `redefinition of x` today (exact #149
//   symptom). After the fix, validation passes.

#[cfg(feature = "validation")]
#[test]
fn shadow_nested_for_validates() {
    shadow_nested_for::WGSL_SOURCE
        .validate()
        .expect("shadow_nested_for should produce valid WGSL after deshadow");
}

#[cfg(feature = "validation")]
#[test]
fn shadow_nested_block_validates() {
    shadow_nested_block::WGSL_SOURCE
        .validate()
        .expect("shadow_nested_block should produce valid WGSL after deshadow");
}

#[cfg(feature = "validation")]
#[test]
fn shadow_impl_method_validates() {
    shadow_impl_method::WGSL_SOURCE
        .validate()
        .expect("shadow_impl_method should produce valid WGSL after deshadow");
}
