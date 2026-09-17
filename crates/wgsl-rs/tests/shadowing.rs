//! Regression tests for wgsl-rs#149 and wgsl-rs#163 (same-scope shadowing —
//! including locals that shadow function parameters — compiled but produced
//! invalid WGSL) and the related findings from the deshadow-pass review.
//!
//! These drive the real `#[wgsl]` macro, render to WGSL, and assert on both
//! specific substrings (to pin the exact rename behavior) and naga validation
//! (to catch the `redefinition of x` errors #149/#163 reported). `validation`
//! is a default feature, so naga runs in normal `cargo test`.
//!
//! Cases covered (behavior as fixed by the deshadow pass in PR #155):
//! - #1: renames must NOT leak into nested scopes that rebind the same name (a
//!   flat rename would silently miscompile; naga accepts the bad output, so
//!   substring asserts are the primary catch here).
//! - #2: shadowing inside `impl` method bodies is deshadowed (the pass recurses
//!   into `Item::Impl`/`ImplItem::Fn`).
//! - #163: a local that shadows a function parameter is renamed — params and
//!   the function body share end-of-scope in WGSL.
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
    // `i_1` (the outer shadowed binding) — a rename leak here would
    // silently miscompile the shader.
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
        /// recurses into `Item::Impl`/`ImplItem::Fn`, so the shadow here
        /// is renamed exactly like in a free function.
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
    // The method body's shadow must be renamed: impl methods are
    // deshadowed just like free functions.
    assert!(
        src.contains("let x_1 = 2.0;"),
        "impl method shadow should be renamed to `x_1`, got: {src}"
    );
    assert!(
        !src.contains("let x = 1.0;\n    let x = 2.0;"),
        "impl method must not emit two `let x` in the same scope, got: {src}"
    );
}

// ===== #163: local shadows a function parameter (exact repro) =====

#[wgsl]
mod shadow_param {
    /// Exact repro from #163: a local in the function body shadows the
    /// parameter. Params and the body share end-of-scope in WGSL, so the
    /// deshadow pass must rename the local.
    pub fn my_func(x: f32) {
        let x = 0.0;
    }
}

#[test]
fn param_shadowing_is_deshadowed() {
    let src = shadow_param::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("let x_1 = 0.0;"),
        "local shadowing param must be renamed to `x_1`, got: {src}"
    );
    assert!(
        !src.contains("let x = 0.0;"),
        "must not emit a same-scope redefinition of `x`, got: {src}"
    );
}

// ===== Naga validation for all modules =====
//
// `validation` is a default feature (Cargo.toml default = ["validation", ...]).
// These run in normal `cargo test`. Each `validate()` call parses the rendered
// WGSL with naga and runs full semantic validation.
//
// Substring asserts and validation are complementary: rename leaks into
// nested scopes produce WGSL naga accepts (silent miscompilation), so the
// substring asserts are the only catch there; same-scope redefinitions fail
// naga outright.

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

#[cfg(feature = "validation")]
#[test]
fn shadow_param_validates() {
    shadow_param::WGSL_SOURCE
        .validate()
        .expect("shadow_param should produce valid WGSL after deshadow");
}
