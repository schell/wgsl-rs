//! Regression test for the same-package/different-target id collision
//! (found in review of PR #173).
//!
//! Integration tests are compiled as a *separate target* of the same
//! package: they see the same `CARGO_PKG_NAME`/`CARGO_PKG_VERSION` as
//! the lib but start their own proc-macro counter at 0. Without target
//! discriminators in module ids, the test target's first module and
//! the lib's first module (`shader_c`) hashed to the same id, so
//! assembling `shader_t2` dropped `shader_c` as a "duplicate" import —
//! unknown-identifier parse errors, exactly the wgsl-rs#165 failure
//! mode but within one package.

use wgsl_rs::wgsl;

#[wgsl]
pub mod shader_t1 {
    use id_collision_app::shader_c::*;

    pub fn shader_t1_func(x: f32) -> f32 {
        shader_c_func(x) + 1.0
    }
}

#[wgsl]
pub mod shader_t2 {
    use super::shader_t1::*;
    use id_collision_app::shader_c::*;

    pub fn shader_t2_func(x: f32) -> f32 {
        shader_t1_func(x) + shader_c_func(x)
    }
}

/// `shader_t2` imports both `shader_t1` (test target, counter 0) and
/// `shader_c` (lib target, counter 0). If the two shared an id, the
/// assembler would drop one during `shader_t2`'s depth-first walk and
/// the surviving module's calls would fail to resolve.
#[test]
fn same_package_cross_target_imports_assemble() {
    let src = shader_t2::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("fn shader_t1_func("),
        "test-target module missing from assembled WGSL:\n{src}"
    );
    assert!(
        src.contains("fn shader_c_func("),
        "lib-target module missing from assembled WGSL:\n{src}"
    );
    assert_eq!(src.matches("fn shader_c_func(").count(), 1, "got:\n{src}");
}
