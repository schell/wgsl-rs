//! Regression test for wgsl-rs#165: shader modules in different crates
//! can collide on module id.
//!
//! `shader_c` imports `shader_a` and `shader_b`, each defined in a
//! *different* crate. With the old per-crate module id counter, both
//! imports got id 0 and the assembler dropped `shader_b` as a
//! "duplicate", leaving `shader_c_func` with a call to an
//! `shader_b_func` that no longer existed — an unknown-identifier
//! parse error. The auto-generated `__validate_wgsl` test fails under
//! the old behavior; the tests below pin the fixed behavior.

use wgsl_rs::wgsl;

#[wgsl]
pub mod shader_c {
    use id_collision_shader_a::shader_a::*;
    use id_collision_shader_b::shader_b::*;

    pub fn shader_c_func(x: f32) -> f32 {
        shader_a_func(x) + shader_b_func(x)
    }
}

/// Both cross-crate imports must survive source assembly: each helper
/// appears exactly once in the assembled WGSL.
#[test]
fn cross_crate_imports_all_assemble() {
    let src = shader_c::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(src.contains("fn shader_a_func("), "got:\n{src}");
    assert!(src.contains("fn shader_b_func("), "got:\n{src}");
    assert_eq!(src.matches("fn shader_a_func(").count(), 1, "got:\n{src}");
    assert_eq!(src.matches("fn shader_b_func(").count(), 1, "got:\n{src}");
}

/// The Rust world must keep working too (the "two worlds" contract):
/// the CPU-side call produces the value the assembled WGSL computes.
#[test]
fn rust_world_shader_c_func() {
    assert_eq!(shader_c::shader_c_func(1.0), 5.0);
}
