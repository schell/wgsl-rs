//! Regression fixture for wgsl-rs#165: `#[wgsl]` modules in different
//! crates used to be assigned overlapping module ids (a per-crate
//! counter), which made the runtime source assembler silently drop one
//! of them as a "duplicate" import.

use wgsl_rs::wgsl;

#[wgsl]
pub mod shader_a {
    pub fn shader_a_func(x: f32) -> f32 {
        x + 1.0
    }
}
