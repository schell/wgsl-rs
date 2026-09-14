//! Regression fixture for wgsl-rs#165 — see `id-collision-shader-a`.

use wgsl_rs::wgsl;

#[wgsl]
pub mod shader_b {
    pub fn shader_b_func(x: f32) -> f32 {
        x + 2.0
    }
}
