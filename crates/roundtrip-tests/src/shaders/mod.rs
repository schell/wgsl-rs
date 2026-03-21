//! Shader modules for roundtrip testing.
//!
//! Each submodule defines one or more `#[wgsl]` compute shader modules that
//! exercise a category of `wgsl_rs::std` builtin functions, along with a
//! [`RoundtripTest`](crate::harness::RoundtripTest) implementation that drives
//! the GPU vs CPU comparison.
//!
//! ## Current coverage
//!
//! - [`trig`] — Trigonometric functions: `sin`, `cos`, `tan`, `asin`, `acos`,
//!   `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
//! - [`exponential`] — Exponential functions: `exp`, `exp2`, `log`, `log2`,
//!   `pow`, `sqrt`, `inverse_sqrt`
//! - [`rounding`] — Rounding functions: `ceil`, `floor`, `round`, `trunc`,
//!   `fract`, `saturate`
//! - [`clamping`] — Clamping and interpolation: `clamp`, `min`, `max`, `mix`,
//!   `smoothstep`, `step`, `select`
//! - [`geometric`] — Geometric and conversion functions: `dot`, `cross`,
//!   `normalize`, `length`, `distance`, `reflect`, `refract`, `face_forward`,
//!   `degrees`, `radians`, `sign`, `abs`, `fma`
//!
//! ## Planned coverage
//!
//! The following categories are intended for future implementation:
//!
//! - **Bitcast** — `bitcast_f32`, `bitcast_u32`, `bitcast_i32`, and vector
//!   variants
//! - **Packing** — `pack4x8snorm`, `unpack2x16float`, and related pack/unpack
//!   functions
//! - **Bit manipulation** — `count_leading_zeros`, `reverse_bits`,
//!   `extract_bits`, `insert_bits`, `first_leading_bit`, `first_trailing_bit`
//! - **Matrix operations** — `determinant`, `transpose`, matrix-vector and
//!   matrix-matrix multiplication
//! - **Atomic operations** — `atomic_add`, `atomic_max`,
//!   `atomic_compare_exchange_weak`, etc. (requires workgroup-scoped tests)
//! - **Derivative functions** — `dpdx`, `dpdy`, `fwidth` and fine/coarse
//!   variants (requires fragment shader tests with `dispatch_fragments`)
//! - **Texture sampling** — `texture_sample`, `texture_load`, etc. (complex
//!   setup with texture data)

pub mod clamping;
pub mod exponential;
pub mod geometric;
pub mod rounding;
pub mod trig;

use crate::harness::RoundtripTest;

/// Returns all available roundtrip tests.
pub fn all_tests() -> Vec<Box<dyn RoundtripTest>> {
    vec![
        Box::new(trig::TrigTest),
        Box::new(exponential::ExponentialTest),
        Box::new(rounding::RoundingTest),
        Box::new(clamping::ClampingTest),
        Box::new(geometric::GeometricTest),
    ]
}
