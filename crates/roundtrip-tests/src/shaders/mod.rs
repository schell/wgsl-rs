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
//! - [`bit_manipulation`] — Bit manipulation: `count_leading_zeros`,
//!   `count_one_bits`, `count_trailing_zeros`, `reverse_bits`,
//!   `first_leading_bit`, `first_trailing_bit`, `extract_bits`, `insert_bits`
//! - [`bitcast`] — Bitcast reinterpretation: `bitcast_f32`, `bitcast_u32`,
//!   `bitcast_i32`, `bitcast_vec4f`, `bitcast_vec4u`
//! - [`packing`] — Pack/unpack quantization: `pack4x8snorm`, `pack4x8unorm`,
//!   `pack2x16snorm`, `pack2x16unorm`, `pack2x16float`, `unpack4x8snorm`,
//!   `unpack4x8unorm`, `unpack2x16snorm`, `unpack2x16unorm`, `unpack2x16float`
//!
//! ## Planned coverage
//!
//! The following categories are intended for future implementation:
//!
//! - **modf, frexp, ldexp** — Struct-returning numeric builtins
//! - **Type conversions** — `f32()`, `u32()`, `i32()` casting functions
//! - **Matrix operations** — `determinant`, `transpose`, matrix-vector and
//!   matrix-matrix multiplication
//! - **Logical builtins** — `all`, `any` on bool vectors
//! - **Vector arithmetic** — Explicit operator tests for `+`, `-`, `*`, `/`,
//!   `%` on vector types
//! - **Atomic operations** — `atomic_add`, `atomic_max`,
//!   `atomic_compare_exchange_weak`, etc. (requires workgroup-scoped tests)
//! - **Synchronization** — `workgroup_barrier`, `storage_barrier`,
//!   `workgroup_uniform_load` (requires multi-invocation compute)
//! - **Derivative functions** — `dpdx`, `dpdy`, `fwidth` and fine/coarse
//!   variants (requires fragment shader tests with `dispatch_fragments`)
//! - **Texture sampling** — `texture_sample`, `texture_load`, etc. (complex
//!   setup with texture data)

pub mod bit_manipulation;
pub mod bitcast;
pub mod clamping;
pub mod exponential;
pub mod geometric;
pub mod packing;
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
        Box::new(bit_manipulation::BitManipulationTest),
        Box::new(bitcast::BitcastTest),
        Box::new(packing::PackingTest),
    ]
}
