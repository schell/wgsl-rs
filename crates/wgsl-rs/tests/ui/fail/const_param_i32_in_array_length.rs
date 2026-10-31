//! `i32` const generic parameters are value-position-only by
//! construction: Rust requires array sizes to evaluate to `usize`
//! (Rust Reference, "Array types"), so rustc itself rejects
//! `[f32; N]` when `N: i32` — before wgsl-rs is involved at all.
//! Design decision #1 (the module must compile as valid Rust) makes
//! rustc the type checker for array-length positions, so wgsl-rs needs
//! no macro-time sign enforcement there. This fixture documents that
//! guard.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod i32_in_array_length {
    pub fn sum<const N: i32>(arr: [f32; N]) -> f32 {
        arr[0]
    }
}

fn main() {}