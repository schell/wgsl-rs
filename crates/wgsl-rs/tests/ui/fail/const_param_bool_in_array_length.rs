//! `bool` const generic parameters are value-position-only by
//! construction: Rust requires array sizes to evaluate to `usize`
//! (Rust Reference, "Array types"), so rustc itself rejects
//! `[f32; B]` when `B: bool` — before wgsl-rs is involved at all.
//! Design decision #1 (the module must compile as valid Rust) makes
//! rustc the type checker for array-length positions, so wgsl-rs needs
//! no macro-time bool enforcement there. This fixture documents that
//! guard; it is the bool parallel of
//! `const_param_i32_in_array_length.rs`.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod bool_in_array_length {
    pub fn sum<const B: bool>(arr: [f32; B]) -> f32 {
        arr[0]
    }
}

fn main() {}
