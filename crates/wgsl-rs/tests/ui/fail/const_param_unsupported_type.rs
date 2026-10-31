//! Const generic parameters are limited to `u32`, `usize`, and `i32`.
//! Other integer types (`u8` here) have no WGSL analog: `u32`/`usize`
//! cover array lengths, `i32` covers value-position constants, and i32
//! is the only signed integer WGSL has.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod unsupported_const_param {
    pub fn scale<const FACTOR: u8>(x: u32) -> u32 {
        x * FACTOR as u32
    }
}

fn main() {}