//! Const generic parameters are limited to `u32`, `usize`, `i32`, and
//! `bool`. Other types (`u8` here) have no WGSL analog: `u32`/`usize`
//! cover array lengths, `i32` covers value-position constants (i32 is
//! the only signed integer WGSL has), and `bool` covers compile-time
//! feature flags.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod unsupported_const_param {
    pub fn scale<const FACTOR: u8>(x: u32) -> u32 {
        x * FACTOR as u32
    }
}

fn main() {}