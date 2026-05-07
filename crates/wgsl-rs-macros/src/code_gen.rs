//! Legacy direct-to-WGSL formatter.
//!
//! As of the IR overhaul, the production code path no longer goes
//! through this formatter — `wgsl-rs-macros` converts parse types to
//! [`wgsl_rs_ir`] and emits constructor functions that build owned IR at
//! runtime. The runtime renderer in `wgsl_rs_ir::render` is the
//! canonical WGSL emitter.
//!
//! This module is kept around because:
//! * Several `monomorphize.rs` tests still validate via the legacy formatter
//!   (using `mono_wgsl`).
//! * The formatter holds source-mapping logic that may be useful for future
//!   error reporting work.
//!
//! Once the legacy tests are migrated to use the IR renderer, this
//! module can be deleted.
#![allow(dead_code)]

use crate::parse::ItemMod;

mod formatter;
pub use formatter::{GenerateCode, GeneratedWgslCode};

/// Generate the WGSL code and a source map back to Rust spans.
pub fn generate_wgsl(module: &ItemMod) -> GeneratedWgslCode {
    let mut code = GeneratedWgslCode::default();
    module.write_code(&mut code);
    code
}
