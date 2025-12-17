//! Formats a WGSL token tree, poorly.

use crate::parse::ItemMod;

mod formatter;
pub use formatter::{GenerateCode, GeneratedWgslCode};

/// Generate the WGSL code and a source map back to Rust spans.
pub fn generate_wgsl(module: ItemMod) -> GeneratedWgslCode {
    let mut code = GeneratedWgslCode::default();
    module.write_code(&mut code);
    code
}
