//! An unclaimed statement macro (not in any extension's MACROS) should be
//! a compile error, not a runtime error.
//!
//! The `extensions = [NoopExt]` list includes an extension, but `NoopExt`
//! does not claim `unclaimed_macro` in its `MACROS` const. The compile-time
//! `const` check should fail with an `E0080` panic.

use wgsl_rs::{ir, wgsl, WgslExtension};

#[macro_export]
macro_rules! unclaimed_macro {
    () => {};
}

pub struct NoopExt;
impl WgslExtension for NoopExt {
    fn modify_ir(_module: &mut ir::Module) {}
}

#[wgsl(crate_path = wgsl_rs, extensions = [super::NoopExt])]
mod ext_macro_shader {
    pub fn main() -> u32 {
        unclaimed_macro!();
        42u32
    }
}

fn main() {}