//! Test that a `WgslExtension` can claim and lower a `Stmt::Macro`.
//!
//! The extension `LowerMyMacro` declares `my_macro` in its `MACROS` const
//! and replaces `Stmt::Macro { name: "my_macro", .. }` with a `Stmt::Local`
//! that initializes a variable to `42u32`.

use wgsl_rs::{ir, wgsl, WgslExtension};

/// CPU-side expansion of `my_macro!`.
///
/// On the CPU, `my_macro!()` expands to a `let` binding.
/// On the GPU (inside `#[wgsl]`), the `LowerMyMacro` extension lowers the
/// `Stmt::Macro` to the same statement in the IR.
#[macro_export]
macro_rules! my_macro {
    () => {
        let result: u32 = 42;
    };
}

pub struct LowerMyMacro;

impl WgslExtension for LowerMyMacro {
    const MACROS: &'static [&'static str] = &["my_macro"];

    fn modify_ir(module: &mut ir::Module) {
        for item in &mut module.items {
            if let ir::Item::Fn(f) = item {
                lower_in_block(&mut f.block);
            }
        }
    }
}

fn lower_in_block(block: &mut ir::Block) {
    for i in 0..block.stmts.len() {
        if let ir::Stmt::Macro { name, .. } = &block.stmts[i] {
            if name == "my_macro" {
                block.stmts[i] = ir::Stmt::Local(ir::Local {
                    mutable: false,
                    name: "result".to_string(),
                    ty: Some(ir::Type::Scalar(ir::ScalarType::U32)),
                    init: Some(ir::Expr::Lit(ir::Lit::Int {
                        digits: "42".to_string(),
                        suffix: "u32".to_string(),
                    })),
                });
            }
        }
    }
}

#[wgsl(crate_path = wgsl_rs, extensions = [super::LowerMyMacro])]
mod ext_macro_shader {
    pub fn main() -> u32 {
        42u32
    }
}

fn main() {
    let source = ext_macro_shader::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        !source.contains("my_macro"),
        "Stmt::Macro should have been lowered by the extension, got:\n{source}"
    );
}