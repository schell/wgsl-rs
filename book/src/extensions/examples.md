# Extension Examples

## NoopExt

The smallest possible extension, useful as a smoke test that the wiring works:

```rust
use wgsl_rs::WgslExtension;
use wgsl_rs::ir;

pub struct NoopExt;

impl WgslExtension for NoopExt {
    fn modify_ir(_module: &mut ir::Module) {}
}
```

```rust
#[wgsl(extensions = [my_crate::NoopExt])]
mod shader {
    // ...
}
```

## SlabItemExt

A derive-driven code generator. It finds structs annotated with `#[derive(SlabItem)]` and injects `slab_read` and `slab_write` helper functions for each.

```rust
use wgsl_rs::WgslExtension;
use wgsl_rs::ir::{Item, Module};

pub struct SlabItemExt;

impl WgslExtension for SlabItemExt {
    fn modify_ir(module: &mut Module) {
        let slab_structs: Vec<String> = module
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Struct(s) => {
                    let derives_slab = s.attrs.iter().any(|a| {
                        a.path == "derive" && a.args.iter().any(|arg| arg == "SlabItem")
                    });
                    derives_slab.then(|| s.name.clone())
                }
                _ => None,
            })
            .collect();

        for name in slab_structs {
            module.items.push(build_slab_read(&name));
            module.items.push(build_slab_write(&name));
        }
    }
}

fn build_slab_read(struct_name: &str) -> Item {
    // Construct an ir::Item::Fn that reads `struct_name` from a slab buffer
    // at a given element index.
    todo!()
}

fn build_slab_write(struct_name: &str) -> Item {
    // Construct an ir::Item::Fn that writes a `struct_name` value into a slab
    // buffer at a given element index.
    todo!()
}
```

## wgsl-rs-layout

`wgsl-rs-layout` is the first real-world extension crate. It is a standalone crate that depends on `wgsl-rs` for its types and implements `WgslExtension` to compute WGSL memory layout for Rust structs. It demonstrates that the extension mechanism is sufficient to build a non-trivial, redistributable tool on top of `wgsl-rs` without forking the transpiler.

See the [Memory Layout](../layout/overview.md) section for full coverage of `wgsl-rs-layout`.

## Statement Macro Lowering: `LowerMyMacro`

An extension can claim a custom statement macro via `MACROS` and replace `Stmt::Macro` nodes with lowered IR. This example from the trybuild test suite defines `my_macro!()` and an extension that lowers it to `let result: u32 = 42;`:

```rust
use wgsl_rs::{ir, wgsl, WgslExtension};

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
```

The shader module uses `my_macro!()` in statement position, and the extension's `modify_ir` replaces it before rendering:

```rust
#[wgsl(extensions = [super::LowerMyMacro])]
mod ext_macro_shader {
    pub fn main() -> u32 {
        my_macro!();
        42u32
    }
}
```

After `modify_ir` runs, the `Stmt::Macro` is replaced by `Stmt::Local`, and the rendered WGSL contains no trace of `my_macro`.