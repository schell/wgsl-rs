# Modifying the IR

`modify_ir` is the single entry point an extension implements. Understanding when and how it runs is essential for writing correct extensions.

## When It Runs

`modify_ir` is invoked by the IR constructor:

1. IR items are built from the transpiled source.
2. **`modify_ir` runs** for each extension, in declaration order.
3. Type instantiation occurs (for templates), substituting `TypeParam` nodes.

This means extensions see the fully constructed IR but operate before any type-parameter substitution. Anything an extension injects containing `TypeParam` nodes will be substituted automatically during instantiation.

`modify_ir` runs on every `wgsl_source()` and `instantiate()` call — not once at definition time.

## Walking the Module

`ir::Module` is:

```rust
pub struct Module {
    pub name: String,
    pub items: Vec<Item>,
    pub attrs: Vec<Attribute>,
}
```

`ir::Item` is an enum with variants: `Struct`, `Fn`, `Const`, `Uniform`, `Storage`, `Workgroup`, `Sampler`, `Texture`, `Impl`, `Enum`.

To iterate and mutate items, match on the variant:

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
            module.items.push(slab_read_fn(&name));
            module.items.push(slab_write_fn(&name));
        }
    }
}

fn slab_read_fn(struct_name: &str) -> Item {
    // Build an ir::Item::Fn that reads a slab item at a given offset.
    // ...
}

fn slab_write_fn(struct_name: &str) -> Item {
    // Build an ir::Item::Fn that writes a slab item at a given offset.
    // ...
}
```

## Type Substitution

Extensions do not need to handle type parameters themselves. If an injected function references a `TypeParam` node, the instantiation pass substitutes it with the concrete type from each `instantiate()` call. Do not attempt to outsmart this by string-replacing type names — operate on IR nodes and let substitution handle generics.

See [IR Attributes](./ir-attributes.md) for how to filter items by attribute, and [Examples](./examples.md) for the full `SlabItemExt` and other worked examples.