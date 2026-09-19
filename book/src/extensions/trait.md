# The `WgslExtension` Trait

The `WgslExtension` trait lets downstream crates inspect and modify a shader's WGSL IR after transpilation but before type instantiation. It is the primary extension point for post-transpile code generation and analysis.

## Definition

The trait lives in `wgsl_rs::extension` and is re-exported at the crate root.

```rust
pub trait WgslExtension {
    const MACROS: &'static [&'static str] = &[];
    fn modify_ir(module: &mut crate::ir::Module);
}
```

Import it directly from the crate root:

```rust
use wgsl_rs::WgslExtension;
```

## Purpose

`modify_ir` receives a mutable reference to the IR module, giving an extension full read/write access to every item, field, function argument, and attribute. Extensions can:

- Inject helper functions derived from `#[derive(...)]` attributes.
- Rewrite or remove items.
- Inspect attributes that are preserved on IR nodes but never rendered to WGSL.
- Lower custom statement macros (see [Statement Macro Lowering](#statement-macro-lowering) below).

Anything an extension adds or rewrites carries through every subsequent `instantiate()` call (see [Template Modules & Instantiation](../generics/templates.md)). `TypeParam` nodes in injected code are substituted automatically.

## Wiring

Extensions are activated via the `extensions` argument on the `wgsl` attribute:

```rust
#[wgsl(extensions = [my_crate::NoopExt, my_crate::SlabItemExt])]
mod shader {
    // ...
}
```

Extensions run in declaration order on every `wgsl_source()` and `instantiate()` call. There is no priority mechanism — order is determined solely by the list order in the attribute.

## Minimal Example

A no-op extension useful as a smoke test:

```rust
use wgsl_rs::WgslExtension;
use wgsl_rs::ir;

pub struct NoopExt;

impl WgslExtension for NoopExt {
    fn modify_ir(_module: &mut ir::Module) {}
}
```

For details on walking the IR and the `SlabItemExt` worked example, see [Modifying the IR](./modify-ir.md).

## Statement Macro Lowering

When the `#[wgsl]` parser encounters a statement macro that is not one of its builtins (`slab_copy!`, `discard!`), it passes it through as an `ir::Stmt::Macro { name, args }` variant instead of rejecting it. An extension can then recognize the macro by name in `modify_ir` and replace the `Stmt::Macro` with lowered IR statements.

Extensions declare which macro names they handle via the `MACROS` associated const:

```rust
pub struct SlabItemExt;

impl WgslExtension for SlabItemExt {
    const MACROS: &'static [&'static str] = &["slab_read", "slab_write"];

    fn modify_ir(module: &mut ir::Module) {
        for item in &mut module.items {
            if let ir::Item::Fn(f) = item {
                lower_in_block(&mut f.block);
            }
        }
    }
}
```

The `#[wgsl]` macro emits a compile-time `const` check ensuring every `Stmt::Macro` in the module is claimed by at least one listed extension. If a macro name is used in the module but no listed extension declares it in `MACROS`, the result is a **compile error** (`E0080`), not a runtime error.

Extensions that do not lower statement macros can leave `MACROS` as the empty default.

### Why statement-position only

`#[wgsl]` runs before `macro_rules!` expand, so expression-position macros are untyped black boxes — the parser can only accept ones it recognizes by name and knows the return type of (like `get!`/`get_mut!`). Statement-position macros don't need return types, so they can be passed through as `Stmt::Macro` and lowered by an extension. This is why downstream statement macros (e.g. crabslab's `slab_read!`/`slab_write!`) are statement macros rather than expressions.

For a full worked example, see [Worked Examples](./examples.md).