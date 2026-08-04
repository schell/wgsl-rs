# The `WgslExtension` Trait

The `WgslExtension` trait lets downstream crates inspect and modify a shader's WGSL IR after transpilation but before type instantiation. It is the primary extension point for post-transpile code generation and analysis.

## Definition

The trait lives in `wgsl_rs::extension` and is re-exported at the crate root.

```rust
pub trait WgslExtension {
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