# IR Attributes

The IR preserves Rust attributes on every node where they appear, making them available for extension inspection.

## The `ir::Attribute` Struct

```rust
pub struct Attribute {
    pub path: String,
    pub args: Vec<String>,
}
```

Each attribute is decomposed into a path and a list of argument strings:

| Rust attribute                          | `path`    | `args`                |
|-----------------------------------------|-----------|-----------------------|
| `#[derive(SlabItem, Clone)]`            | `derive`  | `["SlabItem", "Clone"]` |
| `#[repr(C)]`                            | `repr`    | `["C"]`               |
| `#[inline]`                             | `inline`  | `[]`                  |

## Where Attributes Live

Attributes are preserved on:

- Every `ir::Item` (struct, fn, const, etc.)
- Each `ir::Field` within a struct
- Each `ir::FnArg` within a function
- The `ir::Module` itself

## Never Rendered to WGSL

Attributes exist solely for extension inspection. They are **never** emitted into the final WGSL source. This means an extension can stash metadata on items via attributes and trust that it will not leak into shader output.

## Filtering on Attributes

The common pattern is to find items carrying a specific derive:

```rust
let slab_structs: Vec<&ir::Item> = module
    .items
    .iter()
    .filter(|item| matches!(item, ir::Item::Struct(s) if s.attrs.iter().any(|a| {
        a.path == "derive" && a.args.iter().any(|arg| arg == "SlabItem")
    })))
    .collect();
```

## Intentional Duplication

Some attribute information also appears in dedicated IR fields such as `FnAttrs` and `InterStageIo`. This duplication is intentional: those dedicated fields drive WGSL rendering of entry-point decorators (`@vertex`, `@location`, etc.), while the raw `Attribute` list is preserved verbatim for extensions that want the unfiltered Rust-level view.