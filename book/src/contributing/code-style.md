# Code Style

## Imports

Order imports in two groups, separated by a blank line:

1. Standard library and external crates.
2. `crate::` items.

Within each group, keep imports alphabetical.

```rust
use std::collections::BTreeMap;

use proc_macro2::Span;
use snafu::Snafu;

use crate::ir::Module;
```

## Error Handling

Use `snafu` for errors. Every error variant carries span information so diagnostics point at the originating source location.

```rust
#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(context(false))]
    UnknownType { span: Span, name: String },
}
```

Preserve `proc_macro2::Span` through every AST conversion so errors can report the user's original tokens.

## Naming

| Element      | Convention              | Example            |
|--------------|-------------------------|--------------------|
| Types        | PascalCase              | `WgslExtension`    |
| Functions    | snake_case              | `modify_ir`        |
| Modules      | snake_case              | `ir`               |
| Constants    | SCREAMING_SNAKE_CASE    | `SIZE`, `ALIGN`    |

## Patterns

- Use `TryFrom` / `TryInto` for AST-to-IR conversions so failures carry span context.
- Define one trait per WGSL builtin that has overloaded signatures, rather than one trait with variadic generics.
- Prefer splitting a file into a submodule over adding section-header comments.

## Spans

Never discard `proc_macro2::Span`. Thread it through conversions and store it on IR nodes so that validation and linkage errors can point at the user's source.

## Comments & Modules

- Document every public function with a doc comment.
- Prefer a separate module over a `// === Section ===` header inside a large file.
- Keep non-doc comments sparse; let types and function names carry intent.