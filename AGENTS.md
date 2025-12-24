# wgsl-rs

Rust-to-WGSL transpiler using proc macros.
Workspace with `wgsl-rs`, `wgsl-rs-macros`, and `example` crates.

## Commands

```bash
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p wgsl-rs-macros   # Test specific crate (wgsl-rs-macros in this case)
cargo test -- test_name        # Run a single test by name
cargo fmt && cargo clippy      # Format and lint
cargo run -p example           # Run the example
cargo expand -p example        # Expand the example which uses the `wgsl` macro, showing the generated WGSL_MODULE
```

## Code Style

- **Imports**: Standard lib and external crates first, then `use crate::` for internal modules
- **Errors**: Use `snafu` with span info for IDE integration (`SomethingWrongSnafu { span, note }.fail()?`)
- **Naming**: PascalCase types, snake_case functions/modules, SCREAMING_SNAKE_CASE constants
- **Patterns**: `TryFrom` for AST conversions, traits per WGSL builtin, macros for repetitive impls
- **Spans**: Preserve `proc_macro2::Span` on all parsed types for error mapping back to Rust source
