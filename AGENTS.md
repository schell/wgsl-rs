# wgsl-rs

Rust-to-WGSL transpiler using proc macros.
Workspace with `wgsl-rs`, `wgsl-rs-macros`, `example`, and `xtask` crates.

`wgsl-rs` maintains

## Project Lore

Historical data, insights, affirmations, etc.

### The Two Worlds Problem

A key insight is that `wgsl-rs` maintains two parallel representations:

1. Rust World: The code must compile as valid Rust (design decision #1 in DEVLOG.md) **that runs on the CPU**.
2. WGSL World: The proc-macro transpiles to WGSL **that runs on the GPU**.

These are fundamentally different execution contexts with different memory models, and yet running a `wgsl-rs`
program should produce roughly the same results in both "worlds".

Program setup (or preamble, if you will) and the runtime behavior is expected to be different for each world,
but the results should match, within reason.

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

### xtask - development tools for agents

This repo contains a `cargo xtask` that provides agents with some
shorthand commands for common development tasks.

`cargo xtask wgsl-spec` provides access to the WGSL specification without
overwhelming an agent's context window.

```bash
cargo xtask wgsl-spec toc                        # List WGSL spec table of contents
cargo xtask wgsl-spec section <anchor>           # Fetch a spec section with subsections
cargo xtask wgsl-spec section --shallow <anchor> # Fetch section without subsections
cargo xtask wgsl-spec section <anchor> <sub>     # Fetch a specific subsection
```

## Code Style

- **Imports**: Standard lib and external crates first, then `use crate::` for internal modules
- **Errors**: Use `snafu` with span info for IDE integration (`SomethingWrongSnafu { span, note }.fail()?`)
- **Naming**: PascalCase types, snake_case functions/modules, SCREAMING_SNAKE_CASE constants
- **Patterns**: `TryFrom` for AST conversions, traits per WGSL builtin, macros for repetitive impls
- **Spans**: Preserve `proc_macro2::Span` on all parsed types for error mapping back to Rust source

## DEVLOG.md file

The [DEVLOG](DEVLOG.md) is a set of long-lived development notes.
It is a very informal change-log that also contains thoughts about this library's purpose, requirements, and challenges.

## SESSION.md file

An ephemeral session file is maintained at SESSION.md, which you can use as scratch space for your editing session.
This file is not checked into git and should be used to persist context between editing sessions.
Think of this file as a mini DEVLOG.md, and when the session has ended and goals have been accomplished,
update the [DEVLOG](DEVLOG.md) with a brief, one line summary of the session, then remove the SESSION.md file.
