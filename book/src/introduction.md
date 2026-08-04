# Introduction

Welcome to the **wgsl-rs** operator's manual. This book is the canonical,
user-facing reference for writing GPU shaders with `wgsl-rs`.

## What is wgsl-rs?

With **wgsl-rs** you write a subset of Rust code and it automatically
generates [WGSL](https://www.w3.org/TR/WGSL/) shaders **and** `wgpu` runtime
linkage. Rust code written this way is fully operational — it can be run on the
CPU — while the transpiled WGSL is isomorphic and should generate the same
results on the GPU.

In short, with `wgsl-rs`, you can unit test and run your code on the CPU in
Rust, and use the generated WGSL on the GPU, while sharing the same type
definitions between the two.

Procedural macros are provided by the
[`wgsl-rs-macros`](https://github.com/schell/wgsl-rs/tree/main/crates/wgsl-rs-macros)
crate.

## The Two Worlds Problem

A key insight that shapes everything about `wgsl-rs` is that it maintains **two
parallel representations** of your shader:

1. **Rust World**: The code must compile as valid Rust **that runs on the CPU**.
   This is design decision #1 in the
   [devlog](https://github.com/schell/wgsl-rs/blob/main/DEVLOG.md).
2. **WGSL World**: The proc-macro transpiles to WGSL **that runs on the GPU**.

These are fundamentally different execution contexts with different memory
models, and yet running a `wgsl-rs` program should produce roughly the same
results in both "worlds".

Program setup (or preamble, if you will) and the runtime behavior is expected
to be different for each world, but the results should match, within reason.
This is why `wgsl-rs` provides CPU-side implementations of every WGSL builtin
in `wgsl_rs::std`, and why the roundtrip test harness exists — to verify that
the two worlds agree.

## wgsl-rs vs Rust-GPU

**Maybe — it depends on your needs.**

### Pros of wgsl-rs

- **Lower barrier to entry:** No custom Rust compiler backend required.
- **Works with stable Rust:** No need for nightly or custom toolchains.
- **Editor support:** The `#[wgsl]` macro makes supported syntax explicit, so
  your editor (via rust-analyzer) can help you write valid code.
- **Immediate WGSL output:** Use, inspect, and debug the generated WGSL anywhere
  WGSL is supported, including browsers and non-Rust projects.
- **Human readable WGSL output:** The WGSL that `wgsl-rs` produces is very close
  in structure to the Rust code you write, including binding names and types.
- **Easy interop:** Generated WGSL can be used in any WebGPU environment.

### Cons of wgsl-rs

- **WGSL only:** Only works on platforms that support WGSL.
- **Limited to WebGPU features:** No support for features not present in WGSL
  (e.g., bindless resources).
- **Subset of Rust:** Only a strict subset of Rust is supported.
  - No traits
  - No borrowing
  - Very restricted module support

> **Note:** wgsl-rs and Rust-GPU are not mutually exclusive!
> You can start with wgsl-rs and switch to Rust-GPU when you need more advanced
> features.

## How to Read This Book

- **New to wgsl-rs?** Start with [Installation](./getting-started/installation.md)
  and [Hello, Triangle](./getting-started/hello-triangle.md).
- **Want to see working code?** Jump to the [Examples](./examples/catalog.md)
  chapter, which catalogs all 35+ example modules with both Rust source and
  generated WGSL.
- **Writing a real renderer?** The [wgpu Linkage](./linkage/overview.md) and
  [Generics & Templates](./generics/templates.md) chapters cover building
  pipelines from generic shaders.
- **Building tooling on top of wgsl-rs?** The [Extensions](./extensions/trait.md)
  and [Memory Layout](./layout/overview.md) chapters are for you.
- **Curious about the *why*?** The [Design Decisions](./design-decisions/highlights.md)
  chapter curates highlights from the devlog; the full
  [DEVLOG.md](https://github.com/schell/wgsl-rs/blob/main/DEVLOG.md) has every
  decision recorded with dates and rationale.

## Project Structure

The project is split into a few parts:

| Crate | Purpose |
|-------|---------|
| **`wgsl-rs`** | The `Module`/`Source` type, `wgsl::std`, the `wgsl` macro re-export, extensions, and wgpu linkage. |
| **`wgsl-rs-ir`** | The owned IR (`Module`, `Type`, `Expr`, `Stmt`, `Item`, etc.), `render_module` (IR → WGSL), and `substitute_types`. |
| **`wgsl-rs-macros`** | The `wgsl` procedural macro — parsing and code generation for the supported Rust subset. |
| **`wgsl-rs-layout`** | `WgslLayout` and `Layout` traits for computing WGSL memory layout (§14.4.1). |
| **`wgsl-rs-layout-macros`** | `#[derive(Layout)]` proc-macro. |
| **`example`** | Runnable example modules demonstrating every supported feature. |
| **`xtask`** | Development tools (`wgsl-spec`, `ci`). |
| **`roundtrip-tests`** | Tests ensuring the "two worlds" (CPU and GPU) agree. |
| **`gpu-tests`** | GPU-side test harness. |

There's also a [devlog](https://github.com/schell/wgsl-rs/blob/main/DEVLOG.md)
that explains some of the decisions and tradeoffs made during the making of this
library.

## Funding

This project is funded through
[NGI Zero Commons](https://nlnet.nl/commonsfund/), a fund established by
[NLnet](https://nlnet.nl) with financial support from the European Commission's
[Next Generation Internet](https://ngi.eu) program. Learn more at the
[2025 NLnet project page](https://nlnet.nl/project/Renderling-Ecosystem/).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)

[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

## Sponsor

This work will always be free and open source. If you use it (outright or for
inspiration), please consider donating.

[💰 Sponsor 💝](https://github.com/sponsors/schell)