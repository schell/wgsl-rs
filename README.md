# wgsl-rs

With **wgsl-rs** you write a subset of Rust code and it automatically generates
WGSL shaders and `wgpu` runtime linkage. Rust code written this way is fully
operational (it can be run on the CPU) while the transpiled WGSL is isomorphic
and should generate the same results on the GPU.

In short, with `wgsl-rs`, you can unit test and run your code on the CPU in
Rust, and use the generated WGSL on the GPU, while sharing the same type
definitions between the two.

Procedural macros are provided by the
[`wgsl-rs-macros`](./crates/wgsl-rs-macros) crate.

## Operator's Manual

The canonical user-facing documentation is the
[**Operator's Manual**](./book/src/SUMMARY.md), an mdbook covering installation,
writing shaders, types, generics, validation, the standard library, wgpu
linkage, extensions, memory layout, and a catalog of 35+ runnable examples.

To build the book locally:

```sh
mdbook build book/
```

Or serve it with live reload:

```sh
mdbook serve book/ --open
```

## Roadmap to Beta

There is a project plan for getting to beta
[here](https://github.com/users/schell/projects/3/views/1).

### Can it Hello World?

Yes! See the [example](crates/example/src/main.rs), which transpiles the shader
from [Tour of WGSL](https://google.github.io/tour-of-wgsl/).

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

## Getting Involved

The project is split into a few parts:

- **`wgsl-rs-macros`** — The `wgsl` procedural macro for writing WGSL modules
  in Rust. Handles parsing and code generation for the supported Rust subset.
- **`wgsl-rs`** — The `Source`/`Module` types, `wgsl::std`, the `wgsl` macro
  re-export, extensions, and wgpu linkage.
- **`wgsl-rs-ir`** — The owned IR (`Module`, `Type`, `Expr`, `Stmt`, `Item`),
  `render_module`, and `substitute_types`.
- **`wgsl-rs-layout`** / **`wgsl-rs-layout-macros`** — WGSL memory layout
  computation (`WgslLayout`/`Layout` traits, `#[derive(Layout)]`).

There's also a [devlog](DEVLOG.md) that explains the decisions and tradeoffs
made during development.

Contributions, feedback, and questions are welcome!