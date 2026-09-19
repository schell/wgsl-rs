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
[**Operator's Manual**](https://renderling.xyz/wgsl-rs/manual/index.html) —
an mdbook covering installation, writing shaders, types, generics, validation,
the standard library, wgpu linkage, extensions, memory layout, and a catalog of
35+ runnable examples. The book's sources live in [`book/`](./book) in this
repo.

New to `wgsl-rs`? Start with
[Installation](https://renderling.xyz/wgsl-rs/manual/getting-started/installation.html)
and [Hello, Triangle](https://renderling.xyz/wgsl-rs/manual/getting-started/hello-triangle.html),
or browse the [examples catalog](https://renderling.xyz/wgsl-rs/manual/examples/catalog.html).

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

Yes! This is the canonical `hello_triangle` shader — ordinary Rust that the
`#[wgsl]` macro transpiles to WGSL. It is also valid Rust you can compile,
unit test, and run on the CPU:

```rust
#[wgsl]
pub mod hello_triangle {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];
        let position = POS[vertex_index as usize];
        vec4f(position.x, position.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(get!(FRAME)) / 128.0), 0.0, 1.0)
    }
}
```

See the runnable [example](crates/example/src/main.rs) (transpiled from
[Tour of WGSL](https://google.github.io/tour-of-wgsl/)), or the manual's
[Hello, Triangle](https://renderling.xyz/wgsl-rs/manual/getting-started/hello-triangle.html)
chapter for a line-by-line walkthrough.

## Highlights

Beyond the basics, `wgsl-rs` supports the features real renderers need:

- **Generics** — write generic shader entry points in Rust and
  instantiate them with turbofish at runtime; monomorphization produces concrete WGSL.
  Const generics and generic structs and impls work too.
  ([templates](https://renderling.xyz/wgsl-rs/manual/generics/templates.html))
- **Binding macros** — `uniform!`, `storage!`, `workgroup!`, `texture!`,
  `sampler!` declare bindings once, visible in both Rust and WGSL.
  ([binding macros](https://renderling.xyz/wgsl-rs/manual/writing-shaders/binding-macros.html))
- **Auto validation** — `#[wgsl]` modules get a hidden
  [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) validation test; for
  template modules the test instantiates the template with the types given to
  `validate_with_instantiation_types(T1, T2, ...)` before validating. Either
  way, `cargo test` catches invalid WGSL before it reaches the GPU.
  ([validation](https://renderling.xyz/wgsl-rs/manual/validation/auto-tests.html))
- **wgpu linkage** — generated modules provide bind group layouts, stage
  visibility and pipeline layout helpers, so wiring up `wgpu` requires no
  boilerplate.
  ([linkage](https://renderling.xyz/wgsl-rs/manual/linkage/overview.html))
- **Extensions** — implement `WgslExtension` to hook the IR directly: custom
  attributes, new statement macros, post-processing passes.
  ([extensions](https://renderling.xyz/wgsl-rs/manual/extensions/trait.html))
- **Memory layout** — `WgslLayout` and `#[derive(Layout)]` compute
  WGSL-conformant layouts so CPU structs match GPU buffers exactly.
  ([layout](https://renderling.xyz/wgsl-rs/manual/layout/overview.html))

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
- **`example`** — Runnable example modules demonstrating every supported
  feature.
- **`xtask`** — Development tools (`wgsl-spec`, `ci`).
- **`roundtrip-tests`** — Tests ensuring the "two worlds" (CPU and GPU)
  agree.
- **`gpu-tests`** — GPU-side test harness.

There's also a [devlog](DEVLOG.md) that explains the decisions and tradeoffs
made during development.

Contributions, feedback, and questions are welcome!
