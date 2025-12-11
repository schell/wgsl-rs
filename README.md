# wgsl-rs

**wgsl-rs** lets you write a subset of Rust code and automatically generate WGSL shaders and `wgpu` runtime linkage. With
`wgsl-rs`, you can run your code on the CPU in Rust, and use the generated WGSL on the GPU.

Procedural macros are provided by the [`wgsl-rs-macros`](./crates/wgsl-rs-macros) crate.

---

## Roadmap

- [x] `wgsl` macro for modules
  - [x] Translate a subset of Rust into WGSL
    - [x] Types
      - [x] Concrete scalars
      - [x] `vec{N}<{scalar}>`
      - [x] Vector aliases
      - [x] Arrays
      - [ ] Matrices
      - [ ] Structs
      - [ ] Textures
      - [ ] Atomics
    - [x] Descriptor sets, bindings, etc.
    - [ ] Expand support for all WGSL syntax
  - [x] Glob-importing other `wgsl-rs` modules
  - [x] WGSL standard library for Rust
    - [x] Vector constructors (`vec2f`, `vec3f`, etc.)
    - [x] Binary operators (`+`, `-`, `*`, `/`, etc.)
  - [ ] Generate linkage info
  - [ ] Generate `wgpu` linkage (possibly via a separate crate like `wgsl-rs-wgpu`)

### Can it Hello World?

Yes! See the [example](crates/example/src/main.rs), which transpiles the shader from
[Tour of WGSL](https://google.github.io/tour-of-wgsl/).

---

## Should I use Rust-GPU instead?

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
- **Limited to WebGPU features:** No support for features not present in WGSL (e.g., bindless resources).
- **Subset of Rust:** Only a strict subset of Rust is supported.
  - No traits
  - No borrowing
  - Very restricted module support

> **Note:** wgsl-rs and Rust-GPU are not mutually exclusive!
  You can start with wgsl-rs and switch to Rust-GPU when you need more advanced features.
  I'm working on making them co-habitable.

---

## Getting Involved

The project is split into a few parts:

- **`wgsl-rs-macros`**
  Provides the `wgsl` procedural macro for writing WGSL modules in Rust. Handles
  parsing and code generation for the supported Rust subset.
- **`wgsl-rs`**
  Provides the `Module` type, `wgsl::std`, and exports the `wgsl` macro.

There's also a [devlog](DEVLOG) that explains some of the decisions and tradeoffs made during the making
of this library.

---

Contributions, feedback, and questions are welcome!
