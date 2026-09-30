# Installation

## Prerequisites

- **Rust** 1.87+ (stable toolchain). Install via [rustup](https://rustup.rs/) if you don't already have it.
- **A GPU** with a driver supported by [wgpu](https://wgpu.rs/). Required for roundtrip tests and running example renderers. On macOS, Metal works out of the box.

## Adding wgsl-rs to your project

Add `wgsl-rs` to your `Cargo.toml` with the explicit beta pin — cargo won't
resolve a pre-release version from a bare version requirement:

```sh
cargo add wgsl-rs@0.1.0-beta.1
```

which adds:

```toml
[dependencies]
wgsl-rs = "0.1.0-beta.1"
```

The crate re-exports its proc macros, so you only need the one dependency. If
you prefer to depend on the macro crate directly as well, the equivalent is:

```toml
[dependencies]
wgsl-rs = "0.1.0-beta.1"
wgsl-rs-macros = "0.1.0-beta.1"
```

The `validation` feature is enabled by default and pulls in [naga](https://github.com/gfx-rs/naga) to validate generated WGSL at test time. No extra configuration is required to get it.

## Cargo features

| Feature            | Default | Purpose                                                            |
| ------------------ | ------- | ------------------------------------------------------------------ |
| `validation`       | on      | naga-based WGSL validation; auto-generates `__validate_wgsl` tests. |
| `dispatch-runtime` | off     | CPU-side fragment dispatch runtime for roundtrip testing.          |
| `linkage-wgpu`     | off     | wgpu pipeline/linkage generation from shader modules.              |

See [Cargo Features](./cargo-features.md) for details on each.

## Running the example crate

The repository ships an `example` crate containing 35+ transpiled modules. List them with:

```sh
cargo run -p example -- show
```

Print the generated WGSL for a specific module with:

```sh
cargo run -p example -- source hello_triangle
```

## Verifying your setup

Run the test suite to confirm validation is working:

```sh
cargo test -p example
```

Each `#[wgsl]` module auto-generates a `#[test] fn __validate_wgsl()` that feeds the emitted `WGSL_SOURCE` through naga. If the suite passes, your installation is correct.