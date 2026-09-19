# Cargo Features

`wgsl-rs` exposes three cargo features. Only `validation` is on by default.

## `validation` (default)

Enables naga-based validation of generated WGSL. For every non-template `#[wgsl]` module, the macro auto-generates a `#[test] fn __validate_wgsl()` that compiles the emitted `WGSL_SOURCE` through naga and fails on any validation error.

This is the primary safety net: if your Rust module transpiles but the WGSL is malformed, `cargo test` catches it.

Disable it with `default-features = false`:

```toml
[dependencies]
wgsl-rs = { version = "0.1", default-features = false }
```

Use this when you want no naga dependency at all, e.g. in a build that only consumes `WGSL_SOURCE` text and validates downstream.

## `dispatch-runtime`

Enables the CPU-side fragment dispatch runtime at `wgsl_rs::std::runtime`. It lets you run a fragment shader on the CPU over a set of inputs and compare the output against a GPU render. This is the mechanism used for roundtrip testing: the same shader code runs on both sides and the results are diffed.

Enable it explicitly:

```toml
[dependencies]
wgsl-rs = { version = "0.1", features = ["dispatch-runtime"] }
```

Use this when writing tests that exercise fragment shaders without spinning up a full wgpu pipeline, or when debugging shader logic on the CPU.

## `linkage-wgpu`

Enables wgpu linkage generation at `wgsl_rs::linkage::wgpu`. With this feature on, `#[wgsl]` modules emit the metadata needed to build wgpu render/compute pipelines from the generated shader source, including bind group layouts and entry-point descriptors.

Enable it explicitly:

```toml
[dependencies]
wgsl-rs = { version = "0.1", features = ["linkage-wgpu"] }
```

Use this in application crates that render or compute via wgpu. For pure shader authoring and validation it is not needed.

## Combining features

Features compose freely. A typical application crate enables both runtime features:

```toml
[dependencies]
wgsl-rs = { version = "0.1", features = ["dispatch-runtime", "linkage-wgpu"] }
```

A shader-only library crate leaves everything at defaults:

```toml
[dependencies]
wgsl-rs = "0.1"
```

## Reference

| Feature            | Default | Module path                 | When to use                                     |
| ------------------ | ------- | --------------------------- | ----------------------------------------------- |
| `validation`       | on      | (macro-internal, naga)      | Always, unless you strip naga deliberately.     |
| `dispatch-runtime` | off     | `wgsl_rs::std::runtime`     | CPU-side fragment dispatch and roundtrip tests. |
| `linkage-wgpu`     | off     | `wgsl_rs::linkage::wgpu`    | Building wgpu pipelines from shader modules.    |