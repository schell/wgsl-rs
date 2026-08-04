# Cargo Features

## `wgsl-rs`

| Feature             | Default? | Description                                                                |
|---------------------|----------|----------------------------------------------------------------------------|
| `validation`        | Yes      | Compile-time WGSL validation of generated source                           |
| `dispatch-runtime`  | No       | Enable runtime dispatch support                                            |
| `linkage-wgpu`      | No       | Enable wgpu-based linkage (bind group reflection, pipeline layout)         |

## `wgsl-rs-layout`

| Feature         | Default? | Description                                                       |
|-----------------|----------|-------------------------------------------------------------------|
| `doc-diagrams`  | No       | Enable `generate_svg` for byte-layout SVG diagrams                |

## Usage

```toml
[dependencies]
wgsl-rs = { version = "0.1", default-features = false, features = ["validation", "linkage-wgpu"] }
wgsl-rs-layout = { version = "0.1", features = ["doc-diagrams"] }
```