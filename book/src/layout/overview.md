# wgsl-rs-layout Overview

`wgsl-rs-layout` computes WGSL memory layout for Rust types, implementing the rules in WGSL spec section 14.4.1 ("Alignment and Size"). It answers a single practical question: **where do bytes go in the GPU buffer?**

## Crates

| Crate                    | Kind     | Purpose                                  |
|--------------------------|----------|------------------------------------------|
| `wgsl-rs-layout`         | lib      | `WgslLayout` and `Layout` traits, built-in type impls, SVG diagram generation |
| `wgsl-rs-layout-macros`  | proc-macro | `#[derive(Layout)]` for user structs    |

## Quick Start

Annotate a struct with `#[derive(Layout)]` and assert its WGSL layout constants:

```rust
use wgsl_rs_layout::{Layout, WgslLayout};

#[derive(Layout)]
struct Particle {
    pos: [f32; 3],
    velocity: [f32; 3],
    charge: f32,
}

fn main() {
    assert_eq!(Particle::SIZE, 32);
    assert_eq!(Particle::ALIGN, 16);
}
```

`SIZE` and `ALIGN` are the WGSL-spec size and alignment of the struct, not the Rust layout. Use `FIELDS` to find each field's offset:

```rust
for field in Particle::FIELDS {
    println!("{:>10} offset={:<3} size={:<3} align={}",
        field.name, field.offset, field.size, field.alignment);
}
```

## What It Is Not

The derive computes the **WGSL** memory layout only. It does **not** align the Rust CPU-side representation. For example, a struct with `#[repr(C)]` and a `Vec3f` field has Rust alignment 4, but WGSL `vec3<f32>` has alignment 16. To use a struct as a staging buffer that matches WGSL layout, you must separately ensure the Rust layout matches (for example by padding fields manually) or use a serialization step that writes bytes per `FieldLayout`.

See [Traits](./traits.md), [Derive](./derive.md), and [Field Layout](./field-layout.md) for the details.