# wgpu Linkage Overview

`wgpu` linkage is the bridge between a `#[wgsl]` module and a running
[wgpu](https://github.com/gfx-rs/wgpu) pipeline. Given a shader module, the
linkage analyzer produces everything wgpu needs to create buffers, bind
groups, and pipelines, with sizes computed per WGSL §14.4.1 (not Rust's
`sizeof`).

## Enabling linkage

Linkage lives in the `wgsl-rs-linkage-wgpu` crate, separate from
`wgsl-rs` (which stays wgpu-free). The default features give you the
latest supported wgpu major, exposed at the crate root:

```toml
[dependencies]
wgsl-rs = "0.1"
wgsl-rs-linkage-wgpu = "0.1"
```

Pinned to an older wgpu major? Disable the default and enable the
matching module:

```toml
[dependencies]
wgsl-rs-linkage-wgpu = { version = "0.1", default-features = false, features = ["v29"] }
```

```rust
// Latest: everything at the crate root.
use wgsl_rs_linkage_wgpu::{analyze_wgsl_module, analyze_ir_module, WgpuLinkage};

// Pinned: the same API under the matching module.
use wgsl_rs_linkage_wgpu::v29::analyze_wgsl_module;
```

Both majors may be enabled at once (feature unification); the modules
coexist with distinct, incompatible wgpu types, and the crate root
re-exports the latest enabled major.

See [Cargo Features](../getting-started/cargo-features.md).

## `WgpuLinkage`

`WgpuLinkage` is the main type. It owns the concrete IR it was built from and
renders the WGSL source via `ir::render_module`. Every `#[wgsl]` module ships
an `instantiate()` constructor returning its fully assembled IR — imports
included — and `generate_linkage()` turns that IR into a linkage:

```rust
use wgsl_rs_linkage_wgpu::*;

// A concrete (non-template) module:
let linkage = my_shader::instantiate().generate_linkage();

// A template module (see [Template Modules & Instantiation](../generics/templates.md)):
let linkage = my_shader::instantiate::<f32, u32>().generate_linkage();

// Or hand a concrete Source straight to the analyzer —
// `analyze_wgsl_module` is `instantiate()` + analysis in one call:
let linkage = analyze_wgsl_module(&my_shader::WGSL_SOURCE)?;
```

`generate_linkage()` is an extension-trait method (`IrModuleExt`) exported
from the linkage crate's root; it is the shorthand for `analyze_ir_module`.
Build the IR you hand it with the macro-emitted `instantiate()`, never with
`WGSL_SOURCE.ir_constructor` (own items only — imports silently drop).

## What `WgpuLinkage` provides

| Method | Returns | Description |
|--------|---------|-------------|
| `wgsl_source()` | `String` | The rendered WGSL text. |
| `shader_module(device)` | `wgpu::ShaderModule` | Compiled shader module (no `&Module` arg needed). |
| `vertex_entry(name)` | `Option<&EntryPointInfo>` | Vertex entry point info. |
| `fragment_entry(name)` | `Option<&EntryPointInfo>` | Fragment entry point info. |
| `compute_entry(name)` | `Option<&ComputeEntryInfo>` | Compute entry point info (workgroup size). |
| `bind_group(n)` | `Option<&BindGroupInfo>` | Bind group at `@group(n)`. |
| `bind_groups` | `&BTreeMap<u32, BindGroupInfo>` | All bind groups. |
| `create_bind_group_named(group, device, resources)` | `Result<wgpu::BindGroup, Error>` | Creates a bind group by binding names, caching the layout. |
| `pipeline_layout(&mut self, device, label)` | `wgpu::PipelineLayout` | Pipeline layout, lazily cached. |
| `buffer(name)` | `Option<&BufferDescriptorInfo>` | Find a buffer by its binding name. |

See [Bind Groups & Buffers](./bind-groups.md) and
[Pipeline Layouts](./pipeline-layouts.md) for detail.

## High-level workflow

1. Write your shader in a `#[wgsl]` module.
2. Build the `WgpuLinkage` (via `generate_linkage()` or `analyze_*`).
3. Create the `wgpu::ShaderModule` from the linkage.
4. Create the buffers and bind groups you need.
5. Create the pipeline layout (cached) and the render/compute pipeline.

```rust
use wgsl_rs_linkage_wgpu::*;

let mut linkage = analyze_wgsl_module(&my_shader::WGSL_SOURCE)?;
let shader = linkage.shader_module(&device);

let frame_bg = linkage.create_bind_group_named(0, &device, &[
    ("FRAME", frame_buffer.as_entire_binding()),
])?;

let layout = linkage.pipeline_layout(&device, Some("main_layout"));
let pipeline = device.create_render_pipeline(wgpu::RenderPipelineDescriptor {
    label: Some("main"),
    layout: Some(&layout),
    vertex: linkage.vertex_entry("vs_main").unwrap().vertex_state(&shader),
    fragment: Some(
        linkage.fragment_entry("fs_main").unwrap().fragment_state(
            &shader,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::all(),
            })],
        ),
    ),
    primitive: wgpu::PrimitiveState::default(),
    depth_stencil: None,
    multisample: wgpu::MultisampleState::default(),
    multiview_mask: None,
    cache: None,
});
```

## Sizing

Buffer and binding sizes are computed from the IR using WGSL §14.4.1
alignment and size rules, **not** Rust's `size_of`. This means a Rust struct
with padding or a different layout from its WGSL equivalent still gets the
right wgpu binding size. See [Memory Layout](../layout/overview.md) for the
underlying trait machinery.

## Layout caching

`WgpuLinkage` lazily caches `wgpu::BindGroupLayout`s (per group index) and the
`wgpu::PipelineLayout`. Methods that create these take `&mut self`; the
returned wgpu types are `Arc`-backed, so cloning is cheap. You typically keep
one `WgpuLinkage` per shader and call the `&mut self` methods once at
pipeline-construction time.

## Migrating from the `linkage-wgpu` feature

Earlier betas shipped linkage inside `wgsl-rs` behind a `linkage-wgpu`
cargo feature. It now lives in its own crate so users on different wgpu
majors can each get matching types. To migrate:

1. Remove the `linkage-wgpu` feature from your `wgsl-rs` dependency.
2. Add a dependency on `wgsl-rs-linkage-wgpu` (default features = the
   latest wgpu major; enable the `vNN` feature matching your pinned
   major otherwise).
3. Replace `wgsl_rs::linkage::wgpu::X` with `wgsl_rs_linkage_wgpu::X`
   (or `wgsl_rs_linkage_wgpu::vNN::X` when pinned).

`wgsl-rs` itself no longer depends on wgpu at all; everything else
(`#[wgsl]`, `std`, dispatch, validation) is unchanged.