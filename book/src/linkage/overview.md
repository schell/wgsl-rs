# wgpu Linkage Overview

`wgpu` linkage is the bridge between a `#[wgsl]` module and a running
[wgpu](https://github.com/gfx-rs/wgpu) pipeline. Given a shader module, the
linkage analyzer produces everything wgpu needs to create buffers, bind
groups, and pipelines, with sizes computed per WGSL §14.4.1 (not Rust's
`sizeof`).

## Enabling linkage

Linkage is gated behind the `linkage-wgpu` cargo feature:

```toml
[dependencies]
wgsl-rs = { version = "0.1", features = ["linkage-wgpu"] }
```

See [Cargo Features](../getting-started/cargo-features.md).

## `WgpuLinkage`

`WgpuLinkage` is the main type. It owns the concrete IR it was built from and
renders the WGSL source via `ir::render_module`. You get one of two ways:

```rust
use wgsl_rs::linkage::wgpu::{analyze_wgsl_module, analyze_ir_module, WgpuLinkage};

// From a concrete (non-template) Source:
let linkage: WgpuLinkage = analyze_wgsl_module(&my_shader::SOURCE)?;

// From an instantiated template's IR (see [Template Modules & Instantiation](../generics/templates.md)):
let concrete_ir: ir::Module = template.instantiate::<f32, u32>()?;
let linkage: WgpuLinkage = analyze_ir_module(concrete_ir);
```

`Source` and `ir::Module` both expose `generate_linkage()` — an extension
trait method (`IrModuleExt`) re-exported via `wgsl_rs::*`:

```rust
use wgsl_rs::*;
let linkage = my_shader::SOURCE.generate_linkage()?;
```

## What `WgpuLinkage` provides

| Method | Returns | Description |
|--------|---------|-------------|
| `wgsl_source()` | `String` | The rendered WGSL text. |
| `shader_module(device)` | `wgpu::ShaderModule` | Compiled shader module (no `&Module` arg needed). |
| `entry_point(name)` | `Option<&EntryPointInfo>` | Vertex/fragment/compute entry point info. |
| `bind_group(n)` | `Option<&BindGroupInfo>` | Bind group at `@group(n)`. |
| `bind_groups()` | `&HashMap<u32, BindGroupInfo>` | All bind groups. |
| `create_bind_group_named(group, device, resources)` | `wgpu::BindGroup` | Creates a bind group, caching the layout. |
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
use wgsl_rs::*;

let mut linkage = my_shader::SOURCE.generate_linkage()?;
let shader = linkage.shader_module(&device);

let frame_bg = linkage.create_bind_group_named(0, &device, &[
    ("FRAME", frame_buffer.as_entire_binding()),
])?;

let layout = linkage.pipeline_layout(&mut self, &device, Some("main_layout"));
let pipeline = device.create_render_pipeline(wgpu::RenderPipelineDescriptor {
    label: Some("main"),
    layout: Some(&layout),
    vertex: linkage.entry_point("vs").unwrap().vertex_state(&device, &[], &[]),
    fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: linkage.entry_point("fs").unwrap().name(),
        targets: &[Some(wgpu::TextureFormat::Bgra8Unorm.into())],
    }),
    primitive: wgpu::PrimitiveState::default(),
    depth_stencil: None,
    multisample: wgpu::MultisampleState::default(),
    multiview: None,
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