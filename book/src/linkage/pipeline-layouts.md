# Pipeline Layouts

`WgpuLinkage` builds and caches the `wgpu::PipelineLayout` from the bind group
layouts it derives from the shader. Combined with `EntryPointInfo`, you can
create a complete render or compute pipeline without manually writing layout
descriptors.

## `pipeline_layout`

```rust
let layout: wgpu::PipelineLayout = linkage.pipeline_layout(&mut linkage, &device, Some("main"));
```

- Takes `&mut self` because the result is cached.
- Returns an owned `wgpu::PipelineLayout` (internally `Arc`-backed, so cloning
  is cheap).
- The label argument is passed straight through to wgpu.

Calling `pipeline_layout` again returns a clone of the cached layout.

## Entry points

`linkage.entry_point(name)` returns `Option<&EntryPointInfo>`. The name is the
Rust function name of your `#[vertex]` / `#[fragment]` / `#[compute]` entry
point.

`EntryPointInfo` exposes the wgpu descriptor builders:

| Method | Description |
|--------|-------------|
| `vertex_state(device, buffers, constants)` | `wgpu::VertexState` for this entry point. |
| `fragment_state(device, targets, constants)` | `wgpu::FragmentState`. |
| `compute_state(device, constants)` | `wgpu::ComputeState` (the `module`/`entry_point` pair). |
| `stage()` | The `wgpu::ShaderStages` flag. |
| `name()` | The WGSL entry-point name string. |

The `*_state` builders take the same trailing arguments as the corresponding
wgpu descriptor fields (vertex buffer layouts, color targets, pipeline
constants), so you keep full control over the pipeline descriptor while the
linkage supplies the `module` and `entry_point`.

## Full pipeline creation

```rust
use wgsl_rs::*;

let mut linkage = my_shader::SOURCE.generate_linkage()?;
let shader = linkage.shader_module(&device);

let frame_bg = linkage.create_bind_group_named(0, &device, &[
    ("FRAME", frame_buffer.as_entire_binding()),
    ("ALBEDO", &albedo_view),
    ("SMP", &linear_sampler),
])?;

let pipeline_layout = linkage.pipeline_layout(&mut linkage, &device, Some("main_layout"));

let vs_info = linkage.entry_point("vs").expect("vertex entry point");
let fs_info = linkage.entry_point("fs").expect("fragment entry point");

let pipeline = device.create_render_pipeline(wgpu::RenderPipelineDescriptor {
    label: Some("main_pipeline"),
    layout: Some(&pipeline_layout),
    vertex: vs_info.vertex_state(&device, &vertex_buffer_layouts, &[]),
    fragment: Some(fs_info.fragment_state(&device, &[Some(wgpu::TextureFormat::Bgra8Unorm.into())], &[])),
    primitive: wgpu::PrimitiveState::default(),
    depth_stencil: None,
    multisample: wgpu::MultisampleState::default(),
    multiview: None,
});
```

## Compute pipelines

For compute, use `EntryPointInfo::compute_state`:

```rust
let cs_info = linkage.entry_point("main").expect("compute entry point");
let pipeline = device.create_compute_pipeline(wgpu::ComputePipelineDescriptor {
    label: Some("compute"),
    layout: Some(&pipeline_layout),
    module: &shader,
    entry_point: cs_info.name(),
    compilation_options: Default::default(),
    cache: None,
});
```

`compute_state` returns a small struct with `module` and `entry_point` fields
so you can spread them into the descriptor yourself; `name()` is available
when you want just the string.