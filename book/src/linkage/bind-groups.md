# Bind Groups & Buffers

`WgpuLinkage` exposes the bind groups, bindings, and buffers declared by a
shader via `uniform!`, `storage!`, `workgroup!`, `texture!`, and `sampler!`.

## Bind groups

Each `@group(n)` in WGSL corresponds to a `BindGroupInfo`. Access them by
index or iterate:

```rust
let bg0: &BindGroupInfo = linkage.bind_group(0).expect("group 0 exists");
for (n, bg) in linkage.bind_groups() {
    println!("group {n}: {} bindings", bg.bindings.len());
}
```

`BindGroupInfo` carries:

- The group index.
- The list of bindings (`@binding(i)` entries) with their names, types, and
  visibility (see [Per-binding Shader Stages](./shader-stages.md)).
- A method to create the `wgpu::BindGroupLayout`.

## Creating a bind group

The simplest path: `WgpuLinkage::create_bind_group_named`, which builds the
bind group **and** caches the bind group layout for that group index:

```rust
let frame_bg = linkage.create_bind_group_named(0, &device, &[
    ("FRAME", frame_buffer.as_entire_binding()),
    ("COLOR_TEX", &color_texture_view),
    ("COLOR_SMP", &color_sampler),
])?;
```

Each entry is `("NAME", resource)` where `NAME` is the binding name declared
in the shader (the first argument to `uniform!`/`storage!`/`texture!`/
`sampler!`) and `resource` is a `wgpu::BindingResource`.

For more control, use `BindGroupInfo::create` directly:

```rust
let layout = bg0.create_layout(&device);
let bg = bg0.create(&device, &layout, &[
    frame_buffer.as_entire_binding(),
    &color_texture_view,
    &color_sampler,
]);
```

Resources must be supplied **in binding-index order**, matching the
`@binding(i)` numbering in the generated WGSL.

`create_named` lets you supply resources by name regardless of order:

```rust
let bg = bg0.create_named("renderer", &device, &layout, &[
    ("FRAME", frame_buffer.as_entire_binding()),
    ("COLOR_TEX", &color_texture_view),
]);
```

## Buffers

`uniform!` and `storage!` bindings produce a `BufferDescriptorInfo` entry
that knows the buffer's WGSL size and usage. Find a buffer by its declared
name:

```rust
let frame_buf_info = linkage.buffer("FRAME").expect("FRAME buffer exists");
let frame_buffer = frame_buf_info.create_buffer(&device);
```

`BufferDescriptorInfo::create_buffer(device)` returns a `wgpu::Buffer`
sized per WGSL §14.4.1, with `wgpu::BufferUsages` derived from how the shader
declares the binding (`uniform` vs `storage`, read-only vs read-write).

## Workgroup variables

`workgroup!` bindings do **not** appear in bind groups — they are
workgroup-scoped storage. They do not need wgpu host-side resources.

## Example: full bind group setup

```rust
let frame_info = linkage.buffer("FRAME").unwrap();
let frame_buffer = frame_info.create_buffer(&device);

let frame_bg = linkage.create_bind_group_named(0, &device, &[
    ("FRAME", frame_buffer.as_entire_binding()),
    ("ALBEDO", &albedo_view),
    ("SMP", &linear_sampler),
])?;
```

## Layout caching

`create_bind_group_named` caches the `wgpu::BindGroupLayout` inside the
`WgpuLinkage` so subsequent calls for the same group index return the cached
layout. `BindGroupInfo::create`/`create_named` take an explicit layout, so
you control whether to cache yourself.