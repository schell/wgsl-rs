# Bind Groups & Buffers

`WgpuLinkage` exposes the bind groups, bindings, and buffers declared by a
shader's resource packs: annotated `#[uniform]`, `#[storage]`,
`#[workgroup]`, `#[texture]`, and `#[sampler]` fields.

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

The simplest path: `WgpuLinkage::create_bind_group`, which builds the bind
group **and** caches the bind group layout for that group index:

```rust
let frame_bg = linkage.create_bind_group(0, &device, &[
    frame_buffer.as_entire_binding(),
    &color_texture_view,
    &color_sampler,
])?;
```

Each entry is a `wgpu::BindingResource` supplied **in binding-index order**,
matching the `@binding(i)` numbering in the generated WGSL. Pack field
names are mangled into WGSL globals (`packlower__field`), and binding
positionally keeps those mangled names out of your host code.

For more control, use `BindGroupInfo::create` directly:

```rust
let layout = bg0.create_layout(&device);
let bg = bg0.create(&device, &layout, &[
    frame_buffer.as_entire_binding(),
    &color_texture_view,
    &color_sampler,
]);
```

`create_named` lets you supply resources by name instead of position:

```rust
let bg = bg0.create_named("renderer", &device, &layout, &[
    ("framepack__frame", frame_buffer.as_entire_binding()),
    ("framepack__albedo", &albedo_view),
]);
```

## Buffers

`#[uniform]` and `#[storage]` pack fields produce a `BufferDescriptorInfo`
entry that knows the buffer's WGSL size and usage. Find a buffer by its
mangled global name (the lowercased pack name joined to the field name with
a double underscore):

```rust
let frame_buf_info = linkage
    .buffer("framepack__frame")
    .expect("framepack__frame buffer exists");
let frame_buffer = frame_buf_info.create_buffer(&device);
```

`BufferDescriptorInfo::create_buffer(device)` returns a `wgpu::Buffer`
sized per WGSL §14.4.1, with `wgpu::BufferUsages` derived from how the pack
field declares the binding (`uniform` vs `storage`, read-only vs
read-write).

## Workgroup variables

`#[workgroup]` pack fields do **not** appear in bind groups — they are
workgroup-scoped storage. They do not need wgpu host-side resources.

## Example: full bind group setup

```rust
let frame_info = linkage.buffer("framepack__frame").unwrap();
let frame_buffer = frame_info.create_buffer(&device);

let frame_bg = linkage.create_bind_group(0, &device, &[
    frame_buffer.as_entire_binding(),
    &albedo_view,
    &linear_sampler,
])?;
```

## Layout caching

`create_bind_group` caches the `wgpu::BindGroupLayout` inside the
`WgpuLinkage` so subsequent calls for the same group index return the cached
layout. `BindGroupInfo::create`/`create_named` take an explicit layout, so
you control whether to cache yourself.
