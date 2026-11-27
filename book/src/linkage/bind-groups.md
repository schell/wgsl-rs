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

`create_named` lets you supply resources by name instead of position.
Resource-pack fields accept their generated friendly keys,
`"Pack::field"`, as well as the mangled names:

```rust
let bg = bg0.create_named("renderer", &device, &layout, &[
    (AtlasBindGroup::SLAB, frame_buffer.as_entire_binding()),
    (AtlasBindGroup::TEXTURE, &albedo_view),
]);
```

## Buffers

`#[uniform]` and `#[storage]` pack fields produce a `BufferDescriptorInfo`
entry that knows the buffer's WGSL size and usage. Find a buffer by its
generated friendly key (preferred) or by its mangled global name:

```rust
let slab_buf_info = linkage
    .buffer(AtlasBindGroup::SLAB)
    .expect("AtlasBindGroup::slab buffer exists");
let slab_buffer = slab_buf_info.create_buffer(&device);
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
let frame_info = linkage.buffer(FramePack::FRAME).unwrap();
let frame_buffer = frame_info.create_buffer(&device);

let frame_bg = linkage.create_bind_group(0, &device, &[
    frame_buffer.as_entire_binding(),
    &albedo_view,
    &linear_sampler,
])?;
```

## Typed bind groups: `#[derive(BindGroup)]`

The string-free endgame: derive `BindGroup` on a pack and pass a **named**
resources struct instead of key/value pairs. The derive works on
`#[wgsl]`-module packs (spell it `#[derive(wgsl_rs_linkage_wgpu::BindGroup)]`
inside the module) and on standalone annotated structs in host crates.

```rust
use wgsl_rs_linkage_wgpu::BindGroup;

#[derive(BindGroup)]
pub struct AtlasBindGroup<const FILTERABLE: bool> {
    #[texture(group(2), binding(1))]
    pub texture: Texture2D<f32, FILTERABLE>,
    #[storage(group(2), binding(0))]
    pub slab: Storage<[Vec4f; 4]>,
    #[sampler(group(2), binding(2))]
    pub smp: Sampler<FILTERABLE>,
}
```

The derive generates `AtlasBindGroupResources<'a>`, one named wgpu resource
per bindable field mapped by annotation kind:

| Annotation | Resources field | wgpu binding |
|------------|-----------------|--------------|
| `#[storage]` / `#[uniform]` | `wgpu::BufferBinding<'a>` | `BindingResource::Buffer` |
| `#[texture]` | `&'a wgpu::TextureView` | `BindingResource::TextureView` |
| `#[sampler]` | `&'a wgpu::Sampler` | `BindingResource::Sampler` |
| `#[workgroup]` | *(skipped - no host resource)* | - |

Creation takes the pack's own group and keys, so no strings appear at the
call site:

```rust
let bind_group = <AtlasBindGroup::<true> as BindGroup>::create_bind_group(
    &mut linkage,
    &device,
    AtlasBindGroupResources {
        slab: slab_buffer.as_entire_buffer_binding(),
        texture: &view,
        smp: &sampler,
    },
    Some("atlas_blit"),
);
```

Because bindings resolve through the field attributes (and their
`binding(N)` numbers), reordering pack fields cannot rebind anything.
Layout-only consts vanish from the generated types: every instantiation
shares one `Resources` struct. One `create_bind_group` call builds one bind
group, so every field must select the same group; split multi-group packs
into one pack per group.

A pack instantiation is not statically tied to the matching module
instantiation's linkage (`AtlasBindGroup::<true>` vs
`instantiate::<true>()`); a mismatched pair panics with the linkage's own
error, and wgpu rejects layout mismatches at `set_bind_group` time.

## Layout caching

`create_bind_group` caches the `wgpu::BindGroupLayout` inside the
`WgpuLinkage` so subsequent calls for the same group index return the cached
layout. `BindGroupInfo::create`/`create_named` take an explicit layout, so
you control whether to cache yourself.
