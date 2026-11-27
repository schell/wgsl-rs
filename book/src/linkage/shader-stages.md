# Per-binding Shader Stages

Each wgpu binding has a `wgpu::ShaderStages` visibility flag. The linkage
analyzer computes this per binding by walking the bodies of the shader's
entry-point functions and collecting which bindings each references.

## How visibility is computed

1. For each entry point (`#[vertex]`, `#[fragment]`, `#[compute]`), the
   analyzer walks the function body (and any called functions) and records
   every binding name that is read or written.
2. A binding's `ShaderStages` is the **union** of the stages of the entry
   points that reference it.
3. Bindings **not referenced by any entry point** default to
   `wgpu::ShaderStages::COMPUTE`.

The default reflects WGSL's own default stage for unreferenced bindings and
keeps the common compute-only case working without annotation.

## Example

Given:

```rust
#[wgsl]
pub mod stages_example {
    use wgsl_rs::std::*;

    pub struct StagesExamplePack {
        #[uniform(group(0), binding(0))]
        pub frame: Uniform<Frame>,
        #[storage(group(0), binding(1), read_only)]
        pub vertices: Storage<[Vertex; 1024]>,
        #[storage(group(0), binding(2), read_write)]
        pub counters: Storage<[u32; 4]>,
    }

    #[vertex]
    pub fn vs(..., pack: &StagesExamplePack) -> VertexOutput {
        let v = get!(pack.vertices)[i];
        // ...
    }

    #[fragment]
    pub fn fs(in: VertexOutput, pack: &StagesExamplePack) -> Vec4f {
        let f = get!(pack.frame).time;
        // ...
    }

    #[compute]
    pub fn cs(..., pack: &StagesExamplePack) {
        get_mut!(pack.counters)[0] += 1;
    }
}
```

Resulting visibility:

| Binding | Referenced by | `ShaderStages` |
|---------|---------------|----------------|
| `stagesexamplepack__frame` | `fs` | `FRAGMENT` |
| `stagesexamplepack__vertices` | `vs` | `VERTEX` |
| `stagesexamplepack__counters` | `cs` | `COMPUTE` |

If a binding is referenced from both vertex and fragment stages, the
visibility is `VERTEX | FRAGMENT`.

## Why this matters: read-write storage in vertex stage

Storage pack fields default to `read_write` access unless their annotation
restricts them.
A `read_write` storage buffer visible to the **vertex** stage requires the
[`wgpu::Features::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING`
family / `VERTEX_WRITABLE_STORAGE`](https://wgpu.rs) feature — specifically
`wgpu::Features::VERTEX_WRITABLE_STORAGE` — because not all GPUs support
writes from the vertex stage.

Because the analyzer derives visibility from actual references, a storage
binding touched only by a compute entry point will not force the
vertex-writable-storage feature on. If you instead set `ShaderStages::all()`
globally, wgpu may reject the pipeline on hardware that lacks
vertex-writable-storage.

The per-binding computation keeps the requested feature set as small as
possible: only bindings actually referenced by a vertex-stage entry point
acquire vertex visibility, and only `read_write` storage among those
requires the feature.

## Overriding visibility

The analyzer's per-binding result is what gets passed to
`wgpu::BindGroupLayoutEntry::visibility`. If you need to widen or narrow
visibility (for example, to bind the same layout across multiple shaders),
build the `BindGroupLayout` yourself with explicit `ShaderStages` and pass it
to `BindGroupInfo::create`/`create_named`. See
[Bind Groups & Buffers](./bind-groups.md).