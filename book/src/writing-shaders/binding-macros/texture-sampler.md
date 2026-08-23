# `texture!` and `sampler!`

Declare texture and sampler bindings.

## `texture!`

```rust
texture!(group(N), binding(M), NAME: TextureKind<SampleType>);
```

Generates:

```wgsl
@group(N) @binding(M) var NAME: TextureKind<SampleType>;
```

### Texture Kinds

| Kind | Depth variant |
| --- | --- |
| `Texture1D` | |
| `Texture2D` | `TextureDepth2D` |
| `Texture2DArray` | `TextureDepth2DArray` |
| `Texture3D` | |
| `TextureCube` | `TextureDepthCube` |
| `TextureCubeArray` | `TextureDepthCubeArray` |
| `TextureMultisampled2D` | |

The sample type for color textures is typically `<f32>`. Depth textures need no sample type parameter.

### Storage Textures

Storage textures (`texture_storage_*` in WGSL) are declared with `TextureStorage` types. They take two type parameters: a texel format marker and an access mode marker:

```rust
texture!(group(0), binding(0), OUTPUT: TextureStorage2D<Rgba8unorm, Write>);
```

| Kind | WGSL |
|------|------|
| `TextureStorage1D<F, A>` | `texture_storage_1d<format, access>` |
| `TextureStorage2D<F, A>` | `texture_storage_2d<format, access>` |
| `TextureStorage2DArray<F, A>` | `texture_storage_2d_array<format, access>` |
| `TextureStorage3D<F, A>` | `texture_storage_3d<format, access>` |

Texel format markers (unit structs implementing `WgslTexelFormat`):

| Marker | WGSL format | Value type |
|--------|------------|------------|
| `Rgba8unorm` | `rgba8unorm` | `Vec4f` |
| `Rgba8uint` | `rgba8uint` | `Vec4u` |
| `Rgba16float` | `rgba16float` | `Vec4f` |
| `R32uint` | `r32uint` | `Vec4u` |
| `R32float` | `r32float` | `Vec4f` |
| `Rg32float` | `rg32float` | `Vec4f` |
| `Rgba32float` | `rgba32float` | `Vec4f` |
| `Bgra8unorm` | `bgra8unorm` | `Vec4f` |

Access mode markers:

| Marker | WGSL | Description |
|--------|------|-------------|
| `Read` | `read` | Read-only (load with `texture_load_storage`) |
| `Write` | `write` | Write-only (store with `texture_store`) |
| `ReadWrite` | `read_write` | Both (requires `texture_formats_tier1`) |

> The `enable texture_formats_tier1;` directive is auto-hoisted to the start of the assembled WGSL translation unit when storage textures are present — you don't need to add it manually.

## `sampler!`

```rust
sampler!(group(N), binding(M), NAME: Sampler);
sampler!(group(N), binding(M), NAME: SamplerComparison);
```

Generates:

```wgsl
@group(N) @binding(M) var NAME: sampler;
@group(N) @binding(M) var NAME: sampler_comparison;
```

## Two-Level Binding

Both macros produce a hidden `__NAME` static plus a visible `pub const NAME: &'static ...` so the binding can be passed by value. You reference `NAME` directly in functions:

```rust
#[wgsl]
pub mod texturing {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), ALBEDO: Texture2D<f32>);
    sampler!(group(0), binding(1), LIN: Sampler);

    #[fragment]
    pub fn fs_main(
        #[location(0)] uv: Vec2f,
    ) -> Vec4f {
        texture_sample(ALBEDO, LIN, uv)
    }
}
```

## Passing to Functions

Texture and sampler bindings are passed by value (no `&`) — the visible `NAME` is already a reference:

```rust
pub fn sample_albedo(uv: Vec2f, tex: Texture2D<f32>, smp: Sampler) -> Vec4f {
    texture_sample(tex, smp, uv)
}
```

## Notes

- `SamplerComparison` is used with `textureSampleCompare` and `textureSampleCompareLevel` for shadow maps.
- Pair each texture with its sampler; binding numbers must not collide within a group.