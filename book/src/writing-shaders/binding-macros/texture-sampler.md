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
        textureSample(ALBEDO, LIN, uv)
    }
}
```

## Passing to Functions

Texture and sampler bindings are passed by value (no `&`) — the visible `NAME` is already a reference:

```rust
pub fn sample_albedo(uv: Vec2f, tex: Texture2D<f32>, smp: Sampler) -> Vec4f {
    textureSample(tex, smp, uv)
}
```

## Notes

- `SamplerComparison` is used with `textureSampleCompare` and `textureSampleCompareLevel` for shadow maps.
- Pair each texture with its sampler; binding numbers must not collide within a group.