# Texture & Sampler Functions

`wgsl_rs::std` provides the WGSL texture and sampler builtins plus the
`sampler!` and `texture!` binding macros used to declare them.

## Sampler types

| Type | WGSL | Description |
|------|------|-------------|
| `Sampler` | `sampler` | Filtering sampler. |
| `SamplerComparison` | `sampler_comparison` | Comparison sampler for shadow/PCF sampling. |

See [Binding Macros: `texture!` & `sampler!`](../writing-shaders/binding-macros/texture-sampler.md)
for declaration syntax.

## Sampling functions

Each function has multiple overloads (2D, 2DArray, 3D, Cube, CubeArray, etc.),
implemented as separate Rust functions that map to the same WGSL builtin. The
transpiler picks the right overload from argument types.

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `textureSample(tex, sampler, coords)` | `textureSample` | Filtered sample. Fragment stage only. |
| `textureSample(tex, sampler, coords, offset)` | `textureSample` | With integer texel offset. |
| `textureSampleLevel(tex, sampler, coords, level)` | `textureSampleLevel` | Explicit mip level. Any stage. |
| `textureSampleLevel(tex, sampler, coords, level, offset)` | `textureSampleLevel` | With offset. |
| `textureSampleBias(tex, sampler, coords, bias)` | `textureSampleBias` | Adds mip bias. Fragment stage only. |
| `textureSampleBias(tex, sampler, coords, bias, offset)` | `textureSampleBias` | With offset. |
| `textureSampleGrad(tex, sampler, coords, ddx, ddy)` | `textureSampleGrad` | Explicit gradients. Any stage. |
| `textureSampleGrad(tex, sampler, coords, ddx, ddy, offset)` | `textureSampleGrad` | With offset. |
| `textureSampleCompare(tex, sampler, coords, ref)` | `textureSampleCompare` | Depth comparison. Fragment stage only. |
| `textureSampleCompareLevel(tex, sampler, coords, ref)` | `textureSampleCompareLevel` | Depth comparison, uniform level. Any stage. |
| `textureSampleBaseClampToEdge(tex, sampler, coords)` | `textureSampleBaseClampToEdge` | Sample with coords clamped to `[0,1]`. Any stage. |

> Functions suffixed `...Level` are usable from any stage; plain
> `textureSample` and `textureSampleBias`/`textureSampleCompare` are
> restricted to the fragment stage in WGSL.

## Load / store / query

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `textureLoad(tex, coords)` | `textureLoad` | Load texel at integer coords. |
| `textureLoad(tex, coords, level)` | `textureLoad` | With mip level (2D/3D/array). |
| `textureLoad(tex, coords, sample)` | `textureLoad` | Multisample load. |
| `textureStore(tex, coords, value)` | `textureStore` | Write texel (storage textures). |
| `textureDimensions(tex)` | `textureDimensions` | Dimensions at mip 0. |
| `textureDimensions(tex, level)` | `textureDimensions` | Dimensions at given mip level. |
| `textureNumLayers(tex)` | `textureNumLayers` | Array layer count. |
| `textureNumLevels(tex)` | `textureNumLevels` | Mip level count. |
| `textureNumSamples(tex)` | `textureNumSamples` | Sample count (multisample). |

## Example

```rust
#[wgsl]
pub mod texture_example {
    use wgsl_rs::std::*;

    uniform!(FRAME, Frame);
    texture!(COLOR_TEX, Texture2d);
    sampler!(COLOR_SMP, Sampler);

    pub struct Frame {
        pub time: f32,
        pub resolution: Vec2f,
    }

    #[fragment]
    pub fn fs(in: VertexOutput) -> Vec4f {
        let uv = in.uv;
        let color = textureSample(COLOR_TEX, COLOR_SMP, uv);
        let dim = textureDimensions(COLOR_TEX);
        color * frame.time
    }
}
```

## Overload resolution

Because Rust has no WGSL-style overload sets, each texture builtin is
implemented as a distinct Rust function per texture kind, e.g. there is one
`textureSample` for `Texture2d`, another for `Texture2dArray`, another for
`Texture3d`, another for `TextureCube`, and so on. The transpiler emits the
WGSL `textureSample` builtin regardless of which Rust overload you called —
the overload exists only to type-check on the CPU side.