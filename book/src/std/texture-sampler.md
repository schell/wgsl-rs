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
| `texture_sample(tex, sampler, coords)` | `textureSample` | Filtered sample. Fragment stage only. |
| `texture_sample(tex, sampler, coords, offset)` | `textureSample` | With integer texel offset. |
| `texture_sample_level(tex, sampler, coords, level)` | `textureSampleLevel` | Explicit mip level. Any stage. |
| `texture_sample_level(tex, sampler, coords, level, offset)` | `textureSampleLevel` | With offset. |
| `texture_sample_bias(tex, sampler, coords, bias)` | `textureSampleBias` | Adds mip bias. Fragment stage only. |
| `texture_sample_bias(tex, sampler, coords, bias, offset)` | `textureSampleBias` | With offset. |
| `texture_sample_grad(tex, sampler, coords, ddx, ddy)` | `textureSampleGrad` | Explicit gradients. Any stage. |
| `texture_sample_grad(tex, sampler, coords, ddx, ddy, offset)` | `textureSampleGrad` | With offset. |
| `texture_sample_compare(tex, sampler, coords, ref)` | `textureSampleCompare` | Depth comparison. Fragment stage only. |
| `texture_sample_compare_level(tex, sampler, coords, ref)` | `textureSampleCompareLevel` | Depth comparison, uniform level. Any stage. |
| `texture_sample_base_clamp_to_edge(tex, sampler, coords)` | `textureSampleBaseClampToEdge` | Sample with coords clamped to `[0,1]`. Any stage. |

> Functions suffixed `..._level` are usable from any stage; plain
> `texture_sample` and `texture_sample_bias`/`texture_sample_compare` are
> restricted to the fragment stage in WGSL.

## Load / store / query

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `texture_load(tex, coords)` | `textureLoad` | Load texel at integer coords. |
| `texture_load(tex, coords, level)` | `textureLoad` | With mip level (2D/3D/array). |
| `texture_load(tex, coords, sample)` | `textureLoad` | Multisample load. |
| `texture_store(tex, coords, value)` | `textureStore` | Write texel (storage textures). |
| `texture_dimensions(tex)` | `textureDimensions` | Dimensions at mip 0. |
| `texture_dimensions(tex, level)` | `textureDimensions` | Dimensions at given mip level. |
| `texture_num_layers(tex)` | `textureNumLayers` | Array layer count. |
| `texture_num_levels(tex)` | `textureNumLevels` | Mip level count. |
| `texture_num_samples(tex)` | `textureNumSamples` | Sample count (multisample). |

## Example

```rust
#[wgsl]
pub mod texture_example {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), COLOR_TEX: Texture2D<f32>);
    sampler!(group(0), binding(1), COLOR_SMP: Sampler);

    pub struct FragmentInput {
        #[location(0)]
        pub uv: Vec2f,
    }

    #[fragment]
    pub fn fs_main(input: FragmentInput) -> Vec4f {
        let uv = input.uv;
        let color = texture_sample(COLOR_TEX, COLOR_SMP, uv);
        let dim = texture_dimensions(COLOR_TEX);
        color
    }
}
```

## Overload resolution

Because Rust has no WGSL-style overload sets, each texture builtin is
implemented as a distinct Rust function per texture kind, e.g. there is one
`texture_sample` for `Texture2D`, another for `Texture2DArray`, another for
`Texture3D`, another for `TextureCube`, and so on. The transpiler emits the
WGSL `textureSample` builtin regardless of which Rust overload you called —
the overload exists only to type-check on the CPU side.