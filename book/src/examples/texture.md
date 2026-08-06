# Texture

Demonstrates textures, samplers, and the `textureSample` builtin. Covers the `texture!` and `sampler!` macros and a fragment shader that samples a 2D texture.

## Rust Source

```rust
#[wgsl]
pub mod texture_example {
    //! Demonstrates using textures and texture builtin functions.
    //!
    //! WGSL provides several categories of texture operations:
    //! - **Query functions**: `textureDimensions`, `textureNumLayers`, etc.
    //! - **Load functions**: `textureLoad` - direct texel access without
    //!   filtering
    //! - **Sample functions**: `textureSample` - filtered sampling with a
    //!   sampler
    //! - **Depth comparison**: `textureSampleCompare` - for shadow mapping
    use wgsl_rs::std::*;

    // A 2D texture for color/albedo
    texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
    // A sampler for filtering the texture
    sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);

    // Fragment input with texture coordinates
    pub struct FragmentInput {
        #[location(0)]
        pub uv: Vec2f,
    }

    // Output struct
    pub struct FragmentOutput {
        #[location(0)]
        pub color: Vec4f,
    }

    // Main fragment shader demonstrating texture operations.
    #[fragment]
    pub fn frag_main(input: FragmentInput) -> FragmentOutput {
        // Sample the diffuse texture
        let albedo = texture_sample(DIFFUSE_TEX, TEX_SAMPLER, input.uv);

        FragmentOutput { color: albedo }
    }
}
```

## Generated WGSL

```wgsl
@group(0) @binding(0) var DIFFUSE_TEX: texture_2d<f32>;
@group(0) @binding(1) var TEX_SAMPLER: sampler;

struct FragmentInput {
    @location(0) uv: vec2f
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment fn frag_main(input: FragmentInput) -> FragmentOutput {
    let albedo = textureSample(DIFFUSE_TEX, TEX_SAMPLER, input.uv);
    return FragmentOutput(albedo);
}
```

## Notes

- `texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>)` declares a `texture_2d<f32>` binding.
- `sampler!(...)` declares a `sampler` binding.
- `texture_sample(...)` maps to the WGSL `textureSample` builtin.