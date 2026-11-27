# Texture

Demonstrates textures, samplers, and the `textureSample` builtin. Covers texture and sampler pack fields and a fragment shader that samples a 2D texture.

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

    pub struct TextureExamplePack {
        // A 2D texture for color/albedo
        #[texture(group(0), binding(0))]
        pub diffuse_tex: Texture2D<f32>,
        // A sampler for filtering the texture
        #[sampler(group(0), binding(1))]
        pub tex_sampler: Sampler,
    }

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
    pub fn frag_main(input: FragmentInput, pack: &TextureExamplePack) -> FragmentOutput {
        // Sample the diffuse texture
        let albedo = texture_sample(&pack.diffuse_tex, &pack.tex_sampler, input.uv);

        FragmentOutput { color: albedo }
    }
}
```

## Generated WGSL

Pack fields become mangled WGSL globals (`packlower__field`), and the pack
parameter disappears from the entry point:

```wgsl
@group(0) @binding(0) var textureexamplepack__diffuse_tex: texture_2d<f32>;
@group(0) @binding(1) var textureexamplepack__tex_sampler: sampler;

struct FragmentInput {
    @location(0) uv: vec2f
}

struct FragmentOutput {
    @location(0) color: vec4f
}

@fragment fn frag_main(input: FragmentInput) -> FragmentOutput {
    let albedo = textureSample(textureexamplepack__diffuse_tex, textureexamplepack__tex_sampler, input.uv);
    return FragmentOutput(albedo);
}
```

## Notes

- `#[texture(group(0), binding(0))] pub diffuse_tex: Texture2D<f32>` declares a `texture_2d<f32>` binding.
- `#[sampler(...)]` declares a `sampler` binding.
- `texture_sample(...)` maps to the WGSL `textureSample` builtin.
