use wgsl_rs::wgsl;

// Per WGSL spec 17.7.13, `textureSampleLevel` takes an `f32` level on
// non-depth sampled textures; only depth textures accept `i32`/`u32`
// levels. wgsl-rs must not expose the integer-level overloads (wgsl-rs#157),
// so these calls must fail trait resolution.

#[wgsl]
mod sample_level_i32 {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);
    sampler!(group(0), binding(1), S: Sampler);

    pub fn f() -> Vec4f {
        texture_sample_level(TEX, S, vec2f(0.5, 0.5), 0i32)
    }
}

#[wgsl]
mod sample_level_u32 {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: TextureCube<f32>);
    sampler!(group(0), binding(1), S: Sampler);

    pub fn f() -> Vec4f {
        texture_sample_level(TEX, S, vec3f(0.5, 0.5, 0.5), 0u32)
    }
}

fn main() {}