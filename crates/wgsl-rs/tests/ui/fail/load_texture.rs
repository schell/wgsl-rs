use wgsl_rs::wgsl;

// `load!` on a texture: textures are handles in WGSL, not loadable values.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_texture {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);

    pub fn read_tex(uv: Vec2f) -> Vec4f {
        let t = load!(TEX);
        vec4f(t.x(), t.y(), 0.0, 1.0)
    }
}

fn main() {}