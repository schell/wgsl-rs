use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod builtin_in_pack {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[texture(group(0), binding(0))]
        #[builtin(vertex_index)]
        pub texture: Texture2D<f32>,
    }
}

fn main() {}
