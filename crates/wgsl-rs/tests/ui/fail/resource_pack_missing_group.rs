use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod malformed_pack {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[texture(binding(0))]
        pub texture: Texture2D<f32>,
    }
}

fn main() {}
