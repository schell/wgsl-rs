use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod mixed_io {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[texture(group(0), binding(0))]
        #[location(0)]
        pub texture: Texture2D<f32>,
    }
}

fn main() {}
