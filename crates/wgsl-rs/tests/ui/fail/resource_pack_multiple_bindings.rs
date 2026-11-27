use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod multiple_bindings {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[texture(group(0), binding(0))]
        #[sampler(group(0), binding(1))]
        pub resource: Texture2D<f32>,
    }
}

fn main() {}
