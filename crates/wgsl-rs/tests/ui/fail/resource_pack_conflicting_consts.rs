use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod conflicting_consts {
    use wgsl_rs::std::*;

    pub struct TextureBindGroup<const FILTERABLE: bool> {
        #[texture(group(0), binding(0))]
        pub heights: Texture2D<f32, FILTERABLE>,
        #[sampler(group(0), binding(1))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    #[fragment]
    pub fn first(pack: &TextureBindGroup<true>) -> Vec4f {
        Vec4f::ZERO
    }

    #[fragment]
    pub fn second(pack: &TextureBindGroup<false>) -> Vec4f {
        Vec4f::ZERO
    }
}

fn main() {}
