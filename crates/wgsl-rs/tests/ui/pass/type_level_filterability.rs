//! Pack fields keep concrete filterability in their CPU types, and
//! `#[texture]`/`#[sampler]` field annotations thread it through the WGSL
//! (`filterable: false` on the generated bind group layout entry).

use wgsl_rs::std::*;
use wgsl_rs::wgsl;

#[wgsl]
mod flat_pack {
    use wgsl_rs::std::*;

    pub struct FlatPack {
        #[texture(group(0), binding(0))]
        pub tex: Texture2D<f32, false>,
        #[sampler(group(0), binding(1))]
        pub smp: Sampler<false>,
    }

    #[fragment]
    pub fn main(pack: &FlatPack) -> Vec4f {
        texture_sample(&pack.tex, &pack.smp, vec2f(0.0, 0.0))
    }
}

fn main() {
    let tex: Texture2D<f32, false> = Texture2D::new(1, 1);
    let smp: Sampler<false> = Sampler::new();
    let _: &Texture2D<f32, false> = &tex;
    let _: &Sampler<false> = &smp;
    let _ = flat_pack::instantiate();
}
