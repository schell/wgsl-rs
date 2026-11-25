//! Standalone binding macros preserve concrete filterability in their CPU types.

use wgsl_rs::std::*;
use wgsl_rs::wgsl;

texture!(group(0), binding(0), TEX: Texture2D<f32, false>);
sampler!(group(0), binding(1), SMP: Sampler<false>);

#[wgsl]
mod generic_helper_signatures {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32, false>);
    sampler!(group(0), binding(1), SMP: Sampler<false>);

    pub fn sample<const TEX_FILTERABLE: bool, const SMP_FILTERABLE: bool>(
        _tex: Texture2D<f32, TEX_FILTERABLE>,
        _smp: Sampler<SMP_FILTERABLE>,
    ) -> Vec4f {
        texture_sample(TEX, SMP, vec2f(0.0, 0.0))
    }

    #[fragment]
    pub fn main() -> Vec4f {
        texture_sample(TEX, SMP, vec2f(0.0, 0.0))
    }
}

fn main() {
    let _: &Texture2D<f32, false> = TEX;
    let _: &Sampler<false> = SMP;
    TEX.init(1, 1);
    SMP.init();
}
