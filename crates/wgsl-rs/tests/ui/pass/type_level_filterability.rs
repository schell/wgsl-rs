//! Standalone binding macros preserve concrete filterability in their CPU types.

use wgsl_rs::std::*;

texture!(group(0), binding(0), TEX: Texture2D<f32, false>);
sampler!(group(0), binding(1), SMP: Sampler<false>);

fn main() {
    let _: &Texture2D<f32, false> = TEX;
    let _: &Sampler<false> = SMP;
    TEX.init(1, 1);
    SMP.init();
}
