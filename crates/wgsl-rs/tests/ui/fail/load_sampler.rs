use wgsl_rs::wgsl;

// `load!` on a sampler: samplers are handles in WGSL, not loadable values.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_sampler {
    use wgsl_rs::std::*;

    pub struct SamplerPack {
        #[sampler(group(0), binding(0))]
        pub my_sampler: Sampler,
    }

    pub fn read_sampler(pack: &SamplerPack) -> f32 {
        load!(pack.my_sampler)
    }
}

fn main() {}
