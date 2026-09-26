use wgsl_rs::wgsl;

// `load!` on a sampler: samplers are handles in WGSL, not loadable values.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_sampler {
    use wgsl_rs::std::*;

    sampler!(group(0), binding(0), MY_SAMPLER: Sampler);

    pub fn read_sampler() -> f32 {
        load!(MY_SAMPLER)
    }
}

fn main() {}