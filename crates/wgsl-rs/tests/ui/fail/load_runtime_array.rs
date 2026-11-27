use wgsl_rs::wgsl;

// `load!` targeting a runtime-sized array: runtime arrays are not
// copyable WGSL values.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_runtime_array {
    use wgsl_rs::std::*;

    pub struct ParticlesPack {
        #[storage(group(0), binding(0), read_write)]
        pub particles: Storage<RuntimeArray<f32>>,
    }

    pub fn read_particles(pack: &ParticlesPack) -> f32 {
        load!(pack.particles)[0]
    }
}

fn main() {}
