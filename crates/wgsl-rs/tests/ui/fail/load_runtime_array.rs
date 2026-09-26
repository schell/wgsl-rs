use wgsl_rs::wgsl;

// `load!` targeting a runtime-sized array: runtime arrays are not
// copyable WGSL values.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_runtime_array {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), read_write, PARTICLES: RuntimeArray<f32>);

    pub fn read_particles() -> f32 {
        load!(PARTICLES)[0]
    }
}

fn main() {}