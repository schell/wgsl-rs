use wgsl_rs::wgsl;

// The standalone `sampler!` declaration macro was removed: declare an
// annotated pack field instead.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod removed_sampler_declaration {
    use wgsl_rs::std::*;

    sampler!(group(0), binding(0), S: Sampler);
}

fn main() {}
