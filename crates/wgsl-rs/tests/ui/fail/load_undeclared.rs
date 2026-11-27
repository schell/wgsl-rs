use wgsl_rs::wgsl;

// `load!` on a bare identifier: access-macro targets are resource-pack
// fields, and no such pack parameter exists in this function.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_undeclared {
    use wgsl_rs::std::*;

    pub fn read_missing() -> f32 {
        load!(MISSING)
    }
}

fn main() {}
