use wgsl_rs::wgsl;

// `load!` on an undeclared variable: no matching uniform!/storage!/
// workgroup! declaration in the module.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_undeclared {
    use wgsl_rs::std::*;

    pub fn read_missing() -> f32 {
        load!(MISSING)
    }
}

fn main() {}