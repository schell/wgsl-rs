use wgsl_rs::wgsl;

// One-argument `load!` on a generic module variable: the concrete type is
// not knowable at the declaration, so the two-argument form is required.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_generic_one_arg {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

    pub fn read_frame() -> f32 {
        load!(FRAME)
    }
}

fn main() {}