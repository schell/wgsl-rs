use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod bad_const {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

    pub fn frag_main() -> Vec4f {
        const X: f32 = get!(FRAME, f32);
        vec4f(X, 0.0, 0.0, 1.0)
    }
}

fn main() {}