use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod bad_const {
    use wgsl_rs::std::*;

    pub struct FramePack {
        #[uniform(group(0), binding(0))]
        pub frame: Uniform<f32>,
    }

    pub fn frag_main(pack: &FramePack) -> Vec4f {
        const X: f32 = get!(pack.frame);
        vec4f(X, 0.0, 0.0, 1.0)
    }
}

fn main() {}
