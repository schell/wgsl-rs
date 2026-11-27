use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod uniform_write {
    use wgsl_rs::std::*;

    pub struct FramePack {
        #[uniform(group(0), binding(0))]
        pub time: Uniform<f32>,
    }

    #[fragment]
    pub fn scaled(#[location(0)] _position: Vec4f, frame: &FramePack) -> Vec4f {
        *get_mut!(&frame.time) = 1.0;
        vec4f(0.0, 0.0, 0.0, 1.0)
    }
}

fn main() {}
