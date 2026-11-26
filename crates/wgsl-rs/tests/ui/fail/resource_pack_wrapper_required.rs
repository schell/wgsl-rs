use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod inner_type {
    use wgsl_rs::std::*;

    pub struct FramePack {
        #[uniform(group(0), binding(0))]
        pub time: f32,
    }
}

fn main() {}
