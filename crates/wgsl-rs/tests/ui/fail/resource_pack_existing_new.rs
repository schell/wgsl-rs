use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod existing_new {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[sampler(group(0), binding(0))]
        pub smp: Sampler,
    }

    #[wgsl_ignore]
    impl Pack {
        pub fn new() -> f32 {
            0.0f32
        }
    }
}

fn main() {}
