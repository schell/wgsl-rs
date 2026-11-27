use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod key_collision {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[sampler(group(0), binding(0))]
        pub foo: Sampler,
        #[sampler(group(0), binding(1))]
        pub FOO: Sampler,
    }
}

fn main() {}
