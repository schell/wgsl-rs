use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod duplicate_parameter {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[sampler(group(0), binding(0))]
        pub smp: Sampler,
    }

    pub fn helper(first: &Pack, second: &Pack) {}
}

fn main() {}
