use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod duplicate_address {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[sampler(group(0), binding(0))]
        pub first: Sampler,
        #[sampler(group(0), binding(0))]
        pub second: Sampler,
    }
}

fn main() {}
