use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod duplicate_name {
    use wgsl_rs::std::*;

    pub struct FirstPack {
        #[sampler(group(0), binding(0))]
        pub resource: Sampler,
    }

    pub struct SecondPack {
        #[sampler(group(1), binding(0))]
        pub resource: Sampler,
    }
}

fn main() {}
