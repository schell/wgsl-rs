use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod non_resource {
    use wgsl_rs::std::*;

    pub struct InvalidPack {
        #[sampler(group(0), binding(0))]
        pub sampler: Sampler,
        pub not_a_resource: f32,
    }
}

fn main() {}
