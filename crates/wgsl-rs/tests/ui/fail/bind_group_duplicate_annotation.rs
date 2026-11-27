use wgsl_rs::std::Sampler;
use wgsl_rs_macros::BindGroup;

#[derive(BindGroup)]
pub struct Pack {
    #[sampler(group(0), binding(0))]
    #[sampler(group(0), binding(1))]
    pub a: Sampler,
}

fn main() {}
