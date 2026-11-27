use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod reused_const {
    use wgsl_rs::std::*;

    pub struct PackA<const GROUP: u32> {
        #[sampler(group(GROUP), binding(0))]
        pub a: Sampler,
    }

    pub struct PackB<const GROUP: u32> {
        #[sampler(group(GROUP), binding(1))]
        pub b: Sampler,
    }

    #[fragment]
    pub fn shared<const GROUP: u32>(a: &PackA<GROUP>, b: &PackB<GROUP>) -> Vec4f {
        Vec4f::ZERO
    }
}

fn main() {}
