use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod slot_filterability_collision {
    use wgsl_rs::std::*;

    pub struct MyPack<const GROUP: u32> {
        #[sampler(group(GROUP), binding(0))]
        pub smp: Sampler,
    }

    texture!(group(0), binding(1), TEX: Texture2D<f32, MyPack_GROUP>);
}

fn main() {}
