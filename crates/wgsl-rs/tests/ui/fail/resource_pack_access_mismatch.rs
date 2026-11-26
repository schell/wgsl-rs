use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod access_mismatch {
    use wgsl_rs::std::*;

    pub struct BinPack {
        #[storage(group(0), binding(0), read_only)]
        pub bins: Storage<[f32; 4], ReadWrite>,
    }
}

fn main() {}
