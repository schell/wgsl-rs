use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod unused_type_param {
    use wgsl_rs::std::*;

    pub struct PackA<const GROUP: u32, T> {
        #[uniform(group(GROUP), binding(0))]
        pub data: Vec4f,
    }
}

fn main() {}
