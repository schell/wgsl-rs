use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod arg_count {
    use wgsl_rs::std::*;

    pub struct PackA<const GROUP: u32, T> {
        #[storage(group(GROUP), binding(0))]
        pub data: Storage<RuntimeArray<T>, Read>,
    }

    pub fn helper(pack: &PackA<0>) -> u32 {
        0
    }
}

fn main() {}
