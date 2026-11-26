use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod tied_misuse {
    use wgsl_rs::std::*;

    pub struct DataPack<T> {
        #[storage(group(0), binding(0))]
        pub data: Storage<RuntimeArray<T>, Read>,
    }

    pub fn helper<T>(values: T, pack: &DataPack<T>) {}
}

fn main() {}
