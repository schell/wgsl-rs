use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod workgroup_wrapper {
    use wgsl_rs::std::*;

    pub struct WorkgroupData {
        #[workgroup]
        pub accumulator: Uniform<i32>,
    }
}

fn main() {}
