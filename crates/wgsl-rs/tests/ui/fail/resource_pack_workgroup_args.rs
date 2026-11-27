use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod workgroup_args {
    use wgsl_rs::std::*;

    pub struct WorkgroupData {
        #[workgroup(group(0))]
        pub accumulator: Workgroup<i32>,
    }
}

fn main() {}
