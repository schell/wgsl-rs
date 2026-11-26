use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod slot_collision {
    use wgsl_rs::std::*;

    pub struct PackA<const B_C: u32> {
        #[uniform(group(0), binding(B_C))]
        pub data: Vec4f,
    }

    pub struct PackA_B<const C: u32> {
        #[uniform(group(0), binding(C))]
        pub other: Vec4f,
    }
}

fn main() {}
