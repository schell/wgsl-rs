use wgsl_rs::wgsl;

#[wgsl]
mod trait_default_method_body {
    pub trait Addable {
        fn add(a: Self, b: Self) -> Self {
            b
        }
    }

    impl Addable for u32 {
        fn add(a: u32, b: u32) -> u32 {
            a + b
        }
    }

    pub fn use_it() -> u32 {
        u32::add(1u32, 2u32)
    }
}

fn main() {}
