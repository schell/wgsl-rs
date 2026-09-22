use wgsl_rs::wgsl;

#[wgsl]
mod method_segment_turbofish {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for [u32; 4] {
        fn zero() -> [u32; 4] {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    pub fn rejected() -> [u32; 4] {
        Zeroable::zero::<[u32; 4]>()
    }
}

fn main() {}