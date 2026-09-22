use wgsl_rs::wgsl;

#[wgsl]
mod as_trait_form {
    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    pub fn rejected() -> u32 {
        <u32 as Zeroable>::zero()
    }
}

fn main() {}