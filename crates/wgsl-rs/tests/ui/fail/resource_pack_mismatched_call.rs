use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod mismatched_call {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[sampler(group(0), binding(0))]
        pub smp: Sampler,
    }

    pub fn callee(pack: &Pack, value: u32) {}

    pub fn caller(pack: &Pack) {
        callee(pack);
    }
}

fn main() {}
