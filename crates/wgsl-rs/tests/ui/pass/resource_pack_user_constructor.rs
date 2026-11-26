//! No constructors are generated for packs, so a user-written `new` is
//! free to exist. `#[wgsl_ignore]` keeps the impl CPU-only.

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod user_constructor {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[sampler(group(0), binding(0))]
        pub smp: Sampler,
    }

    #[wgsl_ignore]
    impl Pack {
        pub fn new() -> Self {
            Self {
                smp: Sampler::new(0, 0),
            }
        }
    }
}

fn main() {}
