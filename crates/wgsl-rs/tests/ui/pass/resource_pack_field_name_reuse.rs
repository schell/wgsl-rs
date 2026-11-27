//! Two packs may reuse a field name: mangling namespaces each pack's
//! globals (`firstpack__resource`, `secondpack__resource`).

use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod field_name_reuse {
    use wgsl_rs::std::*;

    pub struct FirstPack {
        #[sampler(group(0), binding(0))]
        pub resource: Sampler,
    }

    pub struct SecondPack {
        #[sampler(group(1), binding(0))]
        pub resource: Sampler,
    }
}

fn main() {}
