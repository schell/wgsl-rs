use wgsl_rs::wgsl;

// Dereferencing a local that holds an accessor value compiles on the
// CPU (the guard derefs to T) but renders `*u` in WGSL, which wgpu rejects
// ("the operand of the `*` operator must be a pointer") — wgsl-rs#153.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod deref_value_local {
    use wgsl_rs::std::*;

    pub struct DerefPack {
        #[uniform(group(0), binding(0))]
        pub u: Uniform<Vec3f>,
    }

    pub fn read_u(pack: &DerefPack) -> Vec3f {
        let u = get!(pack.u);
        *u
    }
}

fn main() {}
