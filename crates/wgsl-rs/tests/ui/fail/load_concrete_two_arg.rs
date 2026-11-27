use wgsl_rs::wgsl;

// The two-argument form of `load!` no longer exists: pack fields hold
// concrete types at their declaration, so the one-argument form is the
// only spelling.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_concrete_two_arg {
    use wgsl_rs::std::*;

    pub struct TimePack {
        #[uniform(group(0), binding(0))]
        pub u_time: Uniform<f32>,
    }

    pub fn read_time(pack: &TimePack) -> f32 {
        load!(pack.u_time, f32)
    }
}

fn main() {}
