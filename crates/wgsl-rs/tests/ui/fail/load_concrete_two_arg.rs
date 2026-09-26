use wgsl_rs::wgsl;

// The two-argument form of `load!` is for generic module variables. On a
// concrete variable it is rejected with a note (the CPU side would fail
// with a method-not-found on `get_typed` otherwise).
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_concrete_two_arg {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U_TIME: f32);

    pub fn read_time() -> f32 {
        load!(U_TIME, f32)
    }
}

fn main() {}