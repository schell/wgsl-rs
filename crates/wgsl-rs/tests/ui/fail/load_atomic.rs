use wgsl_rs::wgsl;

// `load!` on an atomic: WGSL requires atomicLoad, not a value read.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod load_atomic {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), read_write, COUNTER: Atomic<u32>);

    pub fn read_counter() -> u32 {
        load!(COUNTER)
    }
}

fn main() {}