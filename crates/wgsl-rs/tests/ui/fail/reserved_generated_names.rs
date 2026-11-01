use wgsl_rs::wgsl;

/// `instantiate` is the name of the macro-emitted IR constructor, so a
/// user item with that name is rejected with a targeted diagnostic
/// instead of a duplicate-definition error in the consumer crate.
#[wgsl]
mod reserved_generated_names {
    pub fn instantiate() -> u32 {
        1u32
    }
}

#[wgsl]
mod reserved_generated_names_static {
    pub const WGSL_SOURCE: u32 = 1u32;
}

fn main() {}