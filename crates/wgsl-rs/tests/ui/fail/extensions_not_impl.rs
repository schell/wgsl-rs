use wgsl_rs::wgsl;

pub struct NotAnExtension;

#[wgsl(crate_path = wgsl_rs, extensions = [super::NotAnExtension])]
mod bad_ext {
    pub fn main() -> u32 {
        42u32
    }
}

fn main() {}