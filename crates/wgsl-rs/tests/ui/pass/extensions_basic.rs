use wgsl_rs::{ir, wgsl, WgslExtension};

pub struct NoopExt;
impl WgslExtension for NoopExt {
    fn modify_ir(_module: &mut ir::Module) {}
}

#[wgsl(crate_path = wgsl_rs, extensions = [super::NoopExt])]
mod ext_shader {
    pub fn main() -> u32 {
        42u32
    }
}

fn main() {
    let _ = ext_shader::WGSL_SOURCE.wgsl_source().unwrap();
}