use wgsl_rs::wgsl;

// The standalone `uniform!` declaration macro was removed: declare an
// annotated pack field instead.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod removed_uniform_declaration {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: u32);
}

fn main() {}
