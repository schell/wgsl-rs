use wgsl_rs::wgsl;

// The standalone `storage!` declaration macro was removed: declare an
// annotated pack field instead.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod removed_storage_declaration {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), read_write, DATA: [f32; 4]);
}

fn main() {}
