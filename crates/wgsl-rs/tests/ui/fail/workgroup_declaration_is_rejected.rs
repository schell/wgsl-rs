use wgsl_rs::wgsl;

// The standalone `workgroup!` declaration macro was removed: declare an
// annotated pack field instead.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod removed_workgroup_declaration {
    use wgsl_rs::std::*;

    workgroup!(SHARED: u32);
}

fn main() {}
