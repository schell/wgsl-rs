use wgsl_rs::wgsl;

// The standalone `texture!` declaration macro was removed: declare an
// annotated pack field instead.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod removed_texture_declaration {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);
}

fn main() {}
