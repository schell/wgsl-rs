use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod collision {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FILTERABLE: impl Convert<f32>);
    texture!(group(0), binding(1), TEX: Texture2D<f32, FILTERABLE>);
}

fn main() {}
