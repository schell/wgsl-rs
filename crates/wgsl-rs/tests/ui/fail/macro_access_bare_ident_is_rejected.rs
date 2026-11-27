use wgsl_rs::wgsl;

// `get!`/`get_mut!` on a bare identifier: targets must be a resource-pack
// parameter's field. The module-static form no longer exists.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod macro_access_bare_ident {
    use wgsl_rs::std::*;

    pub fn pick(x: u32, y: u32) -> u32 {
        x + y
    }

    pub fn read() -> u32 {
        pick(get!(SHARED), get_mut!(SHARED))
    }
}

fn main() {}
