use wgsl_rs::wgsl;

#[wgsl]
mod mixed_default {
    pub fn f(x: u32) -> u32 {
        match x {
            0 | _ => 1u32,
        }
    }
}

fn main() {}