use wgsl_rs::wgsl;

#[wgsl]
mod bad_float {
    pub fn f() -> f64 {
        let v = 0.0_f64;
        v
    }
}

fn main() {}