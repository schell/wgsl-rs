use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod provider {
    pub fn external_identity<T: Copy>(x: T) -> T {
        x
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod consumer {
    use super::provider::*;

    pub fn local_identity<T: Copy>(x: T) -> T {
        x
    }

    pub fn run() -> f32 {
        let local = local_identity::<f32>(1.0);
        external_identity::<f32>(local)
    }
}

fn main() {
    let _ = consumer::run();
    let _ = consumer::WGSL_MODULE.wgsl_source();
}
