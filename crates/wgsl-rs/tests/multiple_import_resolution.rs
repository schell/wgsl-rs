use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod first_provider {
    pub fn unrelated(x: f32) -> f32 {
        x
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod second_provider {
    pub fn external_identity<T: Copy>(x: T) -> T {
        x
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod consumer {
    #[rustfmt::skip]
    use super::second_provider::*;
    #[rustfmt::skip]
    use crate::first_provider::*;

    pub fn apply() -> f32 {
        let u = unrelated(0.0);
        let _keep = u;
        external_identity::<f32>(2.0)
    }
}

#[test]
fn resolves_template_instantiation_from_correct_import() {
    let _ = consumer::apply();
    let full_src = consumer::WGSL_MODULE.wgsl_source().join("\n");
    assert!(
        full_src.contains("fn external_identity_f32("),
        "Expected instantiated external_identity_f32 in output, got:\n{full_src}"
    );
}
