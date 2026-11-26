//! Regression: parameter shadowing inside an impl-associated function.
//!
//! Renderling's `shader::atlas` module hit this live: an impl method
//!
//! ```ignore
//! impl TextureAddressMode {
//!     pub fn repeat(input: f32) -> f32 {
//!         let mut input = input;
//!         ...
//!     }
//! }
//! ```
//!
//! in a template module (const-parameter entry point, validated via
//! `validate_with_instantiation_types`). The deshadow pass handles this
//! shape and the canonical render route (`instantiate::<T>().wgsl_source()`)
//! renames the local - but the generated `__validate_wgsl_instantiated_N`
//! test renders via the raw `render_module`, bypassing deshadow, so naga
//! rejects `var input = input;` as a redefinition.
//!
//! The free-function shape is covered by `instantiate_wgsl_source_deshadows`
//! in `linkage_wgpu.rs`.
#![cfg(feature = "v30")]

#[allow(unused_imports)]
use wgsl_rs::wgsl;

/// Mirrors the renderling atlas shape: template module, const-parameter
/// entry point, auto-validation instantiation, and an impl method whose
/// body shadows a parameter with a mutable local.
#[wgsl(validate_with_instantiation_types(IS_TRUE))]
pub mod impl_param_shadowing {
    use wgsl_rs::std::*;

    pub const IS_TRUE: bool = true;

    pub struct Wrap {
        pub v: f32,
    }

    impl Wrap {
        /// The exact idiom from `TextureAddressMode::repeat`.
        pub fn shadowed(input: f32) -> f32 {
            let mut input = input;
            input += 1.0;
            input
        }
    }

    #[vertex]
    pub fn vs_main<const FILTERABLE: bool>(#[builtin(vertex_index)] _vi: u32) -> Vec4f {
        let f = Wrap::shadowed(1.0);
        if FILTERABLE {
            vec4f(f, 0.0, 0.0, 1.0)
        } else {
            vec4f(1.0, f, 0.0, 1.0)
        }
    }
}

/// The canonical instantiate route deshadows the impl-method param
/// shadowing: rename applied and naga validation green.
#[test]
fn impl_param_shadowing_deshadows_on_the_instantiate_route() {
    let via_instantiate = impl_param_shadowing::instantiate::<true>().wgsl_source();
    assert!(
        via_instantiate.contains("input_1"),
        "the param-shadowing local must be renamed, got:\n{via_instantiate}"
    );
    wgsl_rs::validate_wgsl_source(&via_instantiate)
        .expect("the deshadowed module source must validate in naga");
}

// NOTE: the generated `impl_param_shadowing::__validate_wgsl_instantiated_0`
// test (from `validate_with_instantiation_types(IS_TRUE)`) is the failing
// half of this regression: it renders via the raw `render_module` and
// rejects the un-deshadowed redefinition. It goes green once the macro
// routes it through `Module::wgsl_source()`, like the concrete-module
// validation test does.
