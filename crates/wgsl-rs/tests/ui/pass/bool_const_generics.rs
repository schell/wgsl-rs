//! Tests for `const B: bool` const generics, the compile-time feature-flag
//! case, across the surfaces that should support them:
//! 1. Free function declared and called locally, with true and false
//!    instantiations
//! 2. Generic struct + impl block over a bool const-generic struct
//! 3. Cross-module template instantiation with a bool const argument
//! 4. Entry-point function with a bool const param (drives
//!    `instantiate::<true>()` / `instantiate::<false>()`)
//!
//! Bool const params are value-position-only by construction: rustc
//! requires array sizes to evaluate to `usize`, so a bool const param can
//! never legally appear in an array length or index. Mangled names spell
//! the value (`fog_scale_true` / `fog_scale_false`), and substituted
//! literals render as `true` / `false`, so both worlds branch
//! identically.
use wgsl_rs::wgsl;

// 1. Generic function with a bool const param, called locally.
#[wgsl]
mod flag_fn {
    use wgsl_rs::std::*;

    pub fn fog_scale<const APPLY_FOG: bool>(c: f32) -> f32 {
        let fogged = c * 0.5;
        select(c, fogged, APPLY_FOG)
    }

    pub fn run_on(c: f32) -> f32 {
        fog_scale::<true>(c)
    }

    pub fn run_off(c: f32) -> f32 {
        fog_scale::<false>(c)
    }
}

// 2. Generic struct with a bool const param, used in impl method bodies.
#[wgsl]
mod flag_struct {
    use wgsl_rs::std::*;

    pub struct Fogged<const APPLY_FOG: bool> {
        pub color: f32,
    }

    impl<const APPLY_FOG: bool> Fogged<APPLY_FOG> {
        pub fn apply(c: f32) -> f32 {
            let fogged = c * 0.5;
            select(c, fogged, APPLY_FOG)
        }
    }

    pub fn run(c: f32) -> f32 {
        Fogged::<true>::apply(c)
    }
}

// 3. Cross-module template instantiation: a bool const-generic function
//    exported from one `#[wgsl]` mod and invoked with a bool argument
//    from another.
#[wgsl]
mod provider {
    use wgsl_rs::std::*;

    pub fn fog_mix<const APPLY_FOG: bool>(a: f32, b: f32) -> f32 {
        select(a, b, APPLY_FOG)
    }
}

#[wgsl]
mod consumer {
    use super::provider::*;

    pub fn run(a: f32, b: f32) -> f32 {
        fog_mix::<true>(a, b)
    }
}

// 4. Entry point with a bool const param, the motivating feature-flag
//    case, driving the `instantiate::<…>()` codegen path. Template
//    modules can't be validated standalone, so the instantiation types
//    list names a module-local const (a single-segment path is a valid
//    const argument per the Rust Reference) and the macro generates an
//    instantiation+validation test for it.
#[wgsl(validate_with_instantiation_types(APPLY_TRUE))]
mod flag_entry_point {
    use wgsl_rs::std::*;

    const APPLY_TRUE: bool = true;

    storage!(group(0), binding(0), read_write, OUTPUT: [u32; 1]);

    #[compute]
    #[workgroup_size(1)]
    pub fn main<const APPLY_FOG: bool>(#[builtin(global_invocation_id)] global_id: Vec3u) {
        if APPLY_FOG {
            get_mut!(OUTPUT)[0] = global_id.x;
        } else {
            get_mut!(OUTPUT)[0] = 0u32;
        }
    }
}

fn main() {
    let _ = flag_fn::run_on(1.0);
    let _ = flag_fn::run_off(1.0);
    let _ = flag_struct::run(2.0);
    let _ = consumer::run(1.0, 2.0);
    let _ = flag_entry_point::instantiate::<true>();
    let _ = flag_fn::WGSL_SOURCE.wgsl_source();

    // Cross-module bool instantiation: the mangled name spells the value.
    let s = consumer::WGSL_SOURCE.wgsl_source().expect("source renders");
    assert!(s.contains("fog_mix_true"), "missing mangled fn: {s}");

    // Same-module bool instantiations render `true` / `false` literals.
    let src = flag_fn::WGSL_SOURCE.wgsl_source().expect("source renders");
    assert!(src.contains("select(c, fogged, true)"), "missing true: {src}");
    assert!(src.contains("select(c, fogged, false)"), "missing false: {src}");

    // Entry-point feature flag through `instantiate`, both ways, validated
    // with naga — mirroring the `__validate_wgsl_instantiated_0` test the
    // `validate_with_instantiation_types` attribute generates (that test
    // is `#[cfg(test)]`, so it doesn't run under trybuild's bin build;
    // this copy does).
    let m = flag_entry_point::instantiate::<true>();
    let src = wgsl_rs::ir::render_module(&m);
    wgsl_rs::validate_wgsl_source(&src).expect("true instantiation must validate");
    assert!(src.contains("if true"), "entry-point true const not rendered: {src}");

    let m = flag_entry_point::instantiate::<false>();
    let src = wgsl_rs::ir::render_module(&m);
    wgsl_rs::validate_wgsl_source(&src).expect("false instantiation must validate");
    assert!(src.contains("if false"), "entry-point false const not rendered: {src}");
}
