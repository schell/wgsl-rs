//! Tests for signed `const N: i32` const generics across the surfaces that
//! should support them:
//! 1. Free function declared and called locally, with a negative and a
//!    positive instantiation
//! 2. Generic struct + impl block over a const-generic struct (value
//!    positions only — Rust requires array sizes to evaluate to `usize`,
//!    so an `i32` const param can never legally appear in an array length
//!    or index)
//! 3. Cross-module template instantiation with a negative const argument
//! 4. Entry-point function with an `i32` const param (drives
//!    `instantiate::<…>()` with a negative value)
//!
//! Negative const arguments mangle with an `n` prefix (rustc symbol-mangling
//! precedent), e.g. `bias_add::<-4>` becomes `bias_add_n4` in WGSL, and
//! substituted literals render parenthesized (`(-4)`) so they stay
//! unambiguous in every expression context.
use wgsl_rs::wgsl;

// 1. Generic function with a signed const param, called locally.
#[wgsl]
mod signed_fn {
    use wgsl_rs::std::*;

    pub fn clamp_to<const MIN: i32, const MAX: i32>(x: i32) -> i32 {
        max(min(x, MAX), MIN)
    }

    pub fn run_neg(x: i32) -> i32 {
        clamp_to::<-8, 8>(x)
    }

    pub fn run_pos(x: i32) -> i32 {
        clamp_to::<0, 100>(x)
    }

    pub fn run_neg_literal() -> i32 {
        clamp_to::<-8, 8>(-9)
    }
}

// 2. Generic struct with signed const params, used in impl method bodies.
#[wgsl]
mod signed_struct {
    use wgsl_rs::std::*;

    pub struct Bounded<const MIN: i32, const MAX: i32> {
        pub value: i32,
    }

    impl<const MIN: i32, const MAX: i32> Bounded<MIN, MAX> {
        pub fn clamp(x: i32) -> i32 {
            max(min(x, MAX), MIN)
        }
    }

    pub fn run(x: i32) -> i32 {
        Bounded::<-4, 4>::clamp(x)
    }
}

// 3. Cross-module template instantiation: a signed const-generic function
//    exported from one `#[wgsl]` mod and invoked with a negative argument
//    from another.
#[wgsl]
mod provider {
    pub fn bias_add<const BIAS: i32>(x: i32) -> i32 {
        x + BIAS
    }
}

#[wgsl]
mod consumer {
    use super::provider::*;

    pub fn run(x: i32) -> i32 {
        bias_add::<-4>(x)
    }
}

// 4. Entry point with a signed const param, driving the
//    `instantiate::<…>()` codegen path with a negative value.
#[wgsl(skip_validation)]
mod signed_entry_point {
    use wgsl_rs::std::*;

    #[compute]
    #[workgroup_size(1)]
    pub fn main<const BIAS: i32>() -> i32 {
        BIAS
    }
}

fn main() {
    let _ = signed_fn::run_neg(-20);
    let _ = signed_fn::run_pos(120);
    let _ = signed_fn::run_neg_literal();
    let _ = signed_struct::run(9);
    let _ = consumer::run(7);
    let _ = signed_entry_point::instantiate::<-4>();
    let _ = signed_fn::WGSL_SOURCE.wgsl_source();

    // Cross-module negative instantiation: the mangled name uses the
    // `n` prefix and the substituted literal renders parenthesized.
    let s = consumer::WGSL_SOURCE.wgsl_source().expect("source renders");
    assert!(s.contains("bias_add_n4"), "missing mangled fn: {s}");
    assert!(s.contains("(-4"), "missing negative literal: {s}");

    // Entry-point negative instantiation through `instantiate`.
    let m = signed_entry_point::instantiate::<-4>();
    let src = wgsl_rs::ir::render_module(&m);
    assert!(src.contains("(-4"), "entry-point negative const not rendered: {src}");
}
