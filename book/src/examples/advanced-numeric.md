# Advanced Numeric

Demonstrates the advanced numeric builtin functions: `modf`, `frexp`, and `ldexp`. `modf` and `frexp` return structs with named fields (`fract`, `whole`, `exp`) that map to WGSL struct member access.

## Rust Source

```rust
#[wgsl]
pub mod advanced_numeric_example {
    //! Demonstrates the advanced numeric builtin functions: `modf`, `frexp`,
    //! and `ldexp`.
    use wgsl_rs::std::*;

    pub fn demo_modf_fract(e: f32) -> f32 {
        let result = modf(e);
        result.fract
    }

    pub fn demo_modf_whole(e: f32) -> f32 {
        modf(e).whole
    }

    pub fn demo_frexp_fract(e: f32) -> f32 {
        frexp(e).fract
    }

    pub fn demo_frexp_exp(e: f32) -> i32 {
        frexp(e).exp
    }

    pub fn demo_ldexp(significand: f32, exponent: i32) -> f32 {
        ldexp(significand, exponent)
    }
}
```

## Generated WGSL

```wgsl
fn demo_modf_fract(e: f32) -> f32 {
    let result = modf(e);
    return result.fract;
}

fn demo_modf_whole(e: f32) -> f32 {
    return modf(e).whole;
}

fn demo_frexp_fract(e: f32) -> f32 {
    return frexp(e).fract;
}

fn demo_frexp_exp(e: f32) -> i32 {
    return frexp(e).exp;
}

fn demo_ldexp(significand: f32, exponent: i32) -> f32 {
    return ldexp(significand, exponent);
}
```

## Notes

- `modf(e)` and `frexp(e)` return structs with `fract`/`whole` and `fract`/`exp` fields respectively, accessed via field access syntax that maps directly to WGSL.