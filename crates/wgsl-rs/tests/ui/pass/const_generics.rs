//! Tests for `const N: usize` const generics across every surface that
//! should support them:
//! 1. Free function declared and called locally
//! 2. Generic struct declared and used locally
//! 3. Generic impl block over a const-generic struct
//! 4. Cross-module template instantiation
//! 5. Entry-point function with a const param (drives `instantiate::<…>()`)
//!
//! We use `usize` for const params (rather than `u32`) because Rust
//! requires `usize` for array lengths (`[u32; N]`), which is the sole
//! legitimate use case for const generics in WGSL. The proc-macro accepts
//! both `u32` and `usize`; we exercise `usize` here because it's the
//! natural Rust idiom and produces valid Rust that type-checks.
use wgsl_rs::wgsl;

// 1. Generic function with a const param, called locally.
#[wgsl(crate_path = wgsl_rs)]
mod const_fn {
    pub fn sum_n<const N: usize>(arr: [u32; N]) -> u32 {
        let mut total: u32 = 0u32;
        for i in 0..N {
            total = total + arr[i];
        }
        total
    }

    pub fn run() -> u32 {
        sum_n::<4>([1u32, 2u32, 3u32, 4u32])
    }
}

// 2. Generic struct with a const param, used locally.
#[wgsl(crate_path = wgsl_rs)]
mod const_struct {
    pub struct Grid<const N: usize> {
        pub cells: [u32; N],
    }

    pub fn run() -> u32 {
        let g = Grid::<4> { cells: [0u32, 0u32, 0u32, 0u32] };
        g.cells[0]
    }
}

// 3. Generic impl block with a const param.
#[wgsl(crate_path = wgsl_rs)]
mod const_impl {
    pub struct Grid<const N: usize> {
        pub cells: [u32; N],
    }

    impl<const N: usize> Grid<N> {
        pub fn first(cells: [u32; N]) -> u32 {
            cells[0]
        }
    }

    pub fn run() -> u32 {
        Grid::<4>::first([0u32, 0u32, 0u32, 0u32])
    }
}

// 4. Cross-module template instantiation: a const-generic function exported
//    from one `#[wgsl]` mod and invoked from another.
#[wgsl(crate_path = wgsl_rs)]
mod provider {
    pub fn external_sum<const N: usize>(arr: [u32; N]) -> u32 {
        let mut total: u32 = 0u32;
        for i in 0..N {
            total = total + arr[i];
        }
        total
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod consumer {
    use super::provider::*;

    pub fn run() -> u32 {
        external_sum::<4>([1u32, 2u32, 3u32, 4u32])
    }
}

// 5. Entry point with a const param, driving the `instantiate::<…>()`
//    codegen path.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod entry_point {
    use wgsl_rs::std::*;

    #[compute]
    #[workgroup_size(1)]
    pub fn main<const N: usize>() -> u32 {
        let arr: [u32; N] = [0u32; N];
        arr[0]
    }
}

fn main() {
    let _ = const_fn::run();
    let _ = const_struct::run();
    let _ = const_impl::run();
    let _ = consumer::run();
    let _ = entry_point::instantiate::<4>();
    let _ = const_fn::WGSL_SOURCE.wgsl_source();
    let _ = consumer::WGSL_SOURCE.wgsl_source();
}