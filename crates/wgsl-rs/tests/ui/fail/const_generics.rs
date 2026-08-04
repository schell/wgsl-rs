//! Failing tests for `const N: u32` const generics across every surface that
//! should eventually support them:
//! 1. Free function declared and called locally
//! 2. Generic struct declared and used locally
//! 3. Generic impl block over a const-generic struct
//! 4. Cross-module template instantiation
//! 5. Entry-point function with a const param (drives `instantiate::<…>()`)
//!
//! Today every mod below fails because the `#[wgsl]` proc-macro rejects
//! `syn::GenericParam::Const` at parse time. When const generics for `u32`
//! land, this file should compile cleanly and the `trybuild.rs` registration
//! flips from `compile_fail` to `pass`.
use wgsl_rs::wgsl;

// 1. Generic function with a const param, called locally.
#[wgsl(crate_path = wgsl_rs)]
mod const_fn {
    pub fn sum_n<const N: u32>(arr: [u32; N]) -> u32 {
        let mut total: u32 = 0u;
        for i in 0..N {
            total = total + arr[i];
        }
        total
    }

    pub fn run() -> u32 {
        sum_n::<4>([1u, 2u, 3u, 4u])
    }
}

// 2. Generic struct with a const param, used locally.
#[wgsl(crate_path = wgsl_rs)]
mod const_struct {
    pub struct Grid<const N: u32> {
        pub cells: [u32; N],
    }

    pub fn run() -> u32 {
        let g = Grid::<4> { cells: [0u, 0u, 0u, 0u] };
        g.cells[0]
    }
}

// 3. Generic impl block with a const param.
#[wgsl(crate_path = wgsl_rs)]
mod const_impl {
    pub struct Grid<const N: u32> {
        pub cells: [u32; N],
    }

    impl<const N: u32> Grid<N> {
        pub fn first(&self) -> u32 {
            self.cells[0]
        }
    }

    pub fn run() -> u32 {
        let g = Grid::<4> { cells: [0u, 0u, 0u, 0u] };
        g.first()
    }
}

// 4. Cross-module template instantiation: a const-generic function exported
//    from one `#[wgsl]` mod and invoked from another.
#[wgsl(crate_path = wgsl_rs)]
mod provider {
    pub fn external_sum<const N: u32>(arr: [u32; N]) -> u32 {
        let mut total: u32 = 0u;
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
        external_sum::<4>([1u, 2u, 3u, 4u])
    }
}

// 5. Entry point with a const param, driving the `instantiate::<…>()`
//    codegen path.
#[wgsl(crate_path = wgsl_rs)]
mod entry_point {
    use wgsl_rs::std::*;

    #[compute]
    pub fn main<const N: u32>() -> u32 {
        let arr: [u32; N] = [0u; N];
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