//! Parsed `for` loops never carry a loop-variable annotation (`for i:
//! u32 in …` is not valid Rust), so the suffix pass infers the loop
//! variable's type from a provable range bound (PR #194 review
//! follow-up).

use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod loops {
    pub fn sum_to(n: u32) -> u32 {
        let mut total: u32 = 0;
        #[wgsl_allow(non_literal_loop_bounds)]
        for i in 0..n {
            total += i;
        }
        total
    }
}

#[test]
fn for_bounds_infer_the_typed_range_bound() {
    // CPU world: the module runs on the host like any Rust code.
    assert_eq!(loops::sum_to(4), 6);

    // WGSL world: the loop bounds are suffixed from `n`'s u32.
    let src = loops::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("var i = 0u;"),
        "loop lower bound should be u32, got:\n{src}"
    );
    assert!(src.contains("i < n;"), "got:\n{src}");
}
