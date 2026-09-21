//! Cross-source template instances must publish their signatures before
//! the calling source's chunk is suffixed (PR #194 review follow-up):
//! `choose::<u32>(select(0, 1, data))` anchors the nested `select`
//! literals from the instance's concrete parameter type.

use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod provider {
    pub fn choose<T: Copy>(x: T) -> T {
        x
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod consumer {
    use super::provider::*;
    use wgsl_rs::std::*;

    pub fn run(data: bool) -> u32 {
        choose::<u32>(select(0, 1, data))
    }
}

#[test]
fn template_call_args_anchor_from_instance_signatures() {
    // CPU world: the consumer runs on the host like any Rust code.
    assert_eq!(consumer::run(true), 1);
    assert_eq!(consumer::run(false), 0);

    // WGSL world: the nested `select` literals are suffixed from
    // `choose_u32`'s concrete `u32` parameter.
    let src = consumer::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("select(0u, 1u, data)"),
        "nested select should be suffixed from the instance's u32 param, got:\n{src}"
    );
    // Sanity: the template was actually instantiated.
    assert!(src.contains("choose_u32"), "got:\n{src}");
}
