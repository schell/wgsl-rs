//! Workgroup variables as `#[workgroup]` pack fields.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod shared {
    use wgsl_rs::std::*;

    pub struct WorkgroupData<T> {
        #[workgroup]
        pub accumulator: Workgroup<T>,
    }

    #[compute]
    #[workgroup_size(4)]
    pub fn accumulate(wg: &WorkgroupData<i32>) {
        let current = load!(wg.accumulator);
        *get_mut!(wg.accumulator) = current + 1;
    }
}

#[test]
fn workgroup_pack_fields_accumulate_on_cpu() {
    let wg = shared::WorkgroupData::<i32> {
        accumulator: Workgroup::new(0),
    };
    shared::accumulate(&wg);
    assert_eq!(*wg.accumulator.get(), 1);
    shared::accumulate(&wg);
    assert_eq!(*wg.accumulator.get(), 2);
}

#[test]
fn workgroup_pack_fields_render_workgroup_vars() {
    assert_eq!(shared::WGSL_SOURCE.module_type_params, &["WorkgroupData_T"]);
    let module = shared::instantiate::<i32>();
    let source = module.wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(
        source.contains("var<workgroup> workgroupdata__accumulator: i32;"),
        "{source}"
    );
    assert!(!source.contains("WorkgroupData"), "{source}");
}
