//! A bool const-param entry point instantiates to a concrete module whose
//! entry point keeps the un-mangled Rust name (`main`).
//!
//! Const-param instantiation does not rename entry points: each
//! `instantiate::<B>()` assembles its own concrete IR module, so there is
//! no name collision to mangle away (only helper template functions
//! mangle, e.g. `fog_scale_true`). `wgsl-rs-linkage-wgpu` builds its
//! compute entry list from the IR fn names and `compute_entry(name)`
//! does an exact-match lookup, so users write lookups with the plain
//! Rust name. This test pins that invariant: if entry points ever start
//! mangling (for example to `main_true`), un-mangled linkage lookups
//! would silently break, and this fails first.

use wgsl_rs::{std::Storage, wgsl};

#[wgsl(validate_with_instantiation_types(APPLY_TRUE))]
mod flag_entry {
    use wgsl_rs::std::*;

    const APPLY_TRUE: bool = true;

    pub struct FlagPack {
        #[storage(group(0), binding(0), read_write)]
        pub output: Storage<[u32; 1]>,
    }

    #[compute]
    #[workgroup_size(1)]
    pub fn main<const APPLY_FOG: bool>(pack: &FlagPack) {
        if APPLY_FOG {
            get_mut!(pack.output)[0] = 1u32;
        } else {
            get_mut!(pack.output)[0] = 0u32;
        }
    }
}

/// The compute entry point names of an instantiated module, in the same
/// form the linkage's `compute_entries` are built from.
fn compute_entry_names(module: &wgsl_rs::ir::Module) -> Vec<String> {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            wgsl_rs::ir::Item::Fn(f)
                if matches!(f.fn_attrs, wgsl_rs::ir::FnAttrs::Compute { .. }) =>
            {
                Some(f.name.to_string())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn instantiated_entry_point_keeps_unmangled_name() {
    let on = flag_entry::instantiate::<true>();
    let on_src = wgsl_rs::ir::render_module(&on);
    wgsl_rs::validate_wgsl_source(&on_src).expect("true instantiation must validate");
    assert_eq!(
        compute_entry_names(&on),
        vec!["main".to_string()],
        "entry point must keep the un-mangled Rust name for linkage lookups"
    );
    assert!(
        on_src.contains("if true"),
        "flag must render a bool literal: {on_src}"
    );

    let off = flag_entry::instantiate::<false>();
    let off_src = wgsl_rs::ir::render_module(&off);
    wgsl_rs::validate_wgsl_source(&off_src).expect("false instantiation must validate");
    assert_eq!(
        compute_entry_names(&off),
        vec!["main".to_string()],
        "entry point must keep the un-mangled Rust name for linkage lookups"
    );
    assert!(
        off_src.contains("if false"),
        "flag must render a bool literal: {off_src}"
    );

    // The CPU world evaluates the same branches at compile time: running
    // both instantiations through the Rust fn must not panic, and the
    // flag must actually change the result.
    let on_pack = flag_entry::FlagPack {
        output: Storage::new([0u32; 1]),
    };
    flag_entry::main::<true>(&on_pack);
    assert_eq!(on_pack.output.get()[0], 1, "CPU true-branch");

    let off_pack = flag_entry::FlagPack {
        output: Storage::new([0u32; 1]),
    };
    flag_entry::main::<false>(&off_pack);
    assert_eq!(off_pack.output.get()[0], 0, "CPU false-branch");
}
