//! Shared pack-owned const slots and independent pack layout selection.

use wgsl_rs::{ir, std::*, wgsl};

#[wgsl(skip_validation)]
mod shared {
    use wgsl_rs::std::*;

    pub struct SharedPack<const GROUP: u32, const FILTERABLE: bool> {
        #[texture(group(GROUP), binding(0))]
        pub texture: Texture2D<f32, FILTERABLE>,
        #[sampler(group(GROUP), binding(1))]
        pub smp: Sampler<FILTERABLE>,
    }

    #[vertex]
    pub fn vertex<const V_GROUP: u32, const V_FILTERABLE: bool, const V_LOCAL: bool>(
        #[builtin(vertex_index)] index: u32,
        pack: &SharedPack<V_GROUP, V_FILTERABLE>,
    ) -> Vec4f {
        let _dimensions = texture_dimensions(&pack.texture);
        let from_shared = V_GROUP as f32 + select(0.0, 1.0, V_FILTERABLE);
        let from_local = select(0.0, 2.0, V_LOCAL);
        vec4f(from_shared + from_local, index as f32, 0.0, 1.0)
    }

    #[fragment]
    pub fn fragment<const F_GROUP: u32, const F_FILTERABLE: bool>(
        #[location(0)] position: Vec4f,
        pack: &SharedPack<F_GROUP, F_FILTERABLE>,
    ) -> Vec4f {
        let sampled = texture_sample(&pack.texture, &pack.smp, position.xy());
        sampled
            + vec4f(
                F_GROUP as f32 + select(0.0, 1.0, F_FILTERABLE),
                0.0,
                0.0,
                0.0,
            )
    }
}

#[test]
fn entry_points_share_pack_values_while_unrelated_consts_stay_independent() {
    assert_eq!(
        shared::WGSL_SOURCE.module_type_params,
        &["SharedPack_GROUP", "SharedPack_FILTERABLE", "vertex_c2"]
    );
    let module = shared::instantiate::<3, false, true>();
    let source = module.wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(source.contains("@group(3) @binding(0)"), "{source}");
    assert!(source.contains("@group(3) @binding(1)"), "{source}");
    assert!(!source.contains("V_GROUP"), "{source}");
    assert!(!source.contains("F_GROUP"), "{source}");
    assert!(!source.contains("V_FILTERABLE"), "{source}");
    assert!(!source.contains("F_FILTERABLE"), "{source}");
    assert!(!source.contains("SharedPack<"), "{source}");

    let pack = shared::SharedPack::<3, false> {
        texture: Texture2D::new(),
        smp: Sampler::new(),
    };
    pack.texture.init(1, 1);
    pack.texture.set_pixel(0, 0, [1.0, 0.0, 0.0, 1.0]);
    pack.smp.init();
    let vertex = shared::vertex::<3, false, true>(0, &pack);
    assert_eq!(vertex, vec4f(5.0, 0.0, 0.0, 1.0));
    let fragment = shared::fragment::<3, false>(vec4f(0.0, 0.0, 0.0, 1.0), &pack);
    assert_eq!(fragment, vec4f(4.0, 0.0, 0.0, 1.0));

    let texture = module
        .items
        .iter()
        .find_map(|item| match item {
            ir::Item::Texture(texture) => Some(texture),
            _ => None,
        })
        .unwrap();
    assert_eq!(texture.filterable.concrete(), Some(false));
}

#[wgsl(skip_validation)]
mod independent {
    use wgsl_rs::std::*;

    #[allow(dead_code)]
    pub struct PackA<const GROUP: u32> {
        #[sampler(group(GROUP), binding(0))]
        pub a: Sampler,
    }

    #[allow(dead_code)]
    pub struct PackB<const GROUP: u32> {
        #[sampler(group(GROUP), binding(0))]
        pub b: Sampler,
    }

    #[vertex]
    pub fn vertex<const V_GROUP: u32>(_pack: &PackA<V_GROUP>) -> Vec4f {
        vec4f(V_GROUP as f32, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn fragment<const F_GROUP: u32>(_pack: &PackB<F_GROUP>) -> Vec4f {
        vec4f(F_GROUP as f32, 0.0, 0.0, 1.0)
    }
}

#[test]
fn separate_packs_have_independent_slots_even_with_matching_param_names() {
    assert_eq!(
        independent::WGSL_SOURCE.module_type_params,
        &["PackA_GROUP", "PackB_GROUP"]
    );
    let source = independent::instantiate::<1, 2>().wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(
        source.contains("@group(1) @binding(0) var packa__a: sampler;"),
        "{source}"
    );
    assert!(
        source.contains("@group(2) @binding(0) var packb__b: sampler;"),
        "{source}"
    );

    let a = independent::PackA::<1> { a: Sampler::new() };
    let b = independent::PackB::<2> { b: Sampler::new() };
    assert_eq!(independent::vertex::<1>(&a), vec4f(1.0, 0.0, 0.0, 1.0));
    assert_eq!(independent::fragment::<2>(&b), vec4f(2.0, 0.0, 0.0, 1.0));

    let collision = std::panic::catch_unwind(independent::instantiate::<1, 1>)
        .expect_err("instantiation must recheck addresses after substitution");
    let message = collision
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| collision.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic");
    assert!(message.contains("duplicate resource address"), "{message}");
    assert!(message.contains("a") && message.contains("b"), "{message}");
}

#[wgsl]
mod fixed {
    use wgsl_rs::std::*;

    pub struct FixedPack<const FILTERABLE: bool> {
        #[texture(group(0), binding(0))]
        pub texture: Texture2D<f32, FILTERABLE>,
        #[sampler(group(0), binding(1))]
        pub smp: Sampler<FILTERABLE>,
    }

    #[fragment]
    pub fn fragment(#[location(0)] position: Vec4f, pack: &FixedPack<false>) -> Vec4f {
        texture_sample(&pack.texture, &pack.smp, position.xy())
    }
}

/// Concrete const arguments specialize immediately: a pack used at a
/// concrete GROUP supplies no generic slot, and its address value must
/// reach the assembled IR as a literal (not an unresolved parameter).
#[wgsl]
mod concrete_group {
    use wgsl_rs::std::*;

    pub struct GroupedPack<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub data: Storage<[Vec4f; 4], Read>,
    }

    #[fragment]
    pub fn shade(#[location(0)] position: Vec4f, pack: &GroupedPack<3>) -> Vec4f {
        let data = get!(pack.data);
        position + data[0] * 0.0
    }
}

#[test]
fn concrete_group_addresses_specialize_into_the_ir() {
    assert!(concrete_group::WGSL_SOURCE.module_type_params.is_empty());
    let module = concrete_group::instantiate();
    let source = module.wgsl_source();
    assert!(
        source.contains("@group(3) @binding(0) var<storage, read> groupedpack__data:"),
        "expected the concrete group to reach the rendered source, got:\n{source}"
    );
    wgsl_rs::validate_wgsl_source(&source).unwrap();

    // The CPU pack still constructs and runs at its concrete layout.
    let pack = concrete_group::GroupedPack::<3> {
        data: Storage::new([vec4f(0.0, 0.0, 0.0, 0.0); 4]),
    };
    let color = concrete_group::shade(vec4f(1.0, 2.0, 3.0, 4.0), &pack);
    assert_eq!(color, vec4f(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn matching_concrete_uses_need_no_generic_slot() {
    assert!(fixed::WGSL_SOURCE.module_type_params.is_empty());
    let module = fixed::instantiate();
    for item in &module.items {
        match item {
            ir::Item::Texture(texture) => {
                assert_eq!(texture.filterable.concrete(), Some(false));
            }
            ir::Item::Sampler(sampler) => {
                assert_eq!(sampler.filterable.concrete(), Some(false));
            }
            _ => {}
        }
    }
    wgsl_rs::validate_wgsl_source(&module.wgsl_source()).unwrap();

    let pack = fixed::FixedPack::<false> {
        texture: Texture2D::new(),
        smp: Sampler::new(),
    };
    pack.texture.init(1, 1);
    pack.texture.set_pixel(0, 0, [0.0, 1.0, 0.0, 1.0]);
    pack.smp.init();
    assert_eq!(
        fixed::fragment(vec4f(0.0, 0.0, 0.0, 1.0), &pack),
        vec4f(0.0, 1.0, 0.0, 1.0)
    );
}
