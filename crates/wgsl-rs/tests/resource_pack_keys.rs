//! Generated binding-key consts on resource packs.
//!
//! Every annotated field gets a `pub const {FIELD}: &'static str =
//! "Pack::field"` spelling, independent of layout-only consts, and every
//! const parameter that selects the group axis gets a `pub const
//! {PARAM}_N` value const so callers never hand-mangle strings.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod atlas {
    use wgsl_rs::std::*;

    #[allow(dead_code)]
    pub struct AtlasBindGroup<const FILTERABLE: bool> {
        #[storage(group(0), binding(0))]
        pub slab: Storage<RuntimeArray<u32>>,
        #[texture(group(0), binding(1))]
        pub texture: Texture2D<f32, FILTERABLE>,
        #[sampler(group(0), binding(2))]
        pub smp: Sampler<FILTERABLE>,
    }
}

#[test]
fn field_consts_are_parameter_independent_friendly_keys() {
    let _constructed = atlas::AtlasBindGroup::<true> {
        slab: Storage::new(RuntimeArray::new()),
        texture: Texture2D::new(),
        smp: Sampler::new(),
    };
    assert_eq!(atlas::AtlasBindGroup::<true>::SLAB, "AtlasBindGroup::slab");
    // Layout-only consts (FILTERABLE) vanish from the key: every
    // instantiation shares one spelling.
    assert_eq!(atlas::AtlasBindGroup::<false>::SLAB, "AtlasBindGroup::slab");
    assert_eq!(
        atlas::AtlasBindGroup::<true>::TEXTURE,
        "AtlasBindGroup::texture"
    );
    assert_eq!(atlas::AtlasBindGroup::<false>::SMP, "AtlasBindGroup::smp");
}

#[wgsl(skip_validation)]
mod grouped {
    use wgsl_rs::std::*;

    #[allow(dead_code)]
    pub struct MyBg<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: Storage<RuntimeArray<Vec4f>, Read>,
    }
}

#[test]
fn group_consts_ride_the_group_axis() {
    let _constructed = grouped::MyBg::<2> {
        slab: Storage::new(RuntimeArray::new()),
    };
    assert_eq!(grouped::MyBg::<0>::GROUP_N, 0);
    assert_eq!(grouped::MyBg::<1>::GROUP_N, 1);
    assert_eq!(grouped::MyBg::<2>::GROUP_N, 2);
    assert_eq!(grouped::MyBg::<2>::SLAB, "MyBg::slab");
}
