//! CPU construction of annotation-driven resource packs.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod packs {
    use wgsl_rs::std::*;

    pub struct TexturePack<const GROUP: u32, const FILTERABLE: bool> {
        #[texture(group(GROUP), binding(4))]
        pub heights: Texture2D<f32, FILTERABLE>,

        #[sampler(binding(5), group(GROUP))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    pub struct BufferPack<const GROUP: usize> {
        #[storage(group(GROUP), binding(2), read_write)]
        pub slab: RuntimeArray<Vec4f>,

        #[uniform(group(GROUP), binding(3))]
        pub config: Vec4f,
    }

    pub struct ReadPack {
        #[storage(group(1), binding(7))]
        pub values: [u32; 4],
    }
}

#[test]
fn constructors_use_annotated_addresses_and_const_values() {
    let textures = packs::TexturePack::<2, false>::new();
    assert_eq!(textures.heights.group(), 2);
    assert_eq!(textures.heights.binding(), 4);
    assert_eq!(textures.heights_smp.group(), 2);
    assert_eq!(textures.heights_smp.binding(), 5);

    let buffers = packs::BufferPack::<3>::new();
    assert_eq!(buffers.slab.group(), 3);
    assert_eq!(buffers.slab.binding(), 2);
    assert_eq!(buffers.config.group, 3);
    assert_eq!(buffers.config.binding, 3);

    let read = packs::ReadPack::new();
    assert_eq!(read.values.group(), 1);
    assert_eq!(read.values.binding(), 7);
}

#[test]
fn constructed_texture_pack_works_with_cpu_builtins() {
    let textures = packs::TexturePack::<0, false>::new();
    textures.heights.init(1, 1);
    textures.heights.set_pixel(0, 0, [0.25, 0.5, 0.75, 1.0]);
    textures.heights_smp.init();

    let color = texture_sample(&textures.heights, &textures.heights_smp, vec2f(0.0, 0.0));
    assert_eq!(color, vec4f(0.25, 0.5, 0.75, 1.0));

    let buffers = packs::BufferPack::<0>::new();
    buffers.slab.set(RuntimeArray {
        data: vec![vec4f(1.0, 2.0, 3.0, 4.0)],
    });
    buffers.config.set(vec4f(4.0, 3.0, 2.0, 1.0));
    assert_eq!(buffers.slab.get().data.len(), 1);
    assert_eq!(*buffers.config.get(), vec4f(4.0, 3.0, 2.0, 1.0));
}

#[test]
fn packs_are_not_emitted_as_wgsl_structs() {
    let source = packs::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(!source.contains("TexturePack"));
    assert!(!source.contains("BufferPack"));
    assert!(!source.contains("ReadPack"));
}
