//! CPU construction of annotation-driven resource packs.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod packs {
    use wgsl_rs::std::*;

    pub struct TexturePack<const TEXTURE_GROUP: u32, const FILTERABLE: bool> {
        #[texture(group(TEXTURE_GROUP), binding(4))]
        pub heights: Texture2D<f32, FILTERABLE>,

        #[sampler(binding(5), group(TEXTURE_GROUP))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    pub struct BufferPack<const BUFFER_GROUP: usize> {
        #[storage(group(BUFFER_GROUP), binding(2), read_write)]
        pub slab: Storage<RuntimeArray<Vec4f>>,

        #[uniform(group(BUFFER_GROUP), binding(3))]
        pub config: Uniform<Vec4f>,
    }

    pub struct ReadPack {
        #[storage(group(1), binding(7))]
        pub values: Storage<[u32; 4], Read>,
    }
}

#[wgsl(skip_validation)]
mod invalid_after_instantiation {
    use wgsl_rs::std::*;

    #[allow(dead_code)]
    pub struct DuplicateAddress<const GROUP: u32> {
        #[sampler(group(GROUP), binding(0))]
        pub first: Sampler,
        #[sampler(group(GROUP), binding(0))]
        pub second: Sampler,
    }

    #[allow(dead_code)]
    pub struct OutOfRange<const OUT_OF_RANGE_GROUP: usize> {
        #[sampler(group(OUT_OF_RANGE_GROUP), binding(0))]
        pub sampler: Sampler,
    }
}

#[test]
fn post_instantiation_addresses_are_range_checked_and_deduplicated() {
    let duplicate = std::panic::catch_unwind(invalid_after_instantiation::instantiate::<1, 0>)
        .expect_err("two resources must not share an instantiated address");
    let message = duplicate
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| duplicate.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic");
    assert!(message.contains("duplicate resource address"), "{message}");

    let invalid = std::panic::catch_unwind(
        invalid_after_instantiation::instantiate::<0, { u32::MAX as usize + 1 }>,
    )
    .expect_err("usize addresses must fit in WGSL's u32 address representation");
    let message = invalid
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| invalid.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic");
    assert!(
        message.contains("outside the u32 address range"),
        "{message}"
    );
}

#[test]
fn packs_construct_by_hand_with_values() {
    let _textures = packs::TexturePack::<2, false> {
        heights: Texture2D::new(),
        heights_smp: Sampler::new(),
    };

    let buffers = packs::BufferPack::<3> {
        slab: Storage::new(RuntimeArray {
            data: vec![vec4f(1.0, 2.0, 3.0, 4.0)],
        }),
        config: Uniform::new(vec4f(4.0, 3.0, 2.0, 1.0)),
    };
    assert_eq!(array_length(&buffers.slab), 1);
    assert_eq!(*buffers.config.get(), vec4f(4.0, 3.0, 2.0, 1.0));

    let read = packs::ReadPack {
        values: Storage::new([9u32, 8, 7, 6]),
    };
    assert_eq!(*read.values.get(), [9u32, 8, 7, 6]);
}

#[test]
fn constructed_texture_pack_works_with_cpu_builtins() {
    let textures = packs::TexturePack::<0, false> {
        heights: Texture2D::new(),
        heights_smp: Sampler::new(),
    };
    textures.heights.init(1, 1);
    textures.heights.set_pixel(0, 0, [0.25, 0.5, 0.75, 1.0]);
    textures.heights_smp.init();

    let color = texture_sample(&textures.heights, &textures.heights_smp, vec2f(0.0, 0.0));
    assert_eq!(color, vec4f(0.25, 0.5, 0.75, 1.0));

    let buffers = packs::BufferPack::<0> {
        slab: Storage::new(RuntimeArray {
            data: vec![vec4f(1.0, 2.0, 3.0, 4.0)],
        }),
        config: Uniform::new(vec4f(4.0, 3.0, 2.0, 1.0)),
    };
    assert_eq!(buffers.slab.get().data.len(), 1);
    assert_eq!(array_length(&buffers.slab), 1);
    assert_eq!(*buffers.config.get(), vec4f(4.0, 3.0, 2.0, 1.0));
}

#[test]
fn packs_are_not_emitted_as_wgsl_structs() {
    assert_eq!(
        packs::WGSL_SOURCE.module_type_params,
        &[
            "TexturePack_TEXTURE_GROUP",
            "TexturePack_FILTERABLE",
            "BufferPack_BUFFER_GROUP"
        ]
    );
    let module = packs::instantiate::<2, false, 3>();
    let source = module.wgsl_source();
    assert!(source.contains("@group(2) @binding(4) var texturepack__heights: texture_2d<f32>;"));
    assert!(source.contains("@group(2) @binding(5) var texturepack__heights_smp: sampler;"));
    assert!(
        source.contains(
            "@group(3) @binding(2) var<storage, read_write> bufferpack__slab: array<vec4f>;"
        ),
        "{source}"
    );
    assert!(source.contains("@group(3) @binding(3) var<uniform> bufferpack__config: vec4f;"));
    assert!(
        source
            .contains("@group(1) @binding(7) var<storage, read> readpack__values: array<u32, 4>;")
    );
    assert!(!source.contains("struct TexturePack"));
    assert!(!source.contains("struct BufferPack"));
    assert!(!source.contains("struct ReadPack"));
}
