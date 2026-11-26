//! Parameterized resource-pack addresses in WGSL and wgpu layouts.
#![cfg(feature = "v30")]

use wgsl_rs::wgsl;
use wgsl_rs_linkage_wgpu::v30 as wg;

#[wgsl(validate_with_instantiation_types(TEST_GROUP, TEST_BINDING, TEST_FILTERABLE))]
mod packs {
    use wgsl_rs::std::*;

    pub const TEST_GROUP: u32 = 4;
    pub const TEST_FILTERABLE: bool = false;
    pub const TEST_BINDING: usize = 7;

    pub struct TexturePack<const GROUP: u32, const FILTERABLE: bool, const BINDING: usize> {
        #[texture(group(GROUP), binding(BINDING))]
        pub texture: Texture2D<f32, FILTERABLE>,

        #[sampler(group(GROUP), binding(8))]
        pub smp: Sampler<FILTERABLE>,
    }

    pub struct BufferPack {
        #[storage(group(2), binding(3), read_only)]
        pub input: RuntimeArray<Vec4f>,
    }
}

#[test]
fn pack_addresses_and_filterability_reach_wgpu_layouts() {
    assert_eq!(
        packs::WGSL_SOURCE.module_type_params,
        &[
            "TexturePack_GROUP",
            "TexturePack_BINDING",
            "TexturePack_FILTERABLE"
        ]
    );

    let cpu = packs::TexturePack::<4, false, 7>::new();
    assert_eq!(cpu.texture.group(), 4);
    assert_eq!(cpu.texture.binding(), 7);
    assert_eq!(cpu.smp.group(), 4);
    assert_eq!(cpu.smp.binding(), 8);
    let _buffers = packs::BufferPack::new();

    for filterable in [true, false] {
        let module = if filterable {
            packs::instantiate::<4, 7, true>()
        } else {
            packs::instantiate::<4, 7, false>()
        };
        let source = module.wgsl_source();
        assert!(
            source.contains("@group(4) @binding(7) var texturepack__texture: texture_2d<f32>;")
        );
        assert!(source.contains("@group(4) @binding(8) var texturepack__smp: sampler;"));
        assert!(
            source.contains(
                "@group(2) @binding(3) var<storage, read> bufferpack__input: array<vec4f>;"
            )
        );
        assert!(!source.contains("struct TexturePack"));

        let linkage = wg::analyze_ir_module(module);
        let bg = linkage.bind_group(4).unwrap();
        assert_eq!(bg.entries[0].binding, 7);
        assert!(matches!(
            bg.entries[0].ty,
            wg::wgpu::BindingType::Texture {
                sample_type: wg::wgpu::TextureSampleType::Float { filterable: actual },
                ..
            } if actual == filterable
        ));
        let expected_sampler = if filterable {
            wg::wgpu::SamplerBindingType::Filtering
        } else {
            wg::wgpu::SamplerBindingType::NonFiltering
        };
        assert!(matches!(
            bg.entries[1].ty,
            wg::wgpu::BindingType::Sampler(actual) if actual == expected_sampler
        ));
    }
}
