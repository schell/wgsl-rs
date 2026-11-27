//! Parameterized resource-pack addresses in WGSL and wgpu layouts.
#![cfg(feature = "v30")]

use wgsl_rs::{std::*, wgsl};
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
        pub input: Storage<RuntimeArray<Vec4f>, Read>,
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

    let cpu = packs::TexturePack::<4, false, 7> {
        texture: Texture2D::new(),
        smp: Sampler::new(),
    };
    cpu.texture.init(1, 1);
    cpu.smp.init();
    let buffers = packs::BufferPack {
        input: Storage::new(RuntimeArray::new()),
    };
    assert_eq!(array_length(&buffers.input), 0);

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

/// Packs expose binding-key consts: `"Pack::field"` spellings that are
/// independent of layout-only consts (FILTERABLE), and a `{PARAM}_N` value
/// const for consts that select the group axis. The friendly keys resolve
/// in `buffer()` lookups alongside the mangled names.
#[test]
fn pack_field_consts_key_bindings_by_friendly_name() {
    // Layout-only consts vanish from the key: every instantiation shares
    // one spelling.
    assert_eq!(
        packs::TexturePack::<4, false, 7>::TEXTURE,
        "TexturePack::texture"
    );
    assert_eq!(
        packs::TexturePack::<6, true, 1>::TEXTURE,
        "TexturePack::texture"
    );
    assert_eq!(packs::TexturePack::<4, false, 7>::SMP, "TexturePack::smp");
    assert_eq!(packs::BufferPack::INPUT, "BufferPack::input");
    // The group rides the const-param axis.
    assert_eq!(packs::TexturePack::<4, false, 7>::GROUP_N, 4);
    assert_eq!(packs::TexturePack::<6, false, 1>::GROUP_N, 6);

    let module = packs::instantiate::<4, 7, false>();
    let linkage = wg::analyze_ir_module(module);
    let buf = linkage
        .buffer(packs::BufferPack::INPUT)
        .expect("friendly key finds the buffer");
    assert_eq!(buf.group, 2);
    assert_eq!(buf.binding, 3);
    assert!(linkage.buffer("bufferpack__input").is_some());
    assert!(linkage.buffer("BufferPack::nope").is_none());
}

/// Helper: returns a `wgpu::Device` for tests, or `None` if no GPU is
/// available. Device-backed tests early-return when this is `None`.
fn test_device() -> Option<wg::wgpu::Device> {
    use futures::executor::block_on;
    let instance = wg::wgpu::Instance::new(wg::wgpu::InstanceDescriptor {
        backends: wg::wgpu::Backends::all(),
        ..wg::wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wg::wgpu::RequestAdapterOptions {
        power_preference: wg::wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let (device, _queue) =
        block_on(adapter.request_device(&wg::wgpu::DeviceDescriptor::default())).ok()?;
    Some(device)
}

/// The design driver: `MyBg<const GROUP: u32>` bound in three groups. The
/// two-axis key `(group, "MyBg::slab")` stays collision-free across all
/// three per-instantiation linkages, and `GROUP_N` selects the group axis.
#[wgsl(validate_with_instantiation_types(SLAB_GROUP))]
mod slabs {
    use wgsl_rs::std::*;

    pub const SLAB_GROUP: u32 = 0;

    #[allow(dead_code)]
    pub struct MyBg<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: Storage<[Vec4f; 4], Read>,
    }
}

#[test]
fn group_param_packs_bind_by_the_same_key_in_three_groups() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    fn bind_at<const GROUP: u32>(device: &wg::wgpu::Device) {
        let module = slabs::instantiate::<GROUP>();
        let mut linkage = wg::analyze_ir_module(module);
        let info = linkage
            .buffer(slabs::MyBg::<GROUP>::SLAB)
            .expect("friendly key finds the slab buffer");
        assert_eq!(info.group, GROUP);
        assert_eq!(info.binding, 0);

        let buffer = info.create_buffer(device);
        let bind_group = linkage
            .create_bind_group_named(
                slabs::MyBg::<GROUP>::GROUP_N,
                device,
                &[(slabs::MyBg::<GROUP>::SLAB, buffer.as_entire_binding())],
            )
            .expect("friendly key creates the bind group");
        let _ = bind_group;
    }

    bind_at::<0>(&device);
    bind_at::<1>(&device);
    bind_at::<2>(&device);
}
