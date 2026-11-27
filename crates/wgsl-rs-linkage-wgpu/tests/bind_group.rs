//! Typed bind-group creation via `#[derive(BindGroup)]`: zero string keys.
#![cfg(feature = "v30")]

use wgsl_rs::{std::*, wgsl};
use wgsl_rs_linkage_wgpu::{BindGroup, v30 as wg};

/// The atlas shape: a buffer, a texture, and a sampler sharing one layout
/// const, with fields deliberately declared out of binding order. The
/// derive is spelled with a full path inside the `#[wgsl]` module; the
/// `#[wgsl]` macro keeps the field annotations for the derive to re-read.
#[wgsl(validate_with_instantiation_types(ATLAS_FILTERABLE))]
mod atlas {
    use wgsl_rs::std::*;

    pub const ATLAS_FILTERABLE: bool = true;

    #[derive(wgsl_rs_linkage_wgpu::BindGroup)]
    #[allow(dead_code)]
    pub struct AtlasBindGroup<const FILTERABLE: bool> {
        #[texture(group(2), binding(1))]
        pub texture: Texture2D<f32, FILTERABLE>,

        #[storage(group(2), binding(0))]
        pub slab: Storage<[Vec4f; 4]>,

        #[sampler(group(2), binding(2))]
        pub smp: Sampler<FILTERABLE>,
    }

    /// Reads every field, so the CPU pack stays exercised.
    #[fragment]
    pub fn blit<const FILTERABLE: bool>(pack: &AtlasBindGroup<FILTERABLE>) -> Vec4f {
        let dims = texture_dimensions(&pack.texture);
        let texel = texture_load(&pack.texture, vec2i(0, 0), 0u32);
        let data = get!(pack.slab);
        vec4f(
            texel.x() + data[0].x() * 0.0 + dims.x() as f32 * 0.0,
            texel.y(),
            texel.z(),
            texel.w(),
        )
    }

    /// Full-screen triangle, so the fragment stage has a vertex feed.
    #[vertex]
    pub fn vtx_fullscreen(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }
}

/// A standalone descriptor: the same layout declared outside any `#[wgsl]`
/// module, sharing the shader pack's name so the generated keys match the
/// module's linkage. Layout-only consts ride along; `MyBg::<1>` *is*
/// group 1.
#[allow(dead_code)]
#[derive(BindGroup)]
pub struct MyBg<const GROUP: u32> {
    #[storage(group(GROUP), binding(0), read_only)]
    pub slab: Storage<[Vec4f; 4], Read>,
}

/// The `usize` flavor of the standalone descriptor. Standalone derives
/// skip the `#[wgsl]` `instantiate` pass, so an oversized group value
/// must be rejected at bind time instead of truncating.
#[allow(dead_code)]
#[derive(BindGroup)]
pub struct MyBgUsize<const GROUP: usize> {
    #[storage(group(GROUP), binding(0), read_only)]
    pub slab: Storage<[Vec4f; 4], Read>,
}

/// The three-slabs driver: one group-param pack instantiated at three
/// groups, driven through the typed path by the same `Resources` struct.
#[wgsl(validate_with_instantiation_types(SLAB_GROUP))]
mod slabs {
    use wgsl_rs::std::*;

    pub const SLAB_GROUP: u32 = 0;

    #[allow(dead_code)]
    #[derive(wgsl_rs_linkage_wgpu::BindGroup)]
    pub struct MyBg<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: Storage<[Vec4f; 4], Read>,
    }
}

/// The `usize` flavor: WGSL group addresses accept `usize` consts for
/// native-Rust compatibility. Everything downstream (the generated group
/// key const and the typed bind-group call) must stay `u32`, so a `usize`
/// suffix on the instantiation value cannot leak into address positions.
#[wgsl(validate_with_instantiation_types(SLAB_GROUP))]
mod slabs_usize {
    use wgsl_rs::std::*;

    pub const SLAB_GROUP: usize = 0;

    #[allow(dead_code)]
    #[derive(wgsl_rs_linkage_wgpu::BindGroup)]
    pub struct SlabBg<const GROUP: usize> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: Storage<[Vec4f; 4], Read>,
    }
}

/// Returns a `wgpu::Device` and `Queue` for tests, or `None` if no GPU is
/// available.
fn test_device() -> Option<(wg::wgpu::Device, wg::wgpu::Queue)> {
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
    let (device, queue) =
        block_on(adapter.request_device(&wg::wgpu::DeviceDescriptor::default())).ok()?;
    Some((device, queue))
}

#[test]
fn typed_path_builds_the_atlas_bind_group_on_device() {
    let Some((device, queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    // The pack stays CPU-constructible and callable alongside the typed
    // GPU path.
    let cpu_pack = atlas::AtlasBindGroup::<true> {
        texture: Texture2D::new(4, 4),
        slab: Storage::new([vec4f(0.0, 0.0, 0.0, 1.0); 4]),
        smp: Sampler::new(),
    };
    let _color = atlas::blit::<true>(&cpu_pack);
    let _position = atlas::vtx_fullscreen(0);

    let module = atlas::instantiate::<true>();
    let mut linkage = wg::analyze_ir_module(module);

    // The §14.4.1-sized buffer, a texture + view, and a sampler.
    let slab_buffer = linkage
        .buffer(atlas::AtlasBindGroup::<true>::SLAB)
        .expect("slab buffer descriptor")
        .create_buffer(&device);
    let texture = device.create_texture(&wg::wgpu::TextureDescriptor {
        label: Some("atlas_source"),
        size: wg::wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wg::wgpu::TextureDimension::D2,
        format: wg::wgpu::TextureFormat::Rgba8Unorm,
        usage: wg::wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wg::wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wg::wgpu::SamplerDescriptor {
        label: Some("atlas_sampler"),
        mag_filter: wg::wgpu::FilterMode::Nearest,
        min_filter: wg::wgpu::FilterMode::Nearest,
        mipmap_filter: wg::wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    // The typed path: named resource fields, zero string keys.
    let bind_group = <atlas::AtlasBindGroup<true> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        atlas::AtlasBindGroupResources {
            slab: slab_buffer.as_entire_buffer_binding(),
            texture: &view,
            smp: &sampler,
        },
        Some("atlas_typed"),
    );
    let _ = bind_group;

    // The layout const vanishes from the resources: the false flavor
    // shares the same resource struct.
    let module = atlas::instantiate::<false>();
    let mut linkage = wg::analyze_ir_module(module);
    let bind_group = <atlas::AtlasBindGroup<false> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        atlas::AtlasBindGroupResources {
            slab: slab_buffer.as_entire_buffer_binding(),
            texture: &view,
            smp: &sampler,
        },
        None,
    );

    // End to end: pipeline + render pass with the typed bind group set at
    // the group the pack itself selects.
    let shader = linkage.shader_module(&device);
    let layout = linkage.pipeline_layout(&device, Some("atlas_pipeline"));
    let pipeline = device.create_render_pipeline(&wg::wgpu::RenderPipelineDescriptor {
        label: Some("atlas_pipeline"),
        layout: Some(&layout),
        vertex: wg::wgpu::VertexState {
            module: &shader,
            entry_point: Some("vtx_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wg::wgpu::FragmentState {
            module: &shader,
            entry_point: Some("blit"),
            targets: &[Some(wg::wgpu::ColorTargetState {
                format: wg::wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wg::wgpu::ColorWrites::all(),
            })],
            compilation_options: Default::default(),
        }),
        primitive: wg::wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wg::wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let target = device.create_texture(&wg::wgpu::TextureDescriptor {
        label: Some("atlas_target"),
        size: wg::wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wg::wgpu::TextureDimension::D2,
        format: wg::wgpu::TextureFormat::Rgba8Unorm,
        usage: wg::wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wg::wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wg::wgpu::RenderPassDescriptor {
            label: Some("atlas_blit"),
            color_attachments: &[Some(wg::wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wg::wgpu::Operations {
                    load: wg::wgpu::LoadOp::Clear(wg::wgpu::Color::TRANSPARENT),
                    store: wg::wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    let _index = queue.submit(Some(encoder.finish()));
}

#[test]
fn typed_path_binds_three_slabs_by_the_same_resources() {
    let Some((device, _queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    fn bind_at<const GROUP: u32>(device: &wg::wgpu::Device) -> wg::wgpu::BindGroup {
        let module = slabs::instantiate::<GROUP>();
        let mut linkage = wg::analyze_ir_module(module);
        let buffer = linkage
            .buffer(slabs::MyBg::<GROUP>::SLAB)
            .expect("slab buffer descriptor")
            .create_buffer(device);
        <slabs::MyBg<GROUP> as BindGroup>::create_bind_group(
            &mut linkage,
            device,
            slabs::MyBgResources {
                slab: buffer.as_entire_buffer_binding(),
            },
            None,
        )
    }

    let _slab0 = bind_at::<0>(&device);
    let _slab1 = bind_at::<1>(&device);
    let _slab2 = bind_at::<2>(&device);
}

#[test]
fn usize_group_params_narrow_to_u32_and_bind_on_device() {
    let Some((device, _queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    // The generated group key const stays `u32` even though the pack's
    // parameter is `usize`, and it carries the instantiation value.
    let _: u32 = slabs_usize::SlabBg::<3>::GROUP_N;
    assert_eq!(slabs_usize::SlabBg::<3>::GROUP_N, 3u32);
    assert_eq!(slabs_usize::SlabBg::<0>::GROUP_N, 0u32);

    // The typed path binds at that group on device: the IR carries
    // `@group(3)` and the derive's narrowed `3u32` matches it.
    let module = slabs_usize::instantiate::<3>();
    let mut linkage = wg::analyze_ir_module(module);
    let buffer = linkage
        .buffer(slabs_usize::SlabBg::<3>::SLAB)
        .expect("slab buffer descriptor")
        .create_buffer(&device);
    let bind_group = <slabs_usize::SlabBg<3> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        slabs_usize::SlabBgResources {
            slab: buffer.as_entire_buffer_binding(),
        },
        Some("usize_group"),
    );
    let _ = bind_group;
}

#[test]
#[should_panic(expected = "group index does not fit in u32")]
fn oversized_usize_group_panics_instead_of_truncating() {
    let Some((device, _queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    // Standalone derives skip the `#[wgsl]` `instantiate` pass, so the
    // bind-time conversion must be checked: an `as u32` cast would
    // truncate `u32::MAX as usize + 1` to 0 and silently bind the wrong
    // group.
    let module = slabs::instantiate::<0>();
    let mut linkage = wg::analyze_ir_module(module);
    let buffer = linkage
        .buffer(slabs::MyBg::<0>::SLAB)
        .expect("slab buffer descriptor")
        .create_buffer(&device);
    const HUGE: usize = u32::MAX as usize + 1;
    let bind_group = <MyBgUsize<HUGE> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        MyBgUsizeResources {
            slab: buffer.as_entire_buffer_binding(),
        },
        None,
    );
    let _ = bind_group;
}

#[test]
fn standalone_descriptor_drives_a_module_linkage() {
    let Some((device, _queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    // The standalone `MyBg` descriptor shares the module pack's name, so
    // its generated keys resolve against the module's linkage.
    let module = slabs::instantiate::<1>();
    let mut linkage = wg::analyze_ir_module(module);
    let buffer = linkage
        .buffer("MyBg::slab")
        .expect("slab buffer descriptor")
        .create_buffer(&device);
    let bind_group = <MyBg<1> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        MyBgResources {
            slab: buffer.as_entire_buffer_binding(),
        },
        Some("standalone"),
    );
    let _ = bind_group;
}

#[test]
#[should_panic(expected = "bind group creation for `MyBg` failed")]
fn mismatched_group_panics_with_the_linkage_error() {
    let Some((device, _queue)) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };

    // The linkage declares group 0 only (SLAB_GROUP); the descriptor asks
    // for group 2: an instantiation mismatch, surfaced as a panic.
    let module = slabs::instantiate::<0>();
    let mut linkage = wg::analyze_ir_module(module);
    let buffer = device.create_buffer(&wg::wgpu::BufferDescriptor {
        label: Some("slab"),
        size: 64,
        usage: wg::wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    <MyBg<2> as BindGroup>::create_bind_group(
        &mut linkage,
        &device,
        MyBgResources {
            slab: buffer.as_entire_buffer_binding(),
        },
        None,
    );
}
