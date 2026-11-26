//! Vertex + fragment resource-pack round-trip with storage and texture data.

#![allow(dead_code)]

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;

#[wgsl(skip_validation)]
pub mod resource_pack_render {
    use wgsl_rs::std::*;

    pub struct SlabBindGroup<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: RuntimeArray<Vec4f>,
    }

    pub struct TextureBindGroup<const GROUP: u32, const FILTERABLE: bool> {
        #[texture(group(GROUP), binding(0))]
        pub heights: Texture2D<f32, FILTERABLE>,

        #[sampler(group(GROUP), binding(1))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    pub fn shade<const GROUP: u32, const FILTERABLE: bool>(
        texture_bg: &TextureBindGroup<GROUP, FILTERABLE>,
        coords: Vec2f,
    ) -> Vec4f {
        texture_sample(&texture_bg.heights, &texture_bg.heights_smp, coords)
    }

    #[vertex]
    pub fn vertex<const GROUP: u32>(
        #[builtin(vertex_index)] vertex_index: u32,
        storage: &SlabBindGroup<GROUP>,
    ) -> Vec4f {
        let count = array_length(&storage.slab);
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x + count as f32 * 0.0, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn fragment<const GROUP: u32, const FILTERABLE: bool>(
        input: FragInput,
        texture: &TextureBindGroup<GROUP, FILTERABLE>,
    ) -> Vec4f {
        let p = input.position;
        let dims = texture_dimensions(&texture.heights);
        let coords = vec2f(p.x / dims.x() as f32, p.y / dims.y() as f32);
        shade(texture, coords)
    }
}

fn pixels() -> Vec<[u8; 4]> {
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let x = index % WIDTH;
            let y = index / WIDTH;
            [
                (x * 40 + 10) as u8,
                (y * 40 + 20) as u8,
                (x * 13 + y * 17) as u8,
                255,
            ]
        })
        .collect()
}

fn create_gpu_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[[u8; 4]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resource_pack_source"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 4),
            rows_per_image: Some(HEIGHT),
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    texture
}

fn cpu_run(filterable: bool, pixels: &[[u8; 4]]) -> Vec<f32> {
    use resource_pack_render as shader;
    use wgsl_rs::std::*;

    let storage = shader::SlabBindGroup::<1>::new();
    storage.slab.set(RuntimeArray {
        data: vec![
            vec4f(1.0, 0.0, 0.0, 1.0),
            vec4f(0.0, 1.0, 0.0, 1.0),
            vec4f(0.0, 0.0, 1.0, 1.0),
        ],
    });
    for index in 0..3 {
        let position = shader::vertex::<1>(index, &storage);
        assert!(position.x >= -1.0 && position.y >= -1.0);
    }

    fn run_texture<const FILTERABLE: bool>(
        texture: &resource_pack_render::TextureBindGroup<0, FILTERABLE>,
        pixels: &[[u8; 4]],
    ) -> Vec<f32> {
        use resource_pack_render as shader;
        use wgsl_rs::std::*;

        texture.heights.init(WIDTH, HEIGHT);
        for (index, pixel) in pixels.iter().enumerate() {
            texture.heights.set_pixel(
                index as u32 % WIDTH,
                index as u32 / WIDTH,
                [
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    1.0,
                ],
            );
        }
        texture.heights_smp.set(SamplerState::default());
        let grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let color = shader::fragment::<0, FILTERABLE>(
                    shader::FragInput {
                        position: builtins.position,
                    },
                    texture,
                );
                [color.x, color.y, color.z, color.w]
            },
        );
        grid.into_iter()
            .flatten()
            .flatten()
            .flat_map(|pixel| pixel.to_vec())
            .collect()
    }

    if filterable {
        let texture = shader::TextureBindGroup::<0, true>::new();
        run_texture(&texture, pixels)
    } else {
        let texture = shader::TextureBindGroup::<0, false>::new();
        run_texture(&texture, pixels)
    }
}

#[allow(clippy::too_many_lines)]
fn gpu_run(
    filterable: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[[u8; 4]],
) -> Vec<f32> {
    use resource_pack_render as shader;

    let module = if filterable {
        shader::instantiate::<1, 0, true>()
    } else {
        shader::instantiate::<1, 0, false>()
    };
    assert_eq!(
        shader::WGSL_SOURCE.module_type_params,
        &[
            "SlabBindGroup_GROUP",
            "TextureBindGroup_GROUP",
            "TextureBindGroup_FILTERABLE"
        ]
    );
    let source = module.wgsl_source();
    assert!(
        source.contains("@group(1) @binding(0) var<storage, read> slabbindgroup__slab"),
        "{source}"
    );
    assert!(
        source.contains("@group(0) @binding(0) var texturebindgroup__heights"),
        "{source}"
    );
    assert!(
        source.contains("@group(0) @binding(1) var texturebindgroup__heights_smp: sampler"),
        "{source}"
    );
    assert!(
        source.contains("arrayLength(&slabbindgroup__slab)"),
        "{source}"
    );
    assert!(!source.contains("SlabBindGroup"), "{source}");
    assert!(!source.contains("TextureBindGroup"), "{source}");

    let mut linkage = wgsl_rs_linkage_wgpu::analyze_ir_module(module);
    let texture_layout = linkage.bind_group(0).unwrap();
    assert!(matches!(
        texture_layout.entries[0].ty,
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: actual },
            view_dimension: wgpu::TextureViewDimension::D2,
            ..
        } if actual == filterable
    ));
    let expected_sampler = if filterable {
        wgpu::SamplerBindingType::Filtering
    } else {
        wgpu::SamplerBindingType::NonFiltering
    };
    assert!(matches!(
        texture_layout.entries[1].ty,
        wgpu::BindingType::Sampler(actual) if actual == expected_sampler
    ));
    assert!(matches!(
        linkage.bind_group(1).unwrap().entries[0].ty,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            ..
        }
    ));

    let source_texture = create_gpu_texture(device, queue, pixels);
    let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("resource_pack_nearest_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let storage_data = [
        [1.0f32, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resource_pack_storage"),
        size: std::mem::size_of_val(&storage_data) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage_buffer, 0, bytemuck::cast_slice(&storage_data));

    let shader_module = linkage.shader_module(device);
    let pipeline_layout = linkage.pipeline_layout(device, Some("resource_pack_pipeline_layout"));
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("resource_pack_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: linkage
            .vertex_entry("vertex")
            .unwrap()
            .vertex_state(&shader_module),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(linkage.fragment_entry("fragment").unwrap().fragment_state(
            &shader_module,
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::all(),
            })],
        )),
        multiview_mask: None,
        cache: None,
    });
    let texture_group = linkage
        .create_bind_group_named(
            0,
            device,
            &[
                (
                    "texturebindgroup__heights",
                    wgpu::BindingResource::TextureView(&source_view),
                ),
                (
                    "texturebindgroup__heights_smp",
                    wgpu::BindingResource::Sampler(&sampler),
                ),
            ],
        )
        .unwrap();
    let storage_group = linkage
        .create_bind_group_named(
            1,
            device,
            &[("slabbindgroup__slab", storage_buffer.as_entire_binding())],
        )
        .unwrap();
    let target =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "resource_pack_target");
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("resource_pack_render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resource_pack_render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &texture_group, &[]);
        pass.set_bind_group(1, &storage_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    harness::read_rgba32float_texture(device, queue, &target, WIDTH, HEIGHT)
        .into_iter()
        .flatten()
        .collect()
}

pub struct ResourcePackTest;

impl RoundtripTest for ResourcePackTest {
    fn name(&self) -> &str {
        "resource_packs"
    }

    fn description(&self) -> &str {
        "vertex + fragment packs with storage, texture, and both filterability values"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let pixels = pixels();
        [true, false]
            .into_iter()
            .map(|filterable| {
                let cpu = cpu_run(filterable, &pixels);
                let gpu = gpu_run(filterable, device, queue, &pixels);
                let labels: Vec<String> = (0..cpu.len())
                    .map(|index| format!("resource_pack[{filterable}][{index}]"))
                    .collect();
                let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                harness::compare_f32_results(
                    if filterable {
                        "resource_pack_filterable"
                    } else {
                        "resource_pack_nonfiltering"
                    },
                    &gpu,
                    &cpu,
                    &refs,
                    1e-5,
                )
            })
            .collect()
    }
}
