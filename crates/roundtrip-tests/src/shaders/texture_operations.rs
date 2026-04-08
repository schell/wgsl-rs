//! Roundtrip tests for texture load/sample operations.
//!
//! Tests: `texture_load` and `texture_sample` on `Texture2D<f32>`.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

#[wgsl]
pub mod texture_load_2d {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);

    #[input]
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> Vec4f {
        let p = input.position;
        texture_load(TEX, vec2i(p.x as i32, p.y as i32), 0u32)
    }
}

#[wgsl]
pub mod texture_sample_2d {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);
    sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);

    #[input]
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> Vec4f {
        let dims = texture_dimensions(TEX);
        let uv = vec2f(
            input.position.x / dims.x() as f32,
            input.position.y / dims.y() as f32,
        );
        texture_sample(TEX, TEX_SAMPLER, uv)
    }
}

/// Builds deterministic RGBA8 test pixels.
fn build_rgba8_pixels() -> Vec<[u8; 4]> {
    let mut pixels = vec![[0u8; 4]; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            pixels[idx] = [
                ((x * 19 + y * 7) % 256) as u8,
                ((x * 11 + y * 23) % 256) as u8,
                ((x * 5 + y * 13) % 256) as u8,
                255u8,
            ];
        }
    }
    pixels
}

/// Writes RGBA8 pixels into a CPU Texture2D<f32> as normalized channels.
fn write_cpu_texture(tex: &wgsl_rs::std::Texture2D<f32>, rgba8: &[[u8; 4]]) {
    tex.init(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            let px = rgba8[idx];
            tex.set_pixel(
                x,
                y,
                [
                    px[0] as f32 / 255.0,
                    px[1] as f32 / 255.0,
                    px[2] as f32 / 255.0,
                    px[3] as f32 / 255.0,
                ],
            );
        }
    }
}

/// Creates a GPU Rgba8Unorm texture and uploads provided pixels.
fn create_gpu_source_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba8: &[[u8; 4]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture_ops_source"),
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
        bytemuck::cast_slice(rgba8),
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

/// Flattens RGBA pixel rows into one f32 vector.
fn flatten_rgba_pixels(pixels: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for px in pixels {
        out.extend_from_slice(px);
    }
    out
}

/// Flattens a `dispatch_fragments` grid into RGBA f32 values.
fn flatten_fragment_grid(grid: &[Vec<Option<[f32; 4]>>]) -> Vec<f32> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let px = grid[y as usize][x as usize].expect("non-helper fragment expected");
            out.extend_from_slice(&px);
        }
    }
    out
}

/// Builds one label per flattened channel value.
fn build_labels(name: &str) -> Vec<String> {
    let mut labels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    let channels = ["r", "g", "b", "a"];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for ch in channels {
                labels.push(format!("{name}[{x},{y}].{ch}"));
            }
        }
    }
    labels
}

/// Renders the texture_load shader and returns RGBA pixels.
fn render_texture_load_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_texture: &wgpu::Texture,
) -> Vec<[f32; 4]> {
    let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let target =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "texture_load_target");
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let module = texture_load_2d::linkage::shader_module(device);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture_load_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("texture_load_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("texture_load_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: texture_load_2d::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(texture_load_2d::linkage::frag_main::fragment_state(
            &module,
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::all(),
            })],
        )),
        multiview_mask: None,
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture_load_bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&source_view),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_load_render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("texture_load_render"),
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
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));

    harness::read_rgba32float_texture(device, queue, &target, WIDTH, HEIGHT)
}

/// Renders the texture_sample shader and returns RGBA pixels.
fn render_texture_sample_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_texture: &wgpu::Texture,
) -> Vec<[f32; 4]> {
    let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("texture_sample_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let target =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "texture_sample_target");
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let module = texture_sample_2d::linkage::shader_module(device);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture_sample_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("texture_sample_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("texture_sample_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: texture_sample_2d::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(texture_sample_2d::linkage::frag_main::fragment_state(
            &module,
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::all(),
            })],
        )),
        multiview_mask: None,
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture_sample_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_sample_render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("texture_sample_render"),
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
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));

    harness::read_rgba32float_texture(device, queue, &target, WIDTH, HEIGHT)
}

pub struct TextureOperationsTest;

impl RoundtripTest for TextureOperationsTest {
    fn name(&self) -> &str {
        "texture_operations"
    }

    fn description(&self) -> &str {
        "texture_load and texture_sample on texture_2d<f32>"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();
        let epsilon = 1e-5;

        let pixels = build_rgba8_pixels();
        let gpu_source_texture = create_gpu_source_texture(device, queue, &pixels);

        let cpu_tex_load: Texture2D<f32> = Texture2D::new(0, 0);
        write_cpu_texture(&cpu_tex_load, &pixels);
        let gpu_load_pixels = render_texture_load_gpu(device, queue, &gpu_source_texture);
        let cpu_load_grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let p = builtins.position;
                let c = texture_load(&cpu_tex_load, vec2i(p.x as i32, p.y as i32), 0u32);
                [c.x, c.y, c.z, c.w]
            },
        );

        let gpu_load = flatten_rgba_pixels(&gpu_load_pixels);
        let cpu_load = flatten_fragment_grid(&cpu_load_grid);
        let load_labels = build_labels("texture_load");
        let load_label_refs: Vec<&str> = load_labels.iter().map(|s| s.as_str()).collect();
        results.push(harness::compare_f32_results(
            "texture_load",
            &gpu_load,
            &cpu_load,
            &load_label_refs,
            epsilon,
        ));

        let cpu_tex_sample: Texture2D<f32> = Texture2D::new(0, 0);
        write_cpu_texture(&cpu_tex_sample, &pixels);
        let cpu_sampler: Sampler = Sampler::new(0, 0);
        cpu_sampler.set(SamplerState::default());
        let gpu_sample_pixels = render_texture_sample_gpu(device, queue, &gpu_source_texture);
        let cpu_sample_grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let p = builtins.position;
                let dims = texture_dimensions(&cpu_tex_sample);
                let uv = vec2f(p.x / dims.x() as f32, p.y / dims.y() as f32);
                let c = texture_sample(&cpu_tex_sample, &cpu_sampler, uv);
                [c.x, c.y, c.z, c.w]
            },
        );

        let gpu_sample = flatten_rgba_pixels(&gpu_sample_pixels);
        let cpu_sample = flatten_fragment_grid(&cpu_sample_grid);
        let sample_labels = build_labels("texture_sample");
        let sample_label_refs: Vec<&str> = sample_labels.iter().map(|s| s.as_str()).collect();
        results.push(harness::compare_f32_results(
            "texture_sample",
            &gpu_sample,
            &cpu_sample,
            &sample_label_refs,
            epsilon,
        ));

        results
    }
}
