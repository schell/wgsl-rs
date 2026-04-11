//! Roundtrip tests for advanced texture builtins.
//!
//! Covers gradient/level/bias/offset sampling, gather, and depth compare
//! variants for 2D and 2D-array textures.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const LAYERS: u32 = 2;

#[wgsl]
pub mod tex2d_variants {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);
    sampler!(group(0), binding(1), S: Sampler);

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
        let ddx = vec2f(1.0 / dims.x() as f32, 0.0);
        let ddy = vec2f(0.0, 1.0 / dims.y() as f32);

        let a = texture_sample_level(TEX, S, uv, 0.0);
        let b = texture_sample_bias(TEX, S, uv, 0.0);
        let c = texture_sample_grad(TEX, S, uv, ddx, ddy);
        let d = texture_sample_level_offset(TEX, S, uv, 0.0, vec2i(1, 0));
        vec4f(a.x, b.y, c.z, d.w)
    }
}

#[wgsl]
pub mod tex2d_gather_variants {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2D<f32>);
    sampler!(group(0), binding(1), S: Sampler);

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
        let g0 = texture_gather(0u32, TEX, S, uv);
        let g1 = texture_gather_offset(1u32, TEX, S, uv, vec2i(1, 0));
        vec4f(g0.x, g0.y, g1.z, g1.w)
    }
}

#[wgsl]
pub mod tex2d_array_variants {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), TEX: Texture2DArray<f32>);
    sampler!(group(0), binding(1), S: Sampler);

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
        let ddx = vec2f(1.0 / dims.x() as f32, 0.0);
        let ddy = vec2f(0.0, 1.0 / dims.y() as f32);
        let mut layer = 0u32;
        if (input.position.x as i32 & 1) != 0 {
            layer = 1u32;
        }

        let a = texture_sample_array(TEX, S, uv, layer);
        let b = texture_sample_level_array(TEX, S, uv, layer, 0.0);
        let c = texture_sample_grad_array(TEX, S, uv, layer, ddx, ddy);
        let d = texture_sample_array_offset(TEX, S, uv, layer, vec2i(1, 0));
        vec4f(a.x, b.y, c.z, d.w)
    }
}

#[wgsl]
pub mod depth2d_compare_variants {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), DEPTH_TEX: TextureDepth2D);
    sampler!(group(0), binding(1), DEPTH_SAMPLER: SamplerComparison);

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
        let dims = texture_dimensions(DEPTH_TEX);
        let uv = vec2f(
            input.position.x / dims.x() as f32,
            input.position.y / dims.y() as f32,
        );
        let depth_ref = 0.5;
        let a = texture_sample_compare(DEPTH_TEX, DEPTH_SAMPLER, uv, depth_ref);
        let b = texture_sample_compare_level(DEPTH_TEX, DEPTH_SAMPLER, uv, depth_ref);
        let c = texture_sample_compare_offset(DEPTH_TEX, DEPTH_SAMPLER, uv, depth_ref, vec2i(1, 0));
        let d = texture_gather_compare(DEPTH_TEX, DEPTH_SAMPLER, uv, depth_ref);
        vec4f(a, b, c, d.x)
    }
}

#[wgsl]
pub mod depth2d_array_compare_variants {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), DEPTH_TEX: TextureDepth2DArray);
    sampler!(group(0), binding(1), DEPTH_SAMPLER: SamplerComparison);

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
        let dims = texture_dimensions(DEPTH_TEX);
        let uv = vec2f(
            input.position.x / dims.x() as f32,
            input.position.y / dims.y() as f32,
        );
        let mut layer = 0u32;
        if (input.position.x as i32 & 1) != 0 {
            layer = 1u32;
        }
        let depth_ref = 0.5;

        let a = texture_sample_compare_array(DEPTH_TEX, DEPTH_SAMPLER, uv, layer, depth_ref);
        let b = texture_sample_compare_level_array(DEPTH_TEX, DEPTH_SAMPLER, uv, layer, depth_ref);
        let c = texture_sample_compare_array_offset(
            DEPTH_TEX,
            DEPTH_SAMPLER,
            uv,
            layer,
            depth_ref,
            vec2i(1, 0),
        );
        let d = texture_gather_compare_array(DEPTH_TEX, DEPTH_SAMPLER, uv, layer, depth_ref);
        vec4f(a, b, c, d.x)
    }
}

fn build_rgba8_pixels(offset: u32) -> Vec<[u8; 4]> {
    let mut pixels = vec![[0u8; 4]; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            pixels[idx] = [
                ((x * 17 + y * 9 + offset) % 256) as u8,
                ((x * 7 + y * 21 + offset) % 256) as u8,
                ((x * 13 + y * 5 + offset) % 256) as u8,
                255,
            ];
        }
    }
    pixels
}

fn build_depth_pixels(offset: f32) -> Vec<f32> {
    let mut pixels = vec![0.0f32; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            pixels[idx] = ((x as f32 * 0.07 + y as f32 * 0.11 + offset) % 1.0).clamp(0.0, 1.0);
        }
    }
    pixels
}

fn write_cpu_texture2d(tex: &wgsl_rs::std::Texture2D<f32>, rgba8: &[[u8; 4]]) {
    tex.init(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let px = rgba8[(y * WIDTH + x) as usize];
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

fn write_cpu_texture2d_array(tex: &wgsl_rs::std::Texture2DArray<f32>, layers: &[Vec<[u8; 4]>]) {
    let mut data = wgsl_rs::std::TextureData2DArray::<f32>::new(WIDTH, HEIGHT, LAYERS);
    for layer in 0..LAYERS {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let px = layers[layer as usize][(y * WIDTH + x) as usize];
                data.layers[layer as usize].set_pixel(
                    x,
                    y,
                    0,
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
    tex.set(data);
}

fn write_cpu_depth2d(tex: &wgsl_rs::std::TextureDepth2D, depth: &[f32]) {
    let mut data = wgsl_rs::std::TextureDataDepth2D::new(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            data.mips[0][y as usize][x as usize] = depth[(y * WIDTH + x) as usize];
        }
    }
    tex.set(data);
}

fn write_cpu_depth2d_array(tex: &wgsl_rs::std::TextureDepth2DArray, layers: &[Vec<f32>]) {
    let mut data = wgsl_rs::std::TextureDataDepth2DArray::new(WIDTH, HEIGHT, LAYERS);
    for layer in 0..LAYERS {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                data.layers[layer as usize].mips[0][y as usize][x as usize] =
                    layers[layer as usize][(y * WIDTH + x) as usize];
            }
        }
    }
    tex.set(data);
}

fn create_gpu_texture2d_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba8: &[[u8; 4]],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("advanced_tex2d_source"),
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

fn create_gpu_texture2d_array_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[Vec<[u8; 4]>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("advanced_tex2d_array_source"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for layer in 0..LAYERS {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&layers[layer as usize]),
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
    }
    texture
}

fn create_gpu_depth2d(device: &wgpu::Device, queue: &wgpu::Queue, depth: &[f32]) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("advanced_depth2d_source"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes_per_row = (WIDTH * 4).next_multiple_of(256);
    let mut padded = vec![0u8; (bytes_per_row * HEIGHT) as usize];
    for y in 0..HEIGHT {
        let src = &depth[(y * WIDTH) as usize..((y + 1) * WIDTH) as usize];
        let dst_offset = (y * bytes_per_row) as usize;
        let dst = &mut padded[dst_offset..dst_offset + (WIDTH * 4) as usize];
        dst.copy_from_slice(bytemuck::cast_slice(src));
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("advanced_depth2d_staging"),
        size: padded.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&staging, 0, &padded);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("advanced_depth2d_upload"),
    });
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    texture
}

fn create_gpu_depth2d_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[Vec<f32>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("advanced_depth2d_array_source"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let bytes_per_row = (WIDTH * 4).next_multiple_of(256);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("advanced_depth2d_array_upload"),
    });
    for layer in 0..LAYERS {
        let mut padded = vec![0u8; (bytes_per_row * HEIGHT) as usize];
        let src_layer = &layers[layer as usize];
        for y in 0..HEIGHT {
            let src = &src_layer[(y * WIDTH) as usize..((y + 1) * WIDTH) as usize];
            let dst_offset = (y * bytes_per_row) as usize;
            let dst = &mut padded[dst_offset..dst_offset + (WIDTH * 4) as usize];
            dst.copy_from_slice(bytemuck::cast_slice(src));
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("advanced_depth2d_array_staging"),
            size: padded.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&staging, 0, &padded);
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit(Some(encoder.finish()));
    texture
}

fn render_one_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    bind_group: &wgpu::BindGroup,
    vertex_state: wgpu::VertexState<'_>,
    fragment_state: wgpu::FragmentState<'_>,
) -> Vec<[f32; 4]> {
    let target = harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, label);
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: vertex_state,
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(fragment_state),
        multiview_mask: None,
        cache: None,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
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
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    harness::read_rgba32float_texture(device, queue, &target, WIDTH, HEIGHT)
}

fn flatten_rgba_pixels(pixels: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for px in pixels {
        out.extend_from_slice(px);
    }
    out
}

fn flatten_grid(grid: &[Vec<Option<[f32; 4]>>]) -> Vec<f32> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            out.extend_from_slice(&grid[y as usize][x as usize].expect("fragment output expected"));
        }
    }
    out
}

fn labels(name: &str) -> Vec<String> {
    let mut labels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            labels.push(format!("{name}[{x},{y}].r"));
            labels.push(format!("{name}[{x},{y}].g"));
            labels.push(format!("{name}[{x},{y}].b"));
            labels.push(format!("{name}[{x},{y}].a"));
        }
    }
    labels
}

fn compare_rgba(
    name: &str,
    gpu: &[[f32; 4]],
    cpu_grid: &[Vec<Option<[f32; 4]>>],
) -> ComparisonResult {
    let g = flatten_rgba_pixels(gpu);
    let c = flatten_grid(cpu_grid);
    let labels = labels(name);
    let refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    harness::compare_f32_results(name, &g, &c, &refs, 1e-5)
}

pub struct AdvancedTextureOperationsTest;

impl RoundtripTest for AdvancedTextureOperationsTest {
    fn name(&self) -> &str {
        "advanced_texture_operations"
    }

    fn description(&self) -> &str {
        "sample_grad/level/bias/offset and gather variants for 2D/2D-array textures"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        let color0 = build_rgba8_pixels(0);
        let color1 = build_rgba8_pixels(47);
        let color_layers = vec![color0.clone(), color1.clone()];
        let gpu_tex2d = create_gpu_texture2d_rgba8(device, queue, &color0);
        let gpu_tex2d_array = create_gpu_texture2d_array_rgba8(device, queue, &color_layers);

        {
            let source_view = gpu_tex2d.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("advanced_tex2d_sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });

            let module = tex2d_variants::linkage::shader_module(device);
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("advanced_tex2d_variants_bgl"),
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
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advanced_tex2d_variants_bg"),
                layout: &bgl,
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

            let gpu = render_one_target(
                device,
                queue,
                "advanced_tex2d_variants",
                &bgl,
                &bg,
                tex2d_variants::linkage::vtx_main::vertex_state(&module),
                tex2d_variants::linkage::frag_main::fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                ),
            );

            let cpu_tex: Texture2D<f32> = Texture2D::new(0, 0);
            write_cpu_texture2d(&cpu_tex, &color0);
            let cpu_sampler: Sampler = Sampler::new(0, 0);
            cpu_sampler.set(SamplerState::default());

            let cpu = dispatch_fragments(
                WIDTH,
                HEIGHT,
                |_, _| (),
                |builtins, _| {
                    let dims = texture_dimensions(&cpu_tex);
                    let uv = vec2f(
                        builtins.position.x / dims.x() as f32,
                        builtins.position.y / dims.y() as f32,
                    );
                    let ddx = vec2f(1.0 / dims.x() as f32, 0.0);
                    let ddy = vec2f(0.0, 1.0 / dims.y() as f32);
                    let a = texture_sample_level(&cpu_tex, &cpu_sampler, uv, 0.0);
                    let b = texture_sample_bias(&cpu_tex, &cpu_sampler, uv, 0.0);
                    let c = texture_sample_grad(&cpu_tex, &cpu_sampler, uv, ddx, ddy);
                    let d =
                        texture_sample_level_offset(&cpu_tex, &cpu_sampler, uv, 0.0, vec2i(1, 0));
                    [a.x, b.y, c.z, d.w]
                },
            );
            results.push(compare_rgba("advanced_tex2d_variants", &gpu, &cpu));
        }

        {
            let source_view = gpu_tex2d.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("advanced_tex2d_gather_sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            let module = tex2d_gather_variants::linkage::shader_module(device);
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("advanced_tex2d_gather_bgl"),
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
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advanced_tex2d_gather_bg"),
                layout: &bgl,
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

            let gpu = render_one_target(
                device,
                queue,
                "advanced_tex2d_gather",
                &bgl,
                &bg,
                tex2d_gather_variants::linkage::vtx_main::vertex_state(&module),
                tex2d_gather_variants::linkage::frag_main::fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                ),
            );

            let cpu_tex: Texture2D<f32> = Texture2D::new(0, 0);
            write_cpu_texture2d(&cpu_tex, &color0);
            let cpu_sampler: Sampler = Sampler::new(0, 0);
            cpu_sampler.set(SamplerState::default());
            let cpu = dispatch_fragments(
                WIDTH,
                HEIGHT,
                |_, _| (),
                |builtins, _| {
                    let dims = texture_dimensions(&cpu_tex);
                    let uv = vec2f(
                        builtins.position.x / dims.x() as f32,
                        builtins.position.y / dims.y() as f32,
                    );
                    let g0 = texture_gather(0u32, &cpu_tex, &cpu_sampler, uv);
                    let g1 = texture_gather_offset(1u32, &cpu_tex, &cpu_sampler, uv, vec2i(1, 0));
                    [g0.x, g0.y, g1.z, g1.w]
                },
            );
            results.push(compare_rgba("advanced_tex2d_gather", &gpu, &cpu));
        }

        {
            let source_view = gpu_tex2d_array.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("advanced_tex2d_array_sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            let module = tex2d_array_variants::linkage::shader_module(device);
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("advanced_tex2d_array_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("advanced_tex2d_array_bg"),
                layout: &bgl,
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

            let gpu = render_one_target(
                device,
                queue,
                "advanced_tex2d_array_variants",
                &bgl,
                &bg,
                tex2d_array_variants::linkage::vtx_main::vertex_state(&module),
                tex2d_array_variants::linkage::frag_main::fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                ),
            );

            let cpu_tex: Texture2DArray<f32> = Texture2DArray::new(0, 0);
            write_cpu_texture2d_array(&cpu_tex, &color_layers);
            let cpu_sampler: Sampler = Sampler::new(0, 0);
            cpu_sampler.set(SamplerState::default());
            let cpu = dispatch_fragments(
                WIDTH,
                HEIGHT,
                |_, _| (),
                |builtins, _| {
                    let dims = texture_dimensions(&cpu_tex);
                    let uv = vec2f(
                        builtins.position.x / dims.x() as f32,
                        builtins.position.y / dims.y() as f32,
                    );
                    let ddx = vec2f(1.0 / dims.x() as f32, 0.0);
                    let ddy = vec2f(0.0, 1.0 / dims.y() as f32);
                    let layer = if (builtins.position.x as i32 & 1) == 0 {
                        0u32
                    } else {
                        1u32
                    };
                    let a = texture_sample_array(&cpu_tex, &cpu_sampler, uv, layer);
                    let b = texture_sample_level_array(&cpu_tex, &cpu_sampler, uv, layer, 0.0);
                    let c = texture_sample_grad_array(&cpu_tex, &cpu_sampler, uv, layer, ddx, ddy);
                    let d =
                        texture_sample_array_offset(&cpu_tex, &cpu_sampler, uv, layer, vec2i(1, 0));
                    [a.x, b.y, c.z, d.w]
                },
            );
            results.push(compare_rgba("advanced_tex2d_array_variants", &gpu, &cpu));
        }

        results
    }
}
