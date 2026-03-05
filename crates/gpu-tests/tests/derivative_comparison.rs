//! GPU vs CPU comparison tests for derivative builtin functions.
//!
//! These tests render fragment shaders that compute derivatives on the GPU,
//! read back the results, and compare them against the CPU dispatch runtime's
//! `dispatch_fragments` output.
//!
//! Requires a GPU (or software rasterizer) — tests are skipped if no adapter
//! is available.

use futures::executor::block_on;
use gpu_tests::{derivative_shader, derivative_variants_shader};
use wgsl_rs::std::*;

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const TEXEL_SIZE: u32 = 16; // 4 f32s * 4 bytes each = 16 bytes per pixel (Rgba32Float)

/// Creates a headless wgpu device, or returns `None` if no adapter is found.
fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some((device, queue))
}

/// Creates a render target texture with `Rgba32Float` format.
fn create_render_target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Reads back a texture into a `Vec<[f32; 4]>` (one entry per pixel,
/// row-major).
fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let bytes_per_row = align_to(width * TEXEL_SIZE, 256);
    let buffer_size = (bytes_per_row * height) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let idx = queue.submit(Some(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("channel send failed");
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();
    receiver
        .recv()
        .expect("channel recv failed")
        .expect("buffer mapping failed");

    let data = staging.slice(..).get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let row_offset = (y * bytes_per_row) as usize;
        for x in 0..width {
            let offset = row_offset + (x * TEXEL_SIZE) as usize;
            let pixel: [f32; 4] =
                *bytemuck::from_bytes(&data[offset..offset + TEXEL_SIZE as usize]);
            pixels.push(pixel);
        }
    }
    drop(data);
    staging.unmap();
    pixels
}

/// Align `value` up to the next multiple of `alignment`.
fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Renders a full-screen triangle with the given shader and returns the pixel
/// data from a single render target.
fn render_single_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let texture = create_render_target(device, width, height);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let module = derivative_shader::linkage::shader_module(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("derivative_test"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("derivative_test"),
        layout: Some(&pipeline_layout),
        vertex: derivative_shader::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(derivative_shader::linkage::frag_main::fragment_state(
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("derivative_test"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));

    read_texture(device, queue, &texture, width, height)
}

/// Renders the derivative_variants_shader and returns (fine_pixels,
/// coarse_pixels).
fn render_variant_targets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let fine_texture = create_render_target(device, width, height);
    let coarse_texture = create_render_target(device, width, height);
    let fine_view = fine_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let coarse_view = coarse_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let module = derivative_variants_shader::linkage::shader_module(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("variant_test"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("variant_test"),
        layout: Some(&pipeline_layout),
        vertex: derivative_variants_shader::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(
            derivative_variants_shader::linkage::frag_main::fragment_state(
                &module,
                &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    }),
                ],
            ),
        ),
        multiview_mask: None,
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("variant_test"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &fine_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &coarse_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));

    let fine = read_texture(device, queue, &fine_texture, width, height);
    let coarse = read_texture(device, queue, &coarse_texture, width, height);
    (fine, coarse)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Compares GPU derivative results against CPU dispatch_fragments for the
    /// basic derivative shader (dpdx, dpdy, fwidth of position).
    #[test]
    fn derivative_gpu_vs_cpu_basic() {
        let Some((device, queue)) = create_device() else {
            eprintln!("No GPU adapter found — skipping GPU comparison test");
            return;
        };

        let gpu_pixels = render_single_target(&device, &queue, WIDTH, HEIGHT);

        // CPU side: dispatch_fragments with the same logic.
        let cpu_grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let position = builtins.position;
                let dx = dpdx(position.x);
                let dy = dpdy(position.y);
                let fwx = fwidth(position.x);
                let fwy = fwidth(position.y);
                [dx, dy, fwx, fwy]
            },
        );

        let epsilon = 1e-4;

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let gpu = gpu_pixels[(y * WIDTH + x) as usize];
                let cpu =
                    cpu_grid[y as usize][x as usize].expect("non-helper pixel should have output");

                for c in 0..4 {
                    let channel_names = [
                        "dpdx(pos.x)",
                        "dpdy(pos.y)",
                        "fwidth(pos.x)",
                        "fwidth(pos.y)",
                    ];
                    assert!(
                        (gpu[c] - cpu[c]).abs() < epsilon,
                        "Mismatch at pixel ({x}, {y}) channel {} ({}): GPU={}, CPU={}",
                        c,
                        channel_names[c],
                        gpu[c],
                        cpu[c],
                    );
                }
            }
        }
    }

    /// Compares GPU derivative results against CPU for fine and coarse
    /// variants.
    #[test]
    fn derivative_gpu_vs_cpu_fine_coarse() {
        let Some((device, queue)) = create_device() else {
            eprintln!("No GPU adapter found — skipping GPU comparison test");
            return;
        };

        let (gpu_fine, gpu_coarse) = render_variant_targets(&device, &queue, WIDTH, HEIGHT);

        // CPU fine.
        let cpu_fine_grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let position = builtins.position;
                [
                    dpdx_fine(position.x),
                    dpdy_fine(position.y),
                    fwidth_fine(position.x),
                    fwidth_fine(position.y),
                ]
            },
        );

        // CPU coarse.
        let cpu_coarse_grid = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let position = builtins.position;
                [
                    dpdx_coarse(position.x),
                    dpdy_coarse(position.y),
                    fwidth_coarse(position.x),
                    fwidth_coarse(position.y),
                ]
            },
        );

        let epsilon = 1e-4;

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let idx = (y * WIDTH + x) as usize;
                let gpu_f = gpu_fine[idx];
                let cpu_f = cpu_fine_grid[y as usize][x as usize]
                    .expect("non-helper pixel should have output");

                for c in 0..4 {
                    assert!(
                        (gpu_f[c] - cpu_f[c]).abs() < epsilon,
                        "Fine mismatch at pixel ({x}, {y}) channel {c}: GPU={}, CPU={}",
                        gpu_f[c],
                        cpu_f[c],
                    );
                }

                let gpu_c = gpu_coarse[idx];
                let cpu_c = cpu_coarse_grid[y as usize][x as usize]
                    .expect("non-helper pixel should have output");

                for c in 0..4 {
                    assert!(
                        (gpu_c[c] - cpu_c[c]).abs() < epsilon,
                        "Coarse mismatch at pixel ({x}, {y}) channel {c}: GPU={}, CPU={}",
                        gpu_c[c],
                        cpu_c[c],
                    );
                }
            }
        }
    }

    /// Sanity check: for linear position values, all derivative variants should
    /// produce the same result (derivative of a linear function is constant).
    #[test]
    fn derivative_gpu_all_variants_agree_for_linear() {
        let Some((device, queue)) = create_device() else {
            eprintln!("No GPU adapter found — skipping GPU comparison test");
            return;
        };

        let basic = render_single_target(&device, &queue, WIDTH, HEIGHT);
        let (fine, coarse) = render_variant_targets(&device, &queue, WIDTH, HEIGHT);

        let epsilon = 1e-4;

        for i in 0..(WIDTH * HEIGHT) as usize {
            for c in 0..4 {
                assert!(
                    (basic[i][c] - fine[i][c]).abs() < epsilon,
                    "basic vs fine mismatch at pixel {i} channel {c}: basic={}, fine={}",
                    basic[i][c],
                    fine[i][c],
                );
                assert!(
                    (basic[i][c] - coarse[i][c]).abs() < epsilon,
                    "basic vs coarse mismatch at pixel {i} channel {c}: basic={}, coarse={}",
                    basic[i][c],
                    coarse[i][c],
                );
            }
        }
    }
}
