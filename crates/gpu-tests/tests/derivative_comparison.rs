//! GPU vs CPU comparison tests for derivative builtin functions.
//!
//! These tests render fragment shaders that compute derivatives on the GPU,
//! read back the results, and compare them against the CPU dispatch runtime's
//! `dispatch_fragments` output.
//!
//! Requires a GPU (or software rasterizer) — tests are skipped if no adapter
//! is available.

use futures::{
    Future, FutureExt,
    channel::oneshot,
    executor::block_on,
    future::{self, Either},
};
use gpu_tests::{derivative_shader, derivative_variants_shader};
use std::{sync::mpsc, thread, time::Duration};
use wgsl_rs::std::*;

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const TEXEL_SIZE: u32 = 16; // 4 f32s * 4 bytes each = 16 bytes per pixel (Rgba32Float)

/// Upper bound on any single GPU operation in this test file. Adapter
/// enumeration, device creation, and buffer mapping should each complete
/// in well under a second on a healthy host; we set a generous ceiling
/// so wedged-driver CI failures surface as skips rather than hangs.
const GPU_TIMEOUT: Duration = Duration::from_secs(60);

/// Sentinel returned when [`race_timeout`] fires before the work future.
#[derive(Debug)]
struct TimedOut;

/// Races `fut` against a deadline of `dur`.
///
/// The deadline is implemented by spawning a detached daemon thread that
/// sleeps for `dur` and then fires a [`oneshot`] channel. If the work
/// future wins the race, the timer thread is not cancelled (it keeps
/// sleeping), but its oneshot sender is dropped on return and the thread
/// will exit harmlessly after its sleep elapses. Crucially the work
/// future is *dropped* on timeout, so a `wgpu` future wedged inside the
/// driver is abandoned cleanly — no permanently-blocked OS thread is
/// left behind, unlike the previous `thread::spawn` + `recv_timeout`
/// pattern.
async fn race_timeout<F, T>(fut: F, dur: Duration) -> Result<T, TimedOut>
where
    F: Future<Output = T> + Send,
{
    let (tx, rx) = oneshot::channel::<()>();
    thread::spawn(move || {
        thread::sleep(dur);
        let _ = tx.send(());
    });
    // `Box::pin` erases the concrete future type and yields a `BoxFuture`,
    // which is `Unpin` — required by `futures::future::select`.
    let work = fut.boxed();
    let timer = rx.fuse();
    match future::select(work, timer).await {
        Either::Left((result, _timer)) => Ok(result),
        Either::Right((_timer_fired, _work)) => Err(TimedOut),
    }
}

/// Creates a headless wgpu device, or returns `None` if no suitable adapter
/// is found, **or** if the underlying `wgpu` adapter/device enumeration
/// hangs longer than [`GPU_TIMEOUT`]. Checks that the adapter supports
/// `Rgba32Float` as a render attachment (some software rasterizers may
/// not).
fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let result = block_on(race_timeout(
        get_adapter_device_and_queue_inner(),
        GPU_TIMEOUT,
    ));
    match result {
        Ok(Some((_adapter, device, queue))) => Some((device, queue)),
        Ok(None) => None,
        Err(TimedOut) => {
            eprintln!(
                "wgpu adapter/device creation timed out after {GPU_TIMEOUT:?} — skipping GPU test"
            );
            None
        }
    }
}

/// Inner async adapter/device acquisition. The returned future is raced
/// against a timer by [`create_device`] so a wedged driver cannot hang
/// the test process — on timeout the future is dropped, abandoning the
/// in-flight `wgpu` request without leaving a blocked thread behind.
async fn get_adapter_device_and_queue_inner() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)>
{
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| eprintln!("{e}"))
        .ok()?;

    // Verify Rgba32Float is renderable on this adapter.
    let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Float);
    if !format_features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    {
        eprintln!("Adapter does not support Rgba32Float as render attachment — skipping GPU test");
        return None;
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|e| eprintln!("{e}"))
        .ok()?;

    Some((adapter, device, queue))
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

    // Run readback on the calling thread. The timeout is enforced
    // natively by `wgpu`'s `PollType::Wait { timeout: Some(..) }`,
    // which returns `PollError::Timeout` instead of blocking forever —
    // no spawned worker thread to leak if the driver wedges.
    match readback_inner(staging, device, idx, width, height, bytes_per_row) {
        Ok(pixels) => pixels,
        Err(reason) => panic!("GPU readback failed: {reason}"),
    }
}

/// Performs the full GPU→CPU readback: map the staging buffer, copy
/// pixels, and unmap. The [`GPU_TIMEOUT`] deadline is passed directly
/// into [`wgpu::Device::poll`], which returns [`wgpu::PollError::Timeout`]
/// rather than blocking indefinitely — so a wedged driver surfaces as a
/// panic without leaving a permanently-blocked OS thread behind.
fn readback_inner(
    staging: wgpu::Buffer,
    device: &wgpu::Device,
    submission_index: wgpu::SubmissionIndex,
    width: u32,
    height: u32,
    bytes_per_row: u32,
) -> Result<Vec<[f32; 4]>, &'static str> {
    let (sender, receiver) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let poll_result = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: Some(GPU_TIMEOUT),
    });
    match poll_result {
        Ok(_) => {}
        Err(wgpu::PollError::Timeout) => return Err("GPU readback timed out"),
        Err(_) => return Err("device.poll returned an error"),
    }
    receiver
        .recv()
        .map_err(|_| "buffer mapping channel closed")?
        .map_err(|_| "buffer mapping failed")?;

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
    Ok(pixels)
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

    // Runtime IR-based wgpu linkage analysis (issue #120).
    let module = &derivative_shader::WGSL_SOURCE;
    let mut linkage = wgsl_rs::linkage::wgpu::analyze_wgsl_module(module).unwrap();
    let module = linkage.shader_module(device);

    let pipeline_layout = linkage.pipeline_layout(device, Some("derivative_test"));

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("derivative_test"),
        layout: Some(&pipeline_layout),
        vertex: linkage
            .vertex_entry("vtx_main")
            .expect("vtx_main entry present")
            .vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(
            linkage
                .fragment_entry("frag_main")
                .expect("frag_main entry present")
                .fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
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

    // Runtime IR-based wgpu linkage analysis (issue #120).
    let module = &derivative_variants_shader::WGSL_SOURCE;
    let mut linkage = wgsl_rs::linkage::wgpu::analyze_wgsl_module(module).unwrap();
    let module = linkage.shader_module(device);

    let pipeline_layout = linkage.pipeline_layout(device, Some("variant_test"));

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("variant_test"),
        layout: Some(&pipeline_layout),
        vertex: linkage
            .vertex_entry("vtx_main")
            .expect("vtx_main entry present")
            .vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(
            linkage
                .fragment_entry("frag_main")
                .expect("frag_main entry present")
                .fragment_state(
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
