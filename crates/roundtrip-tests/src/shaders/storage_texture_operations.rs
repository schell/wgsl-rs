//! Roundtrip tests for storage texture load/store operations.
//!
//! Tests: `texture_store`, `texture_load_storage` on
//! `TextureStorage2D<Rgba8unorm, Write>` and
//! `TextureStorage2D<Rgba8unorm, Read>`.
//!
//! The compute shader reads from a read-only storage texture (INPUT),
//! doubles each channel, and writes to a write-only storage texture
//! (OUTPUT). Both CPU and GPU should produce the same result.

#![allow(dead_code)]

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;

#[wgsl]
pub mod storage_texture_compute {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), INPUT: TextureStorage2D<Rgba8unorm, Read>);
    texture!(group(0), binding(1), OUTPUT: TextureStorage2D<Rgba8unorm, Write>);

    #[compute]
    #[workgroup_size(1)]
    pub fn compute_main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let dims = texture_dimensions(OUTPUT);
        if gid.x() >= dims.x() || gid.y() >= dims.y() {
            return;
        }
        let value = texture_load_storage(INPUT, vec2u(gid.x(), gid.y()));
        let halved = value * 0.5;
        texture_store(OUTPUT, vec2u(gid.x(), gid.y()), halved);
    }
}

/// Builds deterministic RGBA8 test pixels (as normalized f32 values).
fn build_test_pixels() -> Vec<[f32; 4]> {
    let mut pixels = vec![[0.0f32; 4]; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            pixels[idx] = [
                ((x * 19 + y * 7) % 256) as f32 / 255.0,
                ((x * 11 + y * 23) % 256) as f32 / 255.0,
                ((x * 5 + y * 13) % 256) as f32 / 255.0,
                1.0,
            ];
        }
    }
    pixels
}

/// Converts f32 pixels to RGBA8 bytes for GPU upload.
fn f32_pixels_to_rgba8(pixels: &[[f32; 4]]) -> Vec<[u8; 4]> {
    pixels
        .iter()
        .map(|p| {
            [
                (p[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                (p[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                (p[2] * 255.0).round().clamp(0.0, 255.0) as u8,
                (p[3] * 255.0).round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

/// Converts RGBA8 bytes back to f32 pixels for comparison.
fn rgba8_to_f32_pixels(pixels: &[[u8; 4]]) -> Vec<[f32; 4]> {
    pixels
        .iter()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                p[3] as f32 / 255.0,
            ]
        })
        .collect()
}

/// Runs the compute shader on the GPU and reads back the output texture.
fn run_gpu_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_pixels: &[[u8; 4]],
) -> Vec<[f32; 4]> {
    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage_input"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage_output"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Upload input pixels.
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &input_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(input_pixels),
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

    let module = &storage_texture_compute::WGSL_SOURCE;
    let mut linkage = wgsl_rs::linkage::wgpu::analyze_wgsl_module(module).unwrap();
    let shader_module = linkage.shader_module(device);
    let pipeline_layout = linkage.pipeline_layout(device, Some("storage_texture_pipeline_layout"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("storage_texture_compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("compute_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = linkage
        .create_bind_group_named(
            0,
            device,
            &[
                ("INPUT", wgpu::BindingResource::TextureView(&input_view)),
                ("OUTPUT", wgpu::BindingResource::TextureView(&output_view)),
            ],
        )
        .expect("storage texture bind group");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("storage_texture_compute"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("storage_texture_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(WIDTH, HEIGHT, 1);
    }

    // Copy output texture to a buffer for readback.
    // wgpu requires bytes_per_row to be aligned to COPY_BYTES_PER_ROW_ALIGNMENT
    // (256).
    const ALIGNMENT: u32 = 256;
    let bytes_per_pixel = 4u32;
    let unaligned_bpr = WIDTH * bytes_per_pixel;
    let aligned_bpr = unaligned_bpr.div_ceil(ALIGNMENT) * ALIGNMENT;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage_texture_output"),
        size: (aligned_bpr as u64) * (HEIGHT as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bpr),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Read back the output buffer.
    let (sender, receiver) = std::sync::mpsc::channel();
    output_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("channel send failed");
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("wgpu poll failed");
    receiver
        .recv()
        .expect("channel recv failed")
        .expect("buffer mapping failed");

    let data = output_buffer.slice(..).get_mapped_range();
    let mut rgba8: Vec<[u8; 4]> = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        let row_offset = (y * aligned_bpr) as usize;
        for x in 0..WIDTH {
            let offset = row_offset + (x * bytes_per_pixel) as usize;
            rgba8.push([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
        }
    }
    drop(data);
    output_buffer.unmap();
    rgba8_to_f32_pixels(&rgba8)
}

/// Runs the compute shader on the CPU and returns the output pixels.
fn run_cpu_compute(input_pixels: &[[f32; 4]]) -> Vec<[f32; 4]> {
    use wgsl_rs::std::*;

    // Initialize input texture.
    storage_texture_compute::INPUT.init(WIDTH, HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = (y * WIDTH + x) as usize;
            storage_texture_compute::INPUT.set_pixel(
                x,
                y,
                vec4f(
                    input_pixels[idx][0],
                    input_pixels[idx][1],
                    input_pixels[idx][2],
                    input_pixels[idx][3],
                ),
            );
        }
    }

    // Initialize output texture.
    storage_texture_compute::OUTPUT.init(WIDTH, HEIGHT);

    // Run the compute shader.
    dispatch_workgroups((WIDTH, HEIGHT, 1), (1, 1, 1), |builtins| {
        storage_texture_compute::compute_main(builtins.global_invocation_id);
    });

    // Read back output directly from CPU-side data (OUTPUT is write-only,
    // so we can't use texture_load_storage — read the ModuleVar directly).
    let data = storage_texture_compute::OUTPUT.get();
    let mut output = vec![[0.0f32; 4]; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = data.get_pixel(x, y, 0);
            let idx = (y * WIDTH + x) as usize;
            if let Some(p) = pixel {
                output[idx] = [p[0], p[1], p[2], p[3]];
            }
        }
    }
    output
}

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

pub struct StorageTextureOperationsTest;

impl RoundtripTest for StorageTextureOperationsTest {
    fn name(&self) -> &str {
        "storage_texture_operations"
    }

    fn description(&self) -> &str {
        "texture_store and texture_load_storage on texture_storage_2d<rgba8unorm, _>"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let epsilon = 1e-2; // Rgba8Unorm quantizes to 1/255, so allow ~0.004 slack

        let input_f32 = build_test_pixels();
        let input_rgba8 = f32_pixels_to_rgba8(&input_f32);

        let gpu_output = run_gpu_compute(device, queue, &input_rgba8);
        let cpu_output = run_cpu_compute(&input_f32);

        let mut gpu_flat = Vec::with_capacity(gpu_output.len() * 4);
        for px in &gpu_output {
            gpu_flat.extend_from_slice(px);
        }
        let mut cpu_flat = Vec::with_capacity(cpu_output.len() * 4);
        for px in &cpu_output {
            cpu_flat.extend_from_slice(px);
        }
        let labels = build_labels("storage_texture");
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        vec![harness::compare_f32_results(
            "storage_texture_store_load",
            &gpu_flat,
            &cpu_flat,
            &label_refs,
            epsilon,
        )]
    }
}
