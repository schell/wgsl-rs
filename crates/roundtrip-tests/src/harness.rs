//! GPU/CPU roundtrip test harness.
//!
//! Provides utilities for running compute shaders on both GPU and CPU, reading
//! back results, and comparing them within tolerance.

use futures::executor::block_on;

/// Creates a headless wgpu device and queue, or returns `None` if no adapter
/// is available.
pub fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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

    eprintln!(
        "[roundtrip-tests] GPU: {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );

    let (device, queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some((device, queue))
}

/// Align `value` up to the next multiple of `alignment`.
fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Creates an `Rgba32Float` render target texture.
pub fn create_rgba32float_render_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &'static str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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

/// Reads an `Rgba32Float` texture back as row-major pixels.
pub fn read_rgba32float_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    const TEXEL_SIZE: u32 = 16;
    let bytes_per_row = align_to((width * TEXEL_SIZE) as u64, 256) as u32;
    let buffer_size = (bytes_per_row * height) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("roundtrip_rgba32f_readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("roundtrip_rgba32f_readback"),
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
        .expect("wgpu poll failed");
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
            let px: [f32; 4] = *bytemuck::from_bytes(&data[offset..offset + TEXEL_SIZE as usize]);
            pixels.push(px);
        }
    }
    drop(data);
    staging.unmap();
    pixels
}

/// Parameters for a GPU compute dispatch.
pub struct GpuComputeParams<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub shader_source: &'a str,
    pub entry_point: &'a str,
    pub bind_group_layout_entries: &'a [wgpu::BindGroupLayoutEntry],
    pub input_data: &'a [u8],
    pub output_size: u64,
    pub workgroup_count: (u32, u32, u32),
}

/// Runs a compute shader on the GPU and reads back the output storage buffer.
///
/// This is the core GPU execution primitive. It:
/// 1. Creates the shader module from WGSL source
/// 2. Creates the bind group layout and pipeline
/// 3. Uploads `input_data` to the input storage buffer (binding 0)
/// 4. Creates an output storage buffer of `output_size` bytes (binding 1)
/// 5. Dispatches the compute shader
/// 6. Reads back the output buffer contents
pub fn run_gpu_compute(params: &GpuComputeParams<'_>) -> Vec<u8> {
    let GpuComputeParams {
        device,
        queue,
        shader_source,
        entry_point,
        bind_group_layout_entries,
        input_data,
        output_size,
        workgroup_count,
    } = params;
    let output_size = *output_size;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("roundtrip_test"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("roundtrip_test"),
        entries: bind_group_layout_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("roundtrip_test"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("roundtrip_test"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create input buffer and upload data.
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input"),
        size: input_data.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input_buffer, 0, input_data);

    // Create output buffer.
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Zero-initialize the output buffer.
    let zeros = vec![0u8; output_size as usize];
    queue.write_buffer(&output_buffer, 0, &zeros);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("roundtrip_test"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("roundtrip_test"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("roundtrip_test"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
    }

    // Copy output to staging buffer for readback.
    let staging_size = align_to(output_size, 4);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: staging_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);

    let idx = queue.submit(Some(encoder.finish()));

    // Map and read back.
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
    let result = data[..output_size as usize].to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Result of a single roundtrip comparison.
#[derive(Debug)]
pub struct ComparisonResult {
    /// Name of the test.
    pub name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Maximum absolute error across all compared values.
    pub max_error: f32,
    /// Details about any mismatches (empty if passed).
    pub mismatches: Vec<String>,
}

/// Compares two `f32` slices element-by-element within the given tolerance.
///
/// Returns a `ComparisonResult` with details about the comparison. Values that
/// are both NaN are considered matching. Infinities must match exactly.
pub fn compare_f32_results(
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    labels: &[&str],
    epsilon: f32,
) -> ComparisonResult {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU and CPU result lengths differ for {name}"
    );
    assert_eq!(
        gpu.len(),
        labels.len(),
        "result length and label count differ for {name}"
    );

    let mut max_error: f32 = 0.0;
    let mut mismatches = Vec::new();

    for (i, ((g, c), label)) in gpu.iter().zip(cpu.iter()).zip(labels.iter()).enumerate() {
        // Both NaN is fine.
        if g.is_nan() && c.is_nan() {
            continue;
        }
        // One NaN and one not is a mismatch.
        if g.is_nan() || c.is_nan() {
            mismatches.push(format!("  [{i}] {label}: GPU={g}, CPU={c} (NaN mismatch)"));
            continue;
        }
        let err = (g - c).abs();
        max_error = max_error.max(err);
        if err > epsilon {
            mismatches.push(format!("  [{i}] {label}: GPU={g}, CPU={c} (err={err:.2e})"));
        }
    }

    ComparisonResult {
        name: name.to_string(),
        passed: mismatches.is_empty(),
        max_error,
        mismatches,
    }
}

/// Compares two `u32` slices element-by-element for exact equality.
///
/// Returns a `ComparisonResult`. All mismatches are reported since u32
/// comparisons are exact (no epsilon).
pub fn compare_u32_results(
    name: &str,
    gpu: &[u32],
    cpu: &[u32],
    labels: &[&str],
) -> ComparisonResult {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU and CPU result lengths differ for {name}"
    );
    assert_eq!(
        gpu.len(),
        labels.len(),
        "result length and label count differ for {name}"
    );

    let mut mismatches = Vec::new();

    for (i, ((g, c), label)) in gpu.iter().zip(cpu.iter()).zip(labels.iter()).enumerate() {
        if g != c {
            mismatches.push(format!(
                "  [{i}] {label}: GPU=0x{g:08X} ({g}), CPU=0x{c:08X} ({c})"
            ));
        }
    }

    ComparisonResult {
        name: name.to_string(),
        passed: mismatches.is_empty(),
        max_error: if mismatches.is_empty() { 0.0 } else { 1.0 },
        mismatches,
    }
}

/// Compares two `u32` slices where each u32 contains packed sub-values.
///
/// Each u32 is split into bytes and the maximum per-byte difference is
/// checked against `max_byte_diff`. This is useful for pack functions where
/// the WGSL spec allows `±1` LSB rounding differences.
pub fn compare_packed_u32_results(
    name: &str,
    gpu: &[u32],
    cpu: &[u32],
    labels: &[&str],
    max_byte_diff: u8,
) -> ComparisonResult {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU and CPU result lengths differ for {name}"
    );
    assert_eq!(
        gpu.len(),
        labels.len(),
        "result length and label count differ for {name}"
    );

    let mut mismatches = Vec::new();
    let mut max_error: f32 = 0.0;

    for (i, ((g, c), label)) in gpu.iter().zip(cpu.iter()).zip(labels.iter()).enumerate() {
        let g_bytes = g.to_le_bytes();
        let c_bytes = c.to_le_bytes();
        let mut byte_diff = 0u8;
        for (gb, cb) in g_bytes.iter().zip(c_bytes.iter()) {
            let d = (*gb as i16 - *cb as i16).unsigned_abs() as u8;
            byte_diff = byte_diff.max(d);
        }
        max_error = max_error.max(byte_diff as f32);
        if byte_diff > max_byte_diff {
            mismatches.push(format!(
                "  [{i}] {label}: GPU=0x{g:08X}, CPU=0x{c:08X} (max byte diff={byte_diff})"
            ));
        }
    }

    ComparisonResult {
        name: name.to_string(),
        passed: mismatches.is_empty(),
        max_error,
        mismatches,
    }
}

/// The standard bind group layout entries for a read-only input buffer at
/// binding 0 and a read-write output buffer at binding 1.
///
/// This is the common layout used by almost all roundtrip test shaders.
pub const STANDARD_LAYOUT_ENTRIES: &[wgpu::BindGroupLayoutEntry] = &[
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

/// A roundtrip test that can be run by the harness.
pub trait RoundtripTest {
    /// Short name for this test category (e.g., "trig", "exponential").
    fn name(&self) -> &str;

    /// Description of what functions this test covers.
    fn description(&self) -> &str;

    /// Runs the test, returning comparison results.
    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult>;
}
