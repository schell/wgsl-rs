//! CPU/GPU roundtrip for IO attributes on entry-point parameters.
//!
//! The fragment entry point directly consumes a location parameter with an
//! explicit interpolation attribute, covering wgsl-rs#259 in both worlds.

#![allow(dead_code)]

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const MATERIAL: u32 = 7;

#[wgsl]
pub mod direct_location_param {
    use wgsl_rs::std::*;

    pub struct Varyings {
        #[builtin(position)]
        pub clip_position: Vec4f,
        #[location(0)]
        pub pos: Vec3f,
        #[location(1)]
        #[interpolate(flat)]
        pub material: u32,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Varyings {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        Varyings {
            clip_position: vec4f(x, y, 0.0, 1.0),
            pos: vec3f(0.25, 0.5, 0.75),
            material: 7u32,
        }
    }

    #[fragment]
    pub fn frag_main(
        #[location(0)] pos: Vec3f,
        #[location(1)]
        #[interpolate(flat)]
        material: u32,
    ) -> Vec4f {
        vec4f(pos.x * f32(material), pos.y, pos.z, 1.0)
    }
}

/// Renders the direct location parameter shader and reads back its pixels.
fn render_direct_location_param(device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 4]> {
    let texture = harness::create_rgba32float_render_target(
        device,
        WIDTH,
        HEIGHT,
        "direct_location_param_target",
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut linkage =
        wgsl_rs_linkage_wgpu::analyze_wgsl_module(&direct_location_param::WGSL_SOURCE).unwrap();
    let module = linkage.shader_module(device);
    let pipeline_layout = linkage.pipeline_layout(device, Some("direct_location_param"));
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("direct_location_param"),
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
        label: Some("direct_location_param"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("direct_location_param"),
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

    harness::read_rgba32float_texture(device, queue, &texture, WIDTH, HEIGHT)
}

pub struct LocationParamsTest;

impl RoundtripTest for LocationParamsTest {
    fn name(&self) -> &str {
        "location_params"
    }

    fn description(&self) -> &str {
        "Direct fragment location parameter combined with interpolation"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let gpu_pixels = render_direct_location_param(device, queue);
        let gpu: Vec<f32> = gpu_pixels.iter().flatten().copied().collect();
        let cpu: Vec<f32> = (0..(WIDTH * HEIGHT) as usize)
            .flat_map(|_| {
                direct_location_param::frag_main(wgsl_rs::std::vec3f(0.25, 0.5, 0.75), MATERIAL)
                    .to_array()
            })
            .collect();
        let labels = vec!["fragment output"; gpu.len()];
        vec![harness::compare_f32_results(
            "direct location + interpolation parameter",
            &gpu,
            &cpu,
            &labels,
            1e-6,
        )]
    }
}
