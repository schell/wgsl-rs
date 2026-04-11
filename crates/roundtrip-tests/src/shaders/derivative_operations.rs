//! Roundtrip tests for derivative builtin functions.
//!
//! Tests: `dpdx`, `dpdy`, `fwidth` and fine/coarse variants.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;

#[wgsl]
pub mod derivative_basic {
    use wgsl_rs::std::*;

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
        vec4f(dpdx(p.x), dpdy(p.y), fwidth(p.x), fwidth(p.y))
    }
}

#[wgsl]
pub mod derivative_variants {
    use wgsl_rs::std::*;

    #[input]
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[output]
    pub struct Outputs {
        #[location(0)]
        pub fine: Vec4f,
        #[location(1)]
        pub coarse: Vec4f,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> Outputs {
        let p = input.position;
        Outputs {
            fine: vec4f(
                dpdx_fine(p.x),
                dpdy_fine(p.y),
                fwidth_fine(p.x),
                fwidth_fine(p.y),
            ),
            coarse: vec4f(
                dpdx_coarse(p.x),
                dpdy_coarse(p.y),
                fwidth_coarse(p.x),
                fwidth_coarse(p.y),
            ),
        }
    }
}


/// Renders the basic derivative shader to one target and returns pixels.
fn render_basic(device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 4]> {
    let texture =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "derivative_basic_target");
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let module = derivative_basic::linkage::shader_module(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("derivative_basic"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("derivative_basic"),
        layout: Some(&pipeline_layout),
        vertex: derivative_basic::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(derivative_basic::linkage::frag_main::fragment_state(
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
        label: Some("derivative_basic"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("derivative_basic"),
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

/// Renders fine and coarse variant outputs to two targets.
fn render_variants(device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let fine_texture =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "derivative_fine_target");
    let coarse_texture = harness::create_rgba32float_render_target(
        device,
        WIDTH,
        HEIGHT,
        "derivative_coarse_target",
    );
    let fine_view = fine_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let coarse_view = coarse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let module = derivative_variants::linkage::shader_module(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("derivative_variants"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("derivative_variants"),
        layout: Some(&pipeline_layout),
        vertex: derivative_variants::linkage::vtx_main::vertex_state(&module),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(derivative_variants::linkage::frag_main::fragment_state(
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
        )),
        multiview_mask: None,
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("derivative_variants"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("derivative_variants"),
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

    let fine = harness::read_rgba32float_texture(device, queue, &fine_texture, WIDTH, HEIGHT);
    let coarse = harness::read_rgba32float_texture(device, queue, &coarse_texture, WIDTH, HEIGHT);
    (fine, coarse)
}

/// Compares one channel over interior pixels only.
fn compare_channel_interior(
    name: &str,
    gpu_pixels: &[[f32; 4]],
    cpu_grid: &[Vec<Option<[f32; 4]>>],
    channel: usize,
    epsilon: f32,
) -> ComparisonResult {
    let mut gpu = Vec::new();
    let mut cpu = Vec::new();
    let mut labels = Vec::new();

    for y in 1..(HEIGHT - 1) {
        for x in 1..(WIDTH - 1) {
            let idx = (y * WIDTH + x) as usize;
            gpu.push(gpu_pixels[idx][channel]);
            cpu.push(
                cpu_grid[y as usize][x as usize].expect("non-helper fragment expected")[channel],
            );
            labels.push(format!("{name}[{x},{y}]"));
        }
    }

    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    harness::compare_f32_results(name, &gpu, &cpu, &label_refs, epsilon)
}

pub struct DerivativeOperationsTest;

impl RoundtripTest for DerivativeOperationsTest {
    fn name(&self) -> &str {
        "derivative_operations"
    }

    fn description(&self) -> &str {
        "dpdx, dpdy, fwidth, fine/coarse derivatives"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();
        let epsilon = 1e-4;

        let gpu_basic = render_basic(device, queue);
        let cpu_basic = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let p = builtins.position;
                [dpdx(p.x), dpdy(p.y), fwidth(p.x), fwidth(p.y)]
            },
        );

        results.push(compare_channel_interior(
            "derivative_dpdx",
            &gpu_basic,
            &cpu_basic,
            0,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_dpdy",
            &gpu_basic,
            &cpu_basic,
            1,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_x",
            &gpu_basic,
            &cpu_basic,
            2,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_y",
            &gpu_basic,
            &cpu_basic,
            3,
            epsilon,
        ));

        let (gpu_fine, gpu_coarse) = render_variants(device, queue);
        let cpu_fine = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let p = builtins.position;
                [
                    dpdx_fine(p.x),
                    dpdy_fine(p.y),
                    fwidth_fine(p.x),
                    fwidth_fine(p.y),
                ]
            },
        );
        let cpu_coarse = dispatch_fragments(
            WIDTH,
            HEIGHT,
            |_, _| (),
            |builtins, _| {
                let p = builtins.position;
                [
                    dpdx_coarse(p.x),
                    dpdy_coarse(p.y),
                    fwidth_coarse(p.x),
                    fwidth_coarse(p.y),
                ]
            },
        );

        results.push(compare_channel_interior(
            "derivative_dpdx_fine",
            &gpu_fine,
            &cpu_fine,
            0,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_dpdy_fine",
            &gpu_fine,
            &cpu_fine,
            1,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_fine_x",
            &gpu_fine,
            &cpu_fine,
            2,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_fine_y",
            &gpu_fine,
            &cpu_fine,
            3,
            epsilon,
        ));

        results.push(compare_channel_interior(
            "derivative_dpdx_coarse",
            &gpu_coarse,
            &cpu_coarse,
            0,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_dpdy_coarse",
            &gpu_coarse,
            &cpu_coarse,
            1,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_coarse_x",
            &gpu_coarse,
            &cpu_coarse,
            2,
            epsilon,
        ));
        results.push(compare_channel_interior(
            "derivative_fwidth_coarse_y",
            &gpu_coarse,
            &cpu_coarse,
            3,
            epsilon,
        ));

        results
    }
}
