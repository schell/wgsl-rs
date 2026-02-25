//! Fractal Brownian Motion demo.
//!
//! Opens a window and renders an animated FBM shader with domain warping.
//! Uniforms for resolution, mouse position, and elapsed time are updated
//! each frame.
//!
//! Run with: `cargo run -p fbm-example`

mod shader;

use std::{sync::Arc, time::Instant};

use futures::executor::block_on;
use shader::fbm_shader;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

/// Holds wgpu state that lives for the lifetime of the window.
struct GpuContext {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("Failed to create device");

        let size = window.inner_size();
        let surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("no default surface config");
        surface.configure(&device, &surface_config);

        Self {
            surface,
            surface_config,
            device,
            queue,
        }
    }

    /// Reconfigure the surface after a window resize.
    fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }
}

/// Shader pipeline and uniform buffers.
struct FbmPipeline {
    resolution_buffer: wgpu::Buffer,
    mouse_buffer: wgpu::Buffer,
    time_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
}

impl FbmPipeline {
    fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        // Create uniform buffers using the generated helpers.
        let resolution_buffer = fbm_shader::create_u_resolution_buffer(device);
        let mouse_buffer = fbm_shader::create_u_mouse_buffer(device);
        let time_buffer = fbm_shader::create_u_time_buffer(device);

        // Bind group layout and bind group from generated linkage.
        let bg_layout = fbm_shader::linkage::bind_group_0::layout(device);
        let bind_group = fbm_shader::linkage::bind_group_0::create(
            device,
            &bg_layout,
            resolution_buffer.as_entire_binding(),
            mouse_buffer.as_entire_binding(),
            time_buffer.as_entire_binding(),
        );

        // Shader module and render pipeline.
        let module = fbm_shader::linkage::shader_module(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fbm"),
            bind_group_layouts: &[&bg_layout],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fbm"),
            layout: Some(&pipeline_layout),
            vertex: fbm_shader::linkage::vtx_main::vertex_state(&module),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(fbm_shader::linkage::frag_main::fragment_state(
                &module,
                &[Some(wgpu::ColorTargetState {
                    format: gpu.surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::all(),
                })],
            )),
            multiview_mask: None,
            cache: None,
        });

        Self {
            resolution_buffer,
            mouse_buffer,
            time_buffer,
            bind_group,
            render_pipeline,
        }
    }
}

/// Application state.
struct AppInner {
    window: Arc<Window>,
    gpu: GpuContext,
    pipeline: FbmPipeline,
    start_time: Instant,
    mouse_pos: [f32; 2],
}

impl AppInner {
    fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_title("wgsl-rs — Fractal Brownian Motion"),
                )
                .unwrap(),
        );
        let gpu = GpuContext::new(window.clone());
        let pipeline = FbmPipeline::new(&gpu);

        Self {
            window,
            gpu,
            pipeline,
            start_time: Instant::now(),
            mouse_pos: [0.0, 0.0],
        }
    }

    fn render(&mut self) {
        let queue = &self.gpu.queue;

        // Update uniforms.
        let resolution = [
            self.gpu.surface_config.width as f32,
            self.gpu.surface_config.height as f32,
        ];
        queue.write_buffer(
            &self.pipeline.resolution_buffer,
            0,
            bytemuck::bytes_of(&resolution),
        );
        queue.write_buffer(
            &self.pipeline.mouse_buffer,
            0,
            bytemuck::bytes_of(&self.mouse_pos),
        );
        let elapsed = self.start_time.elapsed().as_secs_f32();
        queue.write_buffer(&self.pipeline.time_buffer, 0, bytemuck::bytes_of(&elapsed));

        // Render.
        let texture = self
            .gpu
            .surface
            .get_current_texture()
            .expect("couldn't get current texture");
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fbm") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fbm"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline.render_pipeline);
            pass.set_bind_group(0, &self.pipeline.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        let index = queue.submit(Some(encoder.finish()));
        texture.present();

        self.gpu
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .unwrap();

        self.window.request_redraw();
    }
}

#[derive(Default)]
struct App {
    inner: Option<AppInner>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.inner = Some(AppInner::new(event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(app) = self.inner.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                app.gpu.resize(new_size.width, new_size.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                app.mouse_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::RedrawRequested => {
                app.render();
            }
            _ => {}
        }
    }
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
