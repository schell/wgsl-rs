use wgsl_rs::wgsl;

#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing
    //! color. Original source is [here](https://google.github.io/tour-of-wgsl/).

    // Only glob-imports are supported, but hey, imports work!
    use wgsl_rs::std::*;

    // Define a uniform in both Rust and WGSL using the uniform! macro.
    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];

        let position = POS[vertex_index as usize];
        vec4f(position.x(), position.y(), 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(FRAME) / 128.0), 0.0, 1.0)
    }
}

#[wgsl]
pub mod structs {
    use wgsl_rs::std::*;

    // Mixed builtins and user-defined inputs.
    #[input]
    pub struct MyInputs {
        #[location(0)]
        pub x: Vec4<f32>,

        #[builtin(front_facing)]
        pub y: bool,

        #[location(1)]
        #[interpolate(flat)]
        pub z: u32,

        #[location(2)]
        pub other: f32,
    }

    #[output]
    pub struct MyOutputs {
        #[location(0)]
        pub x: f32,

        #[location(1)]
        pub y: Vec4<f32>,
    }

    #[fragment]
    pub fn frag_shader(in1: MyInputs) -> MyOutputs {
        MyOutputs { x: 0.0, y: in1.x }
    }
}

#[wgsl]
pub mod compute_shader {
    //! A simple compute shader that demonstrates storage buffers.
    use wgsl_rs::std::*;

    // Read-only input buffer
    storage!(group(0), binding(0), INPUT: [f32; 256]);

    // Read-write output buffer
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Compute the index from global invocation ID
        // Note: Storage buffer access requires additional implementation
        // This demonstrates the compute shader structure with storage buffers
        let _idx = global_id.x() as usize;
    }
}

#[wgsl]
#[allow(dead_code)]
pub mod matrix_example {
    //! Demonstrates matrix types and constructors.
    use wgsl_rs::std::*;

    // 4x4 identity matrix constant
    const IDENTITY: Mat4f = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0),
    );

    // 3x3 2D rotation matrix (30 degrees)
    // cos(30°) ≈ 0.866, sin(30°) = 0.5
    const ROTATION_2D: Mat3f = mat3x3f(
        vec3f(0.866, 0.5, 0.0),
        vec3f(-0.5, 0.866, 0.0),
        vec3f(0.0, 0.0, 1.0),
    );

    // 2x2 matrix constant
    const SCALE_2D: Mat2f = mat2x2f(vec2f(2.0, 0.0), vec2f(0.0, 2.0));

    #[vertex]
    pub fn matrix_vertex() -> Vec4f {
        vec4f(0.0, 0.0, 0.0, 1.0)
    }
}

fn validate_and_print_source(source: &str) {
    println!("raw source:\n\n{source}\n\n");

    // Parse the source into a Module.
    let module: naga::Module = naga::front::wgsl::parse_str(source).unwrap();

    // Validate the module.
    // Validation can be made less restrictive by changing the ValidationFlags.
    let result = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module);

    let info = match result {
        Err(e) => {
            panic!("{}", e.emit_to_string(source));
        }
        Ok(i) => i,
    };

    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    println!("naga source:\n\n{wgsl}");
}

/// Test the linkage API when the feature is enabled.
fn print_linkage() {
    println!("\n=== Testing Linkage API ===\n");

    // Test hello_triangle linkage
    println!(
        "hello_triangle::linkage::SHADER_SOURCE length: {}",
        hello_triangle::linkage::SHADER_SOURCE.len()
    );
    println!(
        "hello_triangle bind_group_0 layout entries: {}",
        hello_triangle::linkage::bind_group_0::LAYOUT_ENTRIES.len()
    );
    println!(
        "hello_triangle vtx_main entry point: {}",
        hello_triangle::linkage::vtx_main::ENTRY_POINT
    );
    println!(
        "hello_triangle frag_main entry point: {}",
        hello_triangle::linkage::frag_main::ENTRY_POINT
    );

    // Test compute_shader linkage
    println!(
        "\ncompute_shader::linkage::SHADER_SOURCE length: {}",
        compute_shader::linkage::SHADER_SOURCE.len()
    );
    println!(
        "compute_shader bind_group_0 layout entries: {}",
        compute_shader::linkage::bind_group_0::LAYOUT_ENTRIES.len()
    );
    println!(
        "compute_shader main entry point: {}",
        compute_shader::linkage::main::ENTRY_POINT
    );
    println!(
        "compute_shader main workgroup size: {:?}",
        compute_shader::linkage::main::WORKGROUP_SIZE
    );

    // Test structs linkage
    println!(
        "\nstructs::linkage::SHADER_SOURCE length: {}",
        structs::linkage::SHADER_SOURCE.len()
    );
    println!(
        "structs frag_shader entry point: {}",
        structs::linkage::frag_shader::ENTRY_POINT
    );

    println!("\n=== Linkage API tests passed ===");
}

/// Build the linkage into a working `winit` + `wgpu` app, as a
/// dogfooding test.
fn build_linkage() {
    use std::sync::Arc;

    use futures::executor::block_on;
    use winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    };

    let event_loop = EventLoop::new().unwrap();

    struct WgpuStuff {
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        _adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl WgpuStuff {
        fn new(window: Arc<Window>) -> Self {
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window).unwrap();
            let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }))
            .expect("Failed to find an appropriate adapter");

            let (device, queue) =
                block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .expect("Failed to create device");
            let config = surface
                .get_default_config(&adapter, 800, 600)
                .expect("no default surface config");
            surface.configure(&device, &config);

            Self {
                _instance: instance,
                surface,
                _adapter: adapter,
                device,
                queue,
            }
        }
    }

    struct HelloTriangle {
        frame: u32,
        frame_uniform_buffer: wgpu::Buffer,
        bindgroup: wgpu::BindGroup,
        render_pipeline: wgpu::RenderPipeline,
    }

    impl HelloTriangle {
        fn new(wgpu_stuff: &WgpuStuff) -> Self {
            let device = &wgpu_stuff.device;
            let queue = &wgpu_stuff.queue;
            let frame = 0u32;
            let frame_uniform_buffer = hello_triangle::create_frame_buffer(device);
            queue.write_buffer(&frame_uniform_buffer, 0, &frame.to_ne_bytes());

            let bindgroup_layout = hello_triangle::linkage::bind_group_0::layout(device);
            let bindgroup = hello_triangle::linkage::bind_group_0::create(
                device,
                &bindgroup_layout,
                frame_uniform_buffer.as_entire_binding(),
            );

            let module = hello_triangle::linkage::shader_module(device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hello_triangle"),
                bind_group_layouts: &[&bindgroup_layout],
                immediate_size: 0,
            });
            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hello_triangle"),
                layout: Some(&pipeline_layout),
                vertex: hello_triangle::linkage::vtx_main::vertex_state(&module),
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
                fragment: Some(hello_triangle::linkage::frag_main::fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu_stuff
                            .surface
                            .get_configuration()
                            .expect("missing surface configuration")
                            .format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                )),
                multiview_mask: None,
                cache: None,
            });

            Self {
                frame,
                frame_uniform_buffer,
                bindgroup,
                render_pipeline,
            }
        }
    }

    struct AppInner {
        window: Arc<Window>,
        wgpu_stuff: WgpuStuff,
        hello_triangle: HelloTriangle,
    }

    impl AppInner {
        pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Self {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            let wgpu_stuff = WgpuStuff::new(window.clone());
            let hello_triangle = HelloTriangle::new(&wgpu_stuff);
            Self {
                window,
                wgpu_stuff,
                hello_triangle,
            }
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
            event: winit::event::WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    println!("Closing");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    if let Some(AppInner {
                        window,
                        wgpu_stuff,
                        hello_triangle,
                    }) = self.inner.as_mut()
                    {
                        let device = &wgpu_stuff.device;
                        let queue = &wgpu_stuff.queue;

                        hello_triangle.frame += 1;
                        queue.write_buffer(
                            &hello_triangle.frame_uniform_buffer,
                            0,
                            &hello_triangle.frame.to_ne_bytes(),
                        );

                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("pass"),
                            });
                        let texture = wgpu_stuff
                            .surface
                            .get_current_texture()
                            .expect("couldn't get current texture");
                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &texture
                                            .texture
                                            .create_view(&wgpu::TextureViewDescriptor::default()),
                                        depth_slice: None,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.0,
                                                g: 0.0,
                                                b: 0.0,
                                                a: 0.0,
                                            }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    ..Default::default()
                                });
                            render_pass.set_pipeline(&hello_triangle.render_pipeline);
                            render_pass.set_bind_group(0, &hello_triangle.bindgroup, &[]);
                            render_pass.draw(0..3, 0..1);
                        }

                        let index = queue.submit(Some(encoder.finish()));
                        texture.present();

                        // TODO: maybe do this elsewhere?
                        device
                            .poll(wgpu::PollType::Wait {
                                submission_index: Some(index),
                                timeout: None,
                            })
                            .unwrap();

                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

pub fn main() {
    {
        // hello_triangle
        let source = hello_triangle::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
    {
        // structs
        let source = structs::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
    {
        // compute_shader
        let source = compute_shader::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
    {
        // matrix_example
        let source = matrix_example::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }

    print_linkage();
    build_linkage();
}
