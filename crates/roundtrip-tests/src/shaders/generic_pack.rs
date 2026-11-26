//! CPU/GPU round-trip for type-generic resource packs: one pack type
//! instantiates with different element types, and `array_length` proves the
//! substitution reached both worlds.

#![allow(dead_code)]

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const WIDTH: u32 = 1;
const HEIGHT: u32 = 1;

#[wgsl(skip_validation)]
pub mod generic {
    use wgsl_rs::std::*;

    pub struct DataPack<T> {
        #[storage(group(0), binding(0), read_only)]
        pub data: Storage<RuntimeArray<T>, Read>,
    }

    /// A big triangle covering the whole one-pixel target.
    #[vertex]
    pub fn cover(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = select(-1.0, 3.0, vertex_index == 1u32);
        let y = select(-1.0, 3.0, vertex_index == 2u32);
        vec4f(x, y, 0.0, 1.0)
    }

    /// The element count reaches the color through `array_length`, so a
    /// wrong element-type substitution changes the result.
    #[fragment]
    pub fn fragment<T: Wgsl>(#[builtin(position)] position: Vec4f, pack: &DataPack<T>) -> Vec4f {
        let count = array_length(&pack.data);
        vec4f(count as f32 / 8.0, position.y * 0.0, 0.0, 1.0)
    }
}

fn cpu_run(element_count: u32, is_vec: bool) -> Vec<f32> {
    use generic as shader;
    use wgsl_rs::std::*;

    if is_vec {
        let pack = shader::DataPack::<Vec4f> {
            data: Storage::new(0, 0),
        };
        pack.data.set(RuntimeArray {
            data: vec![vec4f(1.0, 2.0, 3.0, 4.0); element_count as usize],
        });
        let color = shader::fragment::<Vec4f>(vec4f(0.0, 0.0, 0.0, 1.0), &pack);
        vec![color.x, color.y, color.z, color.w]
    } else {
        let pack = shader::DataPack::<f32> {
            data: Storage::new(0, 0),
        };
        pack.data.set(RuntimeArray {
            data: vec![7.0; element_count as usize],
        });
        let color = shader::fragment::<f32>(vec4f(0.0, 0.0, 0.0, 1.0), &pack);
        vec![color.x, color.y, color.z, color.w]
    }
}

fn gpu_run(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], is_vec: bool) -> Vec<f32> {
    use generic as shader;
    use wgsl_rs::std::*;

    let module = if is_vec {
        shader::instantiate::<Vec4f, Vec4f>()
    } else {
        shader::instantiate::<f32, f32>()
    };
    let source = module.wgsl_source();
    assert!(
        if is_vec {
            source.contains("var<storage, read> datapack__data: array<vec4f>;")
        } else {
            source.contains("var<storage, read> datapack__data: array<f32>;")
        },
        "{source}"
    );
    assert!(!source.contains("DataPack"), "{source}");

    let mut linkage = wgsl_rs_linkage_wgpu::analyze_ir_module(module);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generic_pack_storage"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    let bind_group = linkage
        .create_bind_group_named(0, device, &[("datapack__data", buffer.as_entire_binding())])
        .unwrap();

    let shader_module = linkage.shader_module(device);
    let pipeline_layout = linkage.pipeline_layout(device, Some("generic_pack_pipeline_layout"));
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("generic_pack_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: linkage
            .vertex_entry("cover")
            .unwrap()
            .vertex_state(&shader_module),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(linkage.fragment_entry("fragment").unwrap().fragment_state(
            &shader_module,
            &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::all(),
            })],
        )),
        multiview_mask: None,
        cache: None,
    });

    let target =
        harness::create_rgba32float_render_target(device, WIDTH, HEIGHT, "generic_pack_target");
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("generic_pack_render"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("generic_pack_render"),
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
        .into_iter()
        .flatten()
        .collect()
}

pub struct GenericPackTest;

impl RoundtripTest for GenericPackTest {
    fn name(&self) -> &str {
        "generic_pack"
    }

    fn description(&self) -> &str {
        "type-generic storage packs instantiated with vec4f and f32 element types"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        assert_eq!(
            generic::WGSL_SOURCE.module_type_params,
            &["DataPack_T", "fragment_0"]
        );
        let vec_bytes: Vec<u8> = (0..5)
            .flat_map(|_| [1.0f32, 2.0, 3.0, 4.0])
            .flat_map(f32::to_ne_bytes)
            .collect();
        let f32_bytes: Vec<u8> = (0..7).flat_map(|_| 7.0f32.to_ne_bytes()).collect();
        [("vec4f", 5, true, vec_bytes), ("f32", 7, false, f32_bytes)]
            .into_iter()
            .map(|(label, count, is_vec, bytes)| {
                let cpu = cpu_run(count, is_vec);
                let gpu = gpu_run(device, queue, &bytes, is_vec);
                let labels: Vec<String> = (0..cpu.len())
                    .map(|index| format!("generic_pack[{label}][{index}]"))
                    .collect();
                let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                harness::compare_f32_results(
                    &format!("generic_pack_{label}"),
                    &gpu,
                    &cpu,
                    &refs,
                    1e-6,
                )
            })
            .collect()
    }
}
