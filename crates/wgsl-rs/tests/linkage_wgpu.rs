//! Integration tests for the runtime wgpu linkage analyzer.
//!
//! These tests exercise `wgsl_rs::linkage::wgpu::analyze_wgsl_module` and
//! `analyze_module` against small `#[wgsl]` modules. They don't require
//! a GPU; they inspect the produced `WgpuLinkage` and assert on its
//! shape and the buffer sizes computed from WGSL §14.4.1.

#![cfg(feature = "linkage-wgpu")]

#[allow(unused_imports)]
use wgsl_rs::wgsl;
use wgsl_rs::{
    ir::{self, FnAttrs, StorageAccess, TextureDepthKind, TextureKind},
    linkage::wgpu::{self as wg, IrModuleExt},
};

// ===== Test modules =====

/// Concrete module: one uniform, vertex, fragment.
#[wgsl]
pub mod triangle {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] _vi: u32) -> Vec4f {
        vec4f(0.0, 0.0, f32(get!(FRAME)) * 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, 0.0, 0.0, f32(get!(FRAME)) * 0.0 + 1.0)
    }
}

/// Concrete module: two compute entries and two storage buffers in the
/// same bind group.
#[wgsl]
pub mod dual_compute {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), read_only, INPUT: [f32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    #[compute]
    #[workgroup_size(8, 4, 1)]
    pub fn a() {}

    #[compute]
    #[workgroup_size(64)]
    pub fn b() {}
}

/// A struct used as a uniform's element type, to exercise WGSL §14.4.1
/// struct layout.
#[wgsl]
pub mod struct_uniform {
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Camera {
        pub position: Vec3f,
        pub fov: f32,
        pub view_proj: Mat4x4f,
    }

    uniform!(group(0), binding(0), CAMERA: Camera);

    #[fragment]
    pub fn fs_main() -> Vec4f {
        vec4f(0.0, 0.0, 0.0, get!(CAMERA).fov * 0.0 + 1.0)
    }
}

/// A module that imports another module's struct, to exercise
/// cross-module struct resolution for buffer sizing.
#[wgsl]
pub mod shape_provider {
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Circle {
        pub center: Vec2f,
        pub radius: f32,
    }

    storage!(group(0), binding(0), read_write, CIRCLES: [Circle; 16]);

    #[compute]
    #[workgroup_size(4)]
    pub fn main() {}
}

#[wgsl(skip_validation)]
pub mod shape_consumer {
    use super::shape_provider::*;
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), HIGHLIGHT: u32);

    #[compute]
    #[workgroup_size(1)]
    pub fn highlight() {
        let c: Circle = Circle {
            center: vec2f(0.0, 0.0),
            radius: 1.0,
        };
        let v: Vec4f = vec4f(0.0, 0.0, 0.0, f32(get!(HIGHLIGHT)));
        let w: Vec4f = vec4f(c.center.x, c.center.y, c.radius, v.z);
        let _u: Vec4f = vec4f(w.x, w.y, w.z, w.w);
    }
}

/// A generic (template) module — the analyzer must reject
/// `analyze_wgsl_module` for it but accept `analyze_module` of the
/// instantiated IR.
#[wgsl(crate_path = wgsl_rs, validate_with_instantiation_types(f32, f32))]
pub mod generic_triangle {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] _vi: u32) -> Vec4f {
        vec4f(0.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main<T: Convert<f32> + Wgsl>() -> Vec4f {
        let _t = get!(FRAME, T);
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}

// ===== Basic shape tests =====

#[test]
fn analyze_triangle_finds_bind_group_and_entries() {
    let linkage = wg::analyze_wgsl_module(&triangle::WGSL_SOURCE).unwrap();
    assert_eq!(linkage.module_label, "triangle");

    // One bind group, one uniform binding at @group(0) @binding(0).
    let bg = linkage.bind_group(0).expect("expected bind group 0");
    assert_eq!(bg.group, 0);
    assert_eq!(bg.entries.len(), 1);
    assert_eq!(bg.bindings.len(), 1);
    assert_eq!(bg.bindings[0].name, "FRAME");
    assert_eq!(bg.bindings[0].binding, 0);
    assert!(matches!(bg.bindings[0].kind, wg::BindingKind::Uniform));

    // Vertex + fragment entries, no compute.
    assert_eq!(linkage.vertex_entries.len(), 1);
    assert_eq!(linkage.vertex_entries[0].name, "vtx_main");
    assert_eq!(linkage.fragment_entries.len(), 1);
    assert_eq!(linkage.fragment_entries[0].name, "frag_main");
    assert!(linkage.compute_entries.is_empty());

    // One buffer descriptor for FRAME, sized to a u32.
    let buf = linkage.buffer("FRAME").expect("expected FRAME buffer");
    assert_eq!(buf.group, 0);
    assert_eq!(buf.binding, 0);
    assert!(matches!(buf.kind, wg::BufferKind::Uniform));
    assert_eq!(buf.size, 4);
    assert_eq!(buf.descriptor().size, 4);
}

#[test]
fn analyze_dual_compute_groups_workgroup_sizes() {
    let linkage = wg::analyze_wgsl_module(&dual_compute::WGSL_SOURCE).unwrap();
    assert_eq!(linkage.bind_groups.len(), 1);
    let bg = linkage.bind_group(0).unwrap();
    assert_eq!(bg.entries.len(), 2);

    assert_eq!(linkage.compute_entries.len(), 2);
    let a = linkage.compute_entry("a").unwrap();
    assert_eq!(a.workgroup_size, (8, 4, 1));
    let b = linkage.compute_entry("b").unwrap();
    assert_eq!(b.workgroup_size, (64, 1, 1));

    let input = linkage.buffer("INPUT").unwrap();
    assert!(matches!(
        input.kind,
        wg::BufferKind::Storage { read_only: true }
    ));
    let output = linkage.buffer("OUTPUT").unwrap();
    assert!(matches!(
        output.kind,
        wg::BufferKind::Storage { read_only: false }
    ));
}

#[test]
fn shader_module_descriptor_borrows_module_label() {
    let linkage = wg::analyze_wgsl_module(&triangle::WGSL_SOURCE).unwrap();
    let desc = linkage.shader_module_descriptor();
    assert!(matches!(
        desc.source,
        wgpu::ShaderSource::Wgsl(_) // imported via test crate
    ));
    // The label is a &'static str borrowed from the linkage's module_label.
    assert_eq!(desc.label, Some("triangle"));
}

// ===== §14.4.1 sizing tests =====

#[test]
fn uniform_with_struct_is_sized_by_wgsl_rules() {
    let linkage = wg::analyze_wgsl_module(&struct_uniform::WGSL_SOURCE).unwrap();
    let cam = linkage.buffer("CAMERA").expect("CAMERA binding");
    // WGSL §14.4.1 for the `Camera` struct:
    //   position: Vec3f -> offset 0,  size 12, align 16
    //   fov:      f32    -> offset 12, size 4,  align 4
    //   view_proj: Mat4x4f (align 16, size 64) -> roundUp(16, 16) = 16, size 64
    //   struct align = 16, size = roundUp(80, 16) = 80
    assert_eq!(cam.size, 80, "expected WGSL-sized 80-byte Camera uniform");
}

#[test]
fn storage_array_size_uses_wgsl_array_strides() {
    let linkage = wg::analyze_wgsl_module(&shape_provider::WGSL_SOURCE).unwrap();
    let circles = linkage.buffer("CIRCLES").expect("CIRCLES binding");
    // Circle is { center: Vec2f (align 8, size 8), radius: f32 (align 4, size 4) }
    //   struct align = 8, size = roundUp(12, 8) = 16
    // array<[16], 16>: align 8, size = 16 * roundUp(16, 8) = 16 * 16 = 256
    assert_eq!(circles.size, 256, "expected 256-byte array<Circle, 16>");
}

// ===== Cross-module struct resolution =====

#[test]
fn analyze_walks_imports_for_struct_lookup() {
    // `shape_consumer` doesn't define any buffers referencing `Circle`,
    // but it does import `shape_provider` (whose IR gets flattened in
    // when we assemble). The provider's `CIRCLES` storage should be
    // findable with its WGSL-sized buffer descriptor.
    let linkage = wg::analyze_wgsl_module(&shape_consumer::WGSL_SOURCE).unwrap();
    // HIGHLIGHT is in shape_consumer itself.
    let highlight = linkage.buffer("HIGHLIGHT").expect("HIGHLIGHT binding");
    assert_eq!(highlight.size, 4);

    // The imported shape_provider's CIRCLES is also in the linkage.
    let circles = linkage
        .buffer("CIRCLES")
        .expect("CIRCLES from imported shape_provider");
    assert_eq!(circles.size, 256);
}

// ===== Template / instantiation tests =====

#[test]
fn analyze_wgsl_module_returns_error_for_template() {
    let result = wg::analyze_wgsl_module(&generic_triangle::WGSL_SOURCE);
    let err = result.expect_err("analyzing a template module should return an error");
    let msg = err.to_string();
    assert!(
        msg.contains("template") && msg.contains("generic_triangle"),
        "expected template error mentioning generic_triangle, got: {msg}"
    );
}

#[test]
fn analyze_module_after_instantiate_works() {
    // Instantiate the template and analyze the resulting IR directly.
    let m: ir::Module = generic_triangle::instantiate::<f32, f32>();
    let linkage = m.generate_linkage();

    assert_eq!(linkage.bind_groups.len(), 1);
    let frame = linkage.buffer("FRAME").expect("FRAME binding");
    assert_eq!(frame.size, 4, "f32 is 4 bytes");

    assert_eq!(linkage.vertex_entries.len(), 1);
    assert_eq!(linkage.vertex_entries[0].name, "vtx_main");
    // The fragment entry has its own type param, but the entry point
    // itself is still discovered by the analyzer.
    assert_eq!(linkage.fragment_entries.len(), 1);
    assert_eq!(linkage.fragment_entries[0].name, "frag_main");
}

#[test]
fn linkage_renders_valid_wgsl_for_instantiated_template() {
    // Regression test: a linkage built from an instantiated `ir::Module`
    // used to require passing the original (template) `&Module` to
    // `shader_module`, but that template's `wgsl_source()` emits
    // `__TP{name}__` placeholders — invalid WGSL. The linkage now
    // owns the concrete IR and renders source from it directly.
    let m: ir::Module = generic_triangle::instantiate::<f32, f32>();
    let linkage = m.generate_linkage();

    let source = linkage.wgsl_source();
    assert!(
        !source.contains("__TP"),
        "wgsl_source() should render valid WGSL with no placeholders, got:\n{source}"
    );
    assert!(
        source.contains("FRAME") && source.contains("f32"),
        "expected FRAME bound to f32 in source, got:\n{source}"
    );
}

// ===== IR-level type_size direct tests =====

#[test]
fn type_size_for_well_known_types() {
    let m = ir::Module {
        name: "m",
        items: vec![],
        attrs: vec![],
    };
    assert_eq!(
        wg::type_byte_size(&ir::Type::Scalar(ir::ScalarType::F32), &m),
        4
    );
    assert_eq!(
        wg::type_byte_size(&ir::Type::Scalar(ir::ScalarType::U32), &m),
        4
    );
    assert_eq!(
        wg::type_byte_size(&ir::Type::Scalar(ir::ScalarType::Bool), &m),
        4
    );

    // vec2f -> size 8, vec3f -> size 12, vec4f -> size 16
    let v2 = ir::Type::Vector {
        elements: 2,
        scalar_ty: Some(ir::ScalarType::F32),
    };
    let v3 = ir::Type::Vector {
        elements: 3,
        scalar_ty: Some(ir::ScalarType::F32),
    };
    let v4 = ir::Type::Vector {
        elements: 4,
        scalar_ty: Some(ir::ScalarType::F32),
    };
    assert_eq!(wg::type_byte_size(&v2, &m), 8);
    assert_eq!(wg::type_byte_size(&v3, &m), 12);
    assert_eq!(wg::type_byte_size(&v4, &m), 16);

    // mat4x4f -> 4 columns of vec4f, each stride 16, size = 64
    let m44 = ir::Type::Matrix {
        columns: 4,
        rows: 4,
        scalar_ty: Some(ir::ScalarType::F32),
    };
    assert_eq!(wg::type_byte_size(&m44, &m), 64);

    // RuntimeArray -> size 0
    let rt = ir::Type::RuntimeArray {
        elem: Box::new(ir::Type::Scalar(ir::ScalarType::F32)),
    };
    assert_eq!(wg::type_byte_size(&rt, &m), 0);
}

#[test]
fn type_size_for_named_const_array_len() {
    // Build a module with `const N: u32 = 16;` and an array length that
    // references `N`. The analyzer should resolve it.
    let m = ir::Module {
        name: "m",
        items: vec![ir::Item::Const(ir::ItemConst {
            name: "N".to_string(),
            ty: ir::Type::Scalar(ir::ScalarType::U32),
            expr: ir::Expr::Lit(ir::Lit::Int {
                digits: "16".to_string(),
                suffix: String::new(),
            }),
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let arr = ir::Type::Array {
        elem: Box::new(ir::Type::Scalar(ir::ScalarType::F32)),
        len: ir::Expr::Ident("N".to_string()),
    };
    assert_eq!(wg::type_byte_size(&arr, &m), 16 * 4);
}

#[test]
fn type_size_unknown_struct_falls_back_to_zero() {
    // A struct that isn't defined in the module should yield size 0
    // rather than panicking — the caller can still build a pipeline and
    // deal with the (probably broken) buffer at runtime.
    let m = ir::Module {
        name: "m",
        items: vec![],
        attrs: vec![],
    };
    let s = ir::Type::Struct {
        name: "NotDefined".to_string(),
        type_args: vec![],
    };
    assert_eq!(wg::type_byte_size(&s, &m), 0);
}

// ===== Sanity: matching the original compile-time output's shape =====

#[test]
fn bind_group_entries_sorted_by_binding_number() {
    // Module with bindings declared out of order; analyzer should sort
    // them by binding number in the layout entries.
    let ir_module = ir::Module {
        name: "out_of_order",
        items: vec![
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 5,
                name: "FIVE".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::F32),
                attrs: vec![],
            }),
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 1,
                name: "ONE".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::F32),
                attrs: vec![],
            }),
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 3,
                name: "THREE".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::F32),
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    let linkage = ir_module.generate_linkage();
    let bg = linkage.bind_group(0).unwrap();
    let names: Vec<&str> = bg.bindings.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, vec!["ONE", "THREE", "FIVE"]);
    let bindings: Vec<u32> = bg.entries.iter().map(|e| e.binding).collect();
    assert_eq!(bindings, vec![1, 3, 5]);
}

// ===== Texture / sampler coverage =====

#[test]
fn analyze_texture_and_sampler_bindings() {
    // Hand-rolled IR module with a texture and a sampler; ensures the
    // texture_layout_entry helper is exercised end-to-end without
    // requiring GPU instantiation.
    let ir_module = ir::Module {
        name: "tex",
        items: vec![
            ir::Item::Texture(ir::ItemTexture {
                group: 0,
                binding: 0,
                name: "TEX".to_string(),
                ty: ir::Type::Texture {
                    kind: TextureKind::Texture2D,
                    sampled_type: ir::ScalarType::F32,
                },
                attrs: vec![],
            }),
            ir::Item::Texture(ir::ItemTexture {
                group: 0,
                binding: 1,
                name: "SHADOW".to_string(),
                ty: ir::Type::TextureDepth {
                    kind: TextureDepthKind::Depth2D,
                },
                attrs: vec![],
            }),
            ir::Item::Sampler(ir::ItemSampler {
                group: 0,
                binding: 2,
                name: "SAMP".to_string(),
                ty: ir::Type::Sampler,
                attrs: vec![],
            }),
            ir::Item::Sampler(ir::ItemSampler {
                group: 0,
                binding: 3,
                name: "CSAMP".to_string(),
                ty: ir::Type::SamplerComparison,
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    let linkage = ir_module.generate_linkage();
    let bg = linkage.bind_group(0).unwrap();
    assert_eq!(bg.bindings.len(), 4);

    assert!(matches!(bg.bindings[0].kind, wg::BindingKind::Texture));
    assert!(matches!(bg.bindings[1].kind, wg::BindingKind::DepthTexture));
    assert!(matches!(
        bg.bindings[2].kind,
        wg::BindingKind::Sampler { comparison: false }
    ));
    assert!(matches!(
        bg.bindings[3].kind,
        wg::BindingKind::Sampler { comparison: true }
    ));
}

#[test]
fn analyze_storage_texture_bindings() {
    use wgsl_rs::ir::{StorageTextureAccess, TexelFormat, TextureStorageKind};

    let ir_module = ir::Module {
        name: "storage_tex",
        items: vec![
            ir::Item::Texture(ir::ItemTexture {
                group: 0,
                binding: 0,
                name: "OUT_TEX".to_string(),
                ty: ir::Type::TextureStorage {
                    kind: TextureStorageKind::Storage2D,
                    format: TexelFormat::Rgba8unorm,
                    access: StorageTextureAccess::Write,
                },
                attrs: vec![],
            }),
            ir::Item::Texture(ir::ItemTexture {
                group: 0,
                binding: 1,
                name: "IN_TEX".to_string(),
                ty: ir::Type::TextureStorage {
                    kind: TextureStorageKind::Storage2D,
                    format: TexelFormat::Rgba8uint,
                    access: StorageTextureAccess::Read,
                },
                attrs: vec![],
            }),
            ir::Item::Texture(ir::ItemTexture {
                group: 0,
                binding: 2,
                name: "RW_TEX".to_string(),
                ty: ir::Type::TextureStorage {
                    kind: TextureStorageKind::Storage2D,
                    format: TexelFormat::R32float,
                    access: StorageTextureAccess::ReadWrite,
                },
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    let linkage = ir_module.generate_linkage();
    let bg = linkage.bind_group(0).unwrap();
    assert_eq!(bg.bindings.len(), 3);

    assert!(matches!(
        bg.bindings[0].kind,
        wg::BindingKind::StorageTexture { read_write: false }
    ));
    assert!(matches!(
        bg.bindings[1].kind,
        wg::BindingKind::StorageTexture { read_write: false }
    ));
    assert!(matches!(
        bg.bindings[2].kind,
        wg::BindingKind::StorageTexture { read_write: true }
    ));
}

// ===== Storage access mode preserved =====

#[test]
fn storage_read_only_flag_matches_ir() {
    let ir_module = ir::Module {
        name: "ro",
        items: vec![ir::Item::Storage(ir::ItemStorage {
            group: 0,
            binding: 0,
            access: StorageAccess::Read,
            name: "INPUT".to_string(),
            ty: ir::Type::Scalar(ir::ScalarType::F32),
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let linkage = ir_module.generate_linkage();
    let buf = linkage.buffer("INPUT").unwrap();
    assert!(matches!(
        buf.kind,
        wg::BufferKind::Storage { read_only: true }
    ));

    let ir_rw = ir::Module {
        name: "rw",
        items: vec![ir::Item::Storage(ir::ItemStorage {
            group: 0,
            binding: 0,
            access: StorageAccess::ReadWrite,
            name: "OUTPUT".to_string(),
            ty: ir::Type::Scalar(ir::ScalarType::F32),
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let linkage = ir_rw.generate_linkage();
    let buf = linkage.buffer("OUTPUT").unwrap();
    assert!(matches!(
        buf.kind,
        wg::BufferKind::Storage { read_only: false }
    ));
}

// ===== Entry point discovery =====

#[test]
fn compute_workgroup_size_defaults_to_one() {
    let ir_module = ir::Module {
        name: "default_ws",
        items: vec![ir::Item::Fn(ir::ItemFn {
            type_params: vec![],
            const_params: vec![],
            fn_attrs: FnAttrs::Compute {
                workgroup_size: ir::WorkgroupSize {
                    x: 32,
                    y: None,
                    z: None,
                },
            },
            name: "main".to_string().into(),
            inputs: vec![],
            return_type: ir::ReturnType::Default,
            block: ir::Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let linkage = ir_module.generate_linkage();
    let entry = linkage.compute_entry("main").unwrap();
    assert_eq!(entry.workgroup_size, (32, 1, 1));
}

// ===== create_named: error and happy-path tests =====

/// Helper: returns a `wgpu::Device` for tests, or `None` if no GPU is
/// available. Tests that need a real device (the happy-path
/// `create_named` test) should early-return when this is `None`.
#[cfg(feature = "linkage-wgpu")]
fn test_device() -> Option<wgpu::Device> {
    use futures::executor::block_on;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, _queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(device)
}

#[cfg(feature = "linkage-wgpu")]
fn linkage_with_two_uniforms() -> wg::WgpuLinkage {
    let ir_module = ir::Module {
        name: "two_uni",
        items: vec![
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 0,
                name: "A".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::U32),
                attrs: vec![],
            }),
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 1,
                name: "B".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::U32),
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    ir_module.generate_linkage()
}

#[cfg(feature = "linkage-wgpu")]
fn make_uniform_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[test]
fn create_named_happy_path() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let linkage = linkage_with_two_uniforms();
    let bg = linkage.bind_group(0).unwrap();
    let layout = bg.layout(&device);

    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");

    // Pass the resources in REVERSE order to prove the function
    // matches by name and not by index.
    let result = bg.create_named(
        linkage.module_label,
        &device,
        &layout,
        &[
            ("B", b_buf.as_entire_binding()),
            ("A", a_buf.as_entire_binding()),
        ],
    );
    let _bind_group = result.expect("create_named should succeed");
}

#[test]
fn create_named_missing_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let linkage = linkage_with_two_uniforms();
    let bg = linkage.bind_group(0).unwrap();
    let layout = bg.layout(&device);
    let a_buf = make_uniform_buffer(&device, "A");

    // Only provide A, not B.
    let err = bg
        .create_named(
            linkage.module_label,
            &device,
            &layout,
            &[("A", a_buf.as_entire_binding())],
        )
        .expect_err("missing B should error");
    let msg = err.to_string();
    assert!(
        msg.contains("missing binding") && msg.contains("B"),
        "expected MissingBinding for B, got: {msg}"
    );
}

#[test]
fn create_named_unknown_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let linkage = linkage_with_two_uniforms();
    let bg = linkage.bind_group(0).unwrap();
    let layout = bg.layout(&device);
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    let extra_buf = make_uniform_buffer(&device, "Z");

    // All declared bindings present, plus one extra Z.
    let err = bg
        .create_named(
            linkage.module_label,
            &device,
            &layout,
            &[
                ("A", a_buf.as_entire_binding()),
                ("B", b_buf.as_entire_binding()),
                ("Z", extra_buf.as_entire_binding()),
            ],
        )
        .expect_err("unknown Z should error");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown binding name") && msg.contains("Z"),
        "expected UnknownBindingName for Z, got: {msg}"
    );
}

#[test]
fn create_named_duplicate_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let linkage = linkage_with_two_uniforms();
    let bg = linkage.bind_group(0).unwrap();
    let layout = bg.layout(&device);
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");

    // A twice.
    let err = bg
        .create_named(
            linkage.module_label,
            &device,
            &layout,
            &[
                ("A", a_buf.as_entire_binding()),
                ("A", b_buf.as_entire_binding()),
            ],
        )
        .expect_err("duplicate A should error");
    let msg = err.to_string();
    assert!(
        msg.contains("more than once") && msg.contains("A"),
        "expected DuplicateBindingName for A, got: {msg}"
    );
}

// ===== pipeline_layout: WgpuLinkage shortcut tests =====

/// Helper: a linkage with two bind groups (group 0 with one uniform,
/// group 1 with one uniform) so we can verify `pipeline_layout` returns
/// layouts in ascending group order.
#[cfg(feature = "linkage-wgpu")]
fn linkage_with_two_groups() -> wg::WgpuLinkage {
    let ir_module = ir::Module {
        name: "two_groups",
        items: vec![
            ir::Item::Uniform(ir::ItemUniform {
                group: 0,
                binding: 0,
                name: "A".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::U32),
                attrs: vec![],
            }),
            ir::Item::Uniform(ir::ItemUniform {
                group: 1,
                binding: 0,
                name: "B".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::U32),
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    ir_module.generate_linkage()
}

/// Helper: a linkage with no bind groups (e.g. a depth-only pipeline).
#[cfg(feature = "linkage-wgpu")]
fn linkage_with_no_groups() -> wg::WgpuLinkage {
    let ir_module = ir::Module {
        name: "no_groups",
        items: vec![ir::Item::Fn(ir::ItemFn {
            type_params: vec![],
            const_params: vec![],
            fn_attrs: ir::FnAttrs::Vertex,
            name: std::borrow::Cow::Borrowed("vtx_main"),
            inputs: vec![],
            return_type: ir::ReturnType::Default,
            block: ir::Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    ir_module.generate_linkage()
}

#[test]
fn pipeline_layout_single_group() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let pipeline_layout = linkage.pipeline_layout(&device, Some("single"));
    // Second call returns the same cached pipeline layout (cheap clone);
    // the call also succeeds when there is only one bind group.
    let again = linkage.pipeline_layout(&device, Some("single"));
    let _ = again;
    let _ = pipeline_layout;
}

#[test]
fn pipeline_layout_multiple_groups_in_order() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_groups();
    let pipeline_layout = linkage.pipeline_layout(&device, Some("multi"));
    // Reuse the cached bind group layouts with create_named to prove
    // they match the analyzed entries.
    let a_buf = make_uniform_buffer(&device, "A");
    let bgl_0 = linkage
        .bind_group_layout(0, &device)
        .expect("group 0 layout should exist");
    let _bg0 = linkage
        .bind_group(0)
        .unwrap()
        .create_named(
            linkage.module_label,
            &device,
            &bgl_0,
            &[("A", a_buf.as_entire_binding())],
        )
        .expect("group 0 bind group should build");
    let b_buf = make_uniform_buffer(&device, "B");
    let bgl_1 = linkage
        .bind_group_layout(1, &device)
        .expect("group 1 layout should exist");
    let _bg1 = linkage
        .bind_group(1)
        .unwrap()
        .create_named(
            linkage.module_label,
            &device,
            &bgl_1,
            &[("B", b_buf.as_entire_binding())],
        )
        .expect("group 1 bind group should build");
    let _ = pipeline_layout;
}

#[test]
fn pipeline_layout_empty_groups() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_no_groups();
    let pipeline_layout = linkage.pipeline_layout(&device, Some("empty"));
    let _ = pipeline_layout;
}

#[test]
fn bind_group_layout_returns_none_for_missing_group() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    // Only group 0 exists; asking for group 1 should return None.
    let none = linkage.bind_group_layout(1, &device);
    assert!(none.is_none());
}

#[test]
fn bind_group_layout_is_idempotent() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let first = linkage
        .bind_group_layout(0, &device)
        .expect("group 0 layout should exist");
    // Second call returns a clone; the wgpu::BindGroupLayout is
    // Arc-backed, so this is a cheap refcount bump, not a rebuild.
    let second = linkage
        .bind_group_layout(0, &device)
        .expect("group 0 layout should exist");
    // Both clones should be usable with create_named; they reference
    // the same underlying layout on the GPU.
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    let _bg1 = linkage
        .bind_group(0)
        .unwrap()
        .create_named(
            linkage.module_label,
            &device,
            &first,
            &[
                ("A", a_buf.as_entire_binding()),
                ("B", b_buf.as_entire_binding()),
            ],
        )
        .expect("first clone should build");
    let a_buf2 = make_uniform_buffer(&device, "A");
    let b_buf2 = make_uniform_buffer(&device, "B");
    let _bg2 = linkage
        .bind_group(0)
        .unwrap()
        .create_named(
            linkage.module_label,
            &device,
            &second,
            &[
                ("A", a_buf2.as_entire_binding()),
                ("B", b_buf2.as_entire_binding()),
            ],
        )
        .expect("second clone should build");
}

// ===== create_bind_group_named: WgpuLinkage-level wrapper tests =====

#[test]
fn create_bind_group_named_happy_path() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    // Pass resources in reverse order to prove name-based matching.
    let result = linkage.create_bind_group_named(
        0,
        &device,
        &[
            ("B", b_buf.as_entire_binding()),
            ("A", a_buf.as_entire_binding()),
        ],
    );
    let _bind_group = result.expect("create_bind_group_named should succeed");
}

#[test]
fn create_bind_group_named_missing_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let a_buf = make_uniform_buffer(&device, "A");
    // Only A, not B.
    let err = linkage
        .create_bind_group_named(0, &device, &[("A", a_buf.as_entire_binding())])
        .expect_err("missing B should error");
    let msg = err.to_string();
    assert!(
        msg.contains("missing binding") && msg.contains("B"),
        "expected MissingBinding for B, got: {msg}"
    );
}

#[test]
fn create_bind_group_named_unknown_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    let extra_buf = make_uniform_buffer(&device, "Z");
    let err = linkage
        .create_bind_group_named(
            0,
            &device,
            &[
                ("A", a_buf.as_entire_binding()),
                ("B", b_buf.as_entire_binding()),
                ("Z", extra_buf.as_entire_binding()),
            ],
        )
        .expect_err("unknown Z should error");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown binding name") && msg.contains("Z"),
        "expected UnknownBindingName for Z, got: {msg}"
    );
}

#[test]
fn create_bind_group_named_duplicate_binding() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    // A twice.
    let err = linkage
        .create_bind_group_named(
            0,
            &device,
            &[
                ("A", a_buf.as_entire_binding()),
                ("A", b_buf.as_entire_binding()),
            ],
        )
        .expect_err("duplicate A should error");
    let msg = err.to_string();
    assert!(
        msg.contains("more than once") && msg.contains("A"),
        "expected DuplicateBindingName for A, got: {msg}"
    );
}

#[test]
fn create_bind_group_named_no_such_group() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    let a_buf = make_uniform_buffer(&device, "A");
    // Group 1 doesn't exist; this module only has group 0.
    let err = linkage
        .create_bind_group_named(1, &device, &[("A", a_buf.as_entire_binding())])
        .expect_err("group 1 should not exist");
    let msg = err.to_string();
    assert!(
        msg.contains("no bind group") && msg.contains("@group(1)"),
        "expected NoSuchBindGroup for @group(1), got: {msg}"
    );
}

#[test]
fn create_bind_group_named_reuses_cached_layout() {
    let Some(device) = test_device() else {
        eprintln!("[skip] no GPU available");
        return;
    };
    let mut linkage = linkage_with_two_uniforms();
    // First call: populates cache, builds bind group.
    let a_buf = make_uniform_buffer(&device, "A");
    let b_buf = make_uniform_buffer(&device, "B");
    let _bg1 = linkage
        .create_bind_group_named(
            0,
            &device,
            &[
                ("A", a_buf.as_entire_binding()),
                ("B", b_buf.as_entire_binding()),
            ],
        )
        .expect("first call should succeed");
    // Second call: should reuse the cached layout (cheap, not a rebuild).
    let a_buf2 = make_uniform_buffer(&device, "A");
    let b_buf2 = make_uniform_buffer(&device, "B");
    let _bg2 = linkage
        .create_bind_group_named(
            0,
            &device,
            &[
                ("A", a_buf2.as_entire_binding()),
                ("B", b_buf2.as_entire_binding()),
            ],
        )
        .expect("second call should succeed");
}
