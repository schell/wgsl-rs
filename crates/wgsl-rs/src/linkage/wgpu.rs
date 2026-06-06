//! Runtime wgpu linkage analysis for WGSL modules.
//!
//! Given an [`ir::Module`] (produced by a `#[wgsl]` module's
//! `WGSL_MODULE.ir_constructor`, or by [`Module::instantiate`] for
//! template modules), this module extracts the binding and entry-point
//! information needed to build wgpu pipelines.
//!
//! # Why at runtime?
//!
//! The pre-#120 implementation generated all of this at proc-macro
//! expansion time by walking the parse tree in
//! `wgsl-rs-macros/src/linkage.rs`. That had two drawbacks:
//!
//! 1. Template (`#[wgsl]` modules with type parameters) couldn't get any
//!    linkage — the generated WGSL was a template with `__TP{name}__`
//!    placeholders, so a `wgpu::ShaderModule` couldn't be built from it.
//! 2. The proc-macro walked a parse tree that duplicated information already
//!    present in `wgsl-rs-ir`.
//!
//! Walking the runtime IR unifies the two cases. After
//! `Module::instantiate::<...>()` produces a concrete `ir::Module`, the
//! same analyzer used for non-template modules works on the result.
//!
//! # Example
//!
//! ```ignore
//! use wgsl_rs::linkage::wgpu;
//!
//! let module = hello_triangle::WGSL_MODULE;
//! let linkage = wgpu::analyze_wgsl_module(&module);
//! let source = module.wgsl_source();
//! let shader_module = linkage.shader_module(&device, &source);
//! let bg_layout = linkage.bind_group(0).unwrap().layout(&device, Some("bg0"));
//! let bg = linkage.bind_group(0).unwrap().create(
//!     &device, &bg_layout, &[frame_uniform.as_entire_binding()],
//! );
//! let vtx = linkage.vertex_entries.iter().find(|e| e.name == "vtx_main").unwrap()
//!     .vertex_state(&shader_module);
//! ```

use std::collections::BTreeMap;

use wgsl_rs_ir as ir;

use crate::Module;

// ===== Top-level result =====

/// All the wgpu linkage information extracted from a WGSL module.
///
/// Produced by [`analyze_module`] or [`analyze_wgsl_module`]. Owns the
/// string labels that the various `wgpu::*Descriptor` types borrow, so
/// hold this struct for the lifetime of any borrowed descriptor.
#[derive(Clone, Debug)]
pub struct WgpuLinkage {
    /// The module's name (used as a default label for descriptors).
    pub module_label: String,

    /// Bind groups keyed by `@group(N)` index. Each group is sorted by
    /// binding number.
    pub bind_groups: BTreeMap<u32, BindGroupInfo>,

    /// Vertex shader entry points declared in the module.
    pub vertex_entries: Vec<EntryPointInfo>,

    /// Fragment shader entry points declared in the module.
    pub fragment_entries: Vec<EntryPointInfo>,

    /// Compute shader entry points declared in the module.
    pub compute_entries: Vec<ComputeEntryInfo>,

    /// Buffer descriptors for `uniform!` and `storage!` declarations, in
    /// declaration order. Indexed by binding name via [`Self::buffer`].
    pub buffers: Vec<BufferDescriptorInfo>,
}

impl WgpuLinkage {
    /// Returns the bind group for the given `@group(N)` index, if any.
    pub fn bind_group(&self, group: u32) -> Option<&BindGroupInfo> {
        self.bind_groups.get(&group)
    }

    /// Returns the buffer descriptor for a `uniform!` or `storage!`
    /// binding by its declared name, if any.
    pub fn buffer(&self, name: &str) -> Option<&BufferDescriptorInfo> {
        self.buffers.iter().find(|b| b.binding_name == name)
    }

    /// Builds a `wgpu::ShaderModuleDescriptor` for this module using the
    /// given WGSL source string. The source is typically obtained from
    /// `module.wgsl_source()` or `ir::render_module(&instantiated)`.
    pub fn shader_module_descriptor<'a>(
        &'a self,
        source: &'a str,
    ) -> wgpu::ShaderModuleDescriptor<'a> {
        wgpu::ShaderModuleDescriptor {
            label: Some(&self.module_label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        }
    }

    /// Creates a `wgpu::ShaderModule` from the given WGSL source string.
    pub fn shader_module(&self, device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
        device.create_shader_module(self.shader_module_descriptor(source))
    }

    /// Returns the first vertex entry matching `name`, or `None`.
    pub fn vertex_entry(&self, name: &str) -> Option<&EntryPointInfo> {
        self.vertex_entries.iter().find(|e| e.name == name)
    }

    /// Returns the first fragment entry matching `name`, or `None`.
    pub fn fragment_entry(&self, name: &str) -> Option<&EntryPointInfo> {
        self.fragment_entries.iter().find(|e| e.name == name)
    }

    /// Returns the first compute entry matching `name`, or `None`.
    pub fn compute_entry(&self, name: &str) -> Option<&ComputeEntryInfo> {
        self.compute_entries.iter().find(|e| e.name == name)
    }
}

// ===== Bind groups =====

/// A single bind group, including its layout entries and metadata for
/// resource construction.
#[derive(Clone, Debug)]
pub struct BindGroupInfo {
    /// The `@group(N)` index.
    pub group: u32,
    /// The bind group layout entries, sorted by binding number.
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// Per-binding metadata: the declared name, the binding number, and
    /// the kind of resource the binding expects.
    pub bindings: Vec<BindingMeta>,
    /// A default label for descriptors (`"<module>::bind_group_<N>"`).
    pub label: String,
}

/// Metadata for a single binding inside a bind group.
#[derive(Clone, Debug)]
pub struct BindingMeta {
    /// The declared Rust name (e.g. `"FRAME"`, `"DIFFUSE_TEX"`).
    pub name: String,
    /// The `@binding(N)` index.
    pub binding: u32,
    /// What kind of resource the binding expects.
    pub kind: BindingKind,
}

/// The kind of resource a binding expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// A uniform buffer.
    Uniform,
    /// A storage buffer. `read_only` mirrors the WGSL access mode.
    Storage { read_only: bool },
    /// A (filtering) sampler.
    Sampler { comparison: bool },
    /// A sampled texture.
    Texture,
    /// A depth texture.
    DepthTexture,
}

impl BindGroupInfo {
    /// Builds a `wgpu::BindGroupLayoutDescriptor` borrowing this struct.
    /// The returned descriptor borrows from `self`, so `self` must
    /// outlive the descriptor and any wgpu object built from it.
    pub fn layout_descriptor<'a>(
        &'a self,
        extra_label: Option<&'a str>,
    ) -> wgpu::BindGroupLayoutDescriptor<'a> {
        // If a suffix is provided we leak a static string for the label
        // so the descriptor's label type stays `'a` (matching the
        // borrow on `self.entries`). For the common no-suffix case we
        // borrow `self.label` directly.
        match extra_label {
            Some(suffix) => {
                let owned: &'static str = leak_str(&format!("{}::{}", self.label, suffix));
                wgpu::BindGroupLayoutDescriptor {
                    label: Some(owned),
                    entries: &self.entries,
                }
            }
            None => wgpu::BindGroupLayoutDescriptor {
                label: Some(&self.label),
                entries: &self.entries,
            },
        }
    }

    /// Creates the bind group layout on the given device.
    pub fn layout(
        &self,
        device: &wgpu::Device,
        extra_label: Option<&str>,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&self.layout_descriptor(extra_label))
    }

    /// Creates a bind group with one entry per binding in declaration
    /// order. The caller must pass exactly one `BindingResource` per
    /// entry in `self.entries`, in the same order.
    pub fn create<'a>(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        resources: &[wgpu::BindingResource<'a>],
    ) -> wgpu::BindGroup {
        assert_eq!(
            resources.len(),
            self.entries.len(),
            "{}: expected {} binding resource(s), got {}",
            self.label,
            self.entries.len(),
            resources.len(),
        );
        let entries: Vec<wgpu::BindGroupEntry> = self
            .entries
            .iter()
            .zip(resources.iter())
            .map(|(layout_entry, resource)| wgpu::BindGroupEntry {
                binding: layout_entry.binding,
                resource: resource.clone(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.label),
            layout,
            entries: &entries,
        })
    }
}

// ===== Entry points =====

/// A vertex or fragment shader entry point.
#[derive(Clone, Debug)]
pub struct EntryPointInfo {
    /// The entry point name.
    pub name: String,
}

impl EntryPointInfo {
    /// Builds a `wgpu::VertexState` borrowing this entry's name.
    pub fn vertex_state<'a>(&'a self, module: &'a wgpu::ShaderModule) -> wgpu::VertexState<'a> {
        wgpu::VertexState {
            module,
            entry_point: Some(&self.name),
            buffers: &[],
            compilation_options: Default::default(),
        }
    }

    /// Builds a `wgpu::FragmentState` borrowing this entry's name.
    pub fn fragment_state<'a>(
        &'a self,
        module: &'a wgpu::ShaderModule,
        targets: &'a [Option<wgpu::ColorTargetState>],
    ) -> wgpu::FragmentState<'a> {
        wgpu::FragmentState {
            module,
            entry_point: Some(&self.name),
            targets,
            compilation_options: Default::default(),
        }
    }
}

/// A compute shader entry point.
#[derive(Clone, Debug)]
pub struct ComputeEntryInfo {
    /// The entry point name.
    pub name: String,
    /// The `@workgroup_size(X, Y, Z)` dimensions.
    pub workgroup_size: (u32, u32, u32),
    /// A default label for descriptors (`"<module>::<name>"`).
    pub label: String,
}

impl ComputeEntryInfo {
    /// Builds a `wgpu::ComputePipelineDescriptor` borrowing this entry.
    pub fn compute_pipeline_descriptor<'a>(
        &'a self,
        layout: Option<&'a wgpu::PipelineLayout>,
        module: &'a wgpu::ShaderModule,
    ) -> wgpu::ComputePipelineDescriptor<'a> {
        wgpu::ComputePipelineDescriptor {
            label: Some(&self.label),
            layout,
            module,
            entry_point: Some(&self.name),
            compilation_options: Default::default(),
            cache: None,
        }
    }

    /// Creates a compute pipeline on the given device.
    pub fn compute_pipeline(
        &self,
        device: &wgpu::Device,
        layout: Option<&wgpu::PipelineLayout>,
        module: &wgpu::ShaderModule,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&self.compute_pipeline_descriptor(layout, module))
    }
}

// ===== Buffer descriptors =====

/// A pre-built `wgpu::BufferDescriptor` for a `uniform!` or `storage!`
/// declaration. Sizing follows WGSL §14.4.1 ("Alignment and Size").
///
/// `size` is `0` for runtime-sized storage buffers, which signals the
/// caller to choose a buffer size appropriate for the workload (the GPU
/// side stores a runtime array whose length is determined by the buffer
/// size).
#[derive(Clone, Debug)]
pub struct BufferDescriptorInfo {
    /// The declared Rust name of the linkage variable.
    pub binding_name: String,
    /// The `@group(N)` index.
    pub group: u32,
    /// The `@binding(N)` index.
    pub binding: u32,
    /// What kind of buffer this is.
    pub kind: BufferKind,
    /// The byte size, computed per WGSL §14.4.1. `0` for runtime arrays.
    pub size: u64,
    /// The pre-built descriptor, with `usage` already set. The label is
    /// a leaked `&'static str` of the binding name (small, one-time
    /// allocation per binding; matches the original compile-time
    /// `pub const X_BUFFER_DESCRIPTOR` shape).
    pub descriptor: wgpu::BufferDescriptor<'static>,
}

/// What kind of GPU buffer a [`BufferDescriptorInfo`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// A uniform buffer (`var<uniform>`).
    Uniform,
    /// A storage buffer (`var<storage, read[_write]>`).
    Storage { read_only: bool },
}

impl BufferDescriptorInfo {
    /// Creates an empty buffer on the given device using the
    /// pre-computed descriptor. The caller is responsible for populating
    /// the buffer (typically via `queue.write_buffer`).
    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&self.descriptor)
    }
}

// ===== Analysis entry points =====

/// Analyzes an IR module and returns its wgpu linkage.
///
/// This is the core entry point. It expects a concrete IR module (no
/// `Type::TypeParam` nodes); for template modules, call
/// [`Module::instantiate`] first and then pass the result here.
pub fn analyze_module(ir_module: &ir::Module) -> WgpuLinkage {
    let mut linkage = WgpuLinkage {
        module_label: ir_module.name.clone(),
        bind_groups: BTreeMap::new(),
        vertex_entries: Vec::new(),
        fragment_entries: Vec::new(),
        compute_entries: Vec::new(),
        buffers: Vec::new(),
    };

    for item in &ir_module.items {
        match item {
            ir::Item::Uniform(u) => {
                let size = type_byte_size_or_zero(&u.ty, ir_module);
                let entry = wgpu::BindGroupLayoutEntry {
                    binding: u.binding,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };
                linkage
                    .bind_groups
                    .entry(u.group)
                    .or_insert_with(|| BindGroupInfo {
                        group: u.group,
                        entries: Vec::new(),
                        bindings: Vec::new(),
                        label: format!("{}::bind_group_{}", ir_module.name, u.group),
                    })
                    .entries
                    .push(entry);
                linkage
                    .bind_groups
                    .get_mut(&u.group)
                    .unwrap()
                    .bindings
                    .push(BindingMeta {
                        name: u.name.clone(),
                        binding: u.binding,
                        kind: BindingKind::Uniform,
                    });
                linkage.buffers.push(BufferDescriptorInfo {
                    binding_name: u.name.clone(),
                    group: u.group,
                    binding: u.binding,
                    kind: BufferKind::Uniform,
                    size,
                    descriptor: wgpu::BufferDescriptor {
                        label: Some(leak_str(&u.name)),
                        size,
                        usage: wgpu::BufferUsages::UNIFORM
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                });
            }
            ir::Item::Storage(s) => {
                let read_only = matches!(s.access, ir::StorageAccess::Read);
                let size = type_byte_size_or_zero(&s.ty, ir_module);
                let entry = wgpu::BindGroupLayoutEntry {
                    binding: s.binding,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };
                linkage
                    .bind_groups
                    .entry(s.group)
                    .or_insert_with(|| BindGroupInfo {
                        group: s.group,
                        entries: Vec::new(),
                        bindings: Vec::new(),
                        label: format!("{}::bind_group_{}", ir_module.name, s.group),
                    })
                    .entries
                    .push(entry);
                linkage
                    .bind_groups
                    .get_mut(&s.group)
                    .unwrap()
                    .bindings
                    .push(BindingMeta {
                        name: s.name.clone(),
                        binding: s.binding,
                        kind: BindingKind::Storage { read_only },
                    });
                linkage.buffers.push(BufferDescriptorInfo {
                    binding_name: s.name.clone(),
                    group: s.group,
                    binding: s.binding,
                    kind: BufferKind::Storage { read_only },
                    size,
                    descriptor: wgpu::BufferDescriptor {
                        label: Some(leak_str(&s.name)),
                        size,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                });
            }
            ir::Item::Sampler(s) => {
                let comparison = matches!(s.ty, ir::Type::SamplerComparison);
                let sampler_ty = if comparison {
                    wgpu::SamplerBindingType::Comparison
                } else {
                    wgpu::SamplerBindingType::Filtering
                };
                let entry = wgpu::BindGroupLayoutEntry {
                    binding: s.binding,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Sampler(sampler_ty),
                    count: None,
                };
                linkage
                    .bind_groups
                    .entry(s.group)
                    .or_insert_with(|| BindGroupInfo {
                        group: s.group,
                        entries: Vec::new(),
                        bindings: Vec::new(),
                        label: format!("{}::bind_group_{}", ir_module.name, s.group),
                    })
                    .entries
                    .push(entry);
                linkage
                    .bind_groups
                    .get_mut(&s.group)
                    .unwrap()
                    .bindings
                    .push(BindingMeta {
                        name: s.name.clone(),
                        binding: s.binding,
                        kind: BindingKind::Sampler { comparison },
                    });
            }
            ir::Item::Texture(t) => {
                let (entry, kind) = texture_layout_entry(t);
                linkage
                    .bind_groups
                    .entry(t.group)
                    .or_insert_with(|| BindGroupInfo {
                        group: t.group,
                        entries: Vec::new(),
                        bindings: Vec::new(),
                        label: format!("{}::bind_group_{}", ir_module.name, t.group),
                    })
                    .entries
                    .push(entry);
                linkage
                    .bind_groups
                    .get_mut(&t.group)
                    .unwrap()
                    .bindings
                    .push(BindingMeta {
                        name: t.name.clone(),
                        binding: t.binding,
                        kind,
                    });
            }
            ir::Item::Fn(f) => match &f.fn_attrs {
                ir::FnAttrs::Vertex => {
                    linkage.vertex_entries.push(EntryPointInfo {
                        name: f.name.clone(),
                    });
                }
                ir::FnAttrs::Fragment => {
                    linkage.fragment_entries.push(EntryPointInfo {
                        name: f.name.clone(),
                    });
                }
                ir::FnAttrs::Compute { workgroup_size } => {
                    let x = workgroup_size.x;
                    let y = workgroup_size.y.unwrap_or(1);
                    let z = workgroup_size.z.unwrap_or(1);
                    linkage.compute_entries.push(ComputeEntryInfo {
                        name: f.name.clone(),
                        workgroup_size: (x, y, z),
                        label: format!("{}::{}", ir_module.name, f.name),
                    });
                }
                ir::FnAttrs::None => {}
            },
            // Structs, impls, enums, consts, workgroup vars don't directly
            // produce wgpu objects.
            ir::Item::Struct(_)
            | ir::Item::Impl(_)
            | ir::Item::Enum(_)
            | ir::Item::Const(_)
            | ir::Item::Workgroup(_) => {}
        }
    }

    // Sort each bind group's entries by binding number for determinism
    // (matches the compile-time codegen behavior).
    for bg in linkage.bind_groups.values_mut() {
        let mut paired: Vec<_> = bg.entries.drain(..).zip(bg.bindings.drain(..)).collect();
        paired.sort_by_key(|(e, _m)| e.binding);
        bg.entries = paired.iter().map(|(e, _)| *e).collect();
        bg.bindings = paired.into_iter().map(|(_, m)| m).collect();
    }

    linkage
}

/// Analyzes a `wgsl_rs::Module` by first assembling its full IR
/// (including imports and template instantiations), then delegating to
/// [`analyze_module`].
///
/// Panics if the module is a template (has unresolved
/// `Type::TypeParam`s). For templates, call [`Module::instantiate`]
/// first and pass the resulting `ir::Module` to `analyze_module`.
pub fn analyze_wgsl_module(wgsl_module: &Module) -> WgpuLinkage {
    assert!(
        !wgsl_module.is_template(),
        "analyze_wgsl_module called on a template module ('{}'); call Module::instantiate first \
         and pass the result to analyze_module",
        wgsl_module.name,
    );
    let ir_module = assemble_ir(wgsl_module);
    analyze_module(&ir_module)
}

// ===== IR assembly (flattens imports) =====

/// Build a single owned `ir::Module` from a `wgsl_rs::Module` and all
/// its transitive imports and template instantiations. The result is
/// concrete (no `Type::TypeParam`s) and self-contained: every struct
/// referenced by a binding is defined within the assembled module.
///
/// This is the same traversal that `Module::wgsl_source()` performs,
/// but it stops at IR rather than rendering. It exists so the runtime
/// linkage analyzer can correctly size buffers whose types are defined
/// in imported modules.
pub(crate) fn assemble_ir(wgsl_module: &Module) -> ir::Module {
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen: std::collections::HashSet<(u64, String, Vec<String>)> =
        std::collections::HashSet::new();
    let mut items: Vec<ir::Item> = Vec::new();
    collect_items(wgsl_module, &mut visited, &mut seen, &mut items, None);
    ir::Module {
        name: wgsl_module.name.to_string(),
        items,
        attrs: Vec::new(),
    }
}

fn collect_items(
    wgsl_module: &Module,
    visited: &mut std::collections::HashSet<u64>,
    seen: &mut std::collections::HashSet<(u64, String, Vec<String>)>,
    out: &mut Vec<ir::Item>,
    subst: Option<&std::collections::HashMap<String, ir::Type>>,
) {
    // Imports first (depth-first, deduplicated by module id).
    for m in wgsl_module.imports {
        if visited.insert(m.id) {
            collect_items(m, visited, seen, out, None);
        }
    }

    // This module's own items, optionally substituted.
    let mut ir_module = (wgsl_module.ir_constructor)();
    if let Some(s) = subst {
        ir::substitute_types(&mut ir_module, s);
    }
    out.extend(ir_module.items);

    // Cross-module template instantiations.
    for inst in wgsl_module.instantiations {
        let mangled: Vec<String> = inst
            .mangled_type_args
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let type_args = (inst.type_args_constructor)();
        collect_instantiation(
            inst.modules,
            inst.template_name,
            &mangled,
            &type_args,
            out,
            seen,
        );
    }
}

fn collect_instantiation(
    modules: &[&Module],
    template_name: &str,
    mangled_type_args: &[String],
    type_args: &[ir::Type],
    out: &mut Vec<ir::Item>,
    seen: &mut std::collections::HashSet<(u64, String, Vec<String>)>,
) {
    let available: Vec<String> = modules
        .iter()
        .copied()
        .flat_map(|m| m.templates.iter().map(|t| t.name.to_string()))
        .collect();

    let mut matching: Vec<&Module> = modules
        .iter()
        .copied()
        .filter(|m| m.templates.iter().any(|t| t.name == template_name))
        .collect();

    if matching.is_empty() {
        panic!(
            "unable to resolve template '{template_name}' for type args {:?}; available \
             templates: {:?}",
            mangled_type_args, available
        );
    }
    if matching.len() > 1 {
        let names: Vec<&str> = matching.iter().map(|m| m.name).collect();
        panic!(
            "ambiguous template instantiation '{template_name}' for type args {:?}; matching \
             modules: {:?}; available templates: {:?}",
            mangled_type_args, names, available
        );
    }

    let module = matching
        .pop()
        .expect("matching is non-empty after the check above");
    let Some(template) = module.templates.iter().find(|t| t.name == template_name) else {
        panic!(
            "internal error: resolved module '{}' has no template '{}'; available: {:?}",
            module.name, template_name, available
        );
    };

    let key = (
        module.id,
        template_name.to_string(),
        mangled_type_args.to_vec(),
    );
    if !seen.insert(key) {
        return;
    }

    // Recurse into dependencies first.
    for dep in template.dependencies {
        let dep_mangled: Vec<String> = dep
            .type_param_mapping
            .iter()
            .map(|&idx| mangled_type_args[idx].clone())
            .collect();
        let dep_args: Vec<ir::Type> = dep
            .type_param_mapping
            .iter()
            .map(|&idx| type_args[idx].clone())
            .collect();
        collect_instantiation(&[module], dep.callee, &dep_mangled, &dep_args, out, seen);
    }

    let mut subst: std::collections::HashMap<String, ir::Type> = std::collections::HashMap::new();
    for (param, arg) in template.type_params.iter().zip(type_args.iter()) {
        subst.insert((*param).to_string(), arg.clone());
    }

    let mut items = (template.ir_constructor)();
    ir::substitute_items(&mut items, &subst);

    let instance_name = if mangled_type_args.is_empty() {
        template.name.to_string()
    } else {
        let mut components: Vec<&str> = Vec::with_capacity(1 + mangled_type_args.len());
        components.push(template.name);
        for s in mangled_type_args {
            components.push(s.as_str());
        }
        ir::mangle(&components)
    };
    if instance_name != template.name {
        ir::rename_items(&mut items, template.name, &instance_name);
    }
    out.extend(items);
}

// ===== Texture helpers =====

fn texture_layout_entry(item: &ir::ItemTexture) -> (wgpu::BindGroupLayoutEntry, BindingKind) {
    let view_dimension = match &item.ty {
        ir::Type::Texture { kind, .. } => texture_kind_view_dimension(*kind),
        ir::Type::TextureDepth { kind } => texture_depth_view_dimension(*kind),
        _ => unreachable!("ItemTexture validated to be a texture type"),
    };
    let multisampled = match &item.ty {
        ir::Type::Texture { kind, .. } => matches!(kind, ir::TextureKind::TextureMultisampled2D),
        ir::Type::TextureDepth { kind } => {
            matches!(kind, ir::TextureDepthKind::DepthMultisampled2D)
        }
        _ => unreachable!(),
    };
    match &item.ty {
        ir::Type::Texture { sampled_type, .. } => {
            let sample = match sampled_type {
                ir::ScalarType::F32 => wgpu::TextureSampleType::Float { filterable: true },
                ir::ScalarType::I32 => wgpu::TextureSampleType::Sint,
                ir::ScalarType::U32 => wgpu::TextureSampleType::Uint,
                ir::ScalarType::Bool => unreachable!("textures can't be bool-sampled"),
            };
            (
                wgpu::BindGroupLayoutEntry {
                    binding: item.binding,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Texture {
                        sample_type: sample,
                        view_dimension,
                        multisampled,
                    },
                    count: None,
                },
                BindingKind::Texture,
            )
        }
        ir::Type::TextureDepth { .. } => (
            wgpu::BindGroupLayoutEntry {
                binding: item.binding,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension,
                    multisampled,
                },
                count: None,
            },
            BindingKind::DepthTexture,
        ),
        _ => unreachable!(),
    }
}

fn texture_kind_view_dimension(kind: ir::TextureKind) -> wgpu::TextureViewDimension {
    match kind {
        ir::TextureKind::Texture1D => wgpu::TextureViewDimension::D1,
        ir::TextureKind::Texture2D | ir::TextureKind::TextureMultisampled2D => {
            wgpu::TextureViewDimension::D2
        }
        ir::TextureKind::Texture2DArray => wgpu::TextureViewDimension::D2Array,
        ir::TextureKind::Texture3D => wgpu::TextureViewDimension::D3,
        ir::TextureKind::TextureCube => wgpu::TextureViewDimension::Cube,
        ir::TextureKind::TextureCubeArray => wgpu::TextureViewDimension::CubeArray,
    }
}

fn texture_depth_view_dimension(kind: ir::TextureDepthKind) -> wgpu::TextureViewDimension {
    match kind {
        ir::TextureDepthKind::Depth2D | ir::TextureDepthKind::DepthMultisampled2D => {
            wgpu::TextureViewDimension::D2
        }
        ir::TextureDepthKind::Depth2DArray => wgpu::TextureViewDimension::D2Array,
        ir::TextureDepthKind::DepthCube => wgpu::TextureViewDimension::Cube,
        ir::TextureDepthKind::DepthCubeArray => wgpu::TextureViewDimension::CubeArray,
    }
}

// ===== WGSL §14.4.1 type sizing =====

/// Returns the byte size of a host-shareable WGSL type per §14.4.1.
/// `RuntimeArray<T>` returns `0` (caller chooses the buffer size based
/// on workload). User struct lookups require the defining module.
///
/// Returns `0` for non-shareable types (samplers, textures, pointers) and
/// for unresolvable cases (e.g. an array length that isn't a literal or
/// same-module const). In the latter case a panic with location info
/// would be more user-friendly, but we err on the side of "do something
/// reasonable" so that callers can still build pipelines.
pub fn type_byte_size(ty: &ir::Type, module: &ir::Module) -> u64 {
    let layout = type_layout(ty, module);
    layout.size
}

fn type_byte_size_or_zero(ty: &ir::Type, module: &ir::Module) -> u64 {
    type_byte_size(ty, module)
}

/// Layout for a single type (size + align) per WGSL §14.4.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeLayout {
    pub size: u64,
    pub align: u64,
}

fn align_up(val: u64, align: u64) -> u64 {
    if align == 0 {
        val
    } else {
        (val + (align - 1)) & !(align - 1)
    }
}

fn type_layout(ty: &ir::Type, module: &ir::Module) -> TypeLayout {
    match ty {
        ir::Type::Scalar(s) => scalar_layout(*s),
        ir::Type::Vector { elements, .. } => vector_layout(*elements),
        ir::Type::Matrix { columns, rows, .. } => matrix_layout(*columns, *rows),
        ir::Type::Array { elem, len } => {
            let inner = type_layout(elem, module);
            let n = eval_array_len(len, module).unwrap_or(0);
            TypeLayout {
                size: (n as u64) * align_up(inner.size, inner.align.max(1)),
                align: inner.align,
            }
        }
        ir::Type::RuntimeArray { elem } => {
            let inner = type_layout(elem, module);
            TypeLayout {
                size: 0,
                align: inner.align,
            }
        }
        ir::Type::Atomic { elem } => type_layout(elem, module),
        ir::Type::Struct { name, type_args } => struct_layout(name, type_args, module),
        ir::Type::Ptr { elem, .. } => type_layout(elem, module),
        // Samplers, textures, and type parameters aren't host-shareable.
        // Return a zero layout so the analyzer doesn't choke.
        ir::Type::Sampler
        | ir::Type::SamplerComparison
        | ir::Type::Texture { .. }
        | ir::Type::TextureDepth { .. }
        | ir::Type::TypeParam { .. } => TypeLayout { size: 0, align: 1 },
    }
}

fn scalar_layout(s: ir::ScalarType) -> TypeLayout {
    match s {
        ir::ScalarType::F32 | ir::ScalarType::I32 | ir::ScalarType::U32 | ir::ScalarType::Bool => {
            TypeLayout { size: 4, align: 4 }
        }
    }
}

fn vector_layout(elements: u8) -> TypeLayout {
    match elements {
        2 => TypeLayout { size: 8, align: 8 },
        3 => TypeLayout {
            size: 12,
            align: 16,
        },
        4 => TypeLayout {
            size: 16,
            align: 16,
        },
        _ => TypeLayout { size: 0, align: 1 },
    }
}

fn matrix_layout(columns: u8, rows: u8) -> TypeLayout {
    // WGSL §14.4.1: matCxR<T> has:
    //   Align = AlignOf(vecR<T>)
    //   Size  = SizeOf(array<vecR<T>, C>)
    // Each column is stored as a vecR<T> with stride roundUp(SizeOf(vecR<T>),
    // AlignOf(vecR<T>)).
    let row_vec = vector_layout(rows);
    let stride = align_up(row_vec.size, row_vec.align);
    TypeLayout {
        size: (columns as u64) * stride,
        align: row_vec.align,
    }
}

fn struct_layout(name: &str, type_args: &[ir::Type], module: &ir::Module) -> TypeLayout {
    let Some(s) = find_struct(name, type_args, module) else {
        return TypeLayout { size: 0, align: 1 };
    };
    let mut offset: u64 = 0;
    let mut struct_align: u64 = 1;
    for field in &s.fields {
        let fl = type_layout(&field.ty, module);
        struct_align = struct_align.max(fl.align);
        offset = align_up(offset, fl.align);
        offset += fl.size;
    }
    let size = align_up(offset, struct_align);
    TypeLayout {
        size,
        align: struct_align,
    }
}

/// Look up a struct by name (and optional type args) in the module. The
/// struct is matched by base name; type args are currently ignored for
/// lookup (we look up the generic template, then resolve via the
/// `type_args` substitution if any). Generic structs share a base name
/// across instantiations, so callers that need a specific instantiation
/// must pre-substitute the IR before calling.
fn find_struct<'a>(
    name: &str,
    _type_args: &[ir::Type],
    module: &'a ir::Module,
) -> Option<&'a ir::ItemStruct> {
    module.items.iter().find_map(|i| match i {
        ir::Item::Struct(s) if s.name == name => Some(s),
        _ => None,
    })
}

/// Evaluate a fixed-size array length expression. Supports integer
/// literals and named integer constants defined at module scope. Other
/// expressions (binary ops, casts, etc.) return `None` and the caller
/// falls back to size 0 with a clearly-too-small buffer.
fn eval_array_len(expr: &ir::Expr, module: &ir::Module) -> Option<usize> {
    match expr {
        ir::Expr::Lit(ir::Lit::Int { digits, .. }) => digits.parse::<usize>().ok(),
        ir::Expr::Unary {
            op: ir::UnOp::Neg,
            expr,
        } => eval_array_len(expr, module).map(|n| n.wrapping_neg()),
        ir::Expr::Ident(name) => {
            let val = module.items.iter().find_map(|i| match i {
                ir::Item::Const(c) if c.name == *name => Some(eval_const_int(&c.expr, module)),
                _ => None,
            });
            val.flatten()
        }
        _ => None,
    }
}

fn eval_const_int(expr: &ir::Expr, module: &ir::Module) -> Option<usize> {
    eval_array_len(expr, module)
}

/// Convert a `&str` to a `&'static str` by leaking the allocation.
/// Used to produce `'static` labels for `wgpu::BufferDescriptor` and
/// other descriptors whose `L` parameter defaults to `&'static str`.
///
/// This is a small one-time cost per binding analyzed. Bindings are
/// typically few, the strings are short (often just the binding name),
/// and the leaked memory is constant for the lifetime of the program.
fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}
