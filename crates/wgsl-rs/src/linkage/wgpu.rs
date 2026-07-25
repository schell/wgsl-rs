//! Runtime wgpu linkage analysis for WGSL modules.
//!
//! Given an [`ir::Module`] (produced by a `#[wgsl]` source's
//! `WGSL_SOURCE.ir_constructor`, or by `Source::instantiate` for
//! templates), this module extracts the binding and entry-point
//! information needed to build wgpu pipelines.
//!
//! # Why at runtime?
//!
//! The pre-[#120](https://github.com/schell/wgsl-rs/issues/120)
//! implementation generated all of this at proc-macro expansion time by walking
//! the parse tree in `wgsl-rs-macros/src/linkage.rs`. That had two drawbacks:
//!
//! 1. Template modules (`#[wgsl]` modules with type parameters) couldn't get
//!    any linkage — the generated WGSL was a template with `__TP{name}__`
//!    placeholders, so a `wgpu::ShaderModule` couldn't be built from it.
//! 2. The proc-macro walked a parse tree that duplicated information already
//!    present in `wgsl-rs-ir`.
//!
//! Walking the runtime IR unifies the two cases. After
//! `Source::instantiate::<...>()` produces a concrete `ir::Module`, the
//! same analyzer used for non-template sources works on the result.
//!
//! # Example
//!
//! ```ignore
//! use wgsl_rs::linkage::wgpu;
//!
//! let module = hello_triangle::WGSL_SOURCE;
//! let linkage = wgpu::analyze_wgsl_module(&module).unwrap();
//! let shader_module = linkage.shader_module(&device);
//! let bg_layout = linkage.bind_group(0).unwrap().layout(&device);
//! let bg = linkage.bind_group(0).unwrap().create(
//!     &device, &bg_layout, &[frame_uniform.as_entire_binding()],
//! );
//! let vtx = linkage.vertex_entries.iter().find(|e| e.name == "vtx_main").unwrap()
//!     .vertex_state(&shader_module);
//! ```

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
};

use snafu::prelude::*;
use wgsl_rs_ir as ir;

use crate::Source;

/// Errors thrown by the WGPU linkage module.
#[derive(Snafu, Debug)]
pub enum Error {
    #[snafu(display(
        "Unable to resolve template '{template_name}' for type args {:?}; available templates: \
         {:?}",
        mangled_type_args,
        available
    ))]
    TemplateResolution {
        template_name: String,
        mangled_type_args: Vec<String>,
        available: Vec<String>,
    },

    #[snafu(display(
        "{module}::bind_group_{group}: missing binding resource for declared name {name:?}"
    ))]
    MissingBinding {
        module: &'static str,
        group: u32,
        name: String,
    },

    #[snafu(display(
        "{module}::bind_group_{group}: resource provided for unknown binding name {name:?} \
         (declared names: {declared:?})"
    ))]
    UnknownBindingName {
        module: &'static str,
        group: u32,
        name: String,
        declared: Vec<String>,
    },

    #[snafu(display(
        "{module}::bind_group_{group}: binding name {name:?} appears more than once in resources"
    ))]
    DuplicateBindingName {
        module: &'static str,
        group: u32,
        name: String,
    },

    #[snafu(display(
        "{module}: no bind group @group({group}) declared; available groups: {available:?}"
    ))]
    NoSuchBindGroup {
        module: &'static str,
        group: u32,
        available: Vec<u32>,
    },
}

impl From<crate::SourceError<'_>> for Error {
    fn from(err: crate::SourceError<'_>) -> Self {
        match err {
            crate::SourceError::TemplateWgsl {
                uninstantiated_source,
            } => Error::TemplateResolution {
                template_name: format!("<source {}>", uninstantiated_source.name),
                mangled_type_args: uninstantiated_source
                    .module_type_params
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                available: Vec::new(),
            },
            crate::SourceError::TemplateNotFound {
                template_name,
                mangled_type_args,
                available_templates,
            } => Error::TemplateResolution {
                template_name,
                mangled_type_args,
                available: available_templates,
            },
            crate::SourceError::AmbiguousTemplate {
                template_name,
                mangled_type_args,
                matching_source_names,
                available_templates: _,
            } => Error::TemplateResolution {
                template_name,
                mangled_type_args,
                available: matching_source_names,
            },
        }
    }
}

/// All the wgpu linkage information extracted from a WGSL module.
///
/// Produced by [`analyze_ir_module`] or [`analyze_wgsl_module`]. Owns the
/// string labels that the various `wgpu::*Descriptor` types borrow, so
/// hold this struct for the lifetime of any borrowed descriptor.
#[derive(Clone, Debug)]
pub struct WgpuLinkage {
    /// The module's name (used as a default label for descriptors).
    pub module_label: &'static str,

    /// The concrete, fully-assembled IR the linkage was built from. This
    /// is the IR rendered by [`Self::wgsl_source`] to produce the shader
    /// source's WGSL text. For sources built from [`crate::Source`] it
    /// is the result of flattening imports and template instantiations;
    /// for modules built from [`ir::Module`] (e.g. an instantiated
    /// template) it is that module directly. Always concrete: no
    /// `Type::TypeParam` nodes.
    ir: ir::Module,

    /// Bind groups keyed by `@group(N)` index. Each group is sorted by
    /// binding number.
    pub bind_groups: BTreeMap<u32, BindGroupInfo>,

    /// Lazily-built `wgpu::BindGroupLayout` per `@group(N)` index. Empty
    /// until a cache-populating method is called. Populating one does
    /// not populate the others; calling
    /// [`WgpuLinkage::pipeline_layout`] populates all groups at once.
    bind_group_layouts: HashMap<u32, wgpu::BindGroupLayout>,

    /// Lazily-built `wgpu::PipelineLayout` referencing all bind group
    /// layouts. `None` until [`WgpuLinkage::pipeline_layout`] is called.
    pipeline_layout: Option<wgpu::PipelineLayout>,

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

    /// Returns the buffer descriptor for a `uniform!` or `storage!`
    /// binding by its declared name, if any - removing it from the linkage.
    ///
    /// Use this to ensure that you provide data for all buffers in your
    /// program.
    pub fn take_buffer(&mut self, name: &str) -> Option<BufferDescriptorInfo> {
        let mut info = None;
        self.buffers.retain(|bi| {
            if bi.binding_name == name {
                info = Some(bi.clone());
                false
            } else {
                true
            }
        });
        info
    }

    /// Returns the WGSL source for this linkage's source, rendered from
    /// the concrete IR the linkage was built from.
    ///
    /// The linkage's IR is concrete (no `Type::TypeParam`s), so the
    /// rendered text is free of `__TP{name}__` placeholders. For a
    /// linkage built from a template `Source` via
    /// [`analyze_ir_module`], this is the instantiated source.
    pub fn wgsl_source(&self) -> String {
        ir::render_module(&self.ir)
    }

    /// Builds a `wgpu::ShaderModuleDescriptor` for this module using
    /// the concrete IR the linkage was built from. The source string
    /// is owned by the descriptor, so no borrow of `self` is required
    /// to build a wgpu pipeline.
    pub fn shader_module_descriptor(&self) -> wgpu::ShaderModuleDescriptor<'static> {
        wgpu::ShaderModuleDescriptor {
            label: Some(self.module_label),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(self.wgsl_source())),
        }
    }

    /// Creates a `wgpu::ShaderModule` from the concrete IR the linkage
    /// was built from.
    pub fn shader_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(self.shader_module_descriptor())
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

    /// Returns the cached `wgpu::BindGroupLayout` for the given
    /// `@group(N)` index, building and caching it on first access.
    /// Returns `None` if the linkage has no such bind group.
    ///
    /// The returned layout is an owned clone of the cached value; the
    /// cache is not invalidated. Multiple calls are cheap after the
    /// first.
    pub fn bind_group_layout(
        &mut self,
        group: u32,
        device: &wgpu::Device,
    ) -> Option<wgpu::BindGroupLayout> {
        if let Some(layout) = self.bind_group_layouts.get(&group) {
            return Some(layout.clone());
        }
        let info = self.bind_groups.get(&group)?;
        let layout = info.layout(device);
        self.bind_group_layouts.insert(group, layout.clone());
        Some(layout)
    }

    /// Creates a bind group for `@group(N)` by matching each binding's
    /// declared name to a resource. The order of `resources` does not
    /// have to match the binding declaration order; the produced
    /// `BindGroupEntry` list is sorted by binding number, matching the
    /// cached layout.
    ///
    /// Looks up (and builds+caches on first access) the bind group
    /// layout for `group`, then delegates to
    /// [`BindGroupInfo::create_named`].
    ///
    /// # Errors
    ///
    /// - [`Error::NoSuchBindGroup`] if no bind group with the given index is
    ///   declared in this module.
    /// - [`Error::DuplicateBindingName`] if a name appears twice in
    ///   `resources`.
    /// - [`Error::MissingBinding`] if a declared binding has no resource in
    ///   `resources`.
    /// - [`Error::UnknownBindingName`] if `resources` contains a name that
    ///   isn't declared in this bind group.
    pub fn create_bind_group_named<'a>(
        &mut self,
        group: u32,
        device: &wgpu::Device,
        resources: &[(&str, wgpu::BindingResource<'a>)],
    ) -> Result<wgpu::BindGroup, Error> {
        // Build (or reuse cached) the layout for this group first.
        // The cache lookup is cheap and avoids a redundant layout
        // rebuild if the caller has already populated it.
        if !self.bind_group_layouts.contains_key(&group) {
            let Some(info) = self.bind_groups.get(&group) else {
                let mut available: Vec<u32> = self.bind_groups.keys().copied().collect();
                available.sort_unstable();
                return NoSuchBindGroupSnafu {
                    module: self.module_label,
                    group,
                    available,
                }
                .fail();
            };
            self.bind_group_layouts.insert(group, info.layout(device));
        }
        let info = self
            .bind_groups
            .get(&group)
            .expect("just inserted or found");
        let layout = self
            .bind_group_layouts
            .get(&group)
            .expect("just inserted or found");
        info.create_named(self.module_label, device, layout, resources)
    }

    /// Returns the cached `wgpu::PipelineLayout` for this module,
    /// building and caching it (and all bind group layouts it
    /// references) on first access. Subsequent calls return a cheap
    /// Arc-backed clone.
    ///
    /// For modules with no bind groups (e.g. depth-only pipelines), the
    /// cached layout references an empty `bind_group_layouts` slice.
    ///
    /// **Infallible.** The bind groups are already analyzed; this only
    /// drives `wgpu::Device` constructors.
    pub fn pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        label: Option<&str>,
    ) -> wgpu::PipelineLayout {
        if let Some(layout) = &self.pipeline_layout {
            return layout.clone();
        }
        // Build (and cache) every bind group layout we don't already have.
        for (&group, info) in &self.bind_groups {
            self.bind_group_layouts
                .entry(group)
                .or_insert_with(|| info.layout(device));
        }
        // `wgpu::PipelineLayoutDescriptor::bind_group_layouts` is indexed
        // by `@group(N)`: position N must hold group N's layout. Sort the
        // cached layouts by group index so the slice lines up. Gaps (e.g.
        // a module declaring only group 0 and group 2) are filled with
        // `None`, which wgpu accepts as "no bind group at this index".
        let max_group = self.bind_group_layouts.keys().copied().max().unwrap_or(0);
        let mut bg_layouts: Vec<Option<&wgpu::BindGroupLayout>> = (0..=max_group)
            .map(|g| self.bind_group_layouts.get(&g))
            .collect();
        // Trim trailing `None`s so we don't pass unnecessary slots.
        while bg_layouts.last() == Some(&None) {
            bg_layouts.pop();
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &bg_layouts,
            immediate_size: 0,
        });
        self.pipeline_layout = Some(layout.clone());
        layout
    }
}

// ===== ir::Module extension trait =====

/// Extension trait that adds `wgsl-rs` operations to [`wgsl_rs_ir::Module`].
///
/// [`wgsl_rs_ir::Module`] lives in a separate crate and so cannot have
/// inherent methods that reference `wgsl-rs` types. The methods that
/// need `WgpuLinkage` or naga validation live here instead. The trait
/// is re-exported from `wgsl_rs::*` so `m.generate_linkage()` works
/// after `use wgsl_rs;` without an explicit import.
pub trait IrModuleExt {
    /// Builds a [`WgpuLinkage`] from this concrete IR module.
    ///
    /// Equivalent to [`analyze_ir_module`](self).
    fn generate_linkage(&self) -> WgpuLinkage;

    /// Validates the WGSL source rendered from this IR module using naga.
    ///
    /// Equivalent to the free function [`crate::validate_wgsl_source`]
    /// called on the output of [`wgsl_rs_ir::Module::wgsl_source`].
    fn validate(&self) -> Result<(), String>;
}

impl IrModuleExt for wgsl_rs_ir::Module {
    fn generate_linkage(&self) -> WgpuLinkage {
        analyze_ir_module(self.clone())
    }

    fn validate(&self) -> Result<(), String> {
        crate::validate_wgsl_source(&self.wgsl_source())
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
    pub fn layout_descriptor<'a>(&'a self) -> wgpu::BindGroupLayoutDescriptor<'a> {
        wgpu::BindGroupLayoutDescriptor {
            label: Some(&self.label),
            entries: &self.entries,
        }
    }

    /// Creates the bind group layout on the given device.
    pub fn layout(&self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&self.layout_descriptor())
    }

    /// Creates a bind group with one entry per binding, in `self.entries`
    /// order — which is sorted by binding number (see
    /// [`BindGroupInfo::entries`]). The caller must pass exactly one
    /// `BindingResource` per entry, in that same (binding-number) order.
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

    /// Creates a bind group by matching each binding's declared name
    /// to a resource. The order of `resources` does not have to match
    /// the binding declaration order; the produced `BindGroupEntry`
    /// list is sorted by binding number, matching the layout.
    ///
    /// `module_label` is the linkage's `module_label`, used to build
    /// helpful error messages. It is not borrowed into the returned
    /// bind group.
    ///
    /// # Errors
    ///
    /// - [`Error::DuplicateBindingName`] if a name appears twice in
    ///   `resources`.
    /// - [`Error::MissingBinding`] if a declared name has no resource in
    ///   `resources`.
    /// - [`Error::UnknownBindingName`] if `resources` contains a name that
    ///   isn't declared in this bind group.
    pub fn create_named<'a>(
        &self,
        module_label: &'static str,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        resources: &[(&str, wgpu::BindingResource<'a>)],
    ) -> Result<wgpu::BindGroup, Error> {
        use std::collections::HashMap;

        // Build a name → resource map. Detect duplicates as we go.
        let mut by_name: HashMap<&str, wgpu::BindingResource<'a>> =
            HashMap::with_capacity(resources.len());
        for &(name, ref resource) in resources {
            if by_name.insert(name, resource.clone()).is_some() {
                return DuplicateBindingNameSnafu {
                    module: module_label,
                    group: self.group,
                    name: name.to_string(),
                }
                .fail();
            }
        }

        // Every declared binding must be present.
        for binding in &self.bindings {
            if !by_name.contains_key(binding.name.as_str()) {
                return MissingBindingSnafu {
                    module: module_label,
                    group: self.group,
                    name: binding.name.clone(),
                }
                .fail();
            }
        }

        // Every supplied name must be declared. We check this by
        // building the entries from the declared list (not from the
        // supplied slice) and tracking which supplied names we hit.
        let mut hit: HashMap<&str, ()> = HashMap::with_capacity(by_name.len());
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(self.bindings.len());
        for (layout_entry, binding) in self.entries.iter().zip(self.bindings.iter()) {
            let resource = by_name
                .get(binding.name.as_str())
                .expect("checked above")
                .clone();
            hit.insert(binding.name.as_str(), ());
            entries.push(wgpu::BindGroupEntry {
                binding: layout_entry.binding,
                resource,
            });
        }
        for name in by_name.keys() {
            if !hit.contains_key(name) {
                let declared: Vec<String> = self.bindings.iter().map(|b| b.name.clone()).collect();
                return UnknownBindingNameSnafu {
                    module: module_label,
                    group: self.group,
                    name: (*name).to_string(),
                    declared,
                }
                .fail();
            }
        }

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.label),
            layout,
            entries: &entries,
        }))
    }
}

// ===== Entry points =====

/// A vertex or fragment shader entry point.
#[derive(Clone, Debug)]
pub struct EntryPointInfo {
    /// The entry point name. For non-monomorphized functions this is a
    /// `Cow::Borrowed` of a `stringify!`-emitted `'static` literal;
    /// for monomorphized instances (e.g. `id` → `id_f32`) it is a
    /// `Cow::Owned` holding the runtime-computed name.
    pub name: Cow<'static, str>,
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
    /// The entry point name (see [`EntryPointInfo::name`] for the
    /// `Cow<'static, str>` rationale).
    pub name: Cow<'static, str>,
    /// The `@workgroup_size(X, Y, Z)` dimensions.
    pub workgroup_size: (u32, u32, u32),
    /// A default label for descriptors (`"<module>::<name>"`).
    pub label: String,
}

// `ComputeEntryInfo` currently exposes only public fields. Helper
// methods for building compute pipelines live in caller code (the
// `wgpu::ComputePipelineDescriptor` shape is simple enough to assemble
// at the call site, and the previous `compute_pipeline_descriptor` /
// `compute_pipeline` helpers had zero in-tree call sites).

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
    /// The pre-built descriptor's usage flags, ready to plug into a
    /// `wgpu::BufferDescriptor` along with the borrow of
    /// [`Self::binding_name`].
    pub usage: wgpu::BufferUsages,
}

impl BufferDescriptorInfo {
    /// Builds a `wgpu::BufferDescriptor` borrowing from
    /// [`Self::binding_name`]. The caller must keep this
    /// `BufferDescriptorInfo` alive for the duration of any wgpu call
    /// using the returned descriptor.
    pub fn descriptor(&self) -> wgpu::BufferDescriptor<'_> {
        wgpu::BufferDescriptor {
            label: Some(&self.binding_name),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        }
    }

    /// Creates an empty buffer on the given device. The caller is
    /// responsible for populating the buffer (typically via
    /// `queue.write_buffer`).
    pub fn create_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&self.descriptor())
    }
}

/// What kind of GPU buffer a [`BufferDescriptorInfo`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// A uniform buffer (`var<uniform>`).
    Uniform,
    /// A storage buffer (`var<storage, read[_write]>`).
    Storage { read_only: bool },
}

/// Analyzes an IR module and returns its wgpu linkage.
///
/// This is the core entry point. It expects a concrete IR module (no
/// `Type::TypeParam` nodes); for template modules, call
/// `Source::instantiate` first and then pass the result here.
///
/// The passed IR is consumed and stored on the returned linkage so that
/// [`WgpuLinkage::wgsl_source`] and [`WgpuLinkage::shader_module`] can
/// render the same source the linkage was analyzed from — even when
/// the linkage was built from an instantiated template.
///
/// **Infallible.** The input must be concrete (no `Type::TypeParam`
/// nodes); the proc-macro's `instantiate` enforces this at the type
/// level. The infallibility is honest: every failure mode would be a
/// bug in IR construction, not a user-facing error.
pub fn analyze_ir_module(ir_module: wgsl_rs_ir::Module) -> WgpuLinkage {
    let module_label = ir_module.name;
    let mut linkage = WgpuLinkage {
        module_label,
        ir: ir_module,
        bind_groups: BTreeMap::new(),
        bind_group_layouts: HashMap::new(),
        pipeline_layout: None,
        vertex_entries: Vec::new(),
        fragment_entries: Vec::new(),
        compute_entries: Vec::new(),
        buffers: Vec::new(),
    };

    // First pass: collect every binding declaration and every entry
    // point. We also walk each entry-point function body to record
    // which binding names it references so we can compute a per-binding
    // `ShaderStages` visibility in a second pass.
    let mut binding_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut referenced: std::collections::HashMap<String, wgpu::ShaderStages> =
        std::collections::HashMap::new();

    for item in &linkage.ir.items {
        match item {
            ir::Item::Uniform(u) => {
                binding_names.insert(u.name.clone());
            }
            ir::Item::Storage(s) => {
                binding_names.insert(s.name.clone());
            }
            ir::Item::Sampler(s) => {
                binding_names.insert(s.name.clone());
            }
            ir::Item::Texture(t) => {
                binding_names.insert(t.name.clone());
            }
            ir::Item::Fn(f) => {
                let stage = match &f.fn_attrs {
                    ir::FnAttrs::Vertex => wgpu::ShaderStages::VERTEX,
                    ir::FnAttrs::Fragment => wgpu::ShaderStages::FRAGMENT,
                    ir::FnAttrs::Compute { .. } => wgpu::ShaderStages::COMPUTE,
                    ir::FnAttrs::None => continue,
                };
                let mut idents = std::collections::HashSet::new();
                collect_idents_in_block(&mut idents, &f.block);
                for name in idents {
                    if binding_names.contains(&name) {
                        referenced
                            .entry(name)
                            .and_modify(|s| *s |= stage)
                            .or_insert(stage);
                    }
                }
            }
            _ => {}
        }
    }

    for item in &linkage.ir.items {
        match item {
            ir::Item::Uniform(u) => {
                let size = type_byte_size_or_zero(&u.ty, &linkage.ir);
                let entry = wgpu::BindGroupLayoutEntry {
                    binding: u.binding,
                    visibility: visibility_for(&referenced, &u.name),
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
                        label: format!("{}::bind_group_{}", linkage.module_label, u.group),
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
                    usage: wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                });
            }
            ir::Item::Storage(s) => {
                let read_only = matches!(s.access, ir::StorageAccess::Read);
                let size = type_byte_size_or_zero(&s.ty, &linkage.ir);
                let entry = wgpu::BindGroupLayoutEntry {
                    binding: s.binding,
                    visibility: visibility_for(&referenced, &s.name),
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
                        label: format!("{}::bind_group_{}", linkage.module_label, s.group),
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
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
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
                    visibility: visibility_for(&referenced, &s.name),
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
                        label: format!("{}::bind_group_{}", linkage.module_label, s.group),
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
                let (mut entry, kind) = texture_layout_entry(t);
                entry.visibility = visibility_for(&referenced, &t.name);
                linkage
                    .bind_groups
                    .entry(t.group)
                    .or_insert_with(|| BindGroupInfo {
                        group: t.group,
                        entries: Vec::new(),
                        bindings: Vec::new(),
                        label: format!("{}::bind_group_{}", linkage.module_label, t.group),
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
                        label: format!("{}::{}", linkage.module_label, f.name),
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

/// Returns the `ShaderStages` visibility for a binding given the
/// per-binding reference map. If no entry-point function references
/// the binding (e.g. the binding is only declared but unused, or the
/// reference walker missed it), we default to `COMPUTE` — the most
/// common stage for the wgsl-rs roundtrip-test use case. The previous
/// `ShaderStages::all()` default would silently demand the
/// `VERTEX_WRITABLE_STORAGE` feature for any read_write storage, which
/// is almost never what the user wants.
fn visibility_for(
    referenced: &std::collections::HashMap<String, wgpu::ShaderStages>,
    name: &str,
) -> wgpu::ShaderStages {
    referenced
        .get(name)
        .copied()
        .unwrap_or(wgpu::ShaderStages::COMPUTE)
}

/// Walks an `ir::Block`, collecting every `Expr::Ident` name into `out`.
fn collect_idents_in_block(out: &mut std::collections::HashSet<String>, block: &ir::Block) {
    for stmt in &block.stmts {
        collect_idents_in_stmt(out, stmt);
    }
}

fn collect_idents_in_stmt(out: &mut std::collections::HashSet<String>, stmt: &ir::Stmt) {
    match stmt {
        ir::Stmt::Local(l) => {
            if let Some(init) = &l.init {
                collect_idents_in_expr(out, init);
            }
        }
        ir::Stmt::Const(c) => {
            collect_idents_in_expr(out, &c.expr);
        }
        ir::Stmt::Assignment { lhs, rhs } => {
            collect_idents_in_expr(out, lhs);
            collect_idents_in_expr(out, rhs);
        }
        ir::Stmt::CompoundAssignment { lhs, rhs, .. } => {
            collect_idents_in_expr(out, lhs);
            collect_idents_in_expr(out, rhs);
        }
        ir::Stmt::While { condition, body } => {
            collect_idents_in_expr(out, condition);
            collect_idents_in_block(out, body);
        }
        ir::Stmt::Loop { body } => {
            collect_idents_in_block(out, body);
        }
        ir::Stmt::Expr { expr, .. } => {
            collect_idents_in_expr(out, expr);
        }
        ir::Stmt::If(s) => {
            collect_idents_in_expr(out, &s.condition);
            collect_idents_in_block(out, &s.then_block);
            if let Some(else_branch) = &s.else_branch {
                match else_branch {
                    ir::ElseBranch::Block(b) => collect_idents_in_block(out, b),
                    ir::ElseBranch::If(i) => {
                        collect_idents_in_expr(out, &i.condition);
                        collect_idents_in_block(out, &i.then_block);
                        if let Some(else_branch) = &i.else_branch {
                            match else_branch {
                                ir::ElseBranch::Block(b) => collect_idents_in_block(out, b),
                                ir::ElseBranch::If(_) => {
                                    // Shouldn't recurse infinitely; the IR is
                                    // finite.
                                }
                            }
                        }
                    }
                }
            }
        }
        ir::Stmt::Return(Some(e)) => collect_idents_in_expr(out, e),
        ir::Stmt::For(f) => {
            collect_idents_in_expr(out, &f.from);
            collect_idents_in_expr(out, &f.to);
            collect_idents_in_block(out, &f.body);
        }
        ir::Stmt::Switch(s) => {
            collect_idents_in_expr(out, &s.selector);
            for arm in &s.arms {
                for sel in &arm.selectors {
                    match sel {
                        ir::CaseSelector::Literal(_) => {}
                        ir::CaseSelector::Expr(e) => collect_idents_in_expr(out, e),
                        ir::CaseSelector::Default => {}
                    }
                }
                collect_idents_in_block(out, &arm.body);
            }
        }
        ir::Stmt::Block(b) => collect_idents_in_block(out, b),
        ir::Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
        } => {
            collect_idents_in_expr(out, slab);
            collect_idents_in_expr(out, offset);
            collect_idents_in_expr(out, dest);
            collect_idents_in_expr(out, size);
        }
        ir::Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
        } => {
            collect_idents_in_expr(out, slab);
            collect_idents_in_expr(out, offset);
            collect_idents_in_expr(out, src);
            if let Some(s) = size {
                collect_idents_in_expr(out, s);
            }
        }
        ir::Stmt::Break | ir::Stmt::Continue | ir::Stmt::Discard | ir::Stmt::Return(None) => {}
    }
}

fn collect_idents_in_expr(out: &mut std::collections::HashSet<String>, expr: &ir::Expr) {
    match expr {
        ir::Expr::Ident(name) => {
            out.insert(name.clone());
        }
        ir::Expr::Lit(_) => {}
        ir::Expr::Array { elems } => {
            for e in elems {
                collect_idents_in_expr(out, e);
            }
        }
        ir::Expr::Paren(e) => collect_idents_in_expr(out, e),
        ir::Expr::Binary { lhs, rhs, .. } => {
            collect_idents_in_expr(out, lhs);
            collect_idents_in_expr(out, rhs);
        }
        ir::Expr::Unary { expr, .. } => collect_idents_in_expr(out, expr),
        ir::Expr::ArrayIndexing { lhs, index } => {
            collect_idents_in_expr(out, lhs);
            collect_idents_in_expr(out, index);
        }
        ir::Expr::Swizzle { lhs, params, .. } => {
            collect_idents_in_expr(out, lhs);
            if let Some(args) = params {
                for a in args {
                    collect_idents_in_expr(out, a);
                }
            }
        }
        ir::Expr::Cast { lhs, .. } => collect_idents_in_expr(out, lhs),
        ir::Expr::FnCall { params, .. } => {
            for p in params {
                collect_idents_in_expr(out, p);
            }
        }
        ir::Expr::Struct { fields, .. } => {
            for fv in fields {
                collect_idents_in_expr(out, &fv.expr);
            }
        }
        ir::Expr::FieldAccess { base, .. } => collect_idents_in_expr(out, base),
        ir::Expr::TypePath { .. } => {}
        ir::Expr::Reference(e) => collect_idents_in_expr(out, e),
        ir::Expr::ZeroValueArray { len, .. } => collect_idents_in_expr(out, len),
    }
}

/// Analyzes a `wgsl_rs::Source` by first assembling its full IR
/// (including imports and template instantiations), then delegating to
/// [`analyze_ir_module`].
///
/// Errors if the source is a template (has unresolved
/// `Type::TypeParam`s). For templates, call `Source::instantiate`
/// first and pass the resulting `ir::Module` to [`analyze_ir_module`].
///
/// **Fallible.** Returns
/// [`SourceError::TemplateWgsl`](crate::SourceError::TemplateWgsl)-style errors
/// for template sources (caller must `instantiate` first) and
/// [`Error::TemplateResolution`] if a cross-source template instantiation
/// is ambiguous. Callers can `?` this on a possibly-template source, or
/// `unwrap` on a known concrete one.
pub fn analyze_wgsl_module(wgsl_source: &Source) -> Result<WgpuLinkage, Error> {
    if wgsl_source.is_template() {
        return Err(crate::SourceError::TemplateWgsl {
            uninstantiated_source: wgsl_source,
        }
        .into());
    }
    let ir_module = assemble_ir(wgsl_source)?;
    Ok(analyze_ir_module(ir_module))
}

// ===== IR assembly (flattens imports) =====

/// Build a single owned `ir::Module` from a `wgsl_rs::Source` and all
/// its transitive imports and template instantiations. The result is
/// concrete (no `Type::TypeParam`s) and self-contained: every struct
/// referenced by a binding is defined within the assembled source.
///
/// This is the same traversal that `Source::wgsl_source()` performs,
/// but it stops at IR rather than rendering. It exists so the runtime
/// linkage analyzer can correctly size buffers whose types are defined
/// in imported sources.
pub(crate) fn assemble_ir(wgsl_source: &Source) -> Result<ir::Module, Error> {
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen: std::collections::HashSet<(u64, String, Vec<String>)> =
        std::collections::HashSet::new();
    let mut items: Vec<ir::Item> = Vec::new();
    collect_items(wgsl_source, &mut visited, &mut seen, &mut items, None)?;
    Ok(ir::Module {
        name: wgsl_source.name,
        items,
        attrs: Vec::new(),
    })
}

fn collect_items(
    wgsl_source: &Source,
    visited: &mut std::collections::HashSet<u64>,
    seen: &mut std::collections::HashSet<(u64, String, Vec<String>)>,
    out: &mut Vec<ir::Item>,
    subst: Option<&std::collections::HashMap<String, ir::Type>>,
) -> Result<(), Error> {
    // Imports first (depth-first, deduplicated by source id).
    for m in wgsl_source.imports {
        if visited.insert(m.id) {
            collect_items(m, visited, seen, out, None)?;
        }
    }

    // This source's own items, optionally substituted.
    let mut ir_module = (wgsl_source.ir_constructor)();
    if let Some(s) = subst {
        ir::substitute_types(&mut ir_module, s);
    }
    out.extend(ir_module.items);

    // Cross-source template instantiations.
    for inst in wgsl_source.instantiations {
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
        )?;
    }
    Ok(())
}

fn collect_instantiation(
    sources: &[&Source],
    template_name: &str,
    mangled_type_args: &[String],
    type_args: &[ir::Type],
    out: &mut Vec<ir::Item>,
    seen: &mut std::collections::HashSet<(u64, String, Vec<String>)>,
) -> Result<(), Error> {
    let available: Vec<String> = sources
        .iter()
        .copied()
        .flat_map(|m| m.templates.iter().map(|t| t.name.to_string()))
        .collect();

    let mut matching: Vec<&Source> = sources
        .iter()
        .copied()
        .filter(|m| m.templates.iter().any(|t| t.name == template_name))
        .collect();

    snafu::ensure!(
        !matching.is_empty(),
        TemplateResolutionSnafu {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            available,
        }
    );

    if matching.len() > 1 {
        let names: Vec<String> = matching.iter().map(|m| m.name.to_string()).collect();
        return TemplateResolutionSnafu {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            available: names,
        }
        .fail();
    }

    let source = matching
        .pop()
        .expect("matching is non-empty after the check above");
    let Some(template) = source.templates.iter().find(|t| t.name == template_name) else {
        // Invariant: `matching` was built by filtering on
        // `templates.iter().any(|t| t.name == template_name)`, so the
        // sole surviving source must contain the template.
        unreachable!(
            "resolved source '{}' has no template '{}'; available: {:?}",
            source.name, template_name, available
        );
    };

    let key = (
        source.id,
        template_name.to_string(),
        mangled_type_args.to_vec(),
    );
    snafu::ensure!(
        seen.insert(key),
        TemplateResolutionSnafu {
            template_name: template_name.to_owned(),
            mangled_type_args: mangled_type_args.to_vec(),
            available
        }
    );

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
        collect_instantiation(&[source], dep.callee, &dep_mangled, &dep_args, out, seen)?;
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

    Ok(())
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
        ir::Type::Ptr { .. } => TypeLayout { size: 0, align: 1 },
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
        // WGSL array lengths must be non-negative const expressions, so a
        // unary negation is unresolvable — returning `None` falls back to
        // size 0 rather than producing a huge (wraparound) length.
        ir::Expr::Unary {
            op: ir::UnOp::Neg, ..
        } => None,
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
