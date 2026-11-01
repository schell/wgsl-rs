// The per-major wgpu adapter, included by `src/lib.rs` into each
// feature-gated `vNN` module with that major's `wgpu` aliased into
// scope. Because this file is textually included (rather than declared
// as a module), it must stay free of inner attributes (`//!`), and it
// is compiled once per enabled major as the body of `crate::vNN`.
//
// The wgpu-free core (errors, WGSL §14.4.1 sizing, binding
// metadata, and the wgsl-rs#177 stage analysis) lives in
// `crate::analysis`, compiled exactly once per crate.
//
// Given an `ir::Module` (produced by the macro-emitted `instantiate()`
// function — zero arguments for concrete modules, generic
// `instantiate::<T>()` for templates), this module extracts
// the binding and entry-point information needed to build wgpu
// pipelines at runtime (no proc-macro-time code generation), so
// template instantiations get the same linkage analysis as concrete
// sources.
//
// A wgpu major forks this file only when it actually diverges from
// this API: the diverging `vNN` module includes its own copy and drops
// the shared include.
//
// The pinned `wgpu` is re-exported by the including module
// (`pub use wgpuNN as wgpu;` in `src/lib.rs`), so linkage consumers can
// import every type the API produces from one path.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
};

use wgsl_rs_ir as ir;

use wgsl_rs::Source;

// The shared, wgpu-free analysis core, compiled once per crate (not
// once per major). Re-exported here so each major module exposes the
// same surface the pre-combination version crates had.
pub use crate::analysis::{
    BindingKind, BindingMeta, BufferKind, ComputeEntryInfo, Error, Stage, Stages, TypeLayout,
    type_byte_size,
};

use crate::analysis::{
    DuplicateBindingNameSnafu, MissingBindingSnafu, NoSuchBindGroupSnafu, UnknownBindingNameSnafu,
    analyze_stages, binding_kind_label, buffer_kind_label,
};

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
    /// source's WGSL text. Sources built from [`wgsl_rs::Source`] are
    /// assembled by [`Source::assemble_ir`] (flattening imports and
    /// template instantiations), and `ir::Module` inputs (e.g. an
    /// instantiated template) are whatever the caller passed in — build
    /// them with the macro-emitted `instantiate()` so imports are
    /// included, not with `WGSL_SOURCE.ir_constructor` (own items
    /// only). Always concrete: no `Type::TypeParam` nodes.
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

impl core::fmt::Display for WgpuLinkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let WgpuLinkage {
            module_label,
            ir,
            bind_groups,
            bind_group_layouts,
            pipeline_layout,
            vertex_entries,
            fragment_entries,
            compute_entries,
            buffers,
        } = self;

        writeln!(
            f,
            "linkage for '{module_label}' ({} IR items)",
            ir.items.len()
        )?;

        writeln!(f, "  entry points:")?;
        if vertex_entries.is_empty() && fragment_entries.is_empty() && compute_entries.is_empty() {
            writeln!(f, "    (none)")?;
        }
        for e in vertex_entries {
            writeln!(f, "    vertex: {}", e.name)?;
        }
        for e in fragment_entries {
            writeln!(f, "    fragment: {}", e.name)?;
        }
        for e in compute_entries {
            let (x, y, z) = e.workgroup_size;
            writeln!(f, "    compute: {} @workgroup_size({x}, {y}, {z})", e.name)?;
        }

        writeln!(f, "  bind groups:")?;
        if bind_groups.is_empty() {
            writeln!(f, "    (none)")?;
        }
        for (group, info) in bind_groups {
            writeln!(f, "    @group({group}) {}", info.label)?;
            for meta in &info.bindings {
                writeln!(
                    f,
                    "      @binding({}) {} ({})",
                    meta.binding,
                    meta.name,
                    binding_kind_label(&meta.kind)
                )?;
            }
        }

        writeln!(f, "  buffers:")?;
        if buffers.is_empty() {
            writeln!(f, "    (none)")?;
        }
        for b in buffers {
            // `size` is 0 for runtime-sized storage buffers; the buffer's
            // length is then whatever the caller chooses to allocate.
            let size = if b.size == 0 {
                "runtime-sized".to_string()
            } else {
                format!("{} bytes", b.size)
            };
            writeln!(
                f,
                "      @group({}) @binding({}): {} ({}, {size}, {ty})",
                b.group,
                b.binding,
                b.binding_name,
                buffer_kind_label(&b.kind),
                ty = b.ty
            )?;
        }

        // The layouts are built lazily; show how much of the cache is
        // populated. Calling `pipeline_layout` fills every group at once.
        write!(
            f,
            "  cache: {} of {} bind group layouts built, pipeline layout {}",
            bind_group_layouts.len(),
            bind_groups.len(),
            if pipeline_layout.is_some() {
                "built"
            } else {
                "not built"
            }
        )?;

        Ok(())
    }
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
    /// [`analyze_ir_module`], this is the instantiated source. If any
    /// item uses a tier-1 storage texture format, the required `enable
    /// texture_formats_tier1;` directive is prepended — derived from
    /// the items at render time, so no assembly bookkeeping can drop
    /// it.
    pub fn wgsl_source(&self) -> String {
        let mut source = ir::render_module(&self.ir);
        if ir::items_need_tier1_extension(&self.ir.items) {
            source.insert_str(0, "enable texture_formats_tier1;\n\n");
        }
        source
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
/// is exported from this crate's root, so `m.generate_linkage()` works
/// after importing it, without an explicit `use` of the trait's defining
/// module.
pub trait IrModuleExt {
    /// Builds a [`WgpuLinkage`] from this concrete IR module.
    ///
    /// Equivalent to the free function [`analyze_ir_module`].
    fn generate_linkage(&self) -> WgpuLinkage;

    /// Validates the WGSL source rendered from this IR module using naga.
    ///
    /// Equivalent to the free function [`wgsl_rs::validate_wgsl_source`]
    /// called on the output of [`wgsl_rs_ir::Module::wgsl_source`].
    ///
    /// Only available with the `validation` feature enabled.
    #[cfg(feature = "validation")]
    fn validate(&self) -> Result<(), String>;
}

impl IrModuleExt for wgsl_rs_ir::Module {
    fn generate_linkage(&self) -> WgpuLinkage {
        analyze_ir_module(self.clone())
    }

    #[cfg(feature = "validation")]
    fn validate(&self) -> Result<(), String> {
        wgsl_rs::validate_wgsl_source(&self.wgsl_source())
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
    /// The buffer's IR type.
    ///
    /// Here because it's useful for users to see the type from the runtime/CPU
    /// side.
    pub ty: ir::Type,
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

/// Analyzes an IR module and returns its wgpu linkage.
///
/// This is the core entry point. It expects a concrete IR module (no
/// `Type::TypeParam` nodes); for template modules, call the
/// macro-emitted `instantiate::<...>()` function first and then pass the
/// result here.
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
pub fn analyze_ir_module(mut ir_module: wgsl_rs_ir::Module) -> WgpuLinkage {
    let module_label = ir_module.name;
    ir::deshadow_module(&mut ir_module);
    ir::suffix_module(&mut ir_module);
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

    // Binding-use analysis (wgsl-rs#177) lives in the wgpu-free core:
    // it resolves, transitively through helper functions, which stages
    // reference each binding. `visibility_for` below translates each
    // stage set into `wgpu::ShaderStages`.
    let referenced = analyze_stages(&linkage.ir);

    for item in &linkage.ir.items {
        match item {
            ir::Item::Uniform(u) => {
                let size = type_byte_size(&u.ty, &linkage.ir);
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
                    ty: u.ty.clone(),
                });
            }
            ir::Item::Storage(s) => {
                let read_only = matches!(s.access, ir::StorageAccess::Read);
                let size = type_byte_size(&s.ty, &linkage.ir);
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
                    ty: s.ty.clone(),
                });
            }
            ir::Item::Sampler(s) => {
                let comparison = matches!(s.ty, ir::Type::SamplerComparison);
                let sampler_ty = if comparison {
                    wgpu::SamplerBindingType::Comparison
                } else if !s.filterable {
                    wgpu::SamplerBindingType::NonFiltering
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
/// wgpu-free per-binding stage analysis. If no entry-point function
/// references the binding (e.g. the binding is only declared but
/// unused, or the reference walker missed it), we default to
/// `COMPUTE` — the most common stage for the wgsl-rs roundtrip-test
/// use case. The previous `ShaderStages::all()` default would silently
/// demand the `VERTEX_WRITABLE_STORAGE` feature for any read_write
/// storage, which is almost never what the user wants.
fn visibility_for(referenced: &HashMap<String, Stages>, name: &str) -> wgpu::ShaderStages {
    let stages = referenced.get(name).copied().unwrap_or_default();
    let mut visibility = wgpu::ShaderStages::empty();
    if stages.vertex {
        visibility |= wgpu::ShaderStages::VERTEX;
    }
    if stages.fragment {
        visibility |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.compute {
        visibility |= wgpu::ShaderStages::COMPUTE;
    }
    if visibility.is_empty() {
        wgpu::ShaderStages::COMPUTE
    } else {
        visibility
    }
}

/// Analyzes a `wgsl_rs::Source` by first assembling its full IR
/// (including imports and template instantiations), then delegating to
/// [`analyze_ir_module`].
///
/// Errors if the source is a template (has unresolved
/// `Type::TypeParam`s). For templates, call the macro-emitted
/// `instantiate::<...>()` function first and pass the resulting
/// `ir::Module` to [`analyze_ir_module`].
///
/// **Fallible.** Returns
/// [`SourceError::TemplateWgsl`](wgsl_rs::SourceError::TemplateWgsl)-style
/// errors for template sources (caller must `instantiate` first) and
/// [`Error::TemplateResolution`] if a cross-source template instantiation
/// is ambiguous. Callers can `?` this on a possibly-template source, or
/// `unwrap` on a known concrete one.
pub fn analyze_wgsl_module(wgsl_source: &Source) -> Result<WgpuLinkage, Error> {
    if wgsl_source.is_template() {
        return Err(wgsl_rs::SourceError::TemplateWgsl {
            uninstantiated_source: wgsl_source,
        }
        .into());
    }
    let ir_module = wgsl_source.assemble_ir()?;
    Ok(analyze_ir_module(ir_module))
}

// ===== Texture helpers =====

fn texture_layout_entry(item: &ir::ItemTexture) -> (wgpu::BindGroupLayoutEntry, BindingKind) {
    let view_dimension = match &item.ty {
        ir::Type::Texture { kind, .. } => texture_kind_view_dimension(*kind),
        ir::Type::TextureDepth { kind } => texture_depth_view_dimension(*kind),
        ir::Type::TextureStorage { kind, .. } => texture_storage_kind_view_dimension(*kind),
        _ => unreachable!("ItemTexture validated to be a texture type"),
    };
    let multisampled = match &item.ty {
        ir::Type::Texture { kind, .. } => matches!(kind, ir::TextureKind::TextureMultisampled2D),
        ir::Type::TextureDepth { kind } => {
            matches!(kind, ir::TextureDepthKind::DepthMultisampled2D)
        }
        ir::Type::TextureStorage { .. } => false,
        _ => unreachable!(),
    };
    match &item.ty {
        ir::Type::Texture { sampled_type, .. } => {
            let sample = match sampled_type {
                ir::ScalarType::F32 => wgpu::TextureSampleType::Float {
                    filterable: item.filterable,
                },
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
        ir::Type::TextureStorage { format, access, .. } => {
            let wgpu_format = texel_format_to_wgpu(*format);
            let wgpu_access = match access {
                ir::StorageTextureAccess::Read => wgpu::StorageTextureAccess::ReadOnly,
                ir::StorageTextureAccess::Write => wgpu::StorageTextureAccess::WriteOnly,
                ir::StorageTextureAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
            };
            let read_write = matches!(access, ir::StorageTextureAccess::ReadWrite);
            (
                wgpu::BindGroupLayoutEntry {
                    binding: item.binding,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu_access,
                        format: wgpu_format,
                        view_dimension,
                    },
                    count: None,
                },
                BindingKind::StorageTexture { read_write },
            )
        }
        _ => unreachable!(),
    }
}

fn texture_storage_kind_view_dimension(kind: ir::TextureStorageKind) -> wgpu::TextureViewDimension {
    match kind {
        ir::TextureStorageKind::Storage1D => wgpu::TextureViewDimension::D1,
        ir::TextureStorageKind::Storage2D => wgpu::TextureViewDimension::D2,
        ir::TextureStorageKind::Storage2DArray => wgpu::TextureViewDimension::D2Array,
        ir::TextureStorageKind::Storage3D => wgpu::TextureViewDimension::D3,
    }
}

/// Map an IR texel format to the corresponding wgpu texture format.
fn texel_format_to_wgpu(f: ir::TexelFormat) -> wgpu::TextureFormat {
    use ir::TexelFormat::*;
    match f {
        Rgba8unorm => wgpu::TextureFormat::Rgba8Unorm,
        Rgba8snorm => wgpu::TextureFormat::Rgba8Snorm,
        Rgba8uint => wgpu::TextureFormat::Rgba8Uint,
        Rgba8sint => wgpu::TextureFormat::Rgba8Sint,
        Rgba16uint => wgpu::TextureFormat::Rgba16Uint,
        Rgba16sint => wgpu::TextureFormat::Rgba16Sint,
        Rgba16float => wgpu::TextureFormat::Rgba16Float,
        R32uint => wgpu::TextureFormat::R32Uint,
        R32sint => wgpu::TextureFormat::R32Sint,
        R32float => wgpu::TextureFormat::R32Float,
        Rg32uint => wgpu::TextureFormat::Rg32Uint,
        Rg32sint => wgpu::TextureFormat::Rg32Sint,
        Rg32float => wgpu::TextureFormat::Rg32Float,
        Rgba32uint => wgpu::TextureFormat::Rgba32Uint,
        Rgba32sint => wgpu::TextureFormat::Rgba32Sint,
        Rgba32float => wgpu::TextureFormat::Rgba32Float,
        Bgra8unorm => wgpu::TextureFormat::Bgra8Unorm,
        Rgba16unorm => wgpu::TextureFormat::Rgba16Unorm,
        Rgba16snorm => wgpu::TextureFormat::Rgba16Snorm,
        Rg8unorm => wgpu::TextureFormat::Rg8Unorm,
        Rg8snorm => wgpu::TextureFormat::Rg8Snorm,
        Rg8uint => wgpu::TextureFormat::Rg8Uint,
        Rg8sint => wgpu::TextureFormat::Rg8Sint,
        Rg16unorm => wgpu::TextureFormat::Rg16Unorm,
        Rg16snorm => wgpu::TextureFormat::Rg16Snorm,
        Rg16uint => wgpu::TextureFormat::Rg16Uint,
        Rg16sint => wgpu::TextureFormat::Rg16Sint,
        Rg16float => wgpu::TextureFormat::Rg16Float,
        R8unorm => wgpu::TextureFormat::R8Unorm,
        R8snorm => wgpu::TextureFormat::R8Snorm,
        R8uint => wgpu::TextureFormat::R8Uint,
        R8sint => wgpu::TextureFormat::R8Sint,
        R16unorm => wgpu::TextureFormat::R16Unorm,
        R16snorm => wgpu::TextureFormat::R16Snorm,
        R16uint => wgpu::TextureFormat::R16Uint,
        R16sint => wgpu::TextureFormat::R16Sint,
        R16float => wgpu::TextureFormat::R16Float,
        Rgb10a2unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        Rgb10a2uint => wgpu::TextureFormat::Rgb10a2Uint,
        Rg11b10ufloat => wgpu::TextureFormat::Rg11b10Ufloat,
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

#[cfg(test)]
mod binding_visibility {
    //! Tests for per-binding `ShaderStages` visibility, including the
    //! transitive helper-function case from wgsl-rs#177.

    use std::borrow::Cow;

    use super::*;

    fn f32_ty() -> ir::Type {
        ir::Type::Scalar(ir::ScalarType::F32)
    }

    fn zero() -> ir::Expr {
        ir::Expr::Lit(ir::Lit::Float {
            text: "0.0".to_string(),
        })
    }

    fn uniform(name: &str, group: u32, binding: u32) -> ir::Item {
        ir::Item::Uniform(ir::ItemUniform {
            group,
            binding,
            name: name.to_string(),
            ty: f32_ty(),
            attrs: Vec::new(),
        })
    }

    fn call(name: &str, params: Vec<ir::Expr>) -> ir::Expr {
        ir::Expr::FnCall {
            path: ir::FnPath::Ident(name.to_string()),
            type_args: Vec::new(),
            params,
        }
    }

    fn ident(name: &str) -> ir::Expr {
        ir::Expr::Ident(name.to_string())
    }

    fn fn_item(name: &str, fn_attrs: ir::FnAttrs, stmts: Vec<ir::Stmt>) -> ir::Item {
        ir::Item::Fn(ir::ItemFn {
            type_params: Vec::new(),
            const_params: Vec::new(),
            fn_attrs,
            name: Cow::Owned(name.to_string()),
            inputs: Vec::new(),
            return_type: ir::ReturnType::Default,
            block: ir::Block { stmts },
            attrs: Vec::new(),
        })
    }

    fn returns(expr: ir::Expr) -> ir::Stmt {
        ir::Stmt::Return(Some(expr))
    }

    fn analyze(items: Vec<ir::Item>) -> WgpuLinkage {
        analyze_ir_module(ir::Module {
            name: "test_module",
            items,
            attrs: Vec::new(),
        })
    }

    /// Looks up the analyzed visibility for the named binding.
    fn visibility_of(linkage: &WgpuLinkage, name: &str) -> wgpu::ShaderStages {
        for bind_group in linkage.bind_groups.values() {
            if let Some(meta) = bind_group.bindings.iter().find(|b| b.name == name) {
                let entry = bind_group
                    .entries
                    .iter()
                    .find(|e| e.binding == meta.binding)
                    .expect("binding layout entry exists");
                return entry.visibility;
            }
        }
        panic!("binding '{name}' not found in any bind group");
    }

    #[test]
    fn direct_use_gets_entry_stage() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(ident("MY_UNIFORM"))],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::VERTEX
        );
    }

    /// The wgsl-rs#177 repro: the binding is used only inside a helper
    /// called from both entry points, so it must carry both stages.
    #[test]
    fn helper_only_use_carries_entry_point_stages() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "read_uniform",
                ir::FnAttrs::None,
                vec![returns(ident("MY_UNIFORM"))],
            ),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(call("read_uniform", vec![]))],
            ),
            fn_item(
                "frag",
                ir::FnAttrs::Fragment,
                vec![returns(call("read_uniform", vec![]))],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }

    /// A helper called only from `frag` must not leak vertex visibility.
    #[test]
    fn frag_only_helper_stays_fragment() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "read_uniform",
                ir::FnAttrs::None,
                vec![returns(ident("MY_UNIFORM"))],
            ),
            fn_item(
                "frag",
                ir::FnAttrs::Fragment,
                vec![returns(call("read_uniform", vec![]))],
            ),
            fn_item("vert", ir::FnAttrs::Vertex, vec![returns(zero())]),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::FRAGMENT
        );
    }

    /// Mixed use: direct reference in one entry point, helper-only in
    /// the other.
    #[test]
    fn mixed_direct_and_helper_use() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "read_uniform",
                ir::FnAttrs::None,
                vec![returns(ident("MY_UNIFORM"))],
            ),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(ident("MY_UNIFORM"))],
            ),
            fn_item(
                "frag",
                ir::FnAttrs::Fragment,
                vec![returns(call("read_uniform", vec![]))],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }

    /// Reachability must be transitive: entry point -> helper -> helper
    /// -> binding.
    #[test]
    fn helper_chain_is_transitive() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "inner",
                ir::FnAttrs::None,
                vec![returns(ident("MY_UNIFORM"))],
            ),
            fn_item(
                "outer",
                ir::FnAttrs::None,
                vec![returns(call("inner", vec![]))],
            ),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(call("outer", vec![]))],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::VERTEX
        );
    }

    /// A declared-but-unused binding keeps the documented COMPUTE
    /// default.
    #[test]
    fn unused_binding_defaults_to_compute() {
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item("vert", ir::FnAttrs::Vertex, vec![returns(zero())]),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::COMPUTE
        );
    }

    /// A local (or parameter) that shadows a helper-fn name is a value
    /// reference, not a call edge (PR #190 review finding): the helper
    /// is never called, so its bindings must not gain this entry
    /// point's stage.
    #[test]
    fn local_shadowing_fn_name_is_not_a_call_edge() {
        let linkage = analyze(vec![
            uniform("DATA", 0, 0),
            fn_item(
                "read_uniform",
                ir::FnAttrs::None,
                vec![returns(ident("DATA"))],
            ),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![
                    ir::Stmt::Local(ir::Local {
                        mutable: false,
                        name: "read_uniform".to_string(),
                        ty: None,
                        init: Some(zero()),
                    }),
                    returns(ident("read_uniform")),
                ],
            ),
        ]);
        // The helper is unreachable, so DATA falls back to the COMPUTE
        // default rather than spuriously gaining VERTEX.
        assert_eq!(visibility_of(&linkage, "DATA"), wgpu::ShaderStages::COMPUTE);
    }

    /// Same-shaped control: a real call from frag still creates the
    /// edge (FRAGMENT), while vert's same-named local grants nothing.
    #[test]
    fn call_edge_survives_local_shadowing_elsewhere() {
        let linkage = analyze(vec![
            uniform("DATA", 0, 0),
            fn_item(
                "read_uniform",
                ir::FnAttrs::None,
                vec![returns(ident("DATA"))],
            ),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![
                    ir::Stmt::Local(ir::Local {
                        mutable: false,
                        name: "read_uniform".to_string(),
                        ty: None,
                        init: Some(zero()),
                    }),
                    returns(ident("read_uniform")),
                ],
            ),
            fn_item(
                "frag",
                ir::FnAttrs::Fragment,
                vec![returns(call("read_uniform", vec![]))],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "DATA"),
            wgpu::ShaderStages::FRAGMENT
        );
    }

    /// Impl methods are called via mangled `Type_method` names; bindings
    /// used inside them must be reachable from entry points that call
    /// them.
    #[test]
    fn impl_method_helper_is_reachable() {
        let method = ir::ItemFn {
            type_params: Vec::new(),
            const_params: Vec::new(),
            fn_attrs: ir::FnAttrs::None,
            name: Cow::Owned("read_uniform".to_string()),
            inputs: Vec::new(),
            return_type: ir::ReturnType::Default,
            block: ir::Block {
                stmts: vec![returns(ident("MY_UNIFORM"))],
            },
            attrs: Vec::new(),
        };
        let linkage = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            ir::Item::Impl(ir::ItemImpl {
                type_params: Vec::new(),
                const_params: Vec::new(),
                self_ty: "Foo".to_string(),
                items: vec![ir::ImplItem::Fn(method)],
                attrs: Vec::new(),
            }),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(ir::Expr::FnCall {
                    path: ir::FnPath::TypeMethod {
                        ty: "Foo".to_string(),
                        method: "read_uniform".to_string(),
                    },
                    type_args: Vec::new(),
                    params: Vec::new(),
                })],
            ),
        ]);
        assert_eq!(
            visibility_of(&linkage, "MY_UNIFORM"),
            wgpu::ShaderStages::VERTEX
        );
    }
}
