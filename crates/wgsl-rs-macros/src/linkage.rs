//! Generates wgpu linkage code from parsed WGSL modules.
//!
//! This module is only compiled when the `linkage-wgpu` feature is enabled.

use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{FnAttrs, Item, ItemMod, ItemStorage, ItemUniform, StorageAccess};

/// Information about a binding (uniform or storage buffer).
pub struct BindingInfo {
    pub binding: u32,
    pub name: syn::Ident,
    pub kind: BindingKind,
}

/// The kind of binding.
pub enum BindingKind {
    Uniform,
    Storage { read_only: bool },
}

/// Information about a compute shader entry point.
pub struct ComputeEntry {
    pub name: syn::Ident,
    pub workgroup_size: (u32, u32, u32),
}

/// Information about a vertex shader entry point.
pub struct VertexEntry {
    pub name: syn::Ident,
}

/// Information about a fragment shader entry point.
pub struct FragmentEntry {
    pub name: syn::Ident,
}

/// Collected linkage information from a parsed module.
pub struct LinkageInfo {
    pub module_name: syn::Ident,
    pub bind_groups: BTreeMap<u32, Vec<BindingInfo>>,
    pub vertex_entries: Vec<VertexEntry>,
    pub fragment_entries: Vec<FragmentEntry>,
    pub compute_entries: Vec<ComputeEntry>,
}

impl LinkageInfo {
    /// Extract linkage information from a parsed module.
    pub fn from_item_mod(module_name: syn::Ident, parsed: &ItemMod) -> Self {
        let mut info = LinkageInfo {
            module_name,
            bind_groups: BTreeMap::new(),
            vertex_entries: Vec::new(),
            fragment_entries: Vec::new(),
            compute_entries: Vec::new(),
        };

        for item in &parsed.content {
            match item {
                Item::Uniform(u) => info.add_uniform(u),
                Item::Storage(s) => info.add_storage(s),
                Item::Fn(f) => info.add_fn(f),
                _ => {}
            }
        }

        info
    }

    fn add_uniform(&mut self, u: &ItemUniform) {
        let group: u32 = u.group.base10_parse().unwrap_or(0);
        let binding: u32 = u.binding.base10_parse().unwrap_or(0);

        self.bind_groups
            .entry(group)
            .or_default()
            .push(BindingInfo {
                binding,
                name: u.name.clone(),
                kind: BindingKind::Uniform,
            });
    }

    fn add_storage(&mut self, s: &ItemStorage) {
        let group: u32 = s.group.base10_parse().unwrap_or(0);
        let binding: u32 = s.binding.base10_parse().unwrap_or(0);
        let read_only = matches!(s.access, StorageAccess::Read);

        self.bind_groups
            .entry(group)
            .or_default()
            .push(BindingInfo {
                binding,
                name: s.name.clone(),
                kind: BindingKind::Storage { read_only },
            });
    }

    fn add_fn(&mut self, f: &crate::parse::ItemFn) {
        match &f.fn_attrs {
            FnAttrs::Vertex(_) => {
                self.vertex_entries.push(VertexEntry {
                    name: f.ident.clone(),
                });
            }
            FnAttrs::Fragment(_) => {
                self.fragment_entries.push(FragmentEntry {
                    name: f.ident.clone(),
                });
            }
            FnAttrs::Compute {
                ident: _,
                workgroup_size,
            } => {
                let x: u32 = workgroup_size.x.base10_parse().unwrap_or(1);
                let y: u32 = workgroup_size
                    .y
                    .as_ref()
                    .map(|(_, lit)| lit.base10_parse().unwrap_or(1))
                    .unwrap_or(1);
                let z: u32 = workgroup_size
                    .z
                    .as_ref()
                    .map(|(_, lit)| lit.base10_parse().unwrap_or(1))
                    .unwrap_or(1);

                self.compute_entries.push(ComputeEntry {
                    name: f.ident.clone(),
                    workgroup_size: (x, y, z),
                });
            }
            FnAttrs::None => {}
        }
    }
}

/// Generate the linkage module code.
pub fn generate_linkage_module(info: &LinkageInfo, source_lines: &[String]) -> TokenStream {
    let module_name_str = info.module_name.to_string();

    // Join source lines into a single string for SHADER_SOURCE
    let shader_source = source_lines.join("\n");

    // Generate bind group modules
    let bind_group_modules = generate_bind_group_modules(info, &module_name_str);

    // Generate vertex entry point modules
    let vertex_modules = generate_vertex_entry_modules(&info.vertex_entries);

    // Generate fragment entry point modules
    let fragment_modules = generate_fragment_entry_modules(&info.fragment_entries);

    // Generate compute entry point modules
    let compute_modules = generate_compute_entry_modules(&info.compute_entries, &module_name_str);

    quote! {
        pub mod linkage {
            /// The WGSL source code as a single string.
            pub const SHADER_SOURCE: &str = #shader_source;

            /// Creates a shader module descriptor.
            pub fn shader_module_descriptor() -> wgpu::ShaderModuleDescriptor<'static> {
                wgpu::ShaderModuleDescriptor {
                    label: Some(#module_name_str),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
                }
            }

            /// Creates a shader module from the WGSL source.
            pub fn shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
                device.create_shader_module(shader_module_descriptor())
            }

            #bind_group_modules

            #vertex_modules

            #fragment_modules

            #compute_modules
        }
    }
}

/// Generate modules for each bind group.
fn generate_bind_group_modules(info: &LinkageInfo, module_name: &str) -> TokenStream {
    let modules: Vec<TokenStream> = info
        .bind_groups
        .iter()
        .map(|(group, bindings)| {
            let mod_name = format_ident!("bind_group_{}", group);
            let label = format!("{}::bind_group_{}", module_name, group);

            let layout_entries: Vec<TokenStream> = bindings
                .iter()
                .map(|binding| {
                    let binding_num = binding.binding;
                    let binding_type = match &binding.kind {
                        BindingKind::Uniform => quote! {
                            wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
                        },
                        BindingKind::Storage { read_only } => quote! {
                            wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: #read_only },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            }
                        },
                    };

                    quote! {
                        wgpu::BindGroupLayoutEntry {
                            binding: #binding_num,
                            visibility: wgpu::ShaderStages::all(),
                            ty: #binding_type,
                            count: None,
                        }
                    }
                })
                .collect();

            // Generate typed parameters for bind_group function
            // Parameters are in declaration order, but use their correct binding numbers
            let param_names: Vec<syn::Ident> = bindings
                .iter()
                .map(|b| format_ident!("{}", crate::parse::to_snake_case(&b.name.to_string())))
                .collect();

            let param_decls: Vec<TokenStream> = param_names
                .iter()
                .map(|name| quote! { #name: wgpu::BindingResource<'a> })
                .collect();

            let bind_group_entries: Vec<TokenStream> = bindings
                .iter()
                .zip(param_names.iter())
                .map(|(binding_info, param_name)| {
                    let binding_num = binding_info.binding;
                    quote! {
                        wgpu::BindGroupEntry {
                            binding: #binding_num,
                            resource: #param_name,
                        }
                    }
                })
                .collect();

            quote! {
                pub mod #mod_name {
                    /// The bind group layout entries.
                    pub const LAYOUT_ENTRIES: &[wgpu::BindGroupLayoutEntry] = &[
                        #(#layout_entries),*
                    ];

                    /// The bind group layout descriptor.
                    pub const LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
                        wgpu::BindGroupLayoutDescriptor {
                            label: Some(#label),
                            entries: LAYOUT_ENTRIES,
                        };

                    /// Creates a bind group layout.
                    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
                        device.create_bind_group_layout(&LAYOUT_DESCRIPTOR)
                    }

                    /// Creates a bind group with the given entries.
                    ///
                    /// This is a dynamic version that accepts a slice of entries.
                    /// For a safer version with named parameters, use [`create`].
                    pub fn create_dynamic<'a>(
                        device: &wgpu::Device,
                        layout: &wgpu::BindGroupLayout,
                        entries: &[wgpu::BindGroupEntry<'a>],
                    ) -> wgpu::BindGroup {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(#label),
                            layout,
                            entries,
                        })
                    }

                    /// Creates a bind group with the specified bindings.
                    ///
                    /// Each parameter corresponds to a binding in this bind group,
                    /// providing IDE feedback about which resources are expected.
                    pub fn create<'a>(
                        device: &wgpu::Device,
                        layout: &wgpu::BindGroupLayout,
                        #(#param_decls),*
                    ) -> wgpu::BindGroup {
                        create_dynamic(
                            device,
                            layout,
                            &[
                                #(#bind_group_entries),*
                            ],
                        )
                    }
                }
            }
        })
        .collect();

    quote! {
        #(#modules)*
    }
}

/// Generate modules for vertex shader entry points.
fn generate_vertex_entry_modules(entries: &[VertexEntry]) -> TokenStream {
    let modules: Vec<TokenStream> = entries
        .iter()
        .map(|entry| {
            let mod_name = &entry.name;
            let entry_point_str = entry.name.to_string();

            quote! {
                pub mod #mod_name {
                    /// The entry point name for this vertex shader.
                    pub const ENTRY_POINT: &str = #entry_point_str;

                    /// Creates a vertex state for this entry point.
                    pub fn vertex_state(module: &wgpu::ShaderModule) -> wgpu::VertexState<'_> {
                        wgpu::VertexState {
                            module,
                            entry_point: Some(ENTRY_POINT),
                            buffers: &[],
                            compilation_options: Default::default(),
                        }
                    }
                }
            }
        })
        .collect();

    quote! {
        #(#modules)*
    }
}

/// Generate modules for fragment shader entry points.
fn generate_fragment_entry_modules(entries: &[FragmentEntry]) -> TokenStream {
    let modules: Vec<TokenStream> = entries
        .iter()
        .map(|entry| {
            let mod_name = &entry.name;
            let entry_point_str = entry.name.to_string();

            quote! {
                pub mod #mod_name {
                    /// The entry point name for this fragment shader.
                    pub const ENTRY_POINT: &str = #entry_point_str;

                    /// Creates a fragment state for this entry point.
                    pub fn fragment_state<'a>(
                        module: &'a wgpu::ShaderModule,
                        targets: &'a [Option<wgpu::ColorTargetState>],
                    ) -> wgpu::FragmentState<'a> {
                        wgpu::FragmentState {
                            module,
                            entry_point: Some(ENTRY_POINT),
                            targets,
                            compilation_options: Default::default(),
                        }
                    }
                }
            }
        })
        .collect();

    quote! {
        #(#modules)*
    }
}

/// Generate modules for compute shader entry points.
fn generate_compute_entry_modules(entries: &[ComputeEntry], module_name: &str) -> TokenStream {
    let modules: Vec<TokenStream> = entries
        .iter()
        .map(|entry| {
            let mod_name = &entry.name;
            let entry_point_str = entry.name.to_string();
            let label = format!("{}::{}", module_name, entry_point_str);
            let (x, y, z) = entry.workgroup_size;

            quote! {
                pub mod #mod_name {
                    /// The entry point name for this compute shader.
                    pub const ENTRY_POINT: &str = #entry_point_str;

                    /// The workgroup size for this compute shader.
                    pub const WORKGROUP_SIZE: (u32, u32, u32) = (#x, #y, #z);

                    /// Creates a pipeline layout descriptor.
                    pub fn pipeline_layout_descriptor<'a>(
                        bind_group_layouts: &'a [&'a wgpu::BindGroupLayout],
                    ) -> wgpu::PipelineLayoutDescriptor<'a> {
                        wgpu::PipelineLayoutDescriptor {
                            label: Some(#label),
                            bind_group_layouts,
                            immediate_size: 0,
                        }
                    }

                    /// Creates a pipeline layout.
                    pub fn pipeline_layout(
                        device: &wgpu::Device,
                        bind_group_layouts: &[&wgpu::BindGroupLayout],
                    ) -> wgpu::PipelineLayout {
                        device.create_pipeline_layout(&pipeline_layout_descriptor(bind_group_layouts))
                    }

                    /// Creates a compute pipeline descriptor.
                    pub fn compute_pipeline_descriptor<'a>(
                        layout: Option<&'a wgpu::PipelineLayout>,
                        module: &'a wgpu::ShaderModule,
                    ) -> wgpu::ComputePipelineDescriptor<'a> {
                        wgpu::ComputePipelineDescriptor {
                            label: Some(#label),
                            layout,
                            module,
                            entry_point: Some(ENTRY_POINT),
                            compilation_options: Default::default(),
                            cache: None,
                        }
                    }

                    /// Creates a compute pipeline.
                    pub fn compute_pipeline(
                        device: &wgpu::Device,
                        layout: Option<&wgpu::PipelineLayout>,
                        module: &wgpu::ShaderModule,
                    ) -> wgpu::ComputePipeline {
                        device.create_compute_pipeline(&compute_pipeline_descriptor(layout, module))
                    }
                }
            }
        })
        .collect();

    quote! {
        #(#modules)*
    }
}
