//! The wgpu-free core of the linkage crate.
//!
//! Everything in this module is compiled exactly once, independent of
//! any enabled wgpu major: the error type, IR assembly (import
//! flattening and cross-source template instantiation), WGSL §14.4.1
//! buffer sizing, binding metadata, and the shader-stage analysis that
//! resolves which stages reference each binding (transitively through
//! helper functions, wgsl-rs#177). The per-major adapter modules
//! (`crate::v29`, `crate::v30`, ...) translate [`Stages`] sets into
//! `wgpu::ShaderStages`.
//!
//! A bare `--no-default-features` build of this crate compiles this
//! module and nothing else: zero wgpu.

use std::borrow::Cow;

use snafu::prelude::*;
use wgsl_rs_ir as ir;

use wgsl_rs::Source;

/// Errors thrown by the WGPU linkage module.
#[derive(Snafu, Debug)]
// The generated context selectors (`MissingBindingSnafu`, ...) are
// consumed by the per-major adapter modules, which live in different
// modules than this one; the default is private to the defining
// module. `pub(crate)` keeps them out of the public API.
#[snafu(visibility(pub(crate)))]
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
        "{source_name}: reference to generic template '{template_name}' was never resolved to a \
         concrete instantiation{}; supply generic arguments with turbofish at the call site \
         (wgsl-rs#237)",
        member
            .as_ref()
            .map(|m| format!(" (::{m})"))
            .unwrap_or_default()
    ))]
    UnresolvedTemplateRef {
        source_name: String,
        template_name: String,
        member: Option<String>,
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

impl From<wgsl_rs::SourceError<'_>> for Error {
    fn from(err: wgsl_rs::SourceError<'_>) -> Self {
        match err {
            wgsl_rs::SourceError::TemplateWgsl {
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
            wgsl_rs::SourceError::TemplateNotFound {
                template_name,
                mangled_type_args,
                available_templates,
            } => Error::TemplateResolution {
                template_name,
                mangled_type_args,
                available: available_templates,
            },
            wgsl_rs::SourceError::AmbiguousTemplate {
                template_name,
                mangled_type_args,
                matching_source_names,
                available_templates: _,
            } => Error::TemplateResolution {
                template_name,
                mangled_type_args,
                available: matching_source_names,
            },
            wgsl_rs::SourceError::UnresolvedTemplateRef {
                source_name,
                template_name,
                member,
                has_type_args: _,
            } => Error::UnresolvedTemplateRef {
                source_name,
                template_name,
                member,
            },
        }
    }
}

// ===== Shader-stage analysis (wgsl-rs#177) =====

/// A single shader stage, in wgpu-free form.
///
/// The per-major adapter modules translate these into
/// `wgpu::ShaderStages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

/// The set of shader stages that reference a binding, in wgpu-free
/// form.
///
/// The per-major adapter modules translate a set into
/// `wgpu::ShaderStages`; an empty set means the binding is not
/// referenced by any entry point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stages {
    /// Referenced from at least one vertex entry point.
    pub vertex: bool,
    /// Referenced from at least one fragment entry point.
    pub fragment: bool,
    /// Referenced from at least one compute entry point.
    pub compute: bool,
}

impl Stages {
    /// Adds `stage` to this set.
    pub fn insert(&mut self, stage: Stage) {
        match stage {
            Stage::Vertex => self.vertex = true,
            Stage::Fragment => self.fragment = true,
            Stage::Compute => self.compute = true,
        }
    }

    /// True when no stage references the binding.
    pub fn is_empty(&self) -> bool {
        !self.vertex && !self.fragment && !self.compute
    }
}

/// The stage of an entry-point fn, or `None` for helper fns.
pub fn stage_of_attrs(fn_attrs: &ir::FnAttrs) -> Option<Stage> {
    match fn_attrs {
        ir::FnAttrs::Vertex => Some(Stage::Vertex),
        ir::FnAttrs::Fragment => Some(Stage::Fragment),
        ir::FnAttrs::Compute { .. } => Some(Stage::Compute),
        ir::FnAttrs::None => None,
    }
}

/// Value identifiers and call edges collected from a fn body, kept
/// separate so a local (or parameter) that shadows a helper-fn name is
/// not mistaken for a call (PR #190 review finding).
#[derive(Default)]
struct FnRefs {
    /// `Expr::Ident` references: bindings, locals, params.
    idents: std::collections::HashSet<String>,
    /// `Expr::FnCall` callee paths; method calls are mangled
    /// (`Type_method`). Only these are traversed as call-graph edges.
    callees: std::collections::HashSet<String>,
}

/// Records a function's call-graph entry: its (possibly mangled) name
/// and the value identifiers and call edges its body references (see
/// [`collect_refs_in_block`]).
fn record_fn_idents(
    f: &ir::ItemFn,
    name: String,
    fn_names: &mut std::collections::HashSet<String>,
    fn_refs: &mut std::collections::HashMap<String, FnRefs>,
) {
    let mut refs = FnRefs::default();
    collect_refs_in_block(&mut refs, &f.block);
    fn_names.insert(name.clone());
    fn_refs.insert(name, refs);
}

/// Analyzes which shader stages reference each binding, transitively
/// through helper functions (wgsl-rs#177): a binding used only inside
/// a helper called from both the vertex and fragment entry points
/// carries both stages.
///
/// The module must already be deshadowed (plain name matching is safe
/// only then; see `ir::deshadow_module`). Impl methods participate
/// under their mangled render names (`Type_method`), matching how the
/// renderer emits call sites.
///
/// Returns a map from declared binding name to the stages that
/// reference it; unreferenced bindings are absent.
pub fn analyze_stages(module: &ir::Module) -> std::collections::HashMap<String, Stages> {
    let mut binding_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Every function name in the module, including impl methods under
    // their mangled render names (`Type_method`).
    let mut fn_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Per-fn reference sets: value identifiers and call edges, kept
    // separate (see [`FnRefs`]) so a local shadowing a helper-fn name
    // is not mistaken for a call.
    let mut fn_refs: std::collections::HashMap<String, FnRefs> = std::collections::HashMap::new();
    let mut entry_points: Vec<(String, Stage)> = Vec::new();

    for item in &module.items {
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
                record_fn_idents(f, f.name.to_string(), &mut fn_names, &mut fn_refs);
                if let Some(stage) = stage_of_attrs(&f.fn_attrs) {
                    entry_points.push((f.name.to_string(), stage));
                }
            }
            ir::Item::Impl(imp) => {
                for impl_item in &imp.items {
                    let ir::ImplItem::Fn(f) = impl_item else {
                        continue;
                    };
                    // Impl methods are invoked via their mangled render
                    // names (`Type_method`), so key the call graph by the
                    // same mangled name.
                    let mangled = ir::mangle::mangle(&[&imp.self_ty, &f.name]);
                    record_fn_idents(f, mangled.clone(), &mut fn_names, &mut fn_refs);
                    if let Some(stage) = stage_of_attrs(&f.fn_attrs) {
                        entry_points.push((mangled, stage));
                    }
                }
            }
            _ => {}
        }
    }

    // Starting from each entry point, walk the module call graph and
    // union every reachable function's referenced bindings into the
    // entry point's stage. The visited set guards against (spec-
    // illegal) recursive call cycles.
    let mut referenced: std::collections::HashMap<String, Stages> =
        std::collections::HashMap::new();
    for (entry_name, stage) in &entry_points {
        let mut visited: std::collections::HashSet<&String> = std::collections::HashSet::new();
        let mut stack: Vec<&String> = vec![entry_name];
        while let Some(fname) = stack.pop() {
            if !visited.insert(fname) {
                continue;
            }
            let Some(refs) = fn_refs.get(fname) else {
                continue;
            };
            // Bindings are attributed from value identifiers only; call
            // edges come from `callees` only, so a same-named local can
            // never masquerade as a call.
            for name in &refs.idents {
                if binding_names.contains(name) {
                    referenced.entry(name.clone()).or_default().insert(*stage);
                }
            }
            for callee in &refs.callees {
                if fn_names.contains(callee) {
                    stack.push(callee);
                }
            }
        }
    }
    referenced
}

/// Walks an `ir::Block`, collecting value identifiers and call edges
/// into `refs` (see [`FnRefs`]).
fn collect_refs_in_block(refs: &mut FnRefs, block: &ir::Block) {
    for stmt in &block.stmts {
        collect_refs_in_stmt(refs, stmt);
    }
}

fn collect_refs_in_stmt(refs: &mut FnRefs, stmt: &ir::Stmt) {
    match stmt {
        ir::Stmt::Local(l) => {
            if let Some(init) = &l.init {
                collect_refs_in_expr(refs, init);
            }
        }
        ir::Stmt::Const(c) => {
            collect_refs_in_expr(refs, &c.expr);
        }
        ir::Stmt::Assignment { lhs, rhs } => {
            collect_refs_in_expr(refs, lhs);
            collect_refs_in_expr(refs, rhs);
        }
        ir::Stmt::CompoundAssignment { lhs, rhs, .. } => {
            collect_refs_in_expr(refs, lhs);
            collect_refs_in_expr(refs, rhs);
        }
        ir::Stmt::While { condition, body } => {
            collect_refs_in_expr(refs, condition);
            collect_refs_in_block(refs, body);
        }
        ir::Stmt::Loop { body } => {
            collect_refs_in_block(refs, body);
        }
        ir::Stmt::Expr { expr, .. } => {
            collect_refs_in_expr(refs, expr);
        }
        ir::Stmt::If(s) => {
            collect_refs_in_expr(refs, &s.condition);
            collect_refs_in_block(refs, &s.then_block);
            if let Some(else_branch) = &s.else_branch {
                match else_branch {
                    ir::ElseBranch::Block(b) => collect_refs_in_block(refs, b),
                    ir::ElseBranch::If(i) => {
                        collect_refs_in_expr(refs, &i.condition);
                        collect_refs_in_block(refs, &i.then_block);
                        if let Some(else_branch) = &i.else_branch {
                            match else_branch {
                                ir::ElseBranch::Block(b) => collect_refs_in_block(refs, b),
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
        ir::Stmt::Return(Some(e)) => collect_refs_in_expr(refs, e),
        ir::Stmt::For(f) => {
            collect_refs_in_expr(refs, &f.from);
            collect_refs_in_expr(refs, &f.to);
            collect_refs_in_block(refs, &f.body);
        }
        ir::Stmt::Switch(s) => {
            collect_refs_in_expr(refs, &s.selector);
            for arm in &s.arms {
                for sel in &arm.selectors {
                    match sel {
                        ir::CaseSelector::Literal(_) => {}
                        ir::CaseSelector::Expr(e) => collect_refs_in_expr(refs, e),
                        ir::CaseSelector::Default => {}
                    }
                }
                collect_refs_in_block(refs, &arm.body);
            }
        }
        ir::Stmt::Block(b) => collect_refs_in_block(refs, b),
        ir::Stmt::SlabCopy {
            src,
            src_offset,
            dest,
            dest_offset,
            size,
        } => {
            collect_refs_in_expr(refs, src);
            collect_refs_in_expr(refs, src_offset);
            collect_refs_in_expr(refs, dest);
            collect_refs_in_expr(refs, dest_offset);
            collect_refs_in_expr(refs, size);
        }
        ir::Stmt::Break
        | ir::Stmt::Continue
        | ir::Stmt::Discard
        | ir::Stmt::Return(None)
        | ir::Stmt::Macro { .. } => {}
    }
}

fn collect_refs_in_expr(refs: &mut FnRefs, expr: &ir::Expr) {
    match expr {
        ir::Expr::Ident(name) => {
            refs.idents.insert(name.clone());
        }
        ir::Expr::Lit(_) => {}
        ir::Expr::Array { elems } => {
            for e in elems {
                collect_refs_in_expr(refs, e);
            }
        }
        ir::Expr::Paren(e) => collect_refs_in_expr(refs, e),
        ir::Expr::Binary { lhs, rhs, .. } => {
            collect_refs_in_expr(refs, lhs);
            collect_refs_in_expr(refs, rhs);
        }
        ir::Expr::Unary { expr, .. } => collect_refs_in_expr(refs, expr),
        ir::Expr::ArrayIndexing { lhs, index } => {
            collect_refs_in_expr(refs, lhs);
            collect_refs_in_expr(refs, index);
        }
        ir::Expr::Swizzle { lhs, params, .. } => {
            collect_refs_in_expr(refs, lhs);
            if let Some(args) = params {
                for a in args {
                    collect_refs_in_expr(refs, a);
                }
            }
        }
        ir::Expr::Cast { lhs, .. } => collect_refs_in_expr(refs, lhs),
        ir::Expr::FnCall { path, params, .. } => {
            // Record the callee so the linkage analysis can resolve
            // transitive binding use through helper functions
            // (wgsl-rs#177). Method calls are recorded under their
            // mangled render names, matching how impl methods are
            // keyed in the call graph.
            match path {
                ir::FnPath::Ident(name) => {
                    refs.callees.insert(name.clone());
                }
                ir::FnPath::TypeMethod { ty, method } => {
                    refs.callees.insert(ir::mangle::mangle(&[ty, method]));
                }
            }
            for p in params {
                collect_refs_in_expr(refs, p);
            }
        }
        ir::Expr::Struct { fields, .. } => {
            for fv in fields {
                collect_refs_in_expr(refs, &fv.expr);
            }
        }
        ir::Expr::FieldAccess { base, .. } => collect_refs_in_expr(refs, base),
        ir::Expr::TypePath { .. } => {}
        ir::Expr::Reference(e) => collect_refs_in_expr(refs, e),
        ir::Expr::ZeroValueArray { len, .. } => collect_refs_in_expr(refs, len),
    }
}

// ===== Binding metadata =====

/// A human-readable label for a [`BindingKind`].
pub fn binding_kind_label(kind: &BindingKind) -> &'static str {
    match kind {
        BindingKind::Uniform => "uniform",
        BindingKind::Storage { read_only: true } => "storage, read-only",
        BindingKind::Storage { read_only: false } => "storage, read-write",
        BindingKind::Sampler { comparison: true } => "comparison sampler",
        BindingKind::Sampler { comparison: false } => "sampler",
        BindingKind::Texture => "texture",
        BindingKind::DepthTexture => "depth texture",
        BindingKind::StorageTexture { read_write: true } => "storage texture, read-write",
        BindingKind::StorageTexture { read_write: false } => "storage texture",
    }
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
    /// A storage texture. The `read_write` flag distinguishes
    /// `read_write` access from `read`-only or `write`-only.
    StorageTexture { read_write: bool },
}

/// What kind of GPU buffer a `BufferDescriptorInfo` describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// A uniform buffer (`var<uniform>`).
    Uniform,
    /// A storage buffer (`var<storage, read[_write]>`).
    Storage { read_only: bool },
}

/// A human-readable label for a [`BufferKind`].
pub fn buffer_kind_label(kind: &BufferKind) -> &'static str {
    match kind {
        BufferKind::Uniform => "uniform",
        BufferKind::Storage { read_only: true } => "storage, read-only",
        BufferKind::Storage { read_only: false } => "storage, read-write",
    }
}

/// A compute shader entry point.
#[derive(Clone, Debug)]
pub struct ComputeEntryInfo {
    /// The entry point name (see the `EntryPointInfo::name` doc on the
    /// per-major adapter modules for the `Cow<'static, str>`
    /// rationale).
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

// ===== IR assembly (flattens imports) =====

/// Build a single owned `ir::Module` from a `wgsl_rs::Source` and all
/// its transitive imports and template instantiations. The result is
/// concrete (no `Type::TypeParam`s) and self-contained: every struct
/// referenced by a binding is defined within the assembled source.
///
/// This is the same traversal that `Source::wgsl_source()` performs,
/// but it stops at IR rather than rendering. It exists so the runtime
/// linkage analyzer can correctly size buffers whose types are defined
/// in imported sources, and it is usable standalone, with zero wgpu
/// enabled, whenever assembled IR is all you need.
pub fn assemble_ir(wgsl_source: &Source) -> Result<ir::Module, Error> {
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen: std::collections::HashSet<(u64, String, Vec<String>)> =
        std::collections::HashSet::new();
    let mut items: Vec<ir::Item> = Vec::new();
    collect_items(wgsl_source, &mut visited, &mut seen, &mut items, None)?;
    // Note: deshadowing is deferred to `analyze_ir_module`, which is the
    // sole consumer of this assembled IR. Deshadowing here too would be
    // redundant (the pass is idempotent but wastes work).
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
        | ir::Type::TextureStorage { .. }
        | ir::Type::TypeParam { .. }
        | ir::Type::Phantom { .. }
        | ir::Type::AssocType { .. } => TypeLayout { size: 0, align: 1 },
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

#[cfg(test)]
mod stage_analysis {
    //! Unit tests for the wgpu-free core: stage analysis and sizing.
    //! These compile and run in every feature combination, including a
    //! bare `--no-default-features` build with zero wgpu present.

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

    /// Looks up the analyzed stage set for the named binding; an
    /// unreferenced binding yields an empty set.
    fn stages_of(referenced: &std::collections::HashMap<String, Stages>, name: &str) -> Stages {
        referenced.get(name).copied().unwrap_or_default()
    }

    fn analyze(items: Vec<ir::Item>) -> std::collections::HashMap<String, Stages> {
        analyze_stages(&ir::Module {
            name: "test_module",
            items,
            attrs: Vec::new(),
        })
    }

    #[test]
    fn stages_insert_and_is_empty() {
        let mut s = Stages::default();
        assert!(s.is_empty());
        s.insert(Stage::Vertex);
        assert_eq!(
            s,
            Stages {
                vertex: true,
                ..Stages::default()
            }
        );
        s.insert(Stage::Fragment);
        assert!(!s.is_empty());
        assert_eq!(
            s,
            Stages {
                vertex: true,
                fragment: true,
                compute: false,
            }
        );
    }

    #[test]
    fn stage_of_attrs_maps_entry_points() {
        assert_eq!(stage_of_attrs(&ir::FnAttrs::Vertex), Some(Stage::Vertex));
        assert_eq!(
            stage_of_attrs(&ir::FnAttrs::Fragment),
            Some(Stage::Fragment)
        );
        assert_eq!(
            stage_of_attrs(&ir::FnAttrs::Compute {
                workgroup_size: ir::WorkgroupSize {
                    x: 8,
                    y: None,
                    z: None,
                }
            }),
            Some(Stage::Compute)
        );
        assert_eq!(stage_of_attrs(&ir::FnAttrs::None), None);
    }

    #[test]
    fn direct_use_gets_entry_stage() {
        let referenced = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item(
                "vert",
                ir::FnAttrs::Vertex,
                vec![returns(ident("MY_UNIFORM"))],
            ),
        ]);
        assert_eq!(
            stages_of(&referenced, "MY_UNIFORM"),
            Stages {
                vertex: true,
                ..Stages::default()
            }
        );
    }

    /// The wgsl-rs#177 repro: the binding is used only inside a helper
    /// called from both entry points, so it must carry both stages.
    #[test]
    fn helper_only_use_carries_entry_point_stages() {
        let referenced = analyze(vec![
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
            stages_of(&referenced, "MY_UNIFORM"),
            Stages {
                vertex: true,
                fragment: true,
                compute: false,
            }
        );
    }

    /// Reachability must be transitive: entry point -> helper -> helper
    /// -> binding.
    #[test]
    fn helper_chain_is_transitive() {
        let referenced = analyze(vec![
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
            stages_of(&referenced, "MY_UNIFORM"),
            Stages {
                vertex: true,
                ..Stages::default()
            }
        );
    }

    /// A declared-but-unused binding carries no stages at all; the
    /// per-major adapters map "no stages" to the documented COMPUTE
    /// visibility default.
    #[test]
    fn unused_binding_has_no_stages() {
        let referenced = analyze(vec![
            uniform("MY_UNIFORM", 0, 0),
            fn_item("vert", ir::FnAttrs::Vertex, vec![returns(zero())]),
        ]);
        assert_eq!(stages_of(&referenced, "MY_UNIFORM"), Stages::default());
    }

    /// A local (or parameter) that shadows a helper-fn name is a value
    /// reference, not a call edge (PR #190 review finding): the helper
    /// is never called, so its bindings must not gain this entry
    /// point's stage.
    #[test]
    fn local_shadowing_fn_name_is_not_a_call_edge() {
        let referenced = analyze(vec![
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
        assert_eq!(stages_of(&referenced, "DATA"), Stages::default());
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
        let referenced = analyze(vec![
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
            stages_of(&referenced, "MY_UNIFORM"),
            Stages {
                vertex: true,
                ..Stages::default()
            }
        );
    }

    #[test]
    fn type_byte_size_follows_wgsl_14_4_1() {
        let m = ir::Module {
            name: "m",
            items: vec![],
            attrs: vec![],
        };
        assert_eq!(
            type_byte_size(&ir::Type::Scalar(ir::ScalarType::F32), &m),
            4
        );
        let v3 = ir::Type::Vector {
            elements: 3,
            scalar_ty: Some(ir::ScalarType::F32),
        };
        assert_eq!(type_byte_size(&v3, &m), 12);
        let m44 = ir::Type::Matrix {
            columns: 4,
            rows: 4,
            scalar_ty: Some(ir::ScalarType::F32),
        };
        assert_eq!(type_byte_size(&m44, &m), 64);
        let rt = ir::Type::RuntimeArray {
            elem: Box::new(ir::Type::Scalar(ir::ScalarType::F32)),
        };
        assert_eq!(type_byte_size(&rt, &m), 0);
    }
}
