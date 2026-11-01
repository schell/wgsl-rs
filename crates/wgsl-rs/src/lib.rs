//! Keep your WGSL in Rust.
//!
//! This crate answers the following questions:
//! 1. How do I write shaders?
//!   * By writing Rust code in a module annotated by the `#[wgsl_rs]` macro.
//!   * It looks like this <https://renderling.xyz/wgsl-rs/manual/examples/hello-triangle.html#rust-source>.
//! 2. How do I package my shaders?
//!   * They are already packaged in that same module, automatically. No need to
//!     deal with shader source files. Instead, you access the module at
//!     runtime.
//! 3. How do I run my shaders?
//!   * That's still up to you, but if your runtime is [`wgpu`](https://crates.io/crates/wgpu),
//!     then with the [`wgsl-rs-linkage-wgpu`](https://crates.io/crates/wgsl-rs-linkage-wgpu)
//!     crate you can avoid a lot of boiler plate by having your [linkage auto-generated](https://renderling.xyz/wgsl-rs/manual/linkage/overview.html?highlight=generate_linkage#high-level-workflow).
//!
//! For more info see the [Operator's Manual](https://renderling.xyz/wgsl-rs/manual/introduction.html).
use ::std::collections::{HashMap, HashSet};

pub use wgsl_rs_macros::{wgsl, wgsl_allow, wgsl_ignore};

pub mod extension;
pub mod linkage;
pub mod source_error;

pub use extension::{WgslExtension, any_ext_claims, contains_str};

pub use source_error::SourceError;

/// Re-export of the IR crate so the proc-macro and consuming crates can
/// reference IR types via `wgsl_rs::ir::...` without needing to depend on
/// `wgsl-rs-ir` directly.
pub use wgsl_rs_ir as ir;

/// A WGSL source.
///
/// A `Source` is the public-facing spec for a `#[wgsl] mod foo { ... }`
/// block: the macro emits one `pub static foo::WGSL_SOURCE: Source` per
/// module. WGSL itself doesn't support importing modules, but `wgsl-rs`
/// does (glob imports only), so a `Source` can import other `Source`s.
///
/// Concrete and template sources share the same type. Templates are
/// detected at runtime via [`Source::is_template`]; a template source's
/// [`Source::wgsl_source`] returns
/// [`SourceError::TemplateWgsl`] and the caller must use the
/// macro-emitted `instantiate::<…>()` to obtain a concrete
/// [`ir::Module`] first.
///
/// ```rust, ignore
/// #[wgsl]
/// pub mod constants {
///     pub const NUMBER: u32 = 1234;
/// }
/// #[wgsl]
/// pub mod main {
///     use super::constants::*;
///
///     fn add_to_number(n: u32) -> u32 {
///         n + NUMBER
///     }
/// }
#[derive(Clone, Debug)]
pub struct Source {
    /// Identifier for this source, assigned at proc-macro expansion
    /// time from a stable hash of the module's content (its IR
    /// rendering, name, imports, templates, and instantiations) mixed
    /// with the consuming compilation unit's identity (Cargo package
    /// name + version, plus crate/bin target names; wgsl-rs#165,
    /// wgsl-rs#221). Deterministic across builds: identical module
    /// source yields the same id, so adding or moving sibling `#[wgsl]`
    /// mods does not shift it. Distinct content yields distinct ids
    /// probabilistically (64-bit hash). Identical content is
    /// interchangeable, so assembly dedup (diamond import graphs,
    /// same-named sources from different paths) is safe.
    pub id: u64,

    /// Name of the source.
    pub name: &'static str,

    /// Imports of other WGSL sources.
    pub imports: &'static [&'static Source],

    /// Constructor function that builds this module's IR. Called lazily by
    /// `wgsl_source()` / `instantiate()`.
    ///
    /// The returned IR is owned and freshly constructed on each call; the
    /// caller is free to clone, mutate, or substitute over it.
    pub ir_constructor: fn() -> ir::Module,

    /// Generic function templates defined in this module.
    ///
    /// Consuming modules reference these via `instantiations` to produce
    /// monomorphized functions at source-assembly time. The IR is held as
    /// a constructor, just like the parent module's IR.
    pub templates: &'static [GenericTemplate],

    /// Cross-module template instantiations.
    ///
    /// Each entry references a template from an imported module and provides
    /// the concrete IR type arguments to substitute for the type parameters.
    /// These are resolved at source-assembly time in `wgsl_source()`.
    pub instantiations: &'static [TemplateInstantiation],

    /// Module-level type parameters, in instantiation order.
    ///
    /// Two distinct kinds of declaration contribute to this list:
    ///
    /// * `impl Trait` linkage variables, e.g. `uniform!(group(0), binding(0),
    ///   FRAME: impl Convert<f32>)`. Each such declaration contributes the
    ///   variable's identifier (`"FRAME"`).
    /// * Generic shader entry points, e.g. `pub fn frag_main<T: Convert<f32>>()
    ///   -> Vec4f`. Each one of an entry point's type parameters contributes a
    ///   positional name of the form `"<fn_name>_<index>"` (e.g.
    ///   `"frag_main_0"`). Source names like `T` are intentionally not used so
    ///   that two entry points sharing a letter don't collide.
    ///
    /// The two groups appear in declaration order: linkage variables
    /// first, then entry points.
    ///
    /// When this slice is empty, the module is concrete and
    /// [`Self::wgsl_source`] produces a directly usable WGSL source.
    /// When it is non-empty, callers should use the `instantiate`
    /// function generated alongside the source, which uses
    /// `wgsl_rs::linkage::Type<Is = ...>` constraints to check at
    /// compile time that every parameter has been bound.
    pub module_type_params: &'static [&'static str],
}

/// A generic function (or struct) template defined within a module.
///
/// The IR is built lazily by calling `ir_constructor`. The returned items
/// reference unresolved [`ir::Type::TypeParam`] nodes whose names match
/// entries in `type_params` and `Expr::Ident` nodes whose names match
/// entries in `const_params`; substitution at instantiation time replaces
/// them with concrete types and values.
#[derive(Debug)]
pub struct GenericTemplate {
    /// The generic function or struct name (e.g., `"id"`, `"Pair"`).
    pub name: &'static str,
    /// Type parameter names (e.g., `["T"]`, `["M", "L", "N"]`).
    pub type_params: &'static [&'static str],
    /// `const` parameter names (e.g., `["N"]` for `const N: u32`).
    pub const_params: &'static [&'static str],
    /// Constructor that builds the un-instantiated IR for this template.
    pub ir_constructor: fn() -> Vec<ir::Item>,
    /// Transitive generic function calls within this template.
    ///
    /// Each entry records a call to another generic function, with a mapping
    /// from the callee's type params back to this template's type params.
    pub dependencies: &'static [TemplateDependency],
}

/// A transitive dependency from one generic template to another.
#[derive(Clone, Debug)]
pub struct TemplateDependency {
    /// Name of the generic function being called.
    pub callee: &'static str,
    /// Maps each of the callee's type params to one of the caller's type
    /// params, by index. For example, if `quadruple<T>` calls `double::<T>()`,
    /// the mapping is `&[0]` (double's 0th param = quadruple's 0th param).
    pub type_param_mapping: &'static [usize],
}

/// A request to instantiate a generic template with concrete types.
#[derive(Debug)]
pub struct TemplateInstantiation {
    /// Candidate imported sources that may contain the template.
    ///
    /// Resolution must find exactly one matching source; missing or ambiguous
    /// matches are treated as hard failures during `wgsl_source()` assembly.
    pub modules: &'static [&'static Source],
    /// The generic function name to instantiate.
    pub template_name: &'static str,
    /// Constructor that builds the concrete IR types to substitute for the
    /// template's type parameters.
    pub type_args_constructor: fn() -> Vec<ir::Type>,
    /// Identifier-safe mangled type argument names, used for deduplication
    /// keys and the `seen` set (e.g. `"array_f32_4"`, `"ptr_function_f32"`).
    pub mangled_type_args: &'static [&'static str],
    /// Constructor that builds the concrete `u32` const values to
    /// substitute for the template's const parameters.
    pub const_args_constructor: fn() -> Vec<u32>,
    /// Identifier-safe mangled const argument names (e.g. `["4"]`).
    pub mangled_const_args: &'static [&'static str],
}

impl Source {
    /// Returns whether this source is a template — i.e. has unresolved
    /// module-level type parameters.
    ///
    /// Template sources cannot be used as shader sources directly; call
    /// `instantiate(...)` to produce a concrete WGSL source.
    pub fn is_template(&self) -> bool {
        !self.module_type_params.is_empty()
    }

    /// Returns the assembled WGSL source for this source and all its
    /// imports, including cross-source template instantiations.
    ///
    /// If this source is a template, the returned error is
    /// [`SourceError::TemplateWgsl`]; the caller must use the
    /// macro-emitted `instantiate::<…>()` to obtain a concrete
    /// [`ir::Module`] first and then call [`ir::Module::wgsl_source`].
    pub fn wgsl_source(&self) -> Result<String, SourceError<'_>> {
        if self.is_template() {
            return Err(SourceError::TemplateWgsl {
                uninstantiated_source: self,
            });
        }
        // Harvest every template name reachable through the import graph
        // (including templates reachable only via instantiations) so the
        // assembly pass can flag references that were never resolved to
        // a concrete instantiation (wgsl-rs#237). Cross-module references
        // cannot be checked at macro time — the consuming module cannot
        // know that an imported name is generic.
        let mut template_names: HashSet<String> = HashSet::new();
        let mut visited: HashSet<u64> = HashSet::new();
        collect_template_names(self, &mut template_names, &mut visited);
        let mut out = String::new();
        let mut visited_sources: HashSet<u64> = HashSet::new();
        let mut seen: HashSet<(u64, String, Vec<String>, Vec<String>)> = HashSet::new();
        let mut needs_tier1 = false;
        let mut imports = ir::TypeImports::default();
        // Top-level struct and function names actually emitted by
        // rendered source chunks (see `instantiate_template_into`).
        let mut emitted_names: HashSet<String> = HashSet::new();
        self.collect(
            &mut out,
            &mut visited_sources,
            &mut seen,
            None,
            &mut needs_tier1,
            &mut imports,
            &template_names,
            &mut emitted_names,
        )?;
        // WGSL requires `enable` directives at the very start of the
        // translation unit, before any other declarations. Since `collect`
        // assembles multiple modules / instantiations by concatenation, we
        // prepend the directive here after all chunks have been rendered.
        if needs_tier1 {
            out.insert_str(0, "enable texture_formats_tier1;\n\n");
        }
        Ok(out)
    }

    /// Recursively collect this source's WGSL source (and that of its
    /// imports / instantiations) into `out`. When `subst` is `Some`, the
    /// caller's substitution map is applied to *this* source's IR (but not
    /// to imported sources — imports are concrete by construction).
    ///
    /// `visited_sources` tracks source IDs that have already been emitted,
    /// preventing duplicate definitions in diamond import graphs.
    /// `seen` tracks `(source_id, template_name, mangled_type_args,
    /// mangled_const_args)` 4-tuples to deduplicate cross-source template
    /// instantiations. Including const args ensures `foo::<4>` and
    /// `foo::<8>` aren't incorrectly deduplicated.
    /// `imports` accumulates every rendered source's harvested type
    /// environment — globals (consts, linkage declarations), struct
    /// definitions, and user-fn signatures — so later sources anchor
    /// from imported names too (calls, consts, linkage accesses,
    /// struct constructors) via
    /// [`ir::suffix_module_with_type_imports`], with the caller's own
    /// declarations shadowing imported ones.
    #[allow(clippy::too_many_arguments)]
    fn collect(
        &self,
        out: &mut String,
        visited_sources: &mut HashSet<u64>,
        seen: &mut HashSet<(u64, String, Vec<String>, Vec<String>)>,
        subst: Option<&HashMap<String, ir::Type>>,
        needs_tier1: &mut bool,
        imports: &mut ir::TypeImports,
        template_names: &HashSet<String>,
        emitted_names: &mut HashSet<String>,
    ) -> Result<(), SourceError<'_>> {
        // 1. Imports first (depth-first, deduplicated by source ID).
        for m in self.imports {
            if visited_sources.insert(m.id) {
                m.collect(
                    out,
                    visited_sources,
                    seen,
                    None,
                    needs_tier1,
                    imports,
                    template_names,
                    emitted_names,
                )?;
            }
        }

        // 2. Build this source's IR, optionally substituting type params.
        let mut ir_module = (self.ir_constructor)();
        if let Some(s) = subst {
            ir::substitute_types(&mut ir_module, s);
        }
        ir::deshadow_module(&mut ir_module);
        // Flag references to generic templates that were never resolved
        // (wgsl-rs#237): a bare `Id::new(...)` or `Id::NONE` here would
        // render an un-mangled name and fail WGSL validation far from
        // the authoring mistake.
        if let Some(offense) = ir::find_unresolved_template_refs(&ir_module.items, template_names) {
            return Err(SourceError::UnresolvedTemplateRef {
                source_name: self.name.to_string(),
                template_name: offense.template,
                member: offense.member,
                has_type_args: offense.has_type_args,
            });
        }
        // Record this chunk's top-level definitions so later instance
        // chunks can deduplicate against them: a provider's own
        // compile-time monomorphization may already contain the very
        // instantiation a consumer requests.
        for item in &ir_module.items {
            match item {
                ir::Item::Struct(s) => {
                    emitted_names.insert(s.name.to_string());
                }
                ir::Item::Fn(f) => {
                    emitted_names.insert(f.name.to_string());
                }
                _ => {}
            }
        }
        // Pre-seed with the types of cross-source template instances
        // this source calls: instance chunks are appended (and publish
        // their types) after this chunk is suffixed, so a call like
        // `choose::<u32>(select(0, 1, data))` must find the
        // instance's concrete parameter types now.
        for inst in self.instantiations {
            let mangled: Vec<String> = inst
                .mangled_type_args
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let type_args = (inst.type_args_constructor)();
            let mangled_consts: Vec<String> = inst
                .mangled_const_args
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let const_args = (inst.const_args_constructor)();
            imports.extend(instance_type_imports(
                inst.modules,
                inst.template_name,
                &type_args,
                &const_args,
                &mangled,
                &mangled_consts,
            ));
        }
        // Make this source's own globals, structs, and signatures
        // available to later sources, then suffix with every
        // ancestor's types seeded — references to imported names
        // anchor (calls, consts, linkage accesses, struct
        // constructors), and a caller's own declarations shadow
        // imported ones (matching Rust name resolution).
        imports.extend(ir::type_imports(&ir_module));
        ir::suffix_module_with_type_imports(&mut ir_module, imports);
        if ir::items_need_tier1_extension(&ir_module.items) {
            *needs_tier1 = true;
        }
        out.push_str(&ir::render_module(&ir_module));

        // 3. Cross-source template instantiations.
        for inst in self.instantiations {
            let mangled: Vec<String> = inst
                .mangled_type_args
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let type_args = (inst.type_args_constructor)();
            let mangled_consts: Vec<String> = inst
                .mangled_const_args
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let const_args = (inst.const_args_constructor)();
            instantiate_template_into(
                inst.modules,
                inst.template_name,
                &mangled,
                &type_args,
                &mangled_consts,
                &const_args,
                out,
                seen,
                needs_tier1,
                imports,
                template_names,
                emitted_names,
            )?;
        }
        Ok(())
    }
}

/// Harvest every generic template name reachable from `source` through
/// the import graph, including templates reachable only through recorded
/// instantiations. Used by source assembly to detect references that were
/// never resolved to a concrete instantiation (wgsl-rs#237).
fn collect_template_names(
    source: &Source,
    names: &mut HashSet<String>,
    visited: &mut HashSet<u64>,
) {
    if !visited.insert(source.id) {
        return;
    }
    for template in source.templates {
        names.insert(template.name.to_string());
    }
    for import in source.imports {
        collect_template_names(import, names, visited);
    }
    for inst in source.instantiations {
        for module in inst.modules {
            collect_template_names(module, names, visited);
        }
    }
}

/// Resolve `template_name` from the given candidate sources, recursively
/// instantiate its dependencies, then render its substituted IR into
/// `out`. Tracks `(source_id, name, mangled_type_args)` triples in `seen`
/// to avoid duplicate emission.
///
/// Returns `Err` with a structured [`SourceError`] for user-facing
/// configuration errors (template not found, ambiguous match). Returns
/// `Ok` for a successful instantiation, or for the early-return dedup case.
#[allow(clippy::too_many_arguments)]
fn instantiate_template_into<'a>(
    sources: &[&'a Source],
    template_name: &str,
    mangled_type_args: &[String],
    type_args: &[ir::Type],
    mangled_const_args: &[String],
    const_args: &[u32],
    out: &mut String,
    seen: &mut HashSet<(u64, String, Vec<String>, Vec<String>)>,
    needs_tier1: &mut bool,
    imports: &mut ir::TypeImports,
    template_names: &HashSet<String>,
    emitted_names: &mut HashSet<String>,
) -> Result<(), SourceError<'a>> {
    let (source, template) = resolve_template(sources, template_name, mangled_type_args)?;

    let key = (
        source.id,
        template_name.to_string(),
        mangled_type_args.to_vec(),
        mangled_const_args.to_vec(),
    );
    if !seen.insert(key) {
        return Ok(()); // Already instantiated
    }

    // Recursively instantiate dependencies first.
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
        // Dependencies currently share the caller's const args by
        // reference; a full const-param-mapping parallel to
        // `type_param_mapping` is a future extension. For now we pass
        // the const args through unchanged.
        instantiate_template_into(
            &[source],
            dep.callee,
            &dep_mangled,
            &dep_args,
            mangled_const_args,
            const_args,
            out,
            seen,
            needs_tier1,
            imports,
            template_names,
            emitted_names,
        )?;
    }

    let mut items = build_instance_items(
        template,
        type_args,
        const_args,
        mangled_type_args,
        mangled_const_args,
    );

    // Cross-chunk dedup: the provider's own source chunk may already
    // contain this very instantiation — compile-time monomorphization
    // emits concrete items into the provider's IR, and rendering the
    // runtime instance again would double-define the struct (or fn).
    // The content is identical, so the whole chunk is skipped. The
    // check is against names *emitted by rendered chunks*
    // (`emitted_names`), not against `imports`, which is deliberately
    // pre-seeded with instance types before this chunk renders.
    let already_emitted = items.iter().find_map(|item| match item {
        ir::Item::Struct(s) => Some(emitted_names.contains(&s.name)),
        ir::Item::Fn(f) => Some(emitted_names.contains(f.name.as_ref())),
        _ => None,
    });
    if already_emitted == Some(true) {
        return Ok(());
    }

    ir::deshadow_items(&mut items);
    // Flag references to *other* generic templates that this instance's
    // body could not resolve (e.g. a generic fn body calling
    // `Id::<T>::new(...)` — the instance rename only resolves the
    // template's own name, wgsl-rs#237). Without this the broken name
    // reaches WGSL validation as an un-mangled identifier.
    if let Some(offense) = ir::find_unresolved_template_refs(&items, template_names) {
        return Err(SourceError::UnresolvedTemplateRef {
            source_name: source.name.to_string(),
            template_name: offense.template,
            member: offense.member,
            has_type_args: offense.has_type_args,
        });
    }
    // Seed with every ancestor's types (imports and already-
    // instantiated dependency templates), then publish this instance's
    // own (renamed) types for any later chunk.
    ir::suffix_items_with_type_imports(&mut items, imports);
    imports.extend(ir::type_imports_in_items(&items));
    if ir::items_need_tier1_extension(&items) {
        *needs_tier1 = true;
    }
    out.push_str(&ir::render_items(&items));
    Ok(())
}

/// Collect `sources` plus every `Source` reachable from them through
/// the import graph, deduplicated by [`Source::id`] and safe against
/// import cycles.
///
/// Module ids are content hashes (see [`Source::id`]), so a defining
/// module reached through two different re-export paths appears once
/// here — diamond imports cannot manufacture a template ambiguity.
/// Only `Source::imports` is walked: the macro records an
/// instantiation's candidate modules as the importing module's `use`
/// paths, and those same paths also feed `imports`, so the import
/// graph reaches every module a source could have imported from
/// (wgsl-rs#238).
fn reachable_sources<'a>(sources: &[&'a Source]) -> Vec<&'a Source> {
    /// Depth-first visit with an id-based cycle/dedup guard.
    fn visit<'a>(source: &'a Source, out: &mut Vec<&'a Source>, visited: &mut HashSet<u64>) {
        if !visited.insert(source.id) {
            return;
        }
        out.push(source);
        for import in source.imports {
            visit(import, out, visited);
        }
    }

    let mut out = Vec::new();
    let mut visited = HashSet::new();
    for &source in sources {
        visit(source, &mut out, &mut visited);
    }
    out
}

/// Resolve `template_name` to exactly one template among `sources` or
/// any module reachable from them through the import graph, producing
/// the structured errors `instantiate_template_into` reports.
///
/// Searching transitively makes generic templates available through
/// `pub use` re-export chains: a re-exporting module's own template
/// registry is empty, but the defining module is reachable via the
/// re-exporter's `imports` (wgsl-rs#238). A macro-side fix is not
/// possible: proc macros cannot introspect what a sibling module
/// re-exports, so only the literal `use` paths can be recorded as
/// candidates.
fn resolve_template<'a>(
    sources: &[&'a Source],
    template_name: &str,
    mangled_type_args: &[String],
) -> Result<(&'a Source, &'a GenericTemplate), SourceError<'a>> {
    let reachable = reachable_sources(sources);

    let available_templates: Vec<String> = reachable
        .iter()
        .copied()
        .flat_map(|m| m.templates.iter().map(|t| t.name.to_string()))
        .collect();

    let mut matching_sources: Vec<&Source> = reachable
        .into_iter()
        .filter(|m| m.templates.iter().any(|t| t.name == template_name))
        .collect();

    if matching_sources.is_empty() {
        return Err(SourceError::TemplateNotFound {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            available_templates,
        });
    }

    if matching_sources.len() > 1 {
        let source_names: Vec<String> = matching_sources
            .iter()
            .map(|m| m.name.to_string())
            .collect();
        return Err(SourceError::AmbiguousTemplate {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            matching_source_names: source_names,
            available_templates,
        });
    }

    let source = matching_sources
        .pop()
        .expect("matching_sources is guaranteed to be non-empty after checks");

    let Some(template) = source.templates.iter().find(|t| t.name == template_name) else {
        // Invariant: `matching_sources` was built by filtering on
        // `templates.iter().any(|t| t.name == template_name)`, so the
        // sole surviving source must contain the template.
        unreachable!(
            "resolved source '{}' does not contain template '{}'; available templates: {:?}",
            source.name, template_name, available_templates
        );
    };

    Ok((source, template))
}

/// Build a template instance's items: run the template's IR
/// constructor, substitute its type and const parameters, and rename
/// the instance to its mangled name. Uses `ir::mangle` so that names
/// with underscores in either the template name or the type-arg-mangled
/// strings are escaped unambiguously (see issue #112).
fn build_instance_items(
    template: &GenericTemplate,
    type_args: &[ir::Type],
    const_args: &[u32],
    mangled_type_args: &[String],
    mangled_const_args: &[String],
) -> Vec<ir::Item> {
    // Build substitution maps: type params -> concrete types, const
    // params -> concrete u32 values.
    let mut subst: HashMap<String, ir::Type> = HashMap::new();
    for (param, arg) in template.type_params.iter().zip(type_args.iter()) {
        subst.insert((*param).to_string(), arg.clone());
    }
    let mut consts: HashMap<String, u32> = HashMap::new();
    for (param, arg) in template.const_params.iter().zip(const_args.iter()) {
        consts.insert((*param).to_string(), *arg);
    }

    let mut items = (template.ir_constructor)();
    ir::substitute_items(&mut items, &subst);
    ir::substitute_consts_in_items(&mut items, &consts);

    let instance_name = if mangled_type_args.is_empty() && mangled_const_args.is_empty() {
        template.name.to_string()
    } else {
        let mut components: Vec<String> =
            Vec::with_capacity(1 + mangled_type_args.len() + mangled_const_args.len());
        components.push(template.name.to_string());
        for s in mangled_type_args {
            components.push(s.clone());
        }
        for s in mangled_const_args {
            components.push(s.clone());
        }
        let str_components: Vec<&str> = components.iter().map(|s| s.as_str()).collect();
        ir::mangle(&str_components)
    };
    if instance_name != template.name {
        ir::rename_items(&mut items, template.name, &instance_name);
    }
    items
}

/// Harvest the types a template instance would export, for
/// pre-seeding the type-import accumulator before a calling source's
/// chunk is suffixed — instance chunks are appended (and publish their
/// types) after the caller is suffixed, so a call like
/// `choose::<u32>(select(0, 1, data))` must find the instance's
/// concrete parameter types now.
///
/// Resolution failures are ignored here:
/// [`instantiate_template_into`] reports them with full context when
/// the instance is actually rendered.
fn instance_type_imports(
    sources: &[&Source],
    template_name: &str,
    type_args: &[ir::Type],
    const_args: &[u32],
    mangled_type_args: &[String],
    mangled_const_args: &[String],
) -> ir::TypeImports {
    let Ok((_source, template)) = resolve_template(sources, template_name, mangled_type_args)
    else {
        return ir::TypeImports::default();
    };
    let items = build_instance_items(
        template,
        type_args,
        const_args,
        mangled_type_args,
        mangled_const_args,
    );
    ir::type_imports_in_items(&items)
}

#[cfg(feature = "validation")]
impl Source {
    /// Validates the concatenated WGSL source of this source and its imports.
    ///
    /// This method concatenates the WGSL source from this source and all its
    /// imports (recursively), then validates the result using naga.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if validation succeeds, or an error message
    /// describing the validation failure. Template sources return an
    /// `Err(String)` instructing the caller to `instantiate::<…>()`
    /// first (they cannot be validated standalone — their IR contains
    /// `__TP{name}__` placeholders that aren't valid WGSL).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[wgsl]
    /// pub mod my_shader {
    ///     // ...
    /// }
    ///
    /// // Validate at runtime
    /// my_shader::WGSL_SOURCE.validate().expect("WGSL validation failed");
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // Template sources contain unresolved `__TP{name}__` placeholders
        // that aren't valid WGSL identifiers. They cannot be validated
        // standalone — the user must call `instantiate(...)` first.
        if self.is_template() {
            return Err(
                "cannot validate a template source — call instantiate(...) first".to_string(),
            );
        }

        validate_wgsl_source(&self.wgsl_source().expect("concrete source"))
    }
}

/// Validates arbitrary WGSL source text using naga.
///
/// Parses the source, then runs naga's semantic validation. This is useful
/// for validating the output of `instantiate::<...>()` + `ir::render_module()`
/// for generic (template) sources.
///
/// # Returns
///
/// Returns `Ok(())` if validation succeeds, or an error message describing
/// the validation failure.
#[cfg(feature = "validation")]
pub fn validate_wgsl_source(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(source))?;
    Ok(())
}

pub mod std;

#[cfg(test)]
mod std_splats_test {
    #[crate::wgsl(crate_path = crate)]
    pub mod uses_std_splats {
        // The std glob import alone is sufficient in both worlds: the
        // splat and extend calls lower to WGSL constructors through the
        // function mapping table, so no helper functions are assembled.
        use crate::std::*;

        pub fn splat_and_extend() -> Vec4f {
            let v = vec4f(1.0, 2.0, 3.0, 4.0);
            let w = vec4f_splat(2.0);
            let e = vec4f_from_3(vec3f_from_2(vec2f(5.0, 6.0), 7.0), 8.0);
            v + w + e
        }
    }

    #[test]
    fn splats_and_extends_work_in_both_worlds() {
        // CPU world: plain std functions, reached through the std glob.
        assert_eq!(
            uses_std_splats::splat_and_extend().to_array(),
            [8.0, 10.0, 12.0, 14.0]
        );

        // WGSL world: the calls lower to WGSL's splat and mixed
        // constructors; no helper functions are rendered.
        let src = uses_std_splats::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(src.contains("vec4f(2.0)"), "splat lowered: {src}");
        assert!(
            src.contains("vec3f(vec2f(5.0, 6.0), 7.0)"),
            "extend lowered to the mixed constructor: {src}"
        );
        assert!(
            !src.contains("_splat") && !src.contains("_from_"),
            "no helper functions rendered: {src}"
        );
        #[cfg(feature = "validation")]
        uses_std_splats::WGSL_SOURCE.validate().unwrap();
    }
}

#[cfg(feature = "glm")]
#[crate::wgsl(crate_path = crate)]
pub mod glm {
    //! GLM-lineage helpers over the WGSL vector and matrix types.
    //!
    //! Lineage: GLSL → GLM (C++) → glam (Rust) → this module.
    //! [`crate::std`] already provides the WGSL builtin surface as free
    //! functions (`length`, `normalize`, `mix`, `dot`, `cross`, ...), so
    //! this module only adds what GLM added on top of GLSL, as free
    //! functions over the concrete vector types (`Vec3f` etc.).
    //!
    //! In `#[wgsl]` modules, import this module directly; it re-exports
    //! [`crate::std`], so this one glob brings the whole surface:
    //!
    //! ```ignore
    //! use wgsl_rs::glm::*;
    //! ```
    use crate::std::*;

    // Re-export std so a single glob import brings the helpers and the
    // WGSL builtin surface together.
    pub use crate::std::*;

    // -- Length

    /// Returns the squared length of a 2D vector.
    ///
    /// No square root: exact under CPU/GPU roundtrip, unlike
    /// `length(v) * length(v)`.
    pub fn length_squared2(v: Vec2f) -> f32 {
        dot(v, v)
    }

    /// Returns the squared length of a 3D vector.
    ///
    /// No square root: exact under CPU/GPU roundtrip, unlike
    /// `length(v) * length(v)`.
    pub fn length_squared3(v: Vec3f) -> f32 {
        dot(v, v)
    }

    /// Returns the squared length of a 4D vector.
    ///
    /// No square root: exact under CPU/GPU roundtrip, unlike
    /// `length(v) * length(v)`.
    pub fn length_squared4(v: Vec4f) -> f32 {
        dot(v, v)
    }

    // -- Normalization

    /// Returns `v` normalized, or the zero vector if `v` has zero length.
    ///
    /// Unlike WGSL's `normalize`, never produces NaN for zero input.
    pub fn normalize_or_zero2(v: Vec2f) -> Vec2f {
        select(Vec2f::ZERO, normalize(v), length(v) > 0.0)
    }

    /// Returns `v` normalized, or the zero vector if `v` has zero length.
    ///
    /// Unlike WGSL's `normalize`, never produces NaN for zero input.
    pub fn normalize_or_zero3(v: Vec3f) -> Vec3f {
        select(Vec3f::ZERO, normalize(v), length(v) > 0.0)
    }

    /// Returns `v` normalized, or the zero vector if `v` has zero length.
    ///
    /// Unlike WGSL's `normalize`, never produces NaN for zero input.
    pub fn normalize_or_zero4(v: Vec4f) -> Vec4f {
        select(Vec4f::ZERO, normalize(v), length(v) > 0.0)
    }

    // -- Angles

    /// Returns the angle in radians between two 2D vectors.
    ///
    /// Uses the `atan2` form so near-parallel vectors never hit `acos`
    /// domain errors. Returns 0.0 if either vector is zero: the angle is
    /// undefined there, and the raw `atan2(0, 0)` differs between the
    /// worlds (0 on CPU, NaN on Metal), so it is selected away.
    pub fn angle_between2(a: Vec2f, b: Vec2f) -> f32 {
        let defined = length_squared2(a) * length_squared2(b) > 0.0;
        select(0.0, atan2(a.x * b.y - a.y * b.x, dot(a, b)), defined)
    }

    /// Returns the angle in radians between two 3D vectors.
    ///
    /// Uses the `atan2` form so near-parallel vectors never hit `acos`
    /// domain errors. Returns 0.0 if either vector is zero: the angle is
    /// undefined there, and the raw `atan2(0, 0)` differs between the
    /// worlds (0 on CPU, NaN on Metal), so it is selected away.
    pub fn angle_between3(a: Vec3f, b: Vec3f) -> f32 {
        let defined = length_squared3(a) * length_squared3(b) > 0.0;
        select(0.0, atan2(length(cross(a, b)), dot(a, b)), defined)
    }

    // -- Matrix transforms
    //
    // Column-major, matching glam. Handedness and depth range are
    // explicit per function: `_rh`/`_lh` handedness suffixes, `_gl` for
    // the OpenGL-style -1..1 depth range; the non-`_gl` projections use
    // the wgpu-native 0..1 depth range.

    /// Builds a translation matrix that translates by `v`.
    pub fn translation(v: Vec3f) -> Mat4x4f {
        mat4x4f(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(v.x, v.y, v.z, 1.0),
        )
    }

    /// Builds a scaling matrix that scales by `v`.
    pub fn scaling(v: Vec3f) -> Mat4x4f {
        mat4x4f(
            vec4f(v.x, 0.0, 0.0, 0.0),
            vec4f(0.0, v.y, 0.0, 0.0),
            vec4f(0.0, 0.0, v.z, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Builds a rotation matrix around `axis` by `angle` radians
    /// (Rodrigues' formula), matching glam's `from_axis_angle`.
    ///
    /// `axis` must be normalized.
    pub fn rotation_axis_angle(axis: Vec3f, angle: f32) -> Mat4x4f {
        let s = sin(angle);
        let c = cos(angle);
        let dc = 1.0 - c;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;
        mat4x4f(
            vec4f(x * x * dc + c, y * x * dc + z * s, z * x * dc - y * s, 0.0),
            vec4f(x * y * dc - z * s, y * y * dc + c, z * y * dc + x * s, 0.0),
            vec4f(x * z * dc + y * s, y * z * dc - x * s, z * z * dc + c, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Post-multiplies a translation onto `m` (`m * translation(v)`),
    /// GLM's `translate`.
    pub fn translate(m: Mat4x4f, v: Vec3f) -> Mat4x4f {
        m * translation(v)
    }

    /// Post-multiplies a rotation onto `m`
    /// (`m * rotation_axis_angle(axis, angle)`), GLM's `rotate`.
    pub fn rotate(m: Mat4x4f, angle: f32, axis: Vec3f) -> Mat4x4f {
        m * rotation_axis_angle(axis, angle)
    }

    /// Post-multiplies a scale onto `m` (`m * scaling(v)`), GLM's
    /// `scale`.
    pub fn scale(m: Mat4x4f, v: Vec3f) -> Mat4x4f {
        m * scaling(v)
    }

    /// Transforms a 2D point by a `Mat3x3f`, matching glam's
    /// `Mat3::transform_point2`.
    pub fn transform_point2(m: Mat3x3f, v: Vec2f) -> Vec2f {
        (m * vec3f(v.x, v.y, 1.0)).xy()
    }

    /// Transforms a 2D direction vector by a `Mat3x3f` (no translation),
    /// matching glam's `Mat3::transform_vector2`.
    pub fn transform_vector2(m: Mat3x3f, v: Vec2f) -> Vec2f {
        (m * vec3f(v.x, v.y, 0.0)).xy()
    }

    /// Transforms a 3D point by a `Mat4x4f`, matching glam's
    /// `Mat4::transform_point3`.
    ///
    /// Affine: no perspective divide. For the projected form multiply
    /// by a `vec4f` directly.
    pub fn transform_point3(m: Mat4x4f, v: Vec3f) -> Vec3f {
        (m * vec4f(v.x, v.y, v.z, 1.0)).xyz()
    }

    /// Transforms a 3D direction vector by a `Mat4x4f` (no translation),
    /// matching glam's `Mat4::transform_vector3`.
    pub fn transform_vector3(m: Mat4x4f, v: Vec3f) -> Vec3f {
        (m * vec4f(v.x, v.y, v.z, 0.0)).xyz()
    }

    /// Builds a right-handed view matrix looking from `eye` at `center`
    /// with `up`, matching glam's `look_at_rh`: the camera looks down
    /// its negative Z axis.
    pub fn look_at_rh(eye: Vec3f, center: Vec3f, up: Vec3f) -> Mat4x4f {
        let z_axis = normalize(eye - center);
        let x_axis = normalize(cross(up, z_axis));
        let y_axis = cross(z_axis, x_axis);
        mat4x4f(
            vec4f(x_axis.x, y_axis.x, z_axis.x, 0.0),
            vec4f(x_axis.y, y_axis.y, z_axis.y, 0.0),
            vec4f(x_axis.z, y_axis.z, z_axis.z, 0.0),
            vec4f(-dot(x_axis, eye), -dot(y_axis, eye), -dot(z_axis, eye), 1.0),
        )
    }

    /// Builds a left-handed view matrix looking from `eye` at `center`
    /// with `up`, matching glam's `look_at_lh`: the camera looks down
    /// its positive Z axis.
    pub fn look_at_lh(eye: Vec3f, center: Vec3f, up: Vec3f) -> Mat4x4f {
        let z_axis = normalize(center - eye);
        let x_axis = normalize(cross(up, z_axis));
        let y_axis = cross(z_axis, x_axis);
        mat4x4f(
            vec4f(x_axis.x, y_axis.x, z_axis.x, 0.0),
            vec4f(x_axis.y, y_axis.y, z_axis.y, 0.0),
            vec4f(x_axis.z, y_axis.z, z_axis.z, 0.0),
            vec4f(-dot(x_axis, eye), -dot(y_axis, eye), -dot(z_axis, eye), 1.0),
        )
    }

    // -- Projections
    //
    // `fov_y_radians` is the vertical field of view; `near` and `far`
    // are positive clip distances. The non-`_gl` variants use the
    // 0..1 depth range (Vulkan/D3D/wgpu); `_gl` variants use -1..1.
    // The set mirrors glam 0.30, which has no `_lh_gl` variants.

    /// Builds a right-handed perspective projection with a 0..1 depth
    /// range, matching glam's `perspective_rh`.
    pub fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, near: f32, far: f32) -> Mat4x4f {
        let f = 1.0 / tan(fov_y_radians * 0.5);
        mat4x4f(
            vec4f(f / aspect_ratio, 0.0, 0.0, 0.0),
            vec4f(0.0, f, 0.0, 0.0),
            vec4f(0.0, 0.0, far / (near - far), -1.0),
            vec4f(0.0, 0.0, (far * near) / (near - far), 0.0),
        )
    }

    /// Builds a left-handed perspective projection with a 0..1 depth
    /// range, matching glam's `perspective_lh`.
    pub fn perspective_lh(fov_y_radians: f32, aspect_ratio: f32, near: f32, far: f32) -> Mat4x4f {
        let f = 1.0 / tan(fov_y_radians * 0.5);
        mat4x4f(
            vec4f(f / aspect_ratio, 0.0, 0.0, 0.0),
            vec4f(0.0, f, 0.0, 0.0),
            vec4f(0.0, 0.0, far / (far - near), 1.0),
            vec4f(0.0, 0.0, (far * near) / (near - far), 0.0),
        )
    }

    /// Builds a right-handed perspective projection with a -1..1 depth
    /// range, matching glam's `perspective_rh_gl`.
    pub fn perspective_rh_gl(
        fov_y_radians: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Mat4x4f {
        let f = 1.0 / tan(fov_y_radians * 0.5);
        mat4x4f(
            vec4f(f / aspect_ratio, 0.0, 0.0, 0.0),
            vec4f(0.0, f, 0.0, 0.0),
            vec4f(0.0, 0.0, (far + near) / (near - far), -1.0),
            vec4f(0.0, 0.0, (2.0 * far * near) / (near - far), 0.0),
        )
    }

    /// Builds a right-handed orthographic projection with a 0..1 depth
    /// range, matching glam's `orthographic_rh`.
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Mat4x4f {
        let rcp_rx = 1.0 / (right - left);
        let rcp_ry = 1.0 / (top - bottom);
        let rcp_rz = 1.0 / (far - near);
        mat4x4f(
            vec4f(2.0 * rcp_rx, 0.0, 0.0, 0.0),
            vec4f(0.0, 2.0 * rcp_ry, 0.0, 0.0),
            vec4f(0.0, 0.0, -rcp_rz, 0.0),
            vec4f(
                -(right + left) * rcp_rx,
                -(top + bottom) * rcp_ry,
                -near * rcp_rz,
                1.0,
            ),
        )
    }

    /// Builds a left-handed orthographic projection with a 0..1 depth
    /// range, matching glam's `orthographic_lh`.
    pub fn orthographic_lh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Mat4x4f {
        let rcp_rx = 1.0 / (right - left);
        let rcp_ry = 1.0 / (top - bottom);
        let rcp_rz = 1.0 / (far - near);
        mat4x4f(
            vec4f(2.0 * rcp_rx, 0.0, 0.0, 0.0),
            vec4f(0.0, 2.0 * rcp_ry, 0.0, 0.0),
            vec4f(0.0, 0.0, rcp_rz, 0.0),
            vec4f(
                -(right + left) * rcp_rx,
                -(top + bottom) * rcp_ry,
                -near * rcp_rz,
                1.0,
            ),
        )
    }

    /// Builds a right-handed orthographic projection with a -1..1 depth
    /// range, matching glam's `orthographic_rh_gl`.
    pub fn orthographic_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Mat4x4f {
        let rcp_rx = 1.0 / (right - left);
        let rcp_ry = 1.0 / (top - bottom);
        let rcp_rz = 1.0 / (far - near);
        mat4x4f(
            vec4f(2.0 * rcp_rx, 0.0, 0.0, 0.0),
            vec4f(0.0, 2.0 * rcp_ry, 0.0, 0.0),
            vec4f(0.0, 0.0, -2.0 * rcp_rz, 0.0),
            vec4f(
                -(right + left) * rcp_rx,
                -(top + bottom) * rcp_ry,
                -(far + near) * rcp_rz,
                1.0,
            ),
        )
    }

    // -- Matrix inverses
    //
    // Hand-written once, shared by both worlds. Singular matrices
    // produce inf/NaN components (IEEE division), matching glam's
    // documented behavior for `inverse`.

    /// Returns the inverse of a `Mat2x2f`.
    pub fn inverse2x2(m: Mat2x2f) -> Mat2x2f {
        let inv_det = 1.0 / determinant(m);
        let c0 = m[0u32];
        let c1 = m[1u32];
        mat2x2f(vec2f(c1.y, -c0.y) * inv_det, vec2f(-c1.x, c0.x) * inv_det)
    }

    /// Returns the inverse of a `Mat3x3f`.
    pub fn inverse3x3(m: Mat3x3f) -> Mat3x3f {
        let inv_det = 1.0 / determinant(m);
        let c0 = m[0u32];
        let c1 = m[1u32];
        let c2 = m[2u32];
        mat3x3f(
            vec3f(
                c1.y * c2.z - c1.z * c2.y,
                c0.z * c2.y - c0.y * c2.z,
                c0.y * c1.z - c0.z * c1.y,
            ) * inv_det,
            vec3f(
                c1.z * c2.x - c1.x * c2.z,
                c0.x * c2.z - c0.z * c2.x,
                c0.z * c1.x - c0.x * c1.z,
            ) * inv_det,
            vec3f(
                c1.x * c2.y - c1.y * c2.x,
                c0.y * c2.x - c0.x * c2.y,
                c0.x * c1.y - c0.y * c1.x,
            ) * inv_det,
        )
    }

    /// Returns the inverse of a `Mat4x4f`.
    ///
    /// Written as an explicit cofactor expansion: `a_rc` names the
    /// element at row `r`, column `c` (`m[c]` component `r`); `cf_rc`
    /// names the cofactor of that element. The inverse is the adjugate
    /// (the transposed cofactor matrix) divided by the determinant.
    pub fn inverse4x4(m: Mat4x4f) -> Mat4x4f {
        let c0 = m[0u32];
        let c1 = m[1u32];
        let c2 = m[2u32];
        let c3 = m[3u32];
        let a00 = c0.x;
        let a01 = c1.x;
        let a02 = c2.x;
        let a03 = c3.x;
        let a10 = c0.y;
        let a11 = c1.y;
        let a12 = c2.y;
        let a13 = c3.y;
        let a20 = c0.z;
        let a21 = c1.z;
        let a22 = c2.z;
        let a23 = c3.z;
        let a30 = c0.w;
        let a31 = c1.w;
        let a32 = c2.w;
        let a33 = c3.w;

        // Minors M_{0,j} (row 0 and column j deleted), for the Laplace
        // determinant expansion along row 0.
        let m00 = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31);
        let m01 = a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a32 - a22 * a30);
        let m02 = a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30)
            + a13 * (a20 * a31 - a21 * a30);
        let m03 = a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30)
            + a12 * (a20 * a31 - a21 * a30);

        let det = a00 * m00 - a01 * m01 + a02 * m02 - a03 * m03;
        let inv_det = 1.0 / det;

        // Cofactors cf_rc = (-1)^(r+c) * M_{r,c}; column `c` of the
        // inverse is (cf_c0, cf_c1, cf_c2, cf_c3) * inv_det.
        let cf10 = -(a01 * (a22 * a33 - a23 * a32) - a02 * (a21 * a33 - a23 * a31)
            + a03 * (a21 * a32 - a22 * a31));
        let cf11 = a00 * (a22 * a33 - a23 * a32) - a02 * (a20 * a33 - a23 * a30)
            + a03 * (a20 * a32 - a22 * a30);
        let cf12 = -(a00 * (a21 * a33 - a23 * a31) - a01 * (a20 * a33 - a23 * a30)
            + a03 * (a20 * a31 - a21 * a30));
        let cf13 = a00 * (a21 * a32 - a22 * a31) - a01 * (a20 * a32 - a22 * a30)
            + a02 * (a20 * a31 - a21 * a30);
        let cf20 = a01 * (a12 * a33 - a13 * a32) - a02 * (a11 * a33 - a13 * a31)
            + a03 * (a11 * a32 - a12 * a31);
        let cf21 = -(a00 * (a12 * a33 - a13 * a32) - a02 * (a10 * a33 - a13 * a30)
            + a03 * (a10 * a32 - a12 * a30));
        let cf22 = a00 * (a11 * a33 - a13 * a31) - a01 * (a10 * a33 - a13 * a30)
            + a03 * (a10 * a31 - a11 * a30);
        let cf23 = -(a00 * (a11 * a32 - a12 * a31) - a01 * (a10 * a32 - a12 * a30)
            + a02 * (a10 * a31 - a11 * a30));
        let cf30 = -(a01 * (a12 * a23 - a13 * a22) - a02 * (a11 * a23 - a13 * a21)
            + a03 * (a11 * a22 - a12 * a21));
        let cf31 = a00 * (a12 * a23 - a13 * a22) - a02 * (a10 * a23 - a13 * a20)
            + a03 * (a10 * a22 - a12 * a20);
        let cf32 = -(a00 * (a11 * a23 - a13 * a21) - a01 * (a10 * a23 - a13 * a20)
            + a03 * (a10 * a21 - a11 * a20));
        let cf33 = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);

        mat4x4f(
            vec4f(m00, -m01, m02, -m03) * inv_det,
            vec4f(cf10, cf11, cf12, cf13) * inv_det,
            vec4f(cf20, cf21, cf22, cf23) * inv_det,
            vec4f(cf30, cf31, cf32, cf33) * inv_det,
        )
    }
}

#[cfg(all(test, feature = "glm"))]
mod glm_test {
    #[crate::wgsl(crate_path = crate)]
    pub mod uses_glm {
        use crate::glm::*;

        pub fn squared() -> f32 {
            let v = vec3f(1.0, 2.0, 2.0);
            length_squared3(v)
        }

        pub fn zero_stays_zero() -> Vec3f {
            normalize_or_zero3(vec3f(0.0, 0.0, 0.0))
        }

        pub fn unit_normalize() -> Vec3f {
            normalize_or_zero3(vec3f(3.0, 4.0, 0.0))
        }

        pub fn right_angle() -> f32 {
            angle_between3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0))
        }

        pub fn flat_angle() -> f32 {
            angle_between2(vec2f(2.0, 0.0), vec2f(-3.0, 0.0))
        }

        pub fn zero_angle() -> f32 {
            angle_between3(vec3f(0.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0))
        }

        pub fn squared_2_4() -> f32 {
            length_squared2(vec2f(3.0, 4.0)) + length_squared4(vec4f(1.0, 1.0, 1.0, 1.0))
        }

        pub fn norm_or_zero_2_4() -> Vec3f {
            let v2 = normalize_or_zero2(vec2f(0.0, 0.0));
            let v4 = normalize_or_zero4(vec4f(2.0, 2.0, 2.0, 2.0));
            vec3f(v2.x, v2.y, v4.x)
        }
    }

    #[test]
    fn glm_helpers_work_in_both_worlds() {
        use crate::std::vec3f;

        // CPU world: the helpers are plain Rust over the concrete types.
        assert_eq!(uses_glm::squared(), 9.0);
        assert_eq!(uses_glm::zero_stays_zero(), vec3f(0.0, 0.0, 0.0));
        let n = uses_glm::unit_normalize();
        assert!(
            (n.x - 0.6).abs() < 1e-6 && (n.y - 0.8).abs() < 1e-6 && n.z == 0.0,
            "unit normalize: {n:?}"
        );
        assert!(
            (uses_glm::right_angle() - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "right angle: {}",
            uses_glm::right_angle()
        );
        assert!(
            (uses_glm::flat_angle() - std::f32::consts::PI).abs() < 1e-6,
            "flat angle: {}",
            uses_glm::flat_angle()
        );
        // The angle of a zero vector is selected to 0.0 in both worlds.
        assert_eq!(uses_glm::zero_angle(), 0.0);
        assert_eq!(uses_glm::squared_2_4(), 29.0);
        assert_eq!(uses_glm::norm_or_zero_2_4().to_array(), [0.0, 0.0, 0.5]);

        // WGSL world: helpers assembled from the glm module, naga-valid.
        let src = uses_glm::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(src.contains("length_squared3"), "helper assembled: {src}");
        assert!(
            src.contains("normalize_or_zero3"),
            "helper assembled: {src}"
        );
        assert!(src.contains("angle_between3"), "helper assembled: {src}");
        assert!(src.contains("vec3f_ZERO"), "builtin constant path: {src}");
        #[cfg(feature = "validation")]
        uses_glm::WGSL_SOURCE.validate().unwrap();
    }
}

#[cfg(all(test, feature = "glm"))]
mod glm_matrix_test {
    use crate::{glm::*, std::*};

    /// Renders ours vs glam with per-element context in the failure.
    fn assert_mat4_close(actual: Mat4x4f, expected: glam::Mat4, ctx: &str) {
        let eps = 1e-5;
        let e = expected.to_cols_array();
        for c in 0..4usize {
            let col = actual[c].to_array();
            for r in 0..4usize {
                assert!(
                    (col[r] - e[c * 4 + r]).abs() < eps,
                    "{ctx}: column {c}, row {r}: got {}, want {}",
                    col[r],
                    e[c * 4 + r]
                );
            }
        }
    }

    fn to_glam_mat4(m: &Mat4x4f) -> glam::Mat4 {
        glam::Mat4::from_cols(
            m[0u32].into(),
            m[1u32].into(),
            m[2u32].into(),
            m[3u32].into(),
        )
    }

    #[crate::wgsl(crate_path = crate)]
    pub mod uses_glm_matrices {
        use crate::glm::*;

        pub fn build_view() -> Mat4x4f {
            look_at_rh(
                vec3f(3.0, 4.0, 5.0),
                vec3f(0.0, 0.0, 0.0),
                vec3f(0.0, 1.0, 0.0),
            )
        }

        pub fn build_view_lh() -> Mat4x4f {
            look_at_lh(
                vec3f(0.0, 0.0, 5.0),
                vec3f(0.0, 0.0, 0.0),
                vec3f(0.0, 1.0, 0.0),
            )
        }

        pub fn build_proj() -> Mat4x4f {
            perspective_rh(1.2, 1.5, 0.1, 100.0)
        }

        pub fn build_proj_gl() -> Mat4x4f {
            perspective_rh_gl(1.0, 2.0, 0.5, 10.0)
        }

        pub fn build_ortho() -> Mat4x4f {
            orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0)
        }

        pub fn compose() -> Mat4x4f {
            let m = translation(vec3f(1.0, 2.0, 3.0));
            let m = rotate(m, 0.7, vec3f(0.0, 1.0, 0.0));
            scale(m, vec3f(2.0, 2.0, 2.0))
        }

        pub fn moved() -> Vec3f {
            transform_point3(compose(), vec3f(1.0, 1.0, 1.0))
        }

        pub fn aimed() -> Vec4f {
            build_proj() * build_view() * vec4f(1.0, 2.0, 3.0, 1.0)
        }

        pub fn uninverted() -> Mat4x4f {
            inverse4x4(compose()) * compose()
        }

        pub fn inv2() -> Mat2x2f {
            inverse2x2(mat2x2f(vec2f(4.0, 1.0), vec2f(2.0, 2.0)))
        }

        pub fn inv3() -> Mat3x3f {
            inverse3x3(mat3x3f(
                vec3f(1.0, 0.0, 2.0),
                vec3f(0.0, 2.0, 0.0),
                vec3f(0.0, 0.0, 4.0),
            ))
        }

        pub fn point2() -> Vec2f {
            let m = mat3x3f(
                vec3f(2.0, 0.0, 1.0),
                vec3f(0.0, 3.0, 2.0),
                vec3f(0.0, 0.0, 1.0),
            );
            transform_point2(m, vec2f(1.0, 1.0)) + transform_vector2(m, vec2f(1.0, 0.0))
        }
    }

    #[test]
    fn glm_matrix_builders_match_glam() {
        let t = vec3f(1.0, 2.0, 3.0);
        assert_mat4_close(
            translation(t),
            glam::Mat4::from_translation(t.into()),
            "translation",
        );
        assert_mat4_close(scaling(t), glam::Mat4::from_scale(t.into()), "scaling");
        let axis = vec3f(0.0, 1.0, 0.0);
        assert_mat4_close(
            rotation_axis_angle(axis, 0.7),
            glam::Mat4::from_axis_angle(axis.into(), 0.7),
            "rotation_axis_angle",
        );

        // GLM accumulate form: m * builder.
        let m = uses_glm_matrices::compose();
        let g = glam::Mat4::from_translation(vec3f(1.0, 2.0, 3.0).into())
            * glam::Mat4::from_axis_angle(glam::Vec3::Y, 0.7)
            * glam::Mat4::from_scale(glam::Vec3::splat(2.0));
        assert_mat4_close(m, g, "translate/rotate/scale composition");

        assert_mat4_close(
            translate(m, t),
            g * glam::Mat4::from_translation(t.into()),
            "translate accumulate",
        );
    }

    #[test]
    fn glm_view_and_projections_match_glam() {
        let eye = vec3f(3.0, 4.0, 5.0);
        let center = vec3f(0.0, 0.0, 0.0);
        let up = vec3f(0.0, 1.0, 0.0);
        assert_mat4_close(
            look_at_rh(eye, center, up),
            glam::Mat4::look_at_rh(eye.into(), center.into(), up.into()),
            "look_at_rh",
        );
        assert_mat4_close(
            look_at_lh(eye, center, up),
            glam::Mat4::look_at_lh(eye.into(), center.into(), up.into()),
            "look_at_lh",
        );

        let (fov, aspect, near, far) = (1.2f32, 1.5f32, 0.1f32, 100.0f32);
        assert_mat4_close(
            perspective_rh(fov, aspect, near, far),
            glam::Mat4::perspective_rh(fov, aspect, near, far),
            "perspective_rh",
        );
        assert_mat4_close(
            perspective_lh(fov, aspect, near, far),
            glam::Mat4::perspective_lh(fov, aspect, near, far),
            "perspective_lh",
        );
        assert_mat4_close(
            perspective_rh_gl(fov, aspect, near, far),
            glam::Mat4::perspective_rh_gl(fov, aspect, near, far),
            "perspective_rh_gl",
        );

        let (l, r, b, t, n, f) = (-2.0f32, 3.0, -1.0, 4.0, 0.5, 50.0);
        assert_mat4_close(
            orthographic_rh(l, r, b, t, n, f),
            glam::Mat4::orthographic_rh(l, r, b, t, n, f),
            "orthographic_rh",
        );
        assert_mat4_close(
            orthographic_lh(l, r, b, t, n, f),
            glam::Mat4::orthographic_lh(l, r, b, t, n, f),
            "orthographic_lh",
        );
        assert_mat4_close(
            orthographic_rh_gl(l, r, b, t, n, f),
            glam::Mat4::orthographic_rh_gl(l, r, b, t, n, f),
            "orthographic_rh_gl",
        );

        // The wgsl test module's builders are the same checks, asserted
        // through the module functions.
        let eye = glam::Vec3::new(3.0, 4.0, 5.0);
        assert_mat4_close(
            uses_glm_matrices::build_view(),
            glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y),
            "module build_view",
        );
        assert_mat4_close(
            uses_glm_matrices::build_view_lh(),
            glam::Mat4::look_at_lh(
                glam::Vec3::new(0.0, 0.0, 5.0),
                glam::Vec3::ZERO,
                glam::Vec3::Y,
            ),
            "module build_view_lh",
        );
        assert_mat4_close(
            uses_glm_matrices::build_proj(),
            glam::Mat4::perspective_rh(1.2, 1.5, 0.1, 100.0),
            "module build_proj",
        );
        assert_mat4_close(
            uses_glm_matrices::build_proj_gl(),
            glam::Mat4::perspective_rh_gl(1.0, 2.0, 0.5, 10.0),
            "module build_proj_gl",
        );
        assert_mat4_close(
            uses_glm_matrices::build_ortho(),
            glam::Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
            "module build_ortho",
        );

        // aimed() = perspective_rh * look_at_rh * (1, 2, 3, 1).
        let want = glam::Mat4::perspective_rh(1.2, 1.5, 0.1, 100.0)
            * glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y)
            * glam::Vec4::new(1.0, 2.0, 3.0, 1.0);
        let got = uses_glm_matrices::aimed();
        let diff = (got - Vec4f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-4),
            "module aimed: {got:?} vs {want:?}"
        );
    }

    #[test]
    fn glm_transform_point_vector_match_glam() {
        let m = uses_glm_matrices::compose();
        let g = to_glam_mat4(&m);
        let p = vec3f(1.0, 2.0, 3.0);
        let want = g.transform_point3(p.into());
        let got = transform_point3(m, p);
        let diff = (got - Vec3f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "transform_point3: {got:?} vs {want:?}"
        );
        let want = g.transform_vector3(p.into());
        let got = transform_vector3(m, p);
        let diff = (got - Vec3f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "transform_vector3: {got:?} vs {want:?}"
        );

        let m2 = mat3x3f(
            vec3f(2.0, 0.0, 1.0),
            vec3f(0.0, 3.0, 2.0),
            vec3f(0.0, 0.0, 1.0),
        );
        let g2 = glam::Mat3::from_cols(m2[0u32].into(), m2[1u32].into(), m2[2u32].into());
        let p2 = vec2f(1.0, 1.0);
        let want = g2.transform_point2(p2.into());
        let got = transform_point2(m2, p2);
        let diff = (got - Vec2f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "transform_point2: {got:?} vs {want:?}"
        );
        let want = g2.transform_vector2(p2.into());
        let got = transform_vector2(m2, p2);
        let diff = (got - Vec2f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "transform_vector2: {got:?} vs {want:?}"
        );

        // Through the wgsl test module: point2() is
        // transform_point2 + transform_vector2 on the same matrix.
        let want = g2.transform_point2(p2.into()) + g2.transform_vector2(glam::Vec2::X);
        let got = uses_glm_matrices::point2();
        let diff = (got - Vec2f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "module point2: {got:?} vs {want:?}"
        );

        // Through the wgsl test module: moved() is
        // transform_point3(compose(), (1, 1, 1)).
        let want = g.transform_point3(glam::Vec3::new(1.0, 1.0, 1.0));
        let got = uses_glm_matrices::moved();
        let diff = (got - Vec3f::from(want)).to_array();
        assert!(
            diff.iter().all(|d| d.abs() < 1e-5),
            "module moved: {got:?} vs {want:?}"
        );
    }

    #[test]
    fn glm_inverses_match_glam_and_recover_identity() {
        let m = uses_glm_matrices::compose();
        let g = to_glam_mat4(&m);
        assert_mat4_close(inverse4x4(m), g.inverse(), "inverse4x4");

        // inverse * m recovers the identity.
        let recovered = inverse4x4(m) * m;
        for c in 0..4usize {
            let col = recovered[c].to_array();
            for (r, want) in col.iter().zip(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ][c],
            ) {
                assert!((r - want).abs() < 1e-4, "identity recovery: {recovered:?}");
            }
        }

        let m2 = mat2x2f(vec2f(4.0, 1.0), vec2f(2.0, 2.0));
        let g2 = glam::Mat2::from_cols(m2[0u32].into(), m2[1u32].into());
        let want = g2.inverse();
        let got = inverse2x2(m2);
        let e = want.to_cols_array();
        for c in 0..2usize {
            let col = got[c].to_array();
            for r in 0..2usize {
                assert!(
                    (col[r] - e[c * 2 + r]).abs() < 1e-5,
                    "inverse2x2 col {c} row {r}: {} vs {}",
                    col[r],
                    e[c * 2 + r]
                );
            }
        }

        let m3 = mat3x3f(
            vec3f(1.0, 0.0, 2.0),
            vec3f(0.0, 2.0, 0.0),
            vec3f(0.0, 0.0, 4.0),
        );
        let g3 = glam::Mat3::from_cols(m3[0u32].into(), m3[1u32].into(), m3[2u32].into());
        let want = g3.inverse();
        let got = inverse3x3(m3);
        let e = want.to_cols_array();
        for c in 0..3usize {
            let col = got[c].to_array();
            for r in 0..3usize {
                assert!(
                    (col[r] - e[c * 3 + r]).abs() < 1e-5,
                    "inverse3x3 col {c} row {r}: {} vs {}",
                    col[r],
                    e[c * 3 + r]
                );
            }
        }

        // The wgsl-world module exercises the same functions; its source
        // assembles the helpers and naga validates it. inv2()/inv3() and
        // uninverted() assert through the module functions.
        let want =
            glam::Mat2::from_cols(glam::Vec2::new(4.0, 1.0), glam::Vec2::new(2.0, 2.0)).inverse();
        let got = uses_glm_matrices::inv2();
        let e = want.to_cols_array();
        for c in 0..2usize {
            let col = got[c].to_array();
            for r in 0..2usize {
                assert!(
                    (col[r] - e[c * 2 + r]).abs() < 1e-5,
                    "module inv2 col {c} row {r}: {} vs {}",
                    col[r],
                    e[c * 2 + r]
                );
            }
        }
        let want = glam::Mat3::from_cols(
            glam::Vec3::new(1.0, 0.0, 2.0),
            glam::Vec3::new(0.0, 2.0, 0.0),
            glam::Vec3::new(0.0, 0.0, 4.0),
        )
        .inverse();
        let got = uses_glm_matrices::inv3();
        let e = want.to_cols_array();
        for c in 0..3usize {
            let col = got[c].to_array();
            for r in 0..3usize {
                assert!(
                    (col[r] - e[c * 3 + r]).abs() < 1e-5,
                    "module inv3 col {c} row {r}: {} vs {}",
                    col[r],
                    e[c * 3 + r]
                );
            }
        }
        let recovered = uses_glm_matrices::uninverted();
        for c in 0..4usize {
            let col = recovered[c].to_array();
            for (r, want) in col.iter().zip(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ][c],
            ) {
                assert!((r - want).abs() < 1e-4, "module uninverted: {recovered:?}");
            }
        }

        let src = uses_glm_matrices::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(src.contains("fn inverse4x4"), "helper assembled: {src}");
        assert!(src.contains("fn look_at_rh"), "helper assembled: {src}");
        assert!(src.contains("fn perspective_rh"), "helper assembled: {src}");
        #[cfg(feature = "validation")]
        uses_glm_matrices::WGSL_SOURCE.validate().unwrap();
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod test {
    use crate::{GenericTemplate, Source, TemplateDependency, TemplateInstantiation, ir, wgsl};

    #[wgsl(crate_path = crate)]
    pub mod a {
        pub const THREE: u32 = 3;
    }

    #[wgsl(crate_path = crate)]
    pub mod b {
        use super::a::*;

        pub fn add_three_to_x_minus_y(x: u32, y: u32) -> u32 {
            let i: u32 = (x - y) + THREE;
            i
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod c {
        use super::b::*;

        pub fn main() {
            let _u = add_three_to_x_minus_y(1337, 666);
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod vecs {
        use crate::std::*;

        const _VEC3F_ONE: Vec4f = vec4f(0.0, 1.0, 2.0, 3.0);
    }

    #[test]
    fn module_source() {
        let source = c::WGSL_SOURCE.wgsl_source().unwrap();
        c::main();
        assert!(source.contains("const THREE: u32 = 3u;"), "got:\n{source}");
        assert!(
            source.contains("fn add_three_to_x_minus_y(x: u32, y: u32) -> u32"),
            "got:\n{source}"
        );
        assert!(source.contains("fn main()"), "got:\n{source}");
        // Cross-source call arguments are suffixed via the fn-signature
        // accumulator threaded through `collect` (module `c` imports
        // `add_three_to_x_minus_y` from module `b`).
        assert!(
            source.contains("add_three_to_x_minus_y(1337u, 666u)"),
            "got:\n{source}"
        );

        // Verify that imported source outputs are not duplicated.
        // Source C imports B, which imports A. Source A's `THREE` const
        // should appear exactly once in the concatenated output.
        assert_eq!(
            source.matches("const THREE:").count(),
            1,
            "THREE should appear exactly once (no duplicates from transitive imports), \
             got:\n{source}"
        );
    }

    // Cross-module trait impl test:
    // - trait_provider defines a trait and implements it for f32
    // - trait_consumer imports it and uses the impl via a generic function

    #[wgsl(crate_path = crate)]
    pub mod trait_provider {
        pub trait Scalable {
            fn scale(val: Self, factor: f32) -> Self;
        }

        impl Scalable for f32 {
            fn scale(val: f32, factor: f32) -> f32 {
                val * factor
            }
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod trait_consumer {
        use super::trait_provider::*;

        pub fn apply_scale<T: Scalable>(val: T, factor: f32) -> T {
            T::scale(val, factor)
        }

        pub fn go() -> f32 {
            apply_scale::<f32>(2.0, 3.0)
        }
    }

    #[test]
    fn cross_module_trait_impl() {
        // Verify provider source generates f32_scale
        let provider_src = trait_provider::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            provider_src.contains("fn f32_scale("),
            "trait_provider should contain f32_scale, got:\n{provider_src}"
        );

        // Verify consumer source generates the monomorphized `apply_scale<f32>`
        // and that it calls `f32_scale`. Under the robust mangling scheme
        // (issue #112), `apply_scale` has an underscore so the joined name
        // is `_1apply_scale_f32`, not `apply_scale_f32`.
        let consumer_src = trait_consumer::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            consumer_src.contains("fn _1apply_scale_f32("),
            "trait_consumer should contain _1apply_scale_f32, got:\n{consumer_src}"
        );
        assert!(
            consumer_src.contains("f32_scale("),
            "_1apply_scale_f32 should call f32_scale, got:\n{consumer_src}"
        );
        assert!(
            consumer_src.contains("fn f32_scale("),
            "concatenated source should have f32_scale (from import), got:\n{consumer_src}"
        );

        // Verify Rust-side execution works too
        let result = trait_consumer::go();
        assert!(
            (result - 6.0).abs() < f32::EPSILON,
            "2.0 * 3.0 should be 6.0, got {result}"
        );
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_provider {
        pub fn id<T: Copy>(v: T) -> T {
            v
        }
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_left {
        use super::dedupe_provider::*;

        pub fn left() -> f32 {
            id::<f32>(1.0)
        }
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_right {
        use super::dedupe_provider::*;

        pub fn right() -> f32 {
            id::<f32>(2.0)
        }
    }

    #[expect(dead_code)]
    #[wgsl(crate_path = crate)]
    mod dedupe_root {
        #[rustfmt::skip]
        use super::dedupe_left::*;
        #[rustfmt::skip]
        use super::dedupe_right::*;

        pub fn root() -> f32 {
            left() + right()
        }
    }

    #[test]
    fn cross_module_instantiations_are_deduped_globally() {
        let full_src = dedupe_root::WGSL_SOURCE.wgsl_source().unwrap();
        assert_eq!(
            full_src.matches("fn id_f32(").count(),
            1,
            "expected id_f32 to appear once, got:\n{full_src}"
        );
    }

    #[test]
    fn ambiguous_template_provider_errors() {
        // Build hand-crafted sources where two distinct provider sources
        // both define a generic template named "shared". Resolution must
        // return `Err(SourceError::AmbiguousTemplate { .. })`.
        fn empty_source_ir() -> ir::Module {
            ir::Module {
                name: "stub",
                items: vec![],
                attrs: vec![],
            }
        }
        fn shared_template_items() -> Vec<ir::Item> {
            vec![ir::Item::Fn(ir::ItemFn {
                type_params: vec![],
                const_params: vec![],
                fn_attrs: ir::FnAttrs::None,
                name: "shared".to_string().into(),
                inputs: vec![ir::FnArg {
                    inter_stage_io: vec![],
                    name: "x".to_string(),
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                    attrs: vec![],
                }],
                return_type: ir::ReturnType::Type {
                    annotation: ir::ReturnTypeAnnotation::None,
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                },
                block: ir::Block {
                    stmts: vec![ir::Stmt::Return(Some(ir::Expr::Ident("x".to_string())))],
                },
                attrs: vec![],
            })]
        }
        fn f32_args() -> Vec<ir::Type> {
            vec![ir::Type::Scalar(ir::ScalarType::F32)]
        }
        static DEP: [TemplateDependency; 0] = [];
        static A_TEMPLATES: [GenericTemplate; 1] = [GenericTemplate {
            name: "shared",
            type_params: &["T"],
            const_params: &[],
            ir_constructor: shared_template_items,
            dependencies: &DEP,
        }];
        static B_TEMPLATES: [GenericTemplate; 1] = [GenericTemplate {
            name: "shared",
            type_params: &["T"],
            const_params: &[],
            ir_constructor: shared_template_items,
            dependencies: &DEP,
        }];
        static EMPTY_SOURCES: [&Source; 0] = [];
        static EMPTY_INSTS: [TemplateInstantiation; 0] = [];
        static EMPTY_TYPE_PARAMS: [&str; 0] = [];
        static MOD_A: Source = Source {
            id: 0,
            name: "a",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &A_TEMPLATES,
            instantiations: &EMPTY_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };
        static MOD_B: Source = Source {
            id: 1,
            name: "b",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &B_TEMPLATES,
            instantiations: &EMPTY_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };
        static ROOT_INSTS: [TemplateInstantiation; 1] = [TemplateInstantiation {
            modules: &[&MOD_A, &MOD_B],
            template_name: "shared",
            type_args_constructor: f32_args,
            mangled_type_args: &["f32"],
            const_args_constructor: || vec![],
            mangled_const_args: &[],
        }];
        static ROOT: Source = Source {
            id: 2,
            name: "root",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &ROOT_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };

        let err = ROOT.wgsl_source().unwrap_err();
        match &err {
            crate::SourceError::AmbiguousTemplate {
                template_name,
                matching_source_names,
                ..
            } => {
                assert_eq!(template_name, "shared");
                assert!(
                    matching_source_names.contains(&"a".to_string())
                        && matching_source_names.contains(&"b".to_string()),
                    "expected matching sources [a, b], got {matching_source_names:?}"
                );
            }
            other => panic!("expected AmbiguousTemplate, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "got: {msg}");
        assert!(msg.contains("shared"), "got: {msg}");
    }

    // --- Template resolution through `pub use` re-export chains (#238) ---

    mod reexport_resolution {
        //! Hand-crafted source graphs for wgsl-rs#238: generic templates
        //! must resolve through `pub use` re-export chains, multi-hop
        //! chains, diamonds, and import cycles.

        use crate::{GenericTemplate, Source, TemplateDependency, TemplateInstantiation, ir};

        fn empty_source_ir() -> ir::Module {
            ir::Module {
                name: "stub",
                items: vec![],
                attrs: vec![],
            }
        }

        /// The generic fn template `id<T>(x: T) -> T`.
        fn id_template_items() -> Vec<ir::Item> {
            vec![ir::Item::Fn(ir::ItemFn {
                type_params: vec![],
                const_params: vec![],
                fn_attrs: ir::FnAttrs::None,
                name: "id".to_string().into(),
                inputs: vec![ir::FnArg {
                    inter_stage_io: vec![],
                    name: "x".to_string(),
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                    attrs: vec![],
                }],
                return_type: ir::ReturnType::Type {
                    annotation: ir::ReturnTypeAnnotation::None,
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                },
                block: ir::Block {
                    stmts: vec![ir::Stmt::Return(Some(ir::Expr::Ident("x".to_string())))],
                },
                attrs: vec![],
            })]
        }

        fn f32_args() -> Vec<ir::Type> {
            vec![ir::Type::Scalar(ir::ScalarType::F32)]
        }

        fn no_const_args() -> Vec<u32> {
            vec![]
        }

        static DEP: [TemplateDependency; 0] = [];
        static ID_TEMPLATES: [GenericTemplate; 1] = [GenericTemplate {
            name: "id",
            type_params: &["T"],
            const_params: &[],
            ir_constructor: id_template_items,
            dependencies: &DEP,
        }];
        static NO_SOURCES: [&Source; 0] = [];
        static NO_INSTS: [TemplateInstantiation; 0] = [];
        static NO_TYPE_PARAMS: [&str; 0] = [];

        // The defining module.
        static DEFS: Source = Source {
            id: 100,
            name: "defs",
            imports: &NO_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &ID_TEMPLATES,
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        // Identical content to DEFS (same id): the same defining module
        // reached through a second re-export path.
        static DEFS_DUPLICATE: Source = Source {
            id: 100,
            name: "defs",
            imports: &NO_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &ID_TEMPLATES,
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        // A distinct defining module that happens to define a template
        // with the same name.
        static OTHER_DEFS: Source = Source {
            id: 106,
            name: "other_defs",
            imports: &NO_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &ID_TEMPLATES,
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// The single re-export hop: an empty registry, one import.
        static PRELUDE: Source = Source {
            id: 101,
            name: "prelude",
            imports: &[&DEFS],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// A second hop re-exporting the prelude.
        static MID: Source = Source {
            id: 102,
            name: "mid",
            imports: &[&PRELUDE],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// Two re-exporters of the *same* defining module (diamond).
        static HOP_A: Source = Source {
            id: 103,
            name: "hop_a",
            imports: &[&DEFS],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };
        static HOP_B: Source = Source {
            id: 104,
            name: "hop_b",
            imports: &[&DEFS_DUPLICATE],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// Two re-exporters of *distinct* defining modules.
        static PATH_A: Source = Source {
            id: 107,
            name: "path_a",
            imports: &[&DEFS],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };
        static PATH_B: Source = Source {
            id: 108,
            name: "path_b",
            imports: &[&OTHER_DEFS],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// Mutual re-exports: the template lives behind the cycle.
        static CYC_A: Source = Source {
            id: 109,
            name: "cyc_a",
            imports: &[&CYC_B],
            ir_constructor: empty_source_ir,
            templates: &ID_TEMPLATES,
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };
        static CYC_B: Source = Source {
            id: 110,
            name: "cyc_b",
            imports: &[&CYC_A],
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &NO_INSTS,
            module_type_params: &NO_TYPE_PARAMS,
        };

        /// Build a concrete root whose only content is one instantiation
        /// of `id` against the given candidate modules. Declared per-test
        /// so each graph gets its own statics.
        macro_rules! id_root {
            ($root_id:literal, $candidates:expr) => {{
                static INSTS: [TemplateInstantiation; 1] = [TemplateInstantiation {
                    modules: $candidates,
                    template_name: "id",
                    type_args_constructor: f32_args,
                    mangled_type_args: &["f32"],
                    const_args_constructor: no_const_args,
                    mangled_const_args: &[],
                }];
                static ROOT: Source = Source {
                    id: $root_id,
                    name: "root",
                    imports: $candidates,
                    ir_constructor: empty_source_ir,
                    templates: &[],
                    instantiations: &INSTS,
                    module_type_params: &NO_TYPE_PARAMS,
                };
                &ROOT
            }};
        }

        #[test]
        fn template_resolves_through_single_reexport_hop() {
            let root = id_root!(200, &[&PRELUDE]);
            let src = root
                .wgsl_source()
                .expect("id must resolve through the re-export hop");
            assert!(src.contains("id_f32"), "expected id_f32 in:\n{src}");
        }

        #[test]
        fn template_resolves_through_multi_hop_chain() {
            let root = id_root!(201, &[&MID]);
            let src = root
                .wgsl_source()
                .expect("id must resolve through the two-hop chain");
            assert!(src.contains("id_f32"), "expected id_f32 in:\n{src}");
        }

        #[test]
        fn diamond_reexport_of_same_source_is_not_ambiguous() {
            let root = id_root!(202, &[&HOP_A, &HOP_B]);
            let src = root
                .wgsl_source()
                .expect("same defining module through two paths must resolve");
            assert_eq!(
                src.matches("id_f32").count(),
                1,
                "instance must be emitted exactly once, got:\n{src}"
            );
        }

        #[test]
        fn reexport_cycle_terminates_and_resolves() {
            let root = id_root!(203, &[&CYC_B]);
            let src = root
                .wgsl_source()
                .expect("template behind a mutual re-export cycle must resolve");
            assert!(src.contains("id_f32"), "expected id_f32 in:\n{src}");
        }

        #[test]
        fn distinct_definers_behind_hops_are_ambiguous() {
            let root = id_root!(204, &[&PATH_A, &PATH_B]);
            let err = root.wgsl_source().unwrap_err();
            match &err {
                crate::SourceError::AmbiguousTemplate {
                    template_name,
                    matching_source_names,
                    ..
                } => {
                    assert_eq!(template_name, "id");
                    assert!(
                        matching_source_names.contains(&"defs".to_string())
                            && matching_source_names.contains(&"other_defs".to_string()),
                        "expected matching sources [defs, other_defs], got \
                         {matching_source_names:?}"
                    );
                }
                other => panic!("expected AmbiguousTemplate, got {other:?}"),
            }
        }

        #[test]
        fn not_found_lists_reachable_templates_through_hop() {
            static MISSING_INSTS: [TemplateInstantiation; 1] = [TemplateInstantiation {
                modules: &[&PRELUDE],
                template_name: "no_such_template",
                type_args_constructor: f32_args,
                mangled_type_args: &["f32"],
                const_args_constructor: no_const_args,
                mangled_const_args: &[],
            }];
            static MISSING_ROOT: Source = Source {
                id: 206,
                name: "missing_root",
                imports: &[&PRELUDE],
                ir_constructor: empty_source_ir,
                templates: &[],
                instantiations: &MISSING_INSTS,
                module_type_params: &NO_TYPE_PARAMS,
            };
            let err = MISSING_ROOT.wgsl_source().unwrap_err();
            match &err {
                crate::SourceError::TemplateNotFound {
                    template_name,
                    available_templates,
                    ..
                } => {
                    assert_eq!(template_name, "no_such_template");
                    assert!(
                        available_templates.contains(&"id".to_string()),
                        "reachable templates through the hop must be listed, got \
                         {available_templates:?}"
                    );
                }
                other => panic!("expected TemplateNotFound, got {other:?}"),
            }
        }
    }

    // --- Cross-module generic struct tests ---

    #[wgsl(crate_path = crate)]
    mod struct_provider {
        pub struct Pair<T: Copy> {
            pub a: T,
            pub b: T,
        }

        impl<T: Copy + std::ops::Add<Output = T>> Pair<T> {
            pub fn sum(p: Pair<T>) -> T {
                p.a + p.b
            }
        }
    }

    #[wgsl(crate_path = crate)]
    mod struct_consumer {
        use super::struct_provider::*;

        pub fn use_pair() -> f32 {
            let p: Pair<f32> = Pair::<f32> { a: 1.0, b: 2.0 };
            Pair::<f32>::sum(p)
        }
    }

    #[test]
    fn cross_module_generic_struct() {
        // The provider should have a template for "Pair"
        assert!(
            struct_provider::WGSL_SOURCE
                .templates
                .iter()
                .any(|t| t.name == "Pair"),
            "struct_provider should have a template named 'Pair', templates: {:?}",
            struct_provider::WGSL_SOURCE
                .templates
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
        );

        // The consumer should produce WGSL with Pair_f32 and Pair_f32_sum
        let full_src = struct_consumer::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            full_src.contains("Pair_f32"),
            "Expected Pair_f32 in assembled WGSL, got:\n{full_src}"
        );
        assert!(
            full_src.contains("Pair_f32_sum"),
            "Expected Pair_f32_sum in assembled WGSL, got:\n{full_src}"
        );

        // Verify Rust-side execution
        let result = struct_consumer::use_pair();
        assert!(
            (result - 3.0).abs() < f32::EPSILON,
            "1.0 + 2.0 should be 3.0, got {result}"
        );
    }

    // --- Generic linkage / generic entry-point tests ---

    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, f32))]
    pub mod generic_shader {
        use crate::std::*;

        // The linkage variable is generic via `impl Trait` syntax. Its
        // module-level type-parameter name is `FRAME` (the variable's
        // identifier), independent of any entry point's type params.
        uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

        // The fragment entry point has its own type parameter `T`. In the
        // emitted IR this becomes the positional name `frag_main_0`.
        #[fragment]
        pub fn frag_main<T: Convert<f32> + Wgsl + Clone>() -> Vec4f {
            vec4f(1.0, sin(f32(get!(FRAME, T)) / 128.0), 0.0, 1.0)
        }
    }

    #[test]
    fn generic_module_is_template() {
        let m = &generic_shader::WGSL_SOURCE;
        assert!(m.is_template());
        // FRAME first (linkage var), then frag_main_0 (entry-point param 0).
        assert_eq!(m.module_type_params, &["FRAME", "frag_main_0"]);
    }

    #[test]
    fn generic_module_template_source_wgsl_source_errors() {
        // After C2: `Source::wgsl_source` returns Err(TemplateWgsl) for a
        // template source. To get placeholder text, instantiate first.
        assert!(generic_shader::WGSL_SOURCE.wgsl_source().is_err());
    }

    #[test]
    fn generic_module_instantiate_substitutes_placeholders() {
        // Order matches `module_type_params`: FRAME first, then frag_main_0.
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        assert!(
            !src.contains("__TPFRAME__") && !src.contains("__TPfrag_main_0__"),
            "expected no placeholders after instantiation, got:\n{src}"
        );
        assert!(
            src.contains("FRAME") && src.contains("f32"),
            "expected FRAME with f32 type, got:\n{src}"
        );
    }

    #[test]
    fn generic_uniform_is_type_erased_static() {
        // The Rust-side static is `Uniform` (with default
        // WgslTypeVariable), not `Uniform<T>`. Setting and getting via the
        // typed accessors should round-trip.
        generic_shader::FRAME.set_typed::<f32>(42.0f32);
        let v = *generic_shader::FRAME.get_typed::<f32>();
        assert_eq!(v, 42.0);

        // A different concrete type can also be stored simultaneously.
        generic_shader::FRAME.set_typed::<u32>(7u32);
        let u = *generic_shader::FRAME.get_typed::<u32>();
        assert_eq!(u, 7);
        // The f32 value is unaffected.
        let v2 = *generic_shader::FRAME.get_typed::<f32>();
        assert_eq!(v2, 42.0);
    }

    #[cfg(feature = "validation")]
    #[test]
    fn generic_module_validate_rejects_template() {
        // `validate()` returns an error for template sources — they have
        // unresolved placeholders that aren't valid WGSL.
        assert!(
            generic_shader::WGSL_SOURCE.validate().is_err(),
            "template sources should fail validation"
        );
    }

    #[test]
    fn generic_module_instantiate_produces_concrete_ir() {
        // The `instantiate` function produces an `ir::Module` with all type
        // params resolved. Its rendering should be free of placeholders.
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPFRAME__"));
        assert!(!src.contains("__TPfrag_main_0__"));
        assert!(src.contains("f32"));
    }

    #[cfg(feature = "validation")]
    #[test]
    fn generic_module_instantiated_source_is_valid_wgsl() {
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        crate::validate_wgsl_source(&src)
            .expect("instantiated generic_shader should produce valid WGSL");
    }

    // Generic compute shader exercising multiple linkage and entry-point
    // type parameters using `impl Trait` syntax.
    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, u32))]
    pub mod generic_compute {
        use crate::std::*;

        storage!(group(0), binding(0), read_write, INPUT: impl Wgsl + Convert<f32> + Clone);
        storage!(group(0), binding(1), read_write, OUTPUT: impl Wgsl + Convert<f32> + Clone);

        #[compute]
        #[workgroup_size(1)]
        pub fn cs_main() {}
    }

    #[test]
    fn generic_compute_has_two_type_params() {
        let m = &generic_compute::WGSL_SOURCE;
        assert!(m.is_template());
        // Linkage variables in declaration order: INPUT, OUTPUT.
        assert_eq!(m.module_type_params, &["INPUT", "OUTPUT"]);
    }

    #[test]
    fn generic_compute_storages_are_type_erased() {
        generic_compute::INPUT.set_typed::<f32>(1.5f32);
        generic_compute::OUTPUT.set_typed::<u32>(42u32);
        let a = *generic_compute::INPUT.get_typed::<f32>();
        let b = *generic_compute::OUTPUT.get_typed::<u32>();
        assert_eq!(a, 1.5);
        assert_eq!(b, 42);
    }

    #[test]
    fn generic_compute_template_source_errors() {
        // After C2: Source::wgsl_source on a template returns
        // Err(SourceError::TemplateWgsl { .. }).
        let err = generic_compute::WGSL_SOURCE.wgsl_source().unwrap_err();
        match &err {
            crate::SourceError::TemplateWgsl {
                uninstantiated_source,
            } => {
                assert_eq!(uninstantiated_source.name, "generic_compute");
                assert_eq!(
                    uninstantiated_source.module_type_params,
                    &["INPUT", "OUTPUT"]
                );
            }
            _ => unreachable!("expected TemplateWgsl for generic_compute"),
        }
        // The Display impl lists the source name and the type params so
        // users can quickly see what to instantiate.
        let msg = err.to_string();
        assert!(msg.contains("generic_compute"), "got: {msg}");
        assert!(
            msg.contains("INPUT") && msg.contains("OUTPUT"),
            "got: {msg}"
        );
        assert!(msg.contains("instantiate"), "got: {msg}");
    }

    #[test]
    fn generic_compute_instantiate_substitutes_both() {
        let m = generic_compute::instantiate::<f32, u32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPINPUT__") && !src.contains("__TPOUTPUT__"));
        assert!(src.contains("INPUT") && src.contains("OUTPUT"));
        assert!(src.contains("f32") && src.contains("u32"));
    }

    #[test]
    fn generic_compute_instantiate_binds_both_storages() {
        let m = generic_compute::instantiate::<f32, u32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPINPUT__"));
        assert!(!src.contains("__TPOUTPUT__"));
        assert!(src.contains("f32") && src.contains("u32"));
    }

    // --- Multi-entry-point type param disambiguation tests ---

    // Module with two entry points that both use the letter `T` and
    // both access the same linkage variable DATA via `get!`. This tests
    // that the `instantiate` function correctly disambiguates the type
    // params and generates the right `Type<Is = ...>` constraints.
    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, f32, f32))]
    pub mod multi_ep {
        use crate::std::*;

        storage!(group(0), binding(0), read_write, DATA: impl std::any::Any);

        #[compute]
        #[workgroup_size(1)]
        pub fn read_as<T: Wgsl + Convert<f32>>(#[builtin(global_invocation_id)] _gid: Vec3u) {
            let _v = get!(DATA, T);
        }

        #[compute]
        #[workgroup_size(1)]
        pub fn write_as<T: Wgsl + Convert<f32>>(#[builtin(global_invocation_id)] _gid: Vec3u) {
            let mut _v = get_mut!(DATA, T);
        }
    }

    #[test]
    fn multi_ep_instantiate_substitutes_correctly() {
        // Three type params: DATA, T_read_as, T_write_as.
        // Both entry-point type params are substituted with f32.
        let m = multi_ep::instantiate::<f32, f32, f32>();
        let src = ir::render_module(&m);
        assert!(
            !src.contains("__TPDATA__"),
            "expected no placeholder for DATA, got:\n{src}"
        );
        assert!(
            !src.contains("__TPread_as_0__") && !src.contains("__TPwrite_as_0__"),
            "expected no entry-point placeholders, got:\n{src}"
        );
        assert!(
            src.contains("f32"),
            "expected f32 type in output, got:\n{src}"
        );
    }

    #[test]
    fn multi_ep_disambiguated_type_params_are_independent() {
        // Even though both entry points use the letter `T`, the
        // `instantiate` function should have two independent type params
        // (T_read_as and T_write_as). This test verifies the constraints
        // are correct by using different concrete types: DATA = f32 for
        // the read path and DATA = f32 for the write path (they must agree
        // since DATA: Type<Is = T_read_as> and DATA: Type<Is = T_write_as>,
        // so T_read_as must equal T_write_as).
        let m = multi_ep::instantiate::<f32, f32, f32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPDATA__"));
        assert!(src.contains("DATA"));
    }

    // --- Integer inter-stage IO (explicit @interpolate(flat)) ---

    // WGSL §13.3.1.4: user-defined vertex outputs and fragment inputs of
    // integer type must always be declared `@interpolate(flat)`. wgpu 30 /
    // naga 30 no longer imply it, so the renderer emits the attribute
    // explicitly when the source does not. The `__validate_wgsl` test
    // auto-emitted for this module fails under naga 30 if the emission
    // goes missing.
    #[wgsl(crate_path = crate)]
    pub mod integer_io {
        use crate::std::*;

        /// Vertex output / fragment input — a single struct shared across
        /// stages, carrying an integer varying at `@location(1)`.
        pub struct Varyings {
            #[builtin(position)]
            pub clip_position: Vec4f,
            #[location(0)]
            pub uv: Vec2f,
            #[location(1)]
            pub material: u32,
        }

        #[vertex]
        pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Varyings {
            let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
            let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
            Varyings {
                clip_position: vec4f(x, y, 0.0, 1.0),
                uv: vec2f((x + 1.0) / 2.0, (y + 1.0) / 2.0),
                material: vertex_index % 3u32,
            }
        }

        #[fragment]
        pub fn frag_main(input: Varyings) -> Vec4f {
            vec4f(input.uv.x, input.uv.y, f32(input.material), 1.0)
        }
    }

    #[test]
    fn integer_location_io_carries_explicit_flat() {
        let src = integer_io::WGSL_SOURCE
            .wgsl_source()
            .expect("non-template source renders");
        assert!(
            src.contains("@location(1) @interpolate(flat) material: u32"),
            "integer location must carry explicit @interpolate(flat), got:\n{src}"
        );
        assert!(
            !src.contains("@location(0) @interpolate"),
            "float location keeps default interpolation, got:\n{src}"
        );
    }
}
