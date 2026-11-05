//! IR-level source assembly.
//!
//! [`Source::assemble_ir`] is the IR-level counterpart of
//! [`Source::wgsl_source`]: instead of rendering each imported source to
//! WGSL text, it concatenates their IR items — imported sources first
//! (depth-first, deduplicated by [`Source::id`]), then this module's own
//! items, then cross-source template instantiations. The macro-emitted
//! `instantiate()` (concrete modules) and `instantiate::<T>()` (template
//! modules) both return this assembly, so every consumer that builds a
//! pipeline from a module sees every imported item (wgsl-rs#195).
//!
//! The walk mirrors the text path exactly, including its two safeguards:
//! instances that a provider's own compile-time monomorphization already
//! emitted are skipped (the `emitted_names` cross-chunk dedup), and
//! references to generic templates that were never resolved fail with
//! the structured [`SourceError::UnresolvedTemplateRef`] (wgsl-rs#237)
//! instead of leaking an un-mangled identifier into WGSL validation.
//!
//! Template sources assemble too: their own items keep their
//! `Type::TypeParam` nodes, which `instantiate::<T>()` replaces with
//! concrete types after assembly. Imported sources are concrete by
//! construction, so the substitution leaves them untouched.
//!
//! Deshadowing is deliberately deferred, matching the linkage analyzer's
//! expectations: `analyze_ir_module` deshadows the assembled IR, and the
//! text path deshadows per chunk during [`Source::wgsl_source`].
//! Deshadowing here as well would be redundant (the pass is idempotent
//! but wastes work). The #237 checks below do not need deshadowed input:
//! the detector matches type names, which the deshadow pass (a
//! local-variable renamer) never touches.

use std::collections::HashSet;

use crate::{
    Source, SourceError, build_instance_items, collect_template_names, ir, resolve_template,
};

impl Source {
    /// Returns the fully assembled IR for this source and all its
    /// imports, including cross-source template instantiations.
    ///
    /// This is the IR-level counterpart of [`Self::wgsl_source`]: the
    /// same import graph (every edge, any visibility), the same
    /// re-export-aware template resolution, and the same safeguards —
    /// monomorphization cross-chunk dedup and the wgsl-rs#237
    /// unresolved-reference check — but concatenating IR items instead
    /// of rendering text. Template sources are allowed — their own
    /// items keep `Type::TypeParam` nodes for [`ir::substitute_types`]
    /// to replace afterwards.
    ///
    /// Deshadowing is not performed: the linkage analyzer deshadows
    /// during `analyze_ir_module`, and the text path deshadows per
    /// chunk. When you want WGSL text, prefer [`Self::wgsl_source`].
    ///
    /// The returned IR is owned and freshly constructed on each call.
    pub fn assemble_ir(&self) -> Result<ir::Module, SourceError<'_>> {
        let mut visited: HashSet<u64> = HashSet::new();
        // Seed the assembly root's own id: when the root sits inside an
        // import cycle (A -> B -> A), the recursive descent through the
        // imports would otherwise re-enter and emit the root's items a
        // second time before the outer frame appends them (the loop
        // below only marks nodes it descends *into*).
        visited.insert(self.id);
        let mut seen: HashSet<(u64, String, Vec<String>, Vec<String>)> = HashSet::new();
        // Top-level struct and function names actually emitted by
        // assembled source chunks, for the instance cross-chunk dedup.
        let mut emitted_names: HashSet<String> = HashSet::new();
        let mut items: Vec<ir::Item> = Vec::new();
        collect_ir_items(
            self,
            &mut visited,
            &mut seen,
            &mut emitted_names,
            &mut items,
        )?;
        Ok(ir::Module {
            name: self.name,
            items,
            attrs: Vec::new(),
        })
    }
}

/// Concatenate `source` and its transitive imports' IR items into `out`
/// (imports first, depth-first, deduplicated by id), then append this
/// source's own items and its cross-source template instantiations.
fn collect_ir_items<'a>(
    source: &'a Source,
    visited: &mut HashSet<u64>,
    seen: &mut HashSet<(u64, String, Vec<String>, Vec<String>)>,
    emitted_names: &mut HashSet<String>,
    out: &mut Vec<ir::Item>,
) -> Result<(), SourceError<'a>> {
    // Imports first (depth-first, deduplicated by source id).
    for m in source.imports {
        if visited.insert(m.id) {
            collect_ir_items(m, visited, seen, emitted_names, out)?;
        }
    }

    // Public re-export edges fold the same way: downstream modules can
    // name items through them (glob re-exports expose every name of
    // their source, named re-exports exactly one), so the re-exported
    // chunks must be present in the assembly. Without this, an item
    // re-exported through a hop would reference a definition that never
    // lands (wgsl-rs#114).
    for reexport in source.reexports {
        if visited.insert(reexport.id) {
            collect_ir_items(reexport, visited, seen, emitted_names, out)?;
        }
    }
    for named in source.named_exports {
        if visited.insert(named.source.id) {
            collect_ir_items(named.source, visited, seen, emitted_names, out)?;
        }
    }

    // This source's own items. Flag references to generic templates
    // that were never resolved (wgsl-rs#237), mirroring the text path's
    // per-chunk check: the name set is the names *this module* can
    // name, so middle modules' private imports are covered too.
    let ir_module = (source.ir_constructor)();
    let mut template_names: HashSet<String> = HashSet::new();
    collect_template_names(source, &mut template_names);
    if let Some(offense) = ir::find_unresolved_template_refs(&ir_module.items, &template_names) {
        return Err(SourceError::UnresolvedTemplateRef {
            source_name: source.name.to_string(),
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
    out.extend(ir_module.items);

    // Cross-source template instantiations.
    for inst in source.instantiations {
        let mangled: Vec<String> = inst
            .mangled_type_args
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let mangled_consts: Vec<String> = inst
            .mangled_const_args
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let type_args = (inst.type_args_constructor)();
        let const_args = (inst.const_args_constructor)();
        collect_ir_instantiation(
            inst.modules,
            inst.template_name,
            &mangled,
            &mangled_consts,
            &type_args,
            &const_args,
            seen,
            emitted_names,
            out,
        )?;
    }
    Ok(())
}

/// Resolve `template_name` among `sources` — through public re-export
/// edges, exactly like text assembly (wgsl-rs#238) — and append the
/// instantiated items to `out`, dependencies first, deduplicating on
/// `(defining source id, template name, mangled type args, mangled
/// const args)` and against names already emitted by source chunks.
#[allow(clippy::too_many_arguments)]
fn collect_ir_instantiation<'a>(
    sources: &[&'a Source],
    template_name: &str,
    mangled_type_args: &[String],
    mangled_const_args: &[String],
    type_args: &[ir::Type],
    const_args: &[i64],
    seen: &mut HashSet<(u64, String, Vec<String>, Vec<String>)>,
    emitted_names: &mut HashSet<String>,
    out: &mut Vec<ir::Item>,
) -> Result<(), SourceError<'a>> {
    let (source, template) = resolve_template(sources, template_name, mangled_type_args)?;

    let key = (
        source.id,
        template_name.to_string(),
        mangled_type_args.to_vec(),
        mangled_const_args.to_vec(),
    );
    if !seen.insert(key) {
        return Ok(()); // Already assembled.
    }

    // Recursively assemble dependencies first.
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
        // `type_param_mapping` is a future extension, matching text
        // assembly.
        collect_ir_instantiation(
            &[source],
            dep.callee,
            &dep_mangled,
            mangled_const_args,
            &dep_args,
            const_args,
            seen,
            emitted_names,
            out,
        )?;
    }

    let items = build_instance_items(
        template,
        type_args,
        const_args,
        mangled_type_args,
        mangled_const_args,
    );

    // Cross-chunk dedup, mirroring the text path: the provider's own
    // source chunk may already contain this very instantiation —
    // compile-time monomorphization emits concrete items into the
    // provider's IR, and appending the runtime instance again would
    // double-define the struct (or fn). The check is against names
    // *emitted by assembled chunks* (`emitted_names`), not against the
    // whole item list.
    let already_emitted = items.iter().find_map(|item| match item {
        ir::Item::Struct(s) => Some(emitted_names.contains(&s.name)),
        ir::Item::Fn(f) => Some(emitted_names.contains(f.name.as_ref())),
        _ => None,
    });
    if already_emitted == Some(true) {
        return Ok(());
    }

    // Flag references to *other* generic templates that this
    // instance's body could not resolve (wgsl-rs#237), mirroring the
    // text path: the body was written in the defining module, so it is
    // checked against the names *that module* can name. Without this
    // the broken name reaches WGSL validation as an un-mangled
    // identifier.
    let mut template_names: HashSet<String> = HashSet::new();
    collect_template_names(source, &mut template_names);
    if let Some(offense) = ir::find_unresolved_template_refs(&items, &template_names) {
        return Err(SourceError::UnresolvedTemplateRef {
            source_name: source.name.to_string(),
            template_name: offense.template,
            member: offense.member,
            has_type_args: offense.has_type_args,
        });
    }

    out.extend(items);
    Ok(())
}
