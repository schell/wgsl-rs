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
//! Template sources assemble too: their own items keep their
//! `Type::TypeParam` nodes, which `instantiate::<T>()` replaces with
//! concrete types after assembly. Imported sources are concrete by
//! construction, so the substitution leaves them untouched.
//!
//! Deshadowing is deliberately deferred, matching the linkage analyzer's
//! expectations: `analyze_ir_module` deshadows the assembled IR, and the
//! text path deshadows per chunk during [`Source::wgsl_source`].
//! Deshadowing here as well would be redundant (the pass is idempotent
//! but wastes work).

use std::collections::HashSet;

use crate::{Source, SourceError, build_instance_items, ir, resolve_template};

impl Source {
    /// Returns the fully assembled IR for this source and all its
    /// imports, including cross-source template instantiations.
    ///
    /// This is the IR-level counterpart of [`Self::wgsl_source`]: the
    /// same import graph (every edge, any visibility) and the same
    /// re-export-aware template resolution, but concatenating IR items
    /// instead of rendering text. Template sources are allowed — their
    /// own items keep `Type::TypeParam` nodes for
    /// [`ir::substitute_types`] to replace afterwards.
    ///
    /// Deshadowing is not performed: the linkage analyzer deshadows
    /// during `analyze_ir_module`, and the text path deshadows per
    /// chunk. When you want WGSL text, prefer [`Self::wgsl_source`].
    ///
    /// The returned IR is owned and freshly constructed on each call.
    pub fn assemble_ir(&self) -> Result<ir::Module, SourceError<'_>> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut seen: HashSet<(u64, String, Vec<String>, Vec<String>)> = HashSet::new();
        let mut items: Vec<ir::Item> = Vec::new();
        collect_ir_items(self, &mut visited, &mut seen, &mut items)?;
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
    out: &mut Vec<ir::Item>,
) -> Result<(), SourceError<'a>> {
    // Imports first (depth-first, deduplicated by source id).
    for m in source.imports {
        if visited.insert(m.id) {
            collect_ir_items(m, visited, seen, out)?;
        }
    }

    // This source's own items.
    let ir_module = (source.ir_constructor)();
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
            out,
        )?;
    }
    Ok(())
}

/// Resolve `template_name` among `sources` — through public re-export
/// edges, exactly like text assembly (wgsl-rs#238) — and append the
/// instantiated items to `out`, dependencies first, deduplicating on
/// `(defining source id, template name, mangled type args, mangled
/// const args)`.
#[allow(clippy::too_many_arguments)]
fn collect_ir_instantiation<'a>(
    sources: &[&'a Source],
    template_name: &str,
    mangled_type_args: &[String],
    mangled_const_args: &[String],
    type_args: &[ir::Type],
    const_args: &[i64],
    seen: &mut HashSet<(u64, String, Vec<String>, Vec<String>)>,
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
    out.extend(items);
    Ok(())
}
