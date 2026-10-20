//! Errors produced by [`Source`]-level operations.
//!
//! These errors are about the *source* (the spec the user wrote) —
//! distinct from the `wgsl_rs_linkage_wgpu::Error` type, which is about
//! producing a wgpu `WgpuLinkage` from an already-analyzed `ir::Module`
//! in the `wgsl-rs-linkage-wgpu` crates. (That error type lives outside
//! this crate, so it's referenced here as a plain code span rather than
//! an intra-doc link to keep doc builds warning-free.)

use crate::Source;

/// Errors that [`Source::wgsl_source`] can produce.
///
/// The lifetime parameter `'a` is the borrow of the source that triggered
/// the error. A template `Source` is typically a `&'static Source` (it
/// lives in static storage, emitted by the macro), so the error is
/// usually `SourceError<'static>`. The lifetime parameter is present so
/// the error can borrow from `&self` of a non-static source if needed.
#[derive(Debug)]
pub enum SourceError<'a> {
    /// The source is a template (unresolved module-level type parameters);
    /// the caller must use the macro-emitted `instantiate::<…>()` to
    /// obtain a concrete [`crate::ir::Module`] first, then call
    /// [`crate::ir::Module::wgsl_source`] on that.
    TemplateWgsl {
        /// The uninstantiated source that triggered the error.
        uninstantiated_source: &'a Source,
    },
    /// No candidate source declares a template with the requested name.
    /// This is a configuration error in the source graph (e.g. a missing
    /// `wgsl!` module import), not a runtime data error.
    TemplateNotFound {
        template_name: String,
        mangled_type_args: Vec<String>,
        available_templates: Vec<String>,
    },
    /// More than one candidate source declares a template with the
    /// requested name, so the instantiation is ambiguous. Disambiguate
    /// by removing one of the matching sources from the import graph.
    AmbiguousTemplate {
        template_name: String,
        mangled_type_args: Vec<String>,
        matching_source_names: Vec<String>,
        available_templates: Vec<String>,
    },
    /// A reference to a generic template that was never resolved to a
    /// concrete instantiation, detected while assembling WGSL source
    /// (wgsl-rs#237). The macro-time check catches same-module
    /// mistakes; this catches cross-module ones, where the consuming
    /// module's macro cannot know that the referenced name is generic.
    UnresolvedTemplateRef {
        /// Name of the source whose items contain the reference.
        source_name: String,
        /// The referenced template's name (e.g. `Id`).
        template_name: String,
        /// The referenced member, for method calls and associated
        /// constants; `None` for bare calls to generic functions.
        member: Option<String>,
        /// Whether the reference carried explicit type arguments that
        /// could not be resolved (e.g. `Id::<T>::new(...)` inside a
        /// generic context).
        has_type_args: bool,
    },
}

impl<'a> std::fmt::Display for SourceError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::TemplateWgsl {
                uninstantiated_source,
            } => {
                write!(
                    f,
                    "source '{}' is a template (has module-level type parameters {:?}); call the \
                     macro-emitted `instantiate::<…>()` (types in `module_type_params` order) \
                     then `.wgsl_source()` to render concrete WGSL",
                    uninstantiated_source.name, uninstantiated_source.module_type_params,
                )
            }
            SourceError::TemplateNotFound {
                template_name,
                mangled_type_args,
                available_templates,
            } => {
                write!(
                    f,
                    "unable to resolve template '{template_name}' for type args \
                     {mangled_type_args:?}; available templates: {available_templates:?}",
                )
            }
            SourceError::AmbiguousTemplate {
                template_name,
                mangled_type_args,
                matching_source_names,
                available_templates,
            } => {
                write!(
                    f,
                    "ambiguous template instantiation '{template_name}' for type args \
                     {mangled_type_args:?}; matching sources: {matching_source_names:?}; \
                     available templates: {available_templates:?}",
                )
            }
            SourceError::UnresolvedTemplateRef {
                source_name,
                template_name,
                member,
                has_type_args,
            } => {
                let what = match member {
                    Some(member) => format!("'{template_name}::{member}'"),
                    None => format!("'{template_name}'"),
                };
                if *has_type_args {
                    write!(
                        f,
                        "source '{source_name}' references {what} of generic template \
                         '{template_name}' with type arguments that could not be resolved; calls \
                         to generic templates from generic contexts (template bodies, generic \
                         entry points) are not yet supported — call it with concrete type \
                         arguments from a concrete function instead",
                    )
                } else {
                    write!(
                        f,
                        "source '{source_name}' calls generic template {what} without generic \
                         arguments; supply them with turbofish, e.g. \
                         '{template_name}::<ConcreteType>::…'",
                    )
                }
            }
        }
    }
}

impl<'a> std::error::Error for SourceError<'a> {}
