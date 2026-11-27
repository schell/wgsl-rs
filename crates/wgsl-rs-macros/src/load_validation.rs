//! Compile-time validation of access-macro targets (`get!`, `get_mut!`,
//! `load!`).
//!
//! Resource packs are the only module-level binding declaration, so the
//! access macros accept exactly one target shape: a resource-pack
//! parameter's field, `pack.field` or `&pack.field`. Bare identifiers
//! (the old module-static targets) no longer exist; this pass rejects
//! them with span-pointed migration guidance before other passes run.
//!
//! Everything else about a pack target (handle kinds, uniform mutability,
//! storage access mode, atomics, runtime-sized arrays, the `, T` form) is
//! validated against the pack's fields by `resource::BodyLowerer`.

use crate::{
    parse::{Error, Expr, Item, LinkageKind},
    parse_visitor::{ParseVisitorMut, walk_expr},
};

pub(crate) fn validate_items(items: &mut [Item]) -> Result<(), Error> {
    let mut visitor = AccessTargetVisitor;
    for item in items {
        visitor.visit_item(item)?;
    }
    Ok(())
}

/// Read-only visitor that rejects bare-identifier access-macro targets.
struct AccessTargetVisitor;

impl ParseVisitorMut for AccessTargetVisitor {
    fn visit_expr(&mut self, e: &mut Expr) -> Result<(), Error> {
        if let Expr::LinkageAccess {
            kind,
            ident,
            pack_field: None,
            ..
        } = e
        {
            return Err(not_a_pack_field(kind, ident));
        }
        walk_expr(self, e)
    }
}

/// Error for a bare-identifier access-macro target (the removed
/// module-static form).
fn not_a_pack_field(kind: &LinkageKind, ident: &syn::Ident) -> Error {
    let macro_name = match kind {
        LinkageKind::Get => "get",
        LinkageKind::GetMut => "get_mut",
        LinkageKind::Load => "load",
    };
    Error::unsupported(
        ident.span(),
        format!(
            "{macro_name}!({ident}) — `{ident}` is not a resource-pack field; declare an \
             annotated pack field (e.g. `#[uniform(group(0), binding(0))] pub frame: \
             Uniform<u32>`) and target it as `{macro_name}!(pack.field)`"
        ),
    )
}
