//! Compile-time validation of `load!` targets.
//!
//! `load!` copies a module variable's value into a local; in WGSL it
//! renders as a bare value read of the variable. Not every binding kind
//! supports that:
//!
//! * textures and samplers are handles, not values
//! * atomics must be read with `atomicLoad` (`atomic_load(&get!(VAR))`)
//! * runtime-sized arrays are not copyable values
//!
//! These checks run at macro-expansion time, replacing what would
//! otherwise be a runtime wgpu validation failure with a span-pointed
//! compile error.
//!
//! Generic linkage variables (declared with `impl Trait`) fail open:
//! their concrete type is only known at instantiation, and the
//! auto-validation path (`validate_with_instantiation_types`) plus
//! runtime wgpu validation backstop those cases. These checks are a
//! footgun net, not a type system.

use std::collections::HashMap;

use crate::{
    parse::{Error, Expr, Item, LinkageKind, Type, is_wgsl_std_import},
    parse_visitor::{ParseVisitorMut, walk_expr},
};

/// A module variable that a `load!` can target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModuleVarKind {
    /// `uniform!` — always readable.
    Uniform,
    /// `storage!` — wgsl-rs only supports `read_only` and `read_write`
    /// access modes, both of which permit reads.
    Storage,
    /// `workgroup!` — a shared value in the workgroup address space.
    Workgroup,
    /// `texture!` and `sampler!` — handles, not values.
    Handle,
}

/// Everything the validator needs about one declared module variable.
struct ModuleVarInfo {
    kind: ModuleVarKind,
    /// The declared WGSL type. `Type::TypeParam` means the concrete type
    /// is not knowable in this module (an `impl Trait` declaration or a
    /// type-parameterized template context).
    ty: Type,
}

/// The module's declaration table used to validate `load!` targets.
pub(crate) struct ModuleVarTable {
    vars: HashMap<String, ModuleVarInfo>,
    /// Whether the module imports from other modules. Imported module
    /// variables are not visible to this table, so unknown idents fail
    /// open when imports exist.
    has_imports: bool,
}

impl ModuleVarTable {
    /// Build the table from the module's parsed items.
    pub fn from_items(items: &[Item], crate_path: &syn::Path) -> Self {
        let mut vars = HashMap::new();
        let mut has_imports = false;
        for item in items {
            match item {
                Item::Uniform(u) => {
                    vars.insert(
                        u.name.to_string(),
                        ModuleVarInfo {
                            kind: ModuleVarKind::Uniform,
                            ty: u.ty.clone(),
                        },
                    );
                }
                Item::Storage(s) => {
                    vars.insert(
                        s.name.to_string(),
                        ModuleVarInfo {
                            kind: ModuleVarKind::Storage,
                            ty: s.ty.clone(),
                        },
                    );
                }
                Item::Workgroup(w) => {
                    vars.insert(
                        w.name.to_string(),
                        ModuleVarInfo {
                            kind: ModuleVarKind::Workgroup,
                            ty: w.ty.clone(),
                        },
                    );
                }
                Item::Sampler(s) => {
                    vars.insert(
                        s.name.to_string(),
                        ModuleVarInfo {
                            kind: ModuleVarKind::Handle,
                            ty: s.ty.clone(),
                        },
                    );
                }
                Item::Texture(t) => {
                    vars.insert(
                        t.name.to_string(),
                        ModuleVarInfo {
                            kind: ModuleVarKind::Handle,
                            ty: t.ty.clone(),
                        },
                    );
                }
                Item::Use(u) => {
                    for module in &u.modules {
                        // `use <crate_path>::std::*` is the built-in prelude
                        // import; it never carries module variables. Any
                        // other import may be another wgsl module, whose
                        // variables this table cannot see. The check matches
                        // the full crate path so a user module named `std`
                        // is not mistaken for the prelude.
                        if !is_wgsl_std_import(crate_path, module) {
                            has_imports = true;
                        }
                    }
                }
                _ => {}
            }
        }
        Self { vars, has_imports }
    }
}

/// Validate every `load!` access in `items` against `table`.
pub(crate) fn validate_items(table: &ModuleVarTable, items: &mut [Item]) -> Result<(), Error> {
    let mut visitor = LoadVisitor { table };
    for item in items {
        visitor.visit_item(item)?;
    }
    Ok(())
}

/// Read-only visitor that intercepts `load!` accesses and checks them
/// against the module's declaration table.
struct LoadVisitor<'a> {
    table: &'a ModuleVarTable,
}

impl ParseVisitorMut for LoadVisitor<'_> {
    fn visit_expr(&mut self, e: &mut Expr) -> Result<(), Error> {
        if let Expr::LinkageAccess {
            kind: LinkageKind::Load,
            ident,
            type_arg,
        } = e
        {
            self.check(ident, type_arg.as_ref())?;
        }
        walk_expr(self, e)
    }
}

impl LoadVisitor<'_> {
    /// Check one `load!` access against the declaration table.
    fn check(&self, ident: &syn::Ident, type_arg: Option<&syn::Type>) -> Result<(), Error> {
        let Some(info) = self.table.vars.get(&ident.to_string()) else {
            if self.table.has_imports {
                // Could be a module variable imported from another wgsl
                // module, which this table cannot see. Fail open; wgpu
                // validation is the backstop.
                return Ok(());
            }
            return Err(Error::unsupported(
                ident.span(),
                format!(
                    "load!({ident}) — '{ident}' is not a uniform!, storage! or workgroup! \
                     variable declared in this module"
                ),
            ));
        };
        if info.kind == ModuleVarKind::Handle {
            return Err(Error::unsupported(
                ident.span(),
                format!(
                    "load!({ident}) — textures and samplers are handles in WGSL, not loadable \
                     values; use the texture builtins (e.g. textureLoad) instead"
                ),
            ));
        }
        // Declared-type shape errors apply to both forms — the declaration
        // is the source of truth for what the variable holds.
        match &info.ty {
            Type::Atomic { .. } => return Err(not_loadable(ident, "Atomic")),
            Type::RuntimeArray { .. } => return Err(not_loadable(ident, "RuntimeArray")),
            _ => {}
        }
        match type_arg {
            // One-argument form: fine on concrete variables; generic ones
            // need the explicit type.
            None => match &info.ty {
                Type::TypeParam { .. } => Err(Error::unsupported(
                    ident.span(),
                    format!(
                        "load!({ident}) — '{ident}' is a generic module variable; use the \
                         two-argument form load!({ident}, T)"
                    ),
                )),
                _ => Ok(()),
            },
            // Two-argument form: for generic variables only. On a generic
            // declaration, check the syntactically outermost constructor
            // of the type argument — anything else may still be generic
            // (`Vec4<T>`), so fail open to instantiation and wgpu
            // validation.
            Some(ty) => match &info.ty {
                Type::TypeParam { .. } => {
                    if let Some(name) = outermost_path_ident(ty)
                        && (name == "Atomic" || name == "RuntimeArray")
                    {
                        return Err(not_loadable(ident, &name));
                    }
                    Ok(())
                }
                // Concrete variables have no `get_typed` on the CPU side, so
                // the two-argument form would fail to compile there anyway;
                // reject it here with a clearer note.
                _ => Err(Error::unsupported(
                    ident.span(),
                    format!(
                        "load!({ident}, T) — '{ident}' is a concrete module variable; use the \
                         one-argument form load!({ident})"
                    ),
                )),
            },
        }
    }
}

/// Error for a `load!` on a type that cannot be copied in WGSL.
fn not_loadable(ident: &syn::Ident, ty: &str) -> Error {
    let note = if ty == "Atomic" {
        format!(
            "load!({ident}) — atomics must be read with atomicLoad; write \
             atomic_load(&get!({ident}))"
        )
    } else {
        format!(
            "load!({ident}) — runtime-sized arrays are not copyable WGSL values; index into \
             get!({ident}) or copy elements with slab_copy!"
        )
    };
    Error::unsupported(ident.span(), note)
}

/// The outermost identifier of a path type (`Vec4<T>` → `Vec4`), if `ty`
/// is a path type.
fn outermost_path_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}
