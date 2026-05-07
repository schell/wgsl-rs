//! Intermediate representation (IR) for WGSL shader modules.
//!
//! `wgsl-rs-ir` is a standalone crate that defines an owned, serializable
//! representation of a WGSL shader module. It is shared between the
//! `wgsl-rs-macros` proc-macro crate and the `wgsl-rs` runtime library so
//! that:
//!
//! * The proc-macro can convert its `syn`-based parse tree to IR and emit
//!   constructor functions that build IR at runtime.
//! * The runtime library can render IR to WGSL source, perform type
//!   substitution for generic instantiation, and serve as the canonical
//!   representation for cross-module generic templates.
//!
//! All identifiers are stored as `String`, all collections as `Vec<T>`, and
//! all literals as plain Rust types. The IR has no dependency on `syn`,
//! `proc-macro2`, or `quote`.
//!
//! The IR is intentionally semantic and rich. It mirrors the parse tree of
//! `wgsl-rs-macros` and preserves things like field names in struct
//! expressions, un-mangled method names in impl blocks, and Rust-style
//! builtin function names. All WGSL-specific lowering (struct positional
//! args, impl method name mangling, builtin name translation, enum
//! discriminant auto-increment) lives in the [`render`] module.

mod render;
mod substitute;
mod types;

pub use render::{render_items, render_module};
pub use substitute::{rename_items, substitute_items, substitute_types, type_to_ident};
pub use types::*;
