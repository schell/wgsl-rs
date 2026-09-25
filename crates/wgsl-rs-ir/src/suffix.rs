//! Type-directed integer literal suffix insertion.
//!
//! Rust infers the type of an unsuffixed integer literal from context —
//! `0` in a `u32` function is `u32`. WGSL instead defaults unsuffixed
//! integer literals to `i32` wherever the surrounding context does not
//! force a concrete type (most visibly inside polymorphic builtins like
//! `select`), so Rust that compiles cleanly can render to WGSL that naga
//! rejects (wgsl-rs#145).
//!
//! This pass propagates expected types from anchor points down to leaf
//! literals and writes Rust-style suffixes (`u32` / `i32`) onto
//! empty-suffix integer literals, the exact forms [`crate::render`]
//! translates to WGSL `u` / `i`. Anchors:
//!
//! * function return types (`return` statements and trailing expressions),
//! * typed `let`/`var` initializers and `const` items,
//! * assignment and compound-assignment targets (identifiers, fields, array
//!   elements, swizzles),
//! * array literals and struct constructor fields (struct constructors
//!   self-anchor through the struct registry even without an outer annotation),
//! * casts,
//! * binary and comparison operands (a bare literal adopts the provable scalar
//!   type of the other operand),
//! * `for` bounds from the loop variable's type — provable from a range bound,
//!   or probed backward from the body's uses of the variable (`for i in 0..1 {
//!   let x: u32 = i + 1; }` infers `i: u32`, wgsl-rs#154),
//! * `switch` case selectors from the selector's type.
//!
//! Like [`crate::deshadow`], this is a whole-module IR mutation intended
//! to run on freshly built and substituted IR, after deshadowing and
//! immediately before rendering (see the DEVLOG entry for 2026-09-21).
//! Call-argument propagation is part of this pass: user function
//! signatures (seeded across sources via [`fn_signatures`] /
//! [`suffix_module_with_imports`]), same-type builtin groups (`select` /
//! `min` / `max` / `clamp`), and vector constructors.
//!
//! Deliberate exclusions:
//!
//! * Type positions ([`Type::Array`] lengths, [`Expr::ZeroValueArray`] lengths)
//!   are never suffixed — WGSL wants abstract integers there.
//! * Array-index expressions stay bare: WGSL coerces the abstract integer in
//!   `a[0]` to either index type, so there is nothing to fix.
//! * Float expectations never touch integer literals: the IR cannot contain an
//!   integer literal in an `f32` context, because Rust would have rejected the
//!   source.

use std::collections::HashMap;

use crate::{
    mangle::mangle,
    types::{
        AddressSpace, BinOp, Block, CaseSelector, CompoundOp, ElseBranch, Expr, FnPath, ForLoop,
        ImplItem, Item, ItemFn, Lit, Local, Module, ReturnType, ScalarType, Stmt, StmtIf,
        StmtSwitch, Type, UnOp,
    },
};

/// Insert type-directed suffixes onto unsuffixed integer literals across
/// every item in `module`.
///
/// Only literals with an empty suffix are written, so the pass is
/// idempotent and safe to run more than once.
pub fn suffix_module(module: &mut Module) {
    SuffixPass::default().walk_module(module);
}

/// Like [`suffix_module`], but for a bare slice of items — e.g. an
/// instantiated cross-source template that is rendered without a
/// [`Module`] wrapper.
pub fn suffix_items(items: &mut [Item]) {
    SuffixPass::default().walk_items(items);
}

/// Like [`suffix_module`], but seeds the user-function signature registry
/// with `imports` — signatures harvested from other sources via
/// [`fn_signatures`] — so calls to imported functions anchor their
/// arguments too. The module's own signatures shadow imported ones,
/// matching Rust name resolution.
pub fn suffix_module_with_imports(module: &mut Module, imports: &HashMap<String, FnSig>) {
    SuffixPass {
        env: TypeEnv {
            fn_sigs: imports.clone(),
            ..TypeEnv::default()
        },
        ..SuffixPass::default()
    }
    .walk_module(module);
}

/// Like [`suffix_items`], but seeds the user-function signature registry
/// with `imports` — signatures harvested from other sources via
/// [`fn_signatures_in_items`] — so calls to imported functions anchor
/// their arguments too. Used for cross-source template instantiation,
/// where the template's items reference functions defined in other
/// chunks of the assembled translation unit.
pub fn suffix_items_with_imports(items: &mut [Item], imports: &HashMap<String, FnSig>) {
    SuffixPass {
        env: TypeEnv {
            fn_sigs: imports.clone(),
            ..TypeEnv::default()
        },
        ..SuffixPass::default()
    }
    .walk_items(items);
}

/// Like [`suffix_module`], but seeds the type environment with
/// `imports` — globals, struct definitions, and user-fn signatures
/// harvested from other sources via [`type_imports`] — so references
/// to imported names anchor too: `get_mut!(IMPORTED_OUTPUT)[0] =
/// select(0, 1, cond)` anchors from the imported storage's element
/// type, `min(IMPORTED_LIMIT, 0)` from the imported const, and an
/// imported struct's constructor fields from its definition. The
/// module's own declarations shadow imported ones, matching Rust
/// name resolution.
pub fn suffix_module_with_type_imports(module: &mut Module, imports: &TypeImports) {
    SuffixPass {
        env: imports.as_env(),
        return_ty: None,
        ..SuffixPass::default()
    }
    .walk_module(module);
}

/// Like [`suffix_items`], but seeds the type environment with
/// `imports` — see [`suffix_module_with_type_imports`]. Used for
/// cross-source template instantiation, where the template's items
/// reference globals, structs, and functions defined in other chunks
/// of the assembled translation unit.
pub fn suffix_items_with_type_imports(items: &mut [Item], imports: &TypeImports) {
    SuffixPass {
        env: imports.as_env(),
        return_ty: None,
        ..SuffixPass::default()
    }
    .walk_items(items);
}

/// A harvested user-function signature: declared parameter types plus
/// the return type (`None` for `()` returns).
#[derive(Clone, Debug)]
pub struct FnSig {
    /// Declared parameter types, in order.
    pub params: Vec<Type>,
    /// The declared return type, if any.
    pub ret: Option<Type>,
}

/// The signature of one function item.
fn fn_sig(f: &ItemFn) -> FnSig {
    FnSig {
        params: f.inputs.iter().map(|a| a.ty.clone()).collect(),
        ret: match &f.return_type {
            ReturnType::Type { ty, .. } => Some(ty.clone()),
            ReturnType::Default => None,
        },
    }
}

/// Harvest the module's user-function signatures — callee name →
/// [`FnSig`] — for seeding [`suffix_module_with_imports`] /
/// [`suffix_items_with_imports`] on a source that imports or
/// instantiates this module. Free functions key by their name; impl
/// methods key by their mangled render name.
pub fn fn_signatures(module: &Module) -> HashMap<String, FnSig> {
    fn_signatures_in_items(&module.items)
}

/// Like [`fn_signatures`], but for a bare slice of items — e.g. an
/// instantiated cross-source template rendered without a [`Module`]
/// wrapper. Note that signatures are taken from the items as-is, so
/// call any renaming (such as template instance mangling) before
/// harvesting.
pub fn fn_signatures_in_items(items: &[Item]) -> HashMap<String, FnSig> {
    let mut sigs = HashMap::new();
    for item in items {
        match item {
            Item::Fn(f) => {
                sigs.insert(f.name.to_string(), fn_sig(f));
            }
            Item::Impl(imp) => {
                for impl_item in &imp.items {
                    if let ImplItem::Fn(f) = impl_item {
                        sigs.insert(mangle(&[imp.self_ty.as_str(), &f.name]), fn_sig(f));
                    }
                }
            }
            _ => {}
        }
    }
    sigs
}

/// A harvested type environment from other sources' items: the
/// module-level globals (const items, linkage declarations,
/// associated consts under mangled render names), struct
/// definitions, and user-fn signatures a consumer source needs to
/// anchor its literals.
///
/// Built by [`type_imports`] / [`type_imports_in_items`] and threaded
/// depth-first through source assembly (imports render first, so a
/// consumer is suffixed with every ancestor's names seeded); a
/// consumer's own declarations shadow imported ones, matching Rust
/// name resolution. The fields are opaque — construct via the
/// harvest fns, combine via [`TypeImports::extend`], and consume via
/// the `*_with_type_imports` suffix entry points — so the shared IR
/// vocabulary stays unpolluted.
#[derive(Clone, Debug, Default)]
pub struct TypeImports {
    /// Imported global names → types.
    globals: HashMap<String, Type>,
    /// Imported struct definitions by name.
    structs: HashMap<String, StructDef>,
    /// Imported user function signatures: callee name → [`FnSig`].
    /// Free functions key by their name; impl methods key by their
    /// mangled render name.
    fn_sigs: HashMap<String, FnSig>,
}

impl TypeImports {
    /// Extend with another source's harvested imports; later entries
    /// overwrite earlier ones on name collision.
    pub fn extend(&mut self, other: TypeImports) {
        self.globals.extend(other.globals);
        self.structs.extend(other.structs);
        self.fn_sigs.extend(other.fn_sigs);
    }

    /// The imports as a pre-seeded [`TypeEnv`] — used by the
    /// `*_with_type_imports` suffix entry points; the consuming
    /// pass's own collection then shadows these entries.
    fn as_env(&self) -> TypeEnv {
        TypeEnv {
            scopes: Vec::new(),
            globals: self.globals.clone(),
            structs: self.structs.clone(),
            fn_sigs: self.fn_sigs.clone(),
        }
    }
}

/// Harvest a module's type imports — its globals (module consts and
/// linkage declarations), struct definitions, and user-fn signatures —
/// for seeding a consuming source's suffix pass via
/// [`suffix_module_with_type_imports`] /
/// [`suffix_items_with_type_imports`]. Like [`fn_signatures`], but for
/// the whole anchoring environment.
pub fn type_imports(module: &Module) -> TypeImports {
    type_imports_in_items(&module.items)
}

/// Like [`type_imports`], but for a bare slice of items — e.g. an
/// instantiated cross-source template rendered without a [`Module`]
/// wrapper. Types are taken from the items as-is, so apply any
/// renaming (such as template instance mangling) before harvesting.
pub fn type_imports_in_items(items: &[Item]) -> TypeImports {
    let mut env = TypeEnv::default();
    env.collect(items);
    TypeImports {
        globals: env.globals,
        structs: env.structs,
        fn_sigs: env.fn_sigs,
    }
}

/// A collected struct definition: type parameter names plus fields as
/// `(name, type)` pairs.
#[derive(Clone, Debug)]
struct StructDef {
    type_params: Vec<String>,
    fields: Vec<(String, Type)>,
}

/// The suffix pass's type environment: everything needed to resolve
/// the concrete type of an expression bottom-up — the scope stack, the
/// struct registry, the user-function signature registry, and the
/// module-level globals. Collected once per module (plus any imported
/// signatures seeded by the `_with_imports` entry points) before the
/// mutation walk begins.
#[derive(Default)]
struct TypeEnv {
    /// Stack of scope frames mapping local names to their declared
    /// types. `None` marks a name declared in scope whose type is
    /// unprovable — it shadows outer bindings without anchoring from
    /// them. The bottom frame holds the current function's parameters.
    scopes: Vec<HashMap<String, Option<Type>>>,
    /// Struct definitions by name, collected before the walk so struct
    /// constructor fields and field assignments can resolve types.
    structs: HashMap<String, StructDef>,
    /// User function signatures: callee name → [`FnSig`]. Free
    /// functions key by their name; impl methods key by their mangled
    /// render name (`mangle(&[self_ty, method])`).
    fn_sigs: HashMap<String, FnSig>,
    /// Module-level global names → types: `const` items and linkage
    /// declarations (uniforms, storage, workgroup vars), consulted as
    /// the fallback of scope lookup.
    globals: HashMap<String, Type>,
}

impl TypeEnv {
    /// Collect struct definitions, user-function signatures, and global
    /// types from every top-level item. Runs ahead of the mutation
    /// walk; the walk mutates the items, so collection borrows first.
    fn collect(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Struct(s) => {
                    self.structs.insert(
                        s.name.clone(),
                        StructDef {
                            type_params: s.type_params.clone(),
                            fields: s
                                .fields
                                .iter()
                                .map(|f| (f.name.clone(), f.ty.clone()))
                                .collect(),
                        },
                    );
                }
                Item::Fn(f) => {
                    self.fn_sigs.insert(f.name.to_string(), fn_sig(f));
                }
                Item::Const(c) => {
                    self.globals.insert(c.name.clone(), c.ty.clone());
                }
                // Linkage declarations are lowered to `Expr::Ident` on
                // use (`get_mut!(OUTPUT)`), so their declared types
                // must be in the lookup environment.
                Item::Uniform(u) => {
                    self.globals.insert(u.name.clone(), u.ty.clone());
                }
                Item::Storage(s) => {
                    self.globals.insert(s.name.clone(), s.ty.clone());
                }
                Item::Workgroup(w) => {
                    self.globals.insert(w.name.clone(), w.ty.clone());
                }
                // Enums render as a `u32` alias plus per-variant consts
                // under the mangled `Enum_Variant` name, so both the
                // enum itself and its variants anchor as u32 values.
                Item::Enum(e) => {
                    self.globals
                        .insert(e.name.clone(), Type::Scalar(ScalarType::U32));
                    for variant in &e.variants {
                        self.globals.insert(
                            mangle(&[e.name.as_str(), &variant.name]),
                            Type::Scalar(ScalarType::U32),
                        );
                    }
                }
                Item::Impl(imp) => {
                    for impl_item in &imp.items {
                        match impl_item {
                            ImplItem::Fn(f) => {
                                self.fn_sigs
                                    .insert(mangle(&[imp.self_ty.as_str(), &f.name]), fn_sig(f));
                            }
                            // Associated consts render as the mangled
                            // `Type_MEMBER` name — the same key scheme
                            // as methods — so register their declared
                            // type under that key.
                            ImplItem::Const(c) => {
                                self.globals
                                    .insert(mangle(&[imp.self_ty.as_str(), &c.name]), c.ty.clone());
                            }
                            ImplItem::Type(_) => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Look up the declared type of `name`, innermost scope first.
    /// The innermost declaration wins even when its type is
    /// unprovable — a declared-but-unknown name shadows outer
    /// bindings without anchoring from them, since anchoring from a
    /// shadowed outer type can write the wrong suffix.
    /// Module-level globals — const items and linkage declarations —
    /// are the fallback after every scope misses; locals, parameters,
    /// and function-body consts shadow them, matching Rust name
    /// resolution.
    fn lookup(&self, name: &str) -> Option<Type> {
        match self.scopes.iter().rev().find_map(|s| s.get(name)) {
            Some(declared) => declared.clone(),
            None => self.globals.get(name).cloned(),
        }
    }

    /// The concrete type of `expr`, when provable without a full type
    /// checker: typed identifiers from scope, suffixed literals, casts,
    /// constructors, user-fn returns, and field / element accesses
    /// rooted at a typed base.
    ///
    /// The match is exhaustive over [`Expr`]: every variant either has
    /// a derivable rule or an explicit `None` with a comment naming
    /// why, so a new variant cannot silently fall through. `None`
    /// means "unprovable — do not anchor"; the pass never guesses a
    /// type (see the DEVLOG entry for 2026-09-22).
    fn infer(&self, expr: &Expr) -> Option<Type> {
        match expr {
            Expr::Ident(name) => self.lookup(name),
            Expr::Lit(lit) => match lit {
                Lit::Int { suffix, .. } => match suffix.as_str() {
                    "u32" | "usize" => Some(Type::Scalar(ScalarType::U32)),
                    "i32" | "isize" => Some(Type::Scalar(ScalarType::I32)),
                    _ => None,
                },
                // Bool and float literals carry no anchorable integer
                // type — a float literal cannot appear in an integer
                // context in valid Rust, and a bool never anchors one.
                Lit::Bool(_) | Lit::Float { .. } => None,
            },
            Expr::Paren(inner) => self.infer(inner),
            Expr::Cast { ty, .. } => Some((**ty).clone()),
            Expr::FieldAccess { base, field } => match self.infer(base)? {
                Type::Struct { name, type_args } => self.field_ty(&name, &type_args, field),
                // Plain component access on a vector (`v.x`): the
                // Rust-side vector types expose single-component
                // x/y/z/w/r/g/b/a fields and a plain field access
                // lowers here, unlike multi-component swizzles
                // (method calls → `Expr::Swizzle`). A single swizzle
                // character reads one scalar component of the
                // vector's element type.
                Type::Vector {
                    scalar_ty: Some(scalar),
                    ..
                } if field.len() == 1 && "xyzwrgba".contains(field.as_str()) => {
                    Some(Type::Scalar(scalar))
                }
                _ => None,
            },
            Expr::ArrayIndexing { lhs, .. } => match self.infer(lhs)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem } => Some((*elem).clone()),
                // Indexing a vector yields its scalar element —
                // `v[0] = select(0, 1, cond)` anchors from the
                // vector's element type (wgsl-rs#196).
                Type::Vector {
                    scalar_ty: Some(scalar),
                    ..
                } => Some(Type::Scalar(scalar)),
                // Indexing a matrix yields a column vector carrying
                // the matrix's scalar type; `m[i][j]` recurses
                // through the vector case (wgsl-rs#196).
                Type::Matrix {
                    rows,
                    scalar_ty: Some(scalar),
                    ..
                } => Some(Type::Vector {
                    elements: rows,
                    scalar_ty: Some(scalar),
                }),
                // Abstract vectors / matrices (`scalar_ty: None`) and
                // non-indexable bases carry no provable element type.
                _ => None,
            },
            Expr::Swizzle { lhs, swizzle, .. } => match self.infer(lhs)? {
                // A swizzle is vector-valued for lengths 2–4
                // (`v.xy()` → vec2 of the base scalar) and reads a
                // single scalar for a one-component access — the
                // component count decides the shape.
                Type::Vector {
                    scalar_ty: Some(scalar),
                    ..
                } => {
                    let elements = swizzle.len() as u8;
                    if elements == 1 {
                        Some(Type::Scalar(scalar))
                    } else {
                        Some(Type::Vector {
                            elements,
                            scalar_ty: Some(scalar),
                        })
                    }
                }
                // A swizzle on a matrix selects a column (a vector),
                // not a scalar — no scalar expectation is derivable.
                _ => None,
            },
            // Same-type arithmetic operators yield their operands'
            // type; shifts yield the shifted value's type (the count is
            // u32, not the result type); comparisons and logical
            // operators yield `bool`.
            Expr::Binary { lhs, op, rhs } => {
                if is_shift(op) {
                    self.infer(lhs)
                } else if is_arithmetic(op) {
                    // Component-wise scalar-vector arithmetic yields
                    // the vector shape, as `combine_arith` in the
                    // `vector_cmp` lowering resolves it: a vector
                    // operand dominates a scalar one, so
                    // `scale * vec4u(...)` infers as a vector, not the
                    // scalar of its left operand.
                    let lhs_ty = self.infer(lhs);
                    let rhs_ty = self.infer(rhs);
                    match (&lhs_ty, &rhs_ty) {
                        (Some(Type::Vector { .. }), _) => lhs_ty,
                        (_, Some(Type::Vector { .. })) => rhs_ty,
                        _ => lhs_ty.or(rhs_ty),
                    }
                } else {
                    Some(Type::Scalar(ScalarType::Bool))
                }
            }
            // `!`, bitwise complement, and negation preserve the
            // operand type; a deref yields the pointee of a provable
            // pointer operand.
            Expr::Unary { op, expr } => match op {
                UnOp::Not | UnOp::Complement | UnOp::Neg => self.infer(expr),
                UnOp::Deref => match self.infer(expr)? {
                    Type::Ptr { elem, .. } => Some((*elem).clone()),
                    _ => None,
                },
            },
            // Array literals yield their element type, derivable from
            // any single inferable element — Rust unifies the element
            // type across the literal, so a later suffixed element
            // carries the same information as the first; the length is
            // the element count.
            Expr::Array { elems } => {
                let elem = elems.iter().find_map(|e| self.infer(e))?;
                Some(Type::Array {
                    elem: Box::new(elem),
                    len: Expr::Lit(Lit::Int {
                        digits: elems.len().to_string(),
                        suffix: "u32".to_string(),
                    }),
                })
            }
            // `[T; N]()` zero-value arrays carry their full type.
            Expr::ZeroValueArray { elem_type, len } => Some(Type::Array {
                elem: elem_type.clone(),
                len: (**len).clone(),
            }),
            // Struct constructors yield the constructed struct type
            // (with the constructor's own type arguments).
            Expr::Struct {
                name,
                type_args,
                fields: _,
            } => Some(Type::Struct {
                name: name.clone(),
                type_args: type_args.clone(),
            }),
            // Same-type builtin value groups yield their first value
            // argument's type; user functions yield their declared
            // return type. User signatures shadow builtins, matching
            // the walk order.
            Expr::FnCall {
                path,
                type_args,
                params,
            } => {
                let callee = match path {
                    FnPath::Ident(name) => name.clone(),
                    FnPath::TypeMethod { ty, method } => mangle(&[ty, method]),
                };
                if let Some(sig) = self.fn_sigs.get(&callee) {
                    sig.ret.clone()
                } else if let Some(value_indices) = builtin_value_args(&callee) {
                    // The shared value type, searched across all value
                    // arguments — mirroring the walk's operand anchor:
                    // `min(0, n)` with `n: u32` anchors from `n`.
                    value_indices
                        .iter()
                        .find_map(|&i| params.get(i).and_then(|p| self.infer(p)))
                } else if let Some((elements, scalar)) = vec_ctor_shape(&callee, type_args) {
                    Some(Type::Vector {
                        elements,
                        scalar_ty: Some(scalar),
                    })
                } else {
                    // External builtins (texture sampling, atomics,
                    // derivatives, packing, …) have no signature table
                    // here — deliberately unprovable rather than guessed.
                    None
                }
            }
            // Associated constants are registered under their mangled
            // render name (`Type_MEMBER`) at collection time. Built-in
            // vector associated constants (`Vec4u::ZERO`, which lowers
            // to `vec4u::ZERO` in the IR) are not harvested globals —
            // they are always the vector's own shape, so fall back to
            // the type name's encoded shape.
            Expr::TypePath { ty, member } => self
                .globals
                .get(&mangle(&[ty, member]))
                .cloned()
                .or_else(|| {
                    vec_ctor_shape(ty, &[]).map(|(elements, scalar)| Type::Vector {
                        elements,
                        scalar_ty: Some(scalar),
                    })
                }),
            // A reference is a pointer to the pointee; carrying the
            // pointee through lets deref-assignment targets anchor
            // (`let p = &mut get_mut!(OUTPUT)[0]; *p = select(0,1,c)`).
            // References created in function scope are
            // `ptr<function, T>`.
            Expr::Reference(inner) => self.infer(inner).map(|elem| Type::Ptr {
                address_space: AddressSpace::Function,
                elem: Box::new(elem),
            }),
        }
    }

    /// Resolve the declared type of `field` on struct `name`,
    /// substituting the struct's type parameters positionally from
    /// `type_args`.
    fn field_ty(&self, name: &str, type_args: &[Type], field: &str) -> Option<Type> {
        let def = self.structs.get(name)?;
        let mut ty = def
            .fields
            .iter()
            .find(|(field_name, _)| field_name == field)?
            .1
            .clone();
        substitute_params(&mut ty, &def.type_params, type_args);
        Some(ty)
    }
}

/// Scope-tracking state for the suffix pass's mutation walk.
///
/// Structurally mirrors the deshadow pass: one instance walks a whole
/// module, pushing a scope frame per block so that the innermost
/// declaration of a name wins when resolving assignment targets. Type
/// knowledge lives in [`TypeEnv`]; the walk adds the current
/// function's return type, which is walk state rather than environment.
#[derive(Default)]
struct SuffixPass {
    /// The collected type environment.
    env: TypeEnv,
    /// The current function's declared return type, if any.
    return_ty: Option<Type>,
    /// Backward loop-variable inference state (see [`Self::walk_for`]):
    /// a stack of `(variable, anchors)` frames, one per `for` loop whose
    /// untyped variable is currently being probed. Recording targets the
    /// innermost frame matching the name, so same-name shadowing records
    /// to the shadowing loop.
    loop_probes: Vec<(String, Vec<ScalarType>)>,
    /// Depth of conversion contexts (cast operands, shift counts) in
    /// which an expectation must not pin a probed loop variable: a cast
    /// converts rather than unifies, and a shift count is `u32` in WGSL
    /// but independently typed in Rust.
    probe_suppressed: usize,
}

impl SuffixPass {
    /// Walk every item in the module.
    fn walk_module(&mut self, module: &mut Module) {
        self.walk_items(&mut module.items);
    }

    /// First collect the type environment (struct definitions,
    /// function signatures, globals), then walk each top-level item
    /// that can anchor a type expectation: `const` items via their
    /// declared type, and functions (free or impl methods) via their
    /// signatures.
    fn walk_items(&mut self, items: &mut [Item]) {
        self.env.collect(items);
        for item in items {
            match item {
                Item::Const(c) => self.expect(&mut c.expr, Some(&c.ty)),
                Item::Fn(f) => self.walk_fn(f),
                Item::Impl(imp) => {
                    for impl_item in &mut imp.items {
                        match impl_item {
                            ImplItem::Fn(f) => self.walk_fn(f),
                            // Associated consts anchor from their
                            // declared type, like module-level consts.
                            ImplItem::Const(c) => self.expect(&mut c.expr, Some(&c.ty)),
                            // Associated type aliases carry no
                            // expression.
                            ImplItem::Type(_) => {}
                        }
                    }
                }
                // Structs were collected above; uniforms, storage,
                // samplers, textures, and enums carry no expression to
                // suffix.
                _ => {}
            }
        }
    }

    /// Walk one function body with its parameters and return type as the
    /// seed context.
    fn walk_fn(&mut self, f: &mut ItemFn) {
        self.env.scopes.clear();
        self.return_ty = match &f.return_type {
            ReturnType::Type { ty, .. } => Some(ty.clone()),
            ReturnType::Default => None,
        };
        let params: HashMap<String, Option<Type>> = f
            .inputs
            .iter()
            .map(|arg| (arg.name.clone(), Some(arg.ty.clone())))
            .collect();
        self.env.scopes.push(params);
        self.walk_block(&mut f.block);
        self.env.scopes.clear();
        self.return_ty = None;
    }

    /// Walk a block's statements in a fresh scope frame.
    fn walk_block(&mut self, block: &mut Block) {
        self.env.scopes.push(HashMap::new());
        for stmt in &mut block.stmts {
            self.walk_stmt(stmt);
        }
        self.env.scopes.pop();
    }

    /// Apply the anchors that produce a type expectation for a statement.
    fn walk_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Local(local) => self.walk_local(local),
            Stmt::Const(c) => {
                self.expect(&mut c.expr, Some(&c.ty));
                // Register the const's declared type in scope after
                // walking its initializer, like a typed local — later
                // expressions anchor from it
                // (`const C: u32 = 1; select(C, 0, cond)`).
                let name = c.name.clone();
                let ty = Some(c.ty.clone());
                if let Some(scope) = self.env.scopes.last_mut() {
                    scope.insert(name, ty);
                }
            }
            Stmt::Assignment { lhs, rhs } => {
                self.walk_assignment_rhs(lhs, rhs, None);
            }
            Stmt::CompoundAssignment { lhs, op, rhs } => {
                self.walk_assignment_rhs(lhs, rhs, Some(*op));
            }
            Stmt::Return(Some(expr)) => {
                let ret = self.return_ty.clone();
                self.expect(expr, ret.as_ref());
            }
            Stmt::Expr { expr, has_semi } => {
                // A trailing expression is an implicit `return expr;`.
                let expected = if *has_semi {
                    None
                } else {
                    self.return_ty.clone()
                };
                self.expect(expr, expected.as_ref());
            }
            Stmt::If(i) => self.walk_if(i),
            Stmt::While { condition, body } => {
                self.expect(condition, None);
                self.walk_block(body);
            }
            Stmt::Loop { body } => self.walk_block(body),
            Stmt::For(f) => self.walk_for(f),
            Stmt::Switch(s) => self.walk_switch(s),
            Stmt::Block(b) => self.walk_block(b),
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
            } => {
                // The renderer emits a u32 loop counter, compares it
                // against `size`, and adds the offsets to it — all
                // three flow through u32 arithmetic.
                let u32_ty = Type::Scalar(ScalarType::U32);
                self.expect(src_offset, Some(&u32_ty));
                self.expect(dest_offset, Some(&u32_ty));
                self.expect(size, Some(&u32_ty));
                self.expect(src, None);
                self.expect(dest, None);
            }
            // Control flow and extension macros carry no type
            // expectation in this pass.
            Stmt::Return(None)
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Discard
            | Stmt::Macro { .. } => {}
        }
    }

    /// Suffix a local's initializer and register the local's type in the
    /// current scope.
    ///
    /// Typed locals propagate their declared type; un-annotated locals
    /// are still walked so self-anchored expressions (struct
    /// constructors) inside them are suffixed, and a type provable from
    /// the (now suffixed) initializer registers in scope — Rust infers
    /// `let mut y = 0u32;` as `u32`, so a later assignment anchors from
    /// it. Fully-bare initializers register as unprovable (`None`):
    /// the name still shadows outer bindings — anchoring from a
    /// shadowed outer type can write the wrong suffix — but nothing
    /// anchors from it; Rust infers `let x = 0;` as `i32`, matching
    /// WGSL's default, so bare stays valid.
    fn walk_local(&mut self, local: &mut Local) {
        match (&local.ty, &mut local.init) {
            (Some(ty), Some(init)) => self.expect(init, Some(ty)),
            (None, Some(init)) => self.expect(init, None),
            _ => {}
        }
        let ty = match (&local.ty, &local.init) {
            (Some(ty), _) => Some(ty.clone()),
            (None, Some(init)) => self.env.infer(init),
            _ => None,
        };
        if let Some(scope) = self.env.scopes.last_mut() {
            scope.insert(local.name.clone(), ty);
        }
    }

    /// Suffix the RHS of an assignment from the type of the assignment
    /// target, when that is provable (identifier, field, array element,
    /// or swizzle rooted at a typed base).
    fn walk_assignment_rhs(&mut self, lhs: &Expr, rhs: &mut Expr, compound_op: Option<CompoundOp>) {
        // Shift-assign counts are u32 in WGSL, like shift operands —
        // the count never follows the target's type
        // (`x <<= 1` with `x: i32` still needs a `1u` count).
        if matches!(
            compound_op,
            Some(CompoundOp::ShlAssign | CompoundOp::ShrAssign)
        ) {
            let u32_ty = Type::Scalar(ScalarType::U32);
            // Shift-assign counts are independently typed in Rust, like
            // shift counts — see `expect` — so they must not pin a
            // probed loop variable.
            self.probe_suppressed += 1;
            self.expect(rhs, Some(&u32_ty));
            self.probe_suppressed -= 1;
            return;
        }
        let target_ty = self.env.infer(lhs);
        self.expect(rhs, target_ty.as_ref());
    }

    /// Walk an `if` / `else if` / `else` chain.
    fn walk_if(&mut self, i: &mut StmtIf) {
        self.expect(&mut i.condition, None);
        self.walk_block(&mut i.then_block);
        if let Some(else_branch) = &mut i.else_branch {
            match else_branch {
                ElseBranch::Block(b) => self.walk_block(b),
                ElseBranch::If(inner) => self.walk_if(inner),
            }
        }
    }

    /// Walk a `for` loop: bounds inherit the loop variable's type, and
    /// the loop variable is visible in the body.
    ///
    /// Parsed Rust loops never annotate the loop variable (`for i: u32
    /// in …` is not valid Rust — the IR field stays `None`), so its
    /// type is inferred from a provable range bound, mirroring Rust's
    /// own inference. Fully-bare ranges (`for i in 0..8`) stay bare —
    /// `i32` in both Rust and WGSL — *unless the body pins the
    /// variable to another concrete type*: rustc resolves that backward
    /// from the use sites (`for i in 0..1 { let x: u32 = i + 1; }`
    /// infers `i: u32`, wgsl-rs#154), so the pass probes for it. The
    /// body is walked once with the variable registered as an
    /// unprovable scope binding and pushed onto [`Self::loop_probes`];
    /// [`Self::expect`] records every concrete `u32` / `i32`
    /// expectation that lands on it. Unanimous anchors adopt the type:
    /// the bounds get suffixed and the body re-walks with the variable
    /// typed (only empty suffixes are ever written, so the second walk
    /// is idempotent). No anchors, or conflicting ones (source that
    /// would not have compiled as Rust), leave the loop bare — the pass
    /// never guesses (see the DEVLOG entry for 2026-09-22).
    fn walk_for(&mut self, f: &mut ForLoop) {
        let mut var_ty = f
            .var_ty
            .clone()
            .or_else(|| self.env.infer(&f.from).or_else(|| self.env.infer(&f.to)));
        if var_ty.is_none() && !f.body.stmts.is_empty() {
            // Probe walk: the frame registers the variable as unprovable,
            // so its uses resolve to it (never an outer same-name
            // binding) and record onto the probe frame.
            let mut frame = HashMap::new();
            frame.insert(f.var.clone(), None);
            self.env.scopes.push(frame);
            self.loop_probes.push((f.var.clone(), Vec::new()));
            self.walk_block(&mut f.body);
            let probed = self.take_probe();
            self.env.scopes.pop();
            var_ty = probed;
        }
        if let Some(ty) = &var_ty {
            self.expect(&mut f.from, Some(ty));
            self.expect(&mut f.to, Some(ty));
        }
        // The loop variable always gets its own frame: a provable type
        // anchors the body, an unprovable one still shadows any outer
        // binding of the same name (matching Rust's loop scoping).
        let mut frame = HashMap::new();
        frame.insert(f.var.clone(), var_ty);
        self.env.scopes.push(frame);
        self.walk_block(&mut f.body);
        self.env.scopes.pop();
    }

    /// Adopt the probed loop-variable type from the innermost probe
    /// frame: `Some` for a unanimous anchor set, `None` for conflicting
    /// anchors (invalid Rust — the source would not have compiled) or
    /// an empty set, both of which leave the loop bare.
    fn take_probe(&mut self) -> Option<Type> {
        let (_, anchors) = self.loop_probes.pop()?;
        let scalar = *anchors.first()?;
        if anchors.iter().all(|a| *a == scalar) {
            Some(Type::Scalar(scalar))
        } else {
            None
        }
    }

    /// Walk a `switch`: the selector subtree is walked first so nested
    /// call-site and binary anchoring applies (`match x + select(0, 1,
    /// c)` suffixes the select through the binary anchor), then case
    /// selectors inherit the selector's derived type, and arm bodies
    /// are walked for their own anchors.
    fn walk_switch(&mut self, s: &mut StmtSwitch) {
        self.expect(&mut s.selector, None);
        let selector_ty = self.env.infer(&s.selector);
        for arm in &mut s.arms {
            if let Some(ty) = &selector_ty {
                for sel in &mut arm.selectors {
                    match sel {
                        CaseSelector::Literal(l) => suffix_lit(l, ty),
                        CaseSelector::Expr(e) => self.expect(e, Some(ty)),
                        CaseSelector::Default => {}
                    }
                }
            }
            self.walk_block(&mut arm.body);
        }
    }

    /// Propagate expectations into a call's arguments.
    ///
    /// User function signatures are consulted first — they shadow the
    /// builtin rules, mirroring how Rust resolves a local `fn select`
    /// over a glob-imported builtin, and how WGSL resolves user
    /// declarations over predeclared ones. Falls back to builtin rules:
    /// vector constructors carry their element type in the name, and
    /// same-type builtins (`select`, `min`/`max`, `clamp`) propagate one
    /// shared value type across their value arguments.
    fn walk_fn_call(
        &mut self,
        path: &FnPath,
        type_args: &[Type],
        params: &mut [Expr],
        expected: Option<&Type>,
    ) {
        let callee = match path {
            FnPath::Ident(name) => name.clone(),
            FnPath::TypeMethod { ty, method } => mangle(&[ty, method]),
        };
        if let Some(sig) = self.env.fn_sigs.get(&callee).cloned() {
            for (i, arg) in params.iter_mut().enumerate() {
                self.expect(arg, sig.params.get(i));
            }
            return;
        }
        if let Some(elem_ty) = vec_ctor_elem(&callee, type_args) {
            for arg in params {
                self.expect(arg, Some(&elem_ty));
            }
            return;
        }
        if let Some(value_indices) = builtin_value_args(&callee) {
            // The shared value type: an outer expectation first, else a
            // provable value argument (mirrors the binary-operand
            // anchor — `select(x, 0, cond)` with `x: u32`).
            let group_ty = expected.cloned().or_else(|| {
                value_indices
                    .iter()
                    .find_map(|&i| params.get(i).and_then(|a| self.env.infer(a)))
            });
            for (i, arg) in params.iter_mut().enumerate() {
                if value_indices.contains(&i) {
                    self.expect(arg, group_ty.as_ref());
                } else {
                    // `select`'s condition is not a value argument.
                    self.expect(arg, None);
                }
            }
            return;
        }
        for arg in params {
            self.expect(arg, None);
        }
    }

    /// Propagate `expected` into `expr`, suffixing bare integer literals
    /// when the expectation resolves to a concrete scalar integer type.
    ///
    /// `expected` is `None` for contexts with no outer annotation; the
    /// walk still descends so self-anchored sub-expressions (struct
    /// constructors, casts, anchored binary operands) get suffixed.
    fn expect(&mut self, expr: &mut Expr, expected: Option<&Type>) {
        match expr {
            Expr::Lit(lit) => {
                if let Some(ty) = expected {
                    suffix_lit(lit, ty);
                }
            }
            Expr::Paren(inner) => self.expect(inner, expected),
            Expr::Array { elems } => {
                let elem_ty = expected
                    .and_then(|ty| match ty {
                        Type::Array { elem, .. } | Type::RuntimeArray { elem } => {
                            Some((**elem).clone())
                        }
                        _ => None,
                    })
                    // Self-anchor: with no outer expectation, any
                    // provable element carries the shared element
                    // type — Rust unifies the literal, so a suffixed
                    // sibling anchors the bare ones.
                    .or_else(|| elems.iter().find_map(|e| self.env.infer(e)));
                for e in elems {
                    self.expect(e, elem_ty.as_ref());
                }
            }
            Expr::Struct {
                name,
                type_args: ctor_args,
                fields,
            } => {
                // The expectation's type arguments come from an explicit
                // annotation and take precedence; otherwise the
                // constructor's own arguments carry the instantiation.
                let args: Vec<Type> = match expected {
                    Some(Type::Struct {
                        name: exp_name,
                        type_args,
                    }) if exp_name.as_str() == name.as_str() => type_args.clone(),
                    _ => ctor_args.clone(),
                };
                for field in fields {
                    let field_ty = self.env.field_ty(name, &args, &field.member);
                    self.expect(&mut field.expr, field_ty.as_ref());
                }
            }
            Expr::Cast { lhs, ty } => {
                // A cast converts instead of unifying — the operand's
                // type is decoupled from the outer context (`for i in
                // 0..1 { (i as u32); }` keeps `i` i32 in Rust) — so
                // nothing inside a cast pins a probed loop variable.
                self.probe_suppressed += 1;
                self.expect(lhs, Some(ty));
                self.probe_suppressed -= 1;
            }
            Expr::Binary { lhs, op, rhs } => {
                // Shifts are handled separately: the count is `u32` in
                // WGSL, not the result type — Rust infers it
                // independently (`fn f(x: i32) -> i32 { x << 1 }` is
                // valid Rust, but a `1i` count is invalid WGSL).
                if is_shift(op) {
                    let u32_ty = Type::Scalar(ScalarType::U32);
                    // The shifted value shares the outer expectation.
                    self.expect(lhs, expected);
                    // The count is u32 in WGSL but independently typed
                    // in Rust (`x << i` compiles with an i32 count), so
                    // it must not pin a probed loop variable.
                    self.probe_suppressed += 1;
                    self.expect(rhs, Some(&u32_ty));
                    self.probe_suppressed -= 1;
                } else {
                    // Arithmetic operators pass an outer expectation to
                    // their operands (the operands share the result
                    // type); comparisons and logical operators yield
                    // `bool`, so their operands are walked bare. Either
                    // way the operands are walked so nested call-site
                    // and binary anchoring applies
                    // (`let y = select(x, 0, c) + 1;`).
                    let operand_expected = if is_arithmetic(op) { expected } else { None };
                    self.expect(lhs, operand_expected);
                    self.expect(rhs, operand_expected);
                    // Anchor: a bare literal adopts the provable scalar
                    // type of the other operand. Shifts anchor on
                    // neither side (the count is u32; the value follows
                    // the lhs only).
                    if is_arithmetic(op) || is_comparison(op) {
                        let lhs_ty = self.env.infer(lhs);
                        let rhs_ty = self.env.infer(rhs);
                        if let Some(ty) = lhs_ty {
                            self.expect(rhs, Some(&ty));
                        }
                        if let Some(ty) = rhs_ty {
                            self.expect(lhs, Some(&ty));
                        }
                    }
                }
            }
            Expr::FnCall {
                path,
                type_args,
                params,
            } => self.walk_fn_call(path, type_args, params, expected),
            Expr::Unary { op, expr } => {
                // `!` (logical not), bitwise complement, and negation
                // preserve the operand type, so the expectation flows
                // through (`fn f() -> u32 { !0 }` renders `~0u`). A
                // deref's operand is the pointer — its type is not the
                // deref result — so it is walked bare.
                match op {
                    UnOp::Not | UnOp::Complement | UnOp::Neg => self.expect(expr, expected),
                    UnOp::Deref => self.expect(expr, None),
                }
            }
            Expr::ArrayIndexing { lhs, index } => {
                // An outer scalar expectation pins the indexed base's
                // element type (`[select(0, 1, c)][0]` in a `u32`
                // function): carry it into the base as a sequence
                // expectation.
                let seq_expected = expected.map(|ty| Type::RuntimeArray {
                    elem: Box::new(ty.clone()),
                });
                self.expect(lhs, seq_expected.as_ref());
                // Index expressions stay bare (WGSL coerces the
                // abstract integer in `a[0]` to either index type).
                self.expect(index, None);
            }
            Expr::Swizzle { lhs, params, .. } => {
                self.expect(lhs, None);
                if let Some(params) = params {
                    for p in params {
                        self.expect(p, None);
                    }
                }
            }
            Expr::FieldAccess { base, .. } => self.expect(base, None),
            Expr::Reference(inner) => {
                // The reference yields a pointer to the pointee, so an
                // outer pointer expectation carries the pointee type
                // into the referenced expression
                // (`take(&mut [select(0, 1, c)][0])` with
                // `take(p: ptr<function, u32>)` anchors the select).
                let pointee = expected.and_then(|ty| match ty {
                    Type::Ptr { elem, .. } => Some((**elem).clone()),
                    _ => None,
                });
                self.expect(inner, pointee.as_ref());
            }
            Expr::Ident(name) => {
                // Backward loop-var inference (see `walk_for`): a
                // concrete expectation landing on an untyped `for`
                // variable is an anchor rustc would have unified with
                // the variable's inferred type. Record it onto the
                // innermost matching probe frame; the scope lookup must
                // agree the use resolves to an unprovable binding, so
                // typed bindings never record (an adopted variable on
                // the body re-walk, an inner shadowing `let`).
                if self.probe_suppressed == 0
                    && self.env.lookup(name).is_none()
                    && let Some(Type::Scalar(scalar)) = expected
                    && matches!(scalar, ScalarType::U32 | ScalarType::I32)
                    && let Some((_, anchors)) = self
                        .loop_probes
                        .iter_mut()
                        .rev()
                        .find(|(probe, _)| probe == name)
                {
                    anchors.push(*scalar);
                }
            }
            // Type paths and zero-value array lengths (type positions)
            // carry nothing to suffix.
            Expr::TypePath { .. } | Expr::ZeroValueArray { .. } => {}
        }
    }
}

/// Suffix a single bare integer literal from `expected` — the leaf of
/// every expectation path. Literals that already carry a suffix are
/// left alone, as are non-integer expectations a literal cannot have in
/// valid Rust.
fn suffix_lit(lit: &mut Lit, expected: &Type) {
    let Lit::Int { suffix, .. } = lit else {
        return;
    };
    if !suffix.is_empty() {
        return;
    }
    match concrete_scalar(expected) {
        Some(ScalarType::U32) => *suffix = "u32".to_string(),
        Some(ScalarType::I32) => *suffix = "i32".to_string(),
        // A `bool` / `f32` expectation cannot apply to an integer
        // literal in valid Rust; leave it bare.
        Some(ScalarType::F32 | ScalarType::Bool) | None => {}
    }
}

/// The scalar element expectation carried by a vector constructor: the
/// WGSL predeclared `vecN<f|i|u>` shorthand encodes it in the name, and
/// an explicit `vecN<T>` type argument carries it in `type_args[0]`.
///
/// Abstract constructors (`vec3(...)`) return `None` — WGSL's
/// abstract-int defaults match Rust's `i32`, so there is nothing to
/// fix. Bool shorthands (`vec2b`) cannot host integer literals in
/// valid Rust.
fn vec_ctor_elem(name: &str, type_args: &[Type]) -> Option<Type> {
    if let [ty] = type_args {
        return Some(ty.clone());
    }
    let (_, scalar) = vec_ctor_shape(name, type_args)?;
    Some(Type::Scalar(scalar))
}

/// The element count and scalar type of a vector constructor: the
/// `vecN<f|i|u>` shorthand encodes both in the name; an explicit
/// `vecN::<T>` type argument carries the scalar type instead (the
/// name then supplies only the element count — `vec4::<u32>` →
/// `(4, U32)`). Returns `None` for abstract (`vec3`) and bool
/// (`vec3b`) shorthands without a type argument.
fn vec_ctor_shape(name: &str, type_args: &[Type]) -> Option<(u8, ScalarType)> {
    let rest = name.strip_prefix("vec")?;
    // An all-digit tail is the turbofish form (`vec4::<T>`), where the
    // type argument carries the scalar; the shorthand suffix is empty
    // then.
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let (n, suffix) = rest.split_at(digits_end);
    let elements = match n {
        "2" => 2,
        "3" => 3,
        "4" => 4,
        _ => return None,
    };
    let scalar = match type_args {
        [Type::Scalar(scalar)] => *scalar,
        _ => match suffix {
            "i" => ScalarType::I32,
            "u" => ScalarType::U32,
            "f" => ScalarType::F32,
            _ => return None,
        },
    };
    Some((elements, scalar))
}

/// Argument indices of the builtins whose value arguments share one
/// type `T` — `select(f, t, cond)`'s condition is not a value argument.
///
/// Returns `None` for every other name. There is no general signature
/// table for WGSL builtins (`builtin_lookup.rs` translates names only),
/// so this list stays deliberately small: only builtins that appear in
/// wgsl-rs with polymorphic value parameters where a bare literal is
/// otherwise ambiguous.
fn builtin_value_args(name: &str) -> Option<&'static [usize]> {
    Some(match name {
        "select" => &[0, 1],
        "min" | "max" => &[0, 1],
        "clamp" => &[0, 1, 2],
        _ => return None,
    })
}

/// Resolve `ty` to a concrete scalar type, unwrapping single-element
/// wrappers (`atomic<T>`, pointers). Returns `None` for anything the
/// pass cannot prove concrete — type parameters, abstract vectors,
/// structs, and arrays all need more context than a bare [`Type`]
/// carries.
fn concrete_scalar(ty: &Type) -> Option<ScalarType> {
    match ty {
        Type::Scalar(scalar) => Some(*scalar),
        Type::Atomic { elem } | Type::Ptr { elem, .. } => concrete_scalar(elem),
        _ => None,
    }
}

/// Whether `op` produces a result of its operands' type (so an outer
/// expectation flows into the operands).
/// Whether `op` produces a result of its operands' shared type (so an
/// outer expectation flows into the operands). Shifts are excluded —
/// see [`is_shift`].
fn is_arithmetic(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
    )
}

/// Whether `op` is a bit shift. WGSL shift counts are `u32` — the count
/// is *not* the result type, and Rust infers it independently
/// (`fn f(x: i32) -> i32 { x << 1 }` is valid Rust; rendering the count
/// as `1i` is invalid WGSL).
fn is_shift(op: &BinOp) -> bool {
    matches!(op, BinOp::Shl | BinOp::Shr)
}

/// Whether `op` compares two same-typed operands (so one operand's
/// type can anchor the other's literals).
fn is_comparison(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

/// Replace [`Type::TypeParam`] references in `ty` with the corresponding
/// entry of `args` (positional). Unmatched parameters are left in place;
/// the pass no-ops on them downstream.
fn substitute_params(ty: &mut Type, params: &[String], args: &[Type]) {
    match ty {
        Type::TypeParam { name } => {
            if let Some(pos) = params.iter().position(|p| p == name)
                && let Some(arg) = args.get(pos)
            {
                *ty = arg.clone();
            }
        }
        Type::Array { elem, .. }
        | Type::RuntimeArray { elem }
        | Type::Atomic { elem }
        | Type::Ptr { elem, .. }
        | Type::Phantom { elem }
        | Type::AssocType { ty: elem, .. } => substitute_params(elem, params, args),
        Type::Struct { type_args, .. } => {
            for arg in type_args {
                substitute_params(arg, params, args);
            }
        }
        _ => {}
    }
}
