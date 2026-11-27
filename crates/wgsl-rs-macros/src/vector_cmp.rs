//! Vector comparison lowering for wgsl-rs#164.
//!
//! Rust's `==` on vectors produces a `bool` (`PartialEq`), but WGSL's `==`
//! on vectors produces `vecN<bool>` — so wherever a vector comparison
//! feeds a `bool`-typed position (let bindings, returns, conditions) the
//! generated WGSL is invalid. The IR does not track expression types
//! (#145), so this pass walks the parse AST *after* monomorphization —
//! when generic types are concrete — and rewrites comparisons whose
//! operands both infer as vector-typed:
//!
//! * `a == b`  ->  `all(a == b)`
//! * `a != b`  ->  `!(all(a == b))`   (matching `PartialEq::ne`)
//!
//! Inference is deliberately conservative: operands whose types cannot be
//! determined are left untouched (fail-open), so CPU-only code keeps
//! compiling and un-inferable comparisons behave exactly as before. The
//! componentwise escape hatch is `cmp_eq(a, b)`, which lowers at render
//! time via the binary-operator entries in `builtin_lookup` and is never
//! touched by this pass (`FnCall` nodes are not `Binary` nodes).
//!
//! Known limitation, shared with the #160 `!` -> `~` lowering: generic
//! entry-point templates still carry `TypeParam` types when their IR is
//! built lazily at runtime, so a vector `==` inside a runtime-instantiated
//! generic template is not wrapped. Monomorphized functions — the common
//! case — are covered, because `monomorphize` appends their concrete
//! copies to the module content before this pass runs.
//!
//! The same applies to functions imported from other modules
//! (`use provider::*;`): each `#[wgsl]` module expands in isolation and
//! the macro never sees the provider's parse AST, so imported call
//! results are un-inferable and comparisons of them fail open. Fixing
//! that requires cross-module type propagation in the build pipeline.
//!
//! The traversal reuses the [`crate::parse_visitor`] walker (the same
//! machinery monomorphization uses): the pass is a
//! [`parse_visitor::ParseVisitorMut`] that overrides the statement and
//! expression hooks to track locals and wrap comparisons. Operand types
//! are computed by re-reading the (already transformed) operands at each
//! comparison node — a second shallow walk, compile-time only.

use crate::{
    parse::{
        self, BinOp, Block, Expr, FnPath, ImplItem, Item, ItemConst, ItemFn, Lit, ReturnType,
        ScalarType, Stmt, Type, UnOp,
    },
    parse_visitor::{self, ParseVisitorMut},
};
use proc_macro2::{Ident, Span};
use std::collections::HashMap;
use syn::punctuated::Punctuated;

/// Placeholder used to temporarily empty an expression slot while its
/// contents are being rebuilt. Never survives the pass.
const TMP_IDENT: &str = "__wgsl_vector_cmp_tmp";

/// Module-wide symbol information used to infer expression types.
struct Symbols {
    /// Free functions by name -> return type (`None` for `->`-less fns).
    /// Monomorphized instances are included: mono appends their concrete
    /// copies to module content before this pass runs. Functions imported
    /// from other modules are NOT here — each `#[wgsl]` module expands in
    /// isolation, so imported returns are un-inferable and fail open. Impl
    /// methods are indexed separately in `methods` under their mangled
    /// `Type_method` names.
    fns: HashMap<String, Option<Type>>,
    /// Struct fields by struct name: `(field name, type)`.
    structs: HashMap<String, Vec<(String, Type)>>,
    /// Module-level consts by name.
    consts: HashMap<String, Type>,
    /// Impl methods by rendered mangled name (`Type_method`) -> return
    /// type (`None` for `->`-less methods).
    methods: HashMap<String, Option<Type>>,
    /// Linkage variables (`uniform!` / `storage!` / `workgroup!`) by
    /// declared name -> declared WGSL type.
    linkage: HashMap<String, Type>,
}

/// Scoped local-variable types, innermost scope last.
struct Locals {
    scopes: Vec<HashMap<String, Type>>,
}

impl Locals {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn get(&self, name: &str) -> Option<&Type> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }
}

fn scalar_ty(t: ScalarType, span: Span) -> Type {
    Type::Scalar {
        ty: t,
        ident: Ident::new(t.wgsl_name(), span),
    }
}

/// Build a vector type for inference purposes. The ident is a plausible
/// alias (`vec2f`); it is only ever fed back into inference, never
/// rendered.
fn vector_ty(elements: u8, t: ScalarType, span: Span) -> Type {
    let alias = format!("vec{elements}{}", t.short_name());
    Type::Vector {
        elements,
        scalar_ty: t,
        ident: Ident::new(&alias, span),
        scalar: None,
    }
}

/// Parse a WGSL vector constructor alias: `vec2f` -> `(2, F32)`, etc.
fn vector_ctor_alias(name: &str) -> Option<(u8, ScalarType)> {
    let rest = name.strip_prefix("vec")?;
    let mut chars = rest.chars();
    let elements = match chars.next()? {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        _ => return None,
    };
    let scalar = match chars.next()? {
        'f' => ScalarType::F32,
        'i' => ScalarType::I32,
        'u' => ScalarType::U32,
        'b' => ScalarType::Bool,
        _ => return None,
    };
    if chars.next().is_some() {
        return None;
    }
    Some((elements, scalar))
}

/// Parse a WGSL vector *type* alias: `Vec2f` -> `(2, F32)`, etc. Used
/// for associated consts of std vector types (`Vec2f::ZERO`), which are
/// always of the vector's shape.
fn vector_type_alias(name: &str) -> Option<(u8, ScalarType)> {
    let rest = name.strip_prefix("Vec")?;
    let mut chars = rest.chars();
    let elements = match chars.next()? {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        _ => return None,
    };
    let scalar = match chars.next()? {
        'f' => ScalarType::F32,
        'i' => ScalarType::I32,
        'u' => ScalarType::U32,
        'b' => ScalarType::Bool,
        _ => return None,
    };
    if chars.next().is_some() {
        return None;
    }
    Some((elements, scalar))
}

/// WGSL builtin functions that preserve the shape of their first
/// argument: scalar -> scalar, `vecN<T>` -> `vecN<T>`. Used for inference
/// only — never rendered.
const VECTOR_PRESERVING_BUILTINS: &[&str] = &[
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "count_leading_zeros",
    "count_one_bits",
    "count_trailing_zeros",
    "cross",
    "degrees",
    "dpdx",
    "dpdy",
    "exp",
    "exp2",
    "extract_bits",
    "face_forward",
    "first_leading_bit",
    "first_trailing_bit",
    "floor",
    "fma",
    "fract",
    "fwidth",
    "insert_bits",
    "inverse_sqrt",
    "log",
    "log2",
    "max",
    "min",
    "mix",
    "normalize",
    "pow",
    "radians",
    "reflect",
    "refract",
    "reverse_bits",
    "round",
    "saturate",
    "select",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "trunc",
];

/// Combine inferred operand types of an arithmetic/bitwise binary
/// operation: a vector operand dominates an unknown one, two scalars
/// keep the left type.
fn combine_arith(lt: Option<Type>, rt: Option<Type>) -> Option<Type> {
    match (lt, rt) {
        (Some(l @ Type::Vector { .. }), _) => Some(l),
        (_, Some(r @ Type::Vector { .. })) => Some(r),
        (Some(l @ Type::Scalar { .. }), Some(_)) => Some(l),
        _ => None,
    }
}

impl Symbols {
    /// Collect module-wide symbol information (functions, structs, consts,
    /// impl members, linkage declarations).
    fn build(content: &[Item]) -> Self {
        let mut fns = HashMap::new();
        let mut structs = HashMap::new();
        let mut consts = HashMap::new();
        let mut methods = HashMap::new();
        let mut linkage = HashMap::new();
        Self::collect(
            content,
            &mut fns,
            &mut structs,
            &mut consts,
            &mut methods,
            &mut linkage,
        );
        Self {
            fns,
            structs,
            consts,
            methods,
            linkage,
        }
    }

    fn collect(
        content: &[Item],
        fns: &mut HashMap<String, Option<Type>>,
        structs: &mut HashMap<String, Vec<(String, Type)>>,
        consts: &mut HashMap<String, Type>,
        methods: &mut HashMap<String, Option<Type>>,
        linkage: &mut HashMap<String, Type>,
    ) {
        for item in content {
            match item {
                Item::Fn(f) => {
                    let ret = match &f.return_type {
                        ReturnType::Type { ty, .. } => Some((**ty).clone()),
                        ReturnType::Default => None,
                    };
                    fns.insert(f.ident.to_string(), ret);
                }
                Item::Struct(s) => {
                    structs.insert(
                        s.ident.to_string(),
                        s.fields
                            .named
                            .iter()
                            .map(|field| (field.ident.to_string(), field.ty.clone()))
                            .collect(),
                    );
                }
                Item::Const(c) => {
                    consts.insert(c.ident.to_string(), c.ty.clone());
                }
                Item::Uniform(u) => {
                    linkage.insert(u.name.to_string(), u.ty.clone());
                }
                Item::Storage(st) => {
                    linkage.insert(st.name.to_string(), st.ty.clone());
                }
                Item::Workgroup(wg) => {
                    linkage.insert(wg.name.to_string(), wg.ty.clone());
                }
                // Pack fields lower to mangled globals before this pass
                // runs, so comparisons on pack-derived values look up the
                // mangled resource name here.
                Item::ResourcePack(pack) => {
                    for field in &pack.fields {
                        linkage.insert(
                            crate::resource::mangle_resource_name(&pack.ident, &field.ident),
                            field.ty.clone(),
                        );
                    }
                }
                // Nested modules are deliberately NOT collected into the
                // enclosing table: their names must not shadow the
                // enclosing module's (lexical scoping). `rewrite` builds
                // each nested module its own symbol table instead.
                Item::Impl(i) => {
                    // Index impl-method return types by their rendered
                    // mangled name (`Type_method`) so `Type::method()`
                    // calls infer. Only struct-typed impls are indexed;
                    // array/trait impls fail open.
                    if let Type::Struct { ident, .. } = &i.self_ty {
                        for impl_item in &i.items {
                            match impl_item {
                                ImplItem::Fn(f) => {
                                    let ret = match &f.return_type {
                                        ReturnType::Type { ty, .. } => Some((**ty).clone()),
                                        ReturnType::Default => None,
                                    };
                                    methods.insert(format!("{}_{}", ident, f.ident), ret);
                                }
                                ImplItem::Const(c) => {
                                    methods.insert(
                                        format!("{}_{}", ident, c.ident),
                                        Some(c.ty.clone()),
                                    );
                                }
                                ImplItem::Type(_) => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Infer the type of an expression, or `None` when it cannot be
    /// determined (the caller must then fail open).
    fn infer(&self, e: &Expr, locals: &Locals) -> Option<Type> {
        match e {
            Expr::Lit(Lit::Bool(_)) => Some(scalar_ty(ScalarType::Bool, e.span())),
            Expr::Lit(Lit::Float(_)) => Some(scalar_ty(ScalarType::F32, e.span())),
            Expr::Lit(Lit::Int(i)) => {
                let scalar = match i.suffix() {
                    "u32" | "usize" => ScalarType::U32,
                    _ => ScalarType::I32,
                };
                Some(scalar_ty(scalar, e.span()))
            }
            Expr::Ident(id) => locals
                .get(&id.to_string())
                .cloned()
                .or_else(|| self.consts.get(&id.to_string()).cloned())
                // Pack fields lower to bare mangled globals before this
                // pass; a local initialized from one carries its type.
                .or_else(|| self.linkage.get(&id.to_string()).cloned()),
            Expr::Paren { inner, .. } => self.infer(inner, locals),
            Expr::Reference { expr, .. } => self.infer(expr, locals),
            Expr::Unary { op, expr } => match op {
                // `!vecN<bool>` is componentwise in WGSL and stays a bool
                // vector; any other operand makes `!x` a plain bool.
                UnOp::Not(_) => match self.infer(expr, locals) {
                    Some(
                        t @ Type::Vector {
                            scalar_ty: ScalarType::Bool,
                            ..
                        },
                    ) => Some(t),
                    _ => Some(scalar_ty(ScalarType::Bool, e.span())),
                },
                UnOp::Neg(_) | UnOp::Deref(_) => self.infer(expr, locals),
            },
            Expr::Binary { lhs, op, rhs } => match op {
                // Comparisons and logical ops are bool-typed in Rust
                // regardless of operand types.
                BinOp::Eq(_)
                | BinOp::Ne(_)
                | BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
                | BinOp::And(_)
                | BinOp::Or(_) => Some(scalar_ty(ScalarType::Bool, e.span())),
                // Shifts take the lhs's type.
                BinOp::Shl(_) | BinOp::Shr(_) => self.infer(lhs, locals),
                _ => combine_arith(self.infer(lhs, locals), self.infer(rhs, locals)),
            },
            Expr::Cast { ty, .. } => Some((**ty).clone()),
            Expr::Swizzle {
                lhs,
                swizzle,
                params,
                ..
            } => {
                // A swizzle with call arguments (e.g. matrix access via
                // `m.x(i)`) is an indexing operation, not a value.
                if params.as_ref().is_some_and(|p| !p.is_empty()) {
                    return None;
                }
                let n = swizzle.to_string().chars().count();
                let base = self.infer(lhs, locals);
                let elem = match &base {
                    Some(Type::Vector { scalar_ty: s, .. }) => Some(*s),
                    // Matrices are f32-only, so their components are f32.
                    Some(Type::Matrix { .. }) => Some(ScalarType::F32),
                    _ => None,
                };
                match n {
                    // Single component: the base's element scalar.
                    1 => match (&base, elem) {
                        (_, Some(s)) => Some(scalar_ty(s, e.span())),
                        (Some(Type::Scalar { ty, .. }), _) => Some(scalar_ty(*ty, e.span())),
                        _ => None,
                    },
                    // Multi-component swizzles are vectors.
                    2..=4 => Some(vector_ty(
                        n as u8,
                        elem.unwrap_or(ScalarType::F32),
                        e.span(),
                    )),
                    _ => None,
                }
            }
            Expr::ArrayIndexing { lhs, .. } => match self.infer(lhs, locals)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => Some((*elem).clone()),
                // `m[i]` is a column vector; matrices are f32-only.
                Type::Matrix { rows, .. } => Some(vector_ty(rows, ScalarType::F32, e.span())),
                Type::Vector { scalar_ty: s, .. } => Some(scalar_ty(s, e.span())),
                _ => None,
            },
            Expr::FnCall {
                path: FnPath::Ident(f),
                type_args,
                params,
                ..
            } => {
                let name = f.to_string();
                // A user-defined fn of the same name wins over builtins and
                // constructor aliases.
                if let Some(ret) = self.fns.get(&name) {
                    return ret.clone();
                }
                // `all`/`any` reduce a vecN<bool> to bool.
                if name == "all" || name == "any" {
                    return Some(scalar_ty(ScalarType::Bool, e.span()));
                }
                // The componentwise masks `cmp_eq`/`cmp_ne` return a bool
                // vector of the same shape as their operands (#164), so
                // mask-vs-mask comparisons still infer as vector
                // comparisons.
                if (name == "cmp_eq" || name == "cmp_ne") && params.len() == 2 {
                    return match self.infer(&params[0], locals) {
                        Some(Type::Vector { elements, .. }) => {
                            Some(vector_ty(elements, ScalarType::Bool, e.span()))
                        }
                        _ => None,
                    };
                }
                if let Some((elements, scalar)) = vector_ctor_alias(&name) {
                    return Some(vector_ty(elements, scalar, e.span()));
                }
                // `vec2::<f32>(...)` style: the turbofish supplies the
                // scalar type.
                if matches!(name.as_str(), "vec2" | "vec3" | "vec4")
                    && type_args.len() == 1
                    && let Type::Scalar { ty, .. } = &type_args[0]
                    && let Some(elements) = name.chars().nth(3).and_then(|c| c.to_digit(10))
                {
                    return Some(vector_ty(elements as u8, *ty, e.span()));
                }
                // Vector-preserving builtins return the type of their
                // first argument: scalar -> scalar, vecN<T> -> vecN<T>.
                if VECTOR_PRESERVING_BUILTINS.contains(&name.as_str()) && !params.is_empty() {
                    return self.infer(&params[0], locals);
                }
                None
            }
            Expr::FnCall {
                path: FnPath::TypeMethod { ty, method, .. },
                ..
            } => {
                // Impl methods: keyed by the rendered mangled name
                // (`Type_method`). Generic methods and non-struct impls
                // are not indexed and fail open.
                let key = format!("{ty}_{method}");
                self.methods.get(&key).cloned().flatten()
            }
            Expr::Struct { ident, .. } => Some(Type::Struct {
                ident: ident.clone(),
                type_args: Vec::new(),
                const_args: Vec::new(),
            }),
            Expr::FieldAccess { base, field, .. } => match self.infer(base, locals)? {
                Type::Struct { ident, .. } => self
                    .structs
                    .get(&ident.to_string())?
                    .iter()
                    .find(|(name, _)| name == &field.to_string())
                    .map(|(_, ty)| ty.clone()),
                _ => None,
            },
            Expr::LinkageAccess {
                ident, type_arg, ..
            } => {
                // The one-argument form (`get!(VAR)`) resolves against the
                // module's linkage declarations; the two-argument generic
                // form keeps a Rust-side type expression and fails open.
                if type_arg.is_some() {
                    return None;
                }
                self.linkage.get(&ident.to_string()).cloned()
            }
            Expr::TypePath { ty, member, .. } => {
                // `Type::MEMBER`: user impls are indexed under the mangled
                // `Type_member` name; std vector associated consts
                // (`Vec2f::ZERO`) are always of the vector's shape, so
                // fall back to the type alias.
                let key = format!("{ty}_{member}");
                if let Some(t) = self.methods.get(&key) {
                    return t.clone();
                }
                vector_type_alias(&ty.to_string())
                    .map(|(elements, scalar)| vector_ty(elements, scalar, e.span()))
            }
            // Array literals and zero-value arrays: not inferable here.
            _ => None,
        }
    }
} // impl Symbols

/// The #164 rewrite pass. A [`ParseVisitorMut`] walker that carries the
/// module symbol table and the current function's local-variable types.
struct VectorCmpPass<'a> {
    symbols: &'a Symbols,
    locals: Locals,
}

impl<'a> VectorCmpPass<'a> {
    fn new(symbols: &'a Symbols) -> Self {
        Self {
            symbols,
            locals: Locals::new(),
        }
    }

    /// Wrap a vector `==`/`!=` node in `all(...)` / `!(all(...))` when
    /// both operands infer vector-typed. Types are re-read from the
    /// (already transformed) operands.
    fn wrap_vector_cmp(&self, e: &mut Expr) {
        if !matches!(
            e,
            Expr::Binary {
                op: BinOp::Eq(_) | BinOp::Ne(_),
                ..
            }
        ) {
            return;
        }
        let (lt, rt) = match e {
            Expr::Binary { lhs, rhs, .. } => (
                self.symbols.infer(lhs, &self.locals),
                self.symbols.infer(rhs, &self.locals),
            ),
            _ => unreachable!("guarded by the matches! above"),
        };
        if !matches!(
            (&lt, &rt),
            (Some(Type::Vector { .. }), Some(Type::Vector { .. }))
        ) {
            return;
        }

        // Take the node out of `e` so it can be rebuilt without borrow
        // conflicts.
        let span = e.span();
        let inner = std::mem::replace(e, Expr::Ident(Ident::new(TMP_IDENT, span)));
        if let Expr::Binary { lhs, op, rhs } = inner {
            // The inner operator is always `==`: `all(a != b)` means
            // "every component differs", which is not `PartialEq::ne`
            // (`!(self == other)`).
            let was_eq = matches!(op, BinOp::Eq(_));
            let inner_op = if was_eq {
                op
            } else {
                BinOp::Eq(Default::default())
            };
            let mut params = Punctuated::new();
            params.push(Expr::Binary {
                lhs,
                op: inner_op,
                rhs,
            });
            let all = Expr::FnCall {
                path: FnPath::Ident(Ident::new("all", span)),
                type_args: Vec::new(),
                const_args: Vec::new(),
                paren_token: Default::default(),
                params,
            };
            *e = if was_eq {
                all
            } else {
                Expr::Unary {
                    op: UnOp::Not(Default::default()),
                    expr: Box::new(all),
                }
            };
        }
    }
}

impl ParseVisitorMut for VectorCmpPass<'_> {
    fn visit_item(&mut self, item: &mut Item) -> Result<(), parse::Error> {
        match item {
            // Nested modules build their own symbol table: names inside
            // do not resolve against the enclosing module's items.
            Item::Mod(m) => {
                rewrite(&mut m.content);
                Ok(())
            }
            _ => parse_visitor::walk_item(self, item),
        }
    }

    /// A fresh locals scope per function, seeded with the parameters.
    fn visit_fn(&mut self, f: &mut ItemFn) -> Result<(), parse::Error> {
        self.locals = Locals::new();
        for arg in f.inputs.iter() {
            self.locals.insert(&arg.ident.to_string(), arg.ty.clone());
        }
        parse_visitor::walk_fn(self, f)
    }

    /// Module- and impl-level const initializers have no function locals
    /// in scope; reset for the walk and restore after.
    fn visit_const(&mut self, c: &mut ItemConst) -> Result<(), parse::Error> {
        let saved = std::mem::replace(&mut self.locals, Locals::new());
        self.locals = Locals::new();
        let result = parse_visitor::walk_const(self, c);
        self.locals = saved;
        result
    }

    fn visit_block(&mut self, b: &mut Block) -> Result<(), parse::Error> {
        self.locals.push();
        let result = parse_visitor::walk_block(self, b);
        self.locals.pop();
        result
    }

    fn visit_stmt(&mut self, s: &mut Stmt) -> Result<(), parse::Error> {
        match s {
            Stmt::Local(l) => {
                // Transform the initializer first so a `let` bound to a
                // vector comparison records the *wrapped* (bool) type.
                if let Some(init) = &mut l.init {
                    self.visit_expr(&mut init.expr)?;
                }
                let bound = match &l.ty {
                    Some((_, ty)) => Some(ty.clone()),
                    None => l
                        .init
                        .as_ref()
                        .and_then(|i| self.symbols.infer(&i.expr, &self.locals)),
                };
                if let Some(ty) = bound {
                    self.locals.insert(&l.ident.to_string(), ty);
                }
                Ok(())
            }
            Stmt::Const(c) => {
                self.visit_expr(&mut c.expr)?;
                self.locals.insert(&c.ident.to_string(), c.ty.clone());
                Ok(())
            }
            Stmt::For(f) => {
                self.visit_expr(&mut f.from)?;
                self.visit_expr(&mut f.to)?;
                let var_ty = self.symbols.infer(&f.from, &self.locals);
                self.locals.push();
                if let Some(ty) = var_ty {
                    self.locals.insert(&f.ident.to_string(), ty);
                }
                let result = self.visit_block(&mut f.body);
                self.locals.pop();
                result
            }
            _ => parse_visitor::walk_stmt(self, s),
        }
    }

    fn visit_expr(&mut self, e: &mut Expr) -> Result<(), parse::Error> {
        // Descend first (bottom-up), then decide on this node.
        parse_visitor::walk_expr(self, e)?;
        self.wrap_vector_cmp(e);
        Ok(())
    }
}

/// Rewrite vector `==`/`!=` comparisons in every function body of the
/// module (including monomorphized instances and impl methods).
pub(crate) fn rewrite(content: &mut [Item]) {
    let symbols = Symbols::build(content);
    let mut pass = VectorCmpPass::new(&symbols);
    for item in content.iter_mut() {
        // The pass is infallible: every hook returns `Ok(())`.
        let _ = pass.visit_item(item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vector_ctor_aliases() {
        assert_eq!(vector_ctor_alias("vec2f"), Some((2, ScalarType::F32)));
        assert_eq!(vector_ctor_alias("vec3i"), Some((3, ScalarType::I32)));
        assert_eq!(vector_ctor_alias("vec4u"), Some((4, ScalarType::U32)));
        assert_eq!(vector_ctor_alias("vec2b"), Some((2, ScalarType::Bool)));
    }

    #[test]
    fn rejects_non_aliases() {
        assert_eq!(vector_ctor_alias("vec2"), None);
        assert_eq!(vector_ctor_alias("vec5f"), None);
        assert_eq!(vector_ctor_alias("vector"), None);
        assert_eq!(vector_ctor_alias("vec2ff"), None);
        assert_eq!(vector_ctor_alias("dot"), None);
    }
}
