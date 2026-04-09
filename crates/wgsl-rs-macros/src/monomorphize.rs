//! Generic function monomorphization pass.
//!
//! Transforms generic free functions and their call sites into concrete,
//! monomorphized WGSL functions. Since WGSL has no user-defined generics,
//! this pass resolves all type parameters at macro-expansion time by:
//!
//! 1. Collecting generic function templates
//! 2. Discovering concrete instantiation sites (via turbofish syntax)
//! 3. Generating monomorphized copies with mangled names
//! 4. Rewriting call sites to use the mangled names
//!
//! For same-module generics, all type params are resolved at macro time.
//! For cross-module generics (turbofish calls to functions from imported
//! modules), the defining module exports `macro_rules!` macros that produce
//! monomorphized WGSL via `concat!`, and the consuming module invokes them.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use proc_macro2::Span;
use syn::Ident;

use crate::{
    code_gen::{self, GenerateCode},
    parse::{
        Block, CaseSelector, ElseBody, Expr, FnPath, Item, ItemFn, ItemMod, ReturnType, Stmt, Type,
    },
};

/// A cross-module template instantiation that needs to be included in the
/// consuming module's `WGSL_MODULE.source` as a macro invocation.
#[derive(Debug, Clone)]
pub struct CrossModuleInstantiation {
    /// The `use` path prefix that identifies the crate/module containing the
    /// template (e.g., `super::renderer_specialization`).
    pub import_path: syn::Path,
    /// The module name where the template is defined (for macro name
    /// construction).
    pub module_name: String,
    /// The generic function name.
    pub fn_name: String,
    /// The lowercased mangled type argument names.
    pub mangled_type_args: Vec<String>,
}

/// Result of the monomorphization pass.
#[derive(Debug)]
pub struct MonoResult {
    /// Cross-module template instantiations that need macro invocations in
    /// the consuming module's `WGSL_MODULE.source`.
    pub cross_module_instantiations: Vec<CrossModuleInstantiation>,
    /// Template macros to generate for generic functions defined in this
    /// module.
    pub template_macros: Vec<TemplateMacro>,
}

/// Information needed to generate a `#[macro_export] macro_rules!` for a
/// generic function template.
#[derive(Debug)]
pub struct TemplateMacro {
    /// The module name (for macro name prefixing).
    pub module_name: String,
    /// The generic function name.
    pub fn_name: String,
    /// Type parameter names (e.g., `["M", "L", "N"]`).
    pub type_param_names: Vec<String>,
    /// The template WGSL source with `__TP{name}__` placeholders.
    pub template_wgsl: String,
    /// Transitive calls to other generic functions within this template.
    pub dependencies: Vec<TemplateDep>,
}

/// A dependency from one template to another generic function.
#[derive(Debug)]
pub struct TemplateDep {
    /// Name of the callee generic function.
    pub callee: String,
    /// Maps each of the callee's type params to one of the caller's type
    /// params, by index.
    pub type_param_mapping: Vec<usize>,
}

/// Run the monomorphization pass on a parsed module.
///
/// Returns information about cross-module instantiations and template macros
/// that need to be generated.
pub fn run(module: &mut ItemMod) -> Result<MonoResult, crate::parse::Error> {
    let mut ctx = MonoCtx::new(module)?;

    // Generate template macros for generic functions defined in this module
    let template_macros = ctx.generate_template_macros(module)?;

    // Run same-module monomorphization if there are local templates
    if !ctx.templates.is_empty() {
        ctx.discover_instantiations(module)?;
        ctx.process_queue()?;
        ctx.apply(module)?;
    }

    // Resolve cross-module turbofish calls (rewrites call sites and collects
    // instantiation info). This must run before the missing-turbofish check
    // because it clears type_args on cross-module calls.
    let cross_module_instantiations = collect_cross_module_instantiations(module, &ctx.templates)?;

    // Check for any remaining unresolved turbofish calls (missing turbofish
    // on calls to local generic functions)
    for item in &module.content {
        check_unresolved_generic_calls(item, &ctx.templates)?;
    }

    Ok(MonoResult {
        cross_module_instantiations,
        template_macros,
    })
}

/// Maximum number of monomorphized instances to prevent runaway expansion.
const MAX_INSTANTIATIONS: usize = 1024;

/// Identity key for deduplicating instantiations.
/// `Span` is deliberately excluded since it has no `Eq`/`Ord`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct InstKey {
    fn_name: String,
    type_args: Vec<TypeKey>,
}

/// A hashable/orderable representation of a concrete type, for use in dedup
/// keys.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TypeKey {
    Scalar(String),
    Vector(u8, Box<TypeKey>),
    Matrix(u8),
    Array(Box<TypeKey>, String),
    RuntimeArray(Box<TypeKey>),
    Atomic(Box<TypeKey>),
    Struct(String),
    Sampler,
    SamplerComparison,
    Texture(String),
    TextureDepth(String),
    Ptr(String, Box<TypeKey>),
    /// Should not appear in a fully resolved key.
    TypeParam(String),
}

/// A request to instantiate a generic function with specific types.
struct InstRequest {
    key: InstKey,
    /// Span of the call site, for error diagnostics.
    span: Span,
    /// The actual `Type` values to substitute (parallel to key.type_args).
    concrete_types: Vec<Type>,
}

struct MonoCtx {
    /// Generic function templates, keyed by original function name.
    templates: BTreeMap<String, ItemFn>,
    /// Queue of instantiations to process.
    queue: VecDeque<InstRequest>,
    /// Set of already-processed or enqueued keys.
    seen: BTreeSet<InstKey>,
    /// Generated concrete functions to insert into the module.
    generated: Vec<Item>,
    /// Reserved function names (from non-generic items) for collision
    /// detection.
    reserved_names: BTreeSet<String>,
}

impl MonoCtx {
    /// Partition module items into templates and concrete items, collecting
    /// reserved names.
    fn new(module: &ItemMod) -> Result<Self, crate::parse::Error> {
        let mut templates = BTreeMap::new();
        let mut reserved_names = BTreeSet::new();

        for item in &module.content {
            match item {
                Item::Fn(f) => {
                    if f.type_params.is_empty() {
                        reserved_names.insert(f.ident.to_string());
                    } else {
                        templates.insert(f.ident.to_string(), f.as_ref().clone());
                    }
                }
                Item::Impl(impl_item) => {
                    for ii in &impl_item.items {
                        match ii {
                            crate::parse::ImplItem::Fn(f) => {
                                let mangled = format!("{}_{}", impl_item.self_ty, f.ident);
                                if !reserved_names.insert(mangled.clone()) {
                                    return Err(crate::parse::Error::unsupported(
                                        f.ident.span(),
                                        format!(
                                            "duplicate impl method '{mangled}': another impl \
                                             block already defines a method with this name"
                                        ),
                                    ));
                                }
                            }
                            crate::parse::ImplItem::Const(c) => {
                                let mangled = format!("{}_{}", impl_item.self_ty, c.ident);
                                if !reserved_names.insert(mangled.clone()) {
                                    return Err(crate::parse::Error::unsupported(
                                        c.ident.span(),
                                        format!(
                                            "duplicate impl constant '{mangled}': another impl \
                                             block already defines a constant with this name"
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(MonoCtx {
            templates,
            queue: VecDeque::new(),
            seen: BTreeSet::new(),
            generated: Vec::new(),
            reserved_names,
        })
    }

    /// Generate template macros for generic functions defined in this module.
    ///
    /// For each generic function, produces a `TemplateMacro` containing the
    /// WGSL source with `__TP{name}__` placeholders that can be used for
    /// cross-module instantiation via string substitution.
    fn generate_template_macros(
        &self,
        module: &ItemMod,
    ) -> Result<Vec<TemplateMacro>, crate::parse::Error> {
        let module_name = module.ident.to_string();
        let mut macros = Vec::new();

        for template in self.templates.values() {
            // Build a substitution map that replaces type params with
            // placeholder Type::Struct identifiers (e.g., M → __TPM__).
            // This ensures both Type::TypeParam and FnPath::TypeMethod.ty
            // get replaced with placeholders in the generated WGSL.
            let placeholder_subst: BTreeMap<String, Type> = template
                .type_params
                .iter()
                .map(|id| {
                    let name = id.to_string();
                    let placeholder = format!("__TP{name}__");
                    (
                        name,
                        Type::Struct {
                            ident: Ident::new(&placeholder, id.span()),
                        },
                    )
                })
                .collect();

            // Clone the template and substitute placeholders
            let mut tmpl_fn = template.clone();
            tmpl_fn.type_params.clear();

            // Mangle the function name with placeholders
            let param_placeholders: Vec<String> = template
                .type_params
                .iter()
                .map(|id| format!("__TP{}__", id))
                .collect();
            let mangled_name = std::iter::once(template.ident.to_string())
                .chain(param_placeholders.into_iter())
                .collect::<Vec<_>>()
                .join("_");
            tmpl_fn.ident = Ident::new(&mangled_name, template.ident.span());

            // Substitute types in inputs
            for pair in tmpl_fn.inputs.iter_mut() {
                substitute_type(&mut pair.ty, &placeholder_subst);
            }
            // Substitute in return type
            if let ReturnType::Type { ty, .. } = &mut tmpl_fn.return_type {
                substitute_type(ty, &placeholder_subst);
            }
            // Substitute in block
            substitute_block(&mut tmpl_fn.block, &placeholder_subst);

            // Generate WGSL
            let mut code = code_gen::GeneratedWgslCode::default();
            tmpl_fn.write_code(&mut code);
            let template_wgsl = code.source_lines().join("\n");

            // Compute transitive dependencies by scanning the original
            // (pre-substitution) template body for turbofish calls to other
            // generic functions.
            let type_param_names: Vec<String> = template
                .type_params
                .iter()
                .map(|id| id.to_string())
                .collect();
            let dependencies =
                collect_template_dependencies(&template.block, &self.templates, &type_param_names);

            macros.push(TemplateMacro {
                module_name: module_name.clone(),
                fn_name: template.ident.to_string(),
                type_param_names,
                template_wgsl,
                dependencies,
            });
        }

        Ok(macros)
    }

    /// Walk all concrete items to find turbofish call sites.
    fn discover_instantiations(&mut self, module: &ItemMod) -> Result<(), crate::parse::Error> {
        for item in &module.content {
            match item {
                Item::Fn(f) if f.type_params.is_empty() => {
                    self.collect_from_fn(f)?;
                }
                Item::Impl(impl_item) => {
                    for ii in &impl_item.items {
                        if let crate::parse::ImplItem::Fn(f) = ii {
                            self.collect_from_fn(f)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn collect_from_fn(&mut self, f: &ItemFn) -> Result<(), crate::parse::Error> {
        self.collect_from_block(&f.block)
    }

    fn collect_from_block(&mut self, block: &Block) -> Result<(), crate::parse::Error> {
        for stmt in &block.stmt {
            self.collect_from_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_from_stmt(&mut self, stmt: &Stmt) -> Result<(), crate::parse::Error> {
        match stmt {
            Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    self.collect_from_expr(&init.expr)?;
                }
            }
            Stmt::Const(c) => {
                self.collect_from_expr(&c.expr)?;
            }
            Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
                self.collect_from_expr(lhs)?;
                self.collect_from_expr(rhs)?;
            }
            Stmt::While {
                condition, body, ..
            } => {
                self.collect_from_expr(condition)?;
                self.collect_from_block(body)?;
            }
            Stmt::Loop { body, .. } => {
                self.collect_from_block(body)?;
            }
            Stmt::Expr { expr, .. } => {
                self.collect_from_expr(expr)?;
            }
            Stmt::If(if_stmt) => {
                self.collect_from_if(if_stmt)?;
            }
            Stmt::For(for_loop) => {
                self.collect_from_expr(&for_loop.from)?;
                self.collect_from_expr(&for_loop.to)?;
                self.collect_from_block(&for_loop.body)?;
            }
            Stmt::Switch(switch) => {
                self.collect_from_expr(&switch.selector)?;
                for arm in &switch.arms {
                    self.collect_from_block(&arm.body)?;
                }
            }
            Stmt::Return { expr, .. } => {
                if let Some(e) = expr {
                    self.collect_from_expr(e)?;
                }
            }
            Stmt::Block(block) => {
                self.collect_from_block(block)?;
            }
            Stmt::SlabRead {
                slab,
                offset,
                dest,
                size,
                ..
            } => {
                self.collect_from_expr(slab)?;
                self.collect_from_expr(offset)?;
                self.collect_from_expr(dest)?;
                self.collect_from_expr(size)?;
            }
            Stmt::SlabWrite {
                slab,
                offset,
                src,
                size,
                ..
            } => {
                self.collect_from_expr(slab)?;
                self.collect_from_expr(offset)?;
                self.collect_from_expr(src)?;
                if let Some(s) = size {
                    self.collect_from_expr(s)?;
                }
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
        }
        Ok(())
    }

    fn collect_from_if(
        &mut self,
        if_stmt: &crate::parse::StmtIf,
    ) -> Result<(), crate::parse::Error> {
        self.collect_from_expr(&if_stmt.condition)?;
        self.collect_from_block(&if_stmt.then_block)?;
        if let Some(else_branch) = &if_stmt.else_branch {
            match &else_branch.body {
                ElseBody::Block(block) => self.collect_from_block(block)?,
                ElseBody::If(nested_if) => self.collect_from_if(nested_if)?,
            }
        }
        Ok(())
    }

    fn collect_from_expr(&mut self, expr: &Expr) -> Result<(), crate::parse::Error> {
        match expr {
            Expr::FnCall {
                path,
                type_args,
                params,
                ..
            } => {
                if !type_args.is_empty() {
                    let fn_name = match path {
                        FnPath::Ident(id) => id.to_string(),
                        FnPath::TypeMethod { ty, method, .. } => {
                            format!("{}_{}", ty, method)
                        }
                    };
                    let span = match path {
                        FnPath::Ident(id) => id.span(),
                        FnPath::TypeMethod { ty, .. } => ty.span(),
                    };

                    // Validate this is a known generic function
                    if !self.templates.contains_key(&fn_name) {
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "'{fn_name}' is not a generic function, but was called with type \
                                 arguments"
                            ),
                        ));
                    }

                    let template = &self.templates[&fn_name];
                    if type_args.len() != template.type_params.len() {
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "'{fn_name}' expects {} type argument(s), but {} were provided",
                                template.type_params.len(),
                                type_args.len()
                            ),
                        ));
                    }

                    // Only enqueue if all type args are concrete (no TypeParam)
                    if type_args.iter().all(|ta| !contains_type_param(ta)) {
                        let key = InstKey {
                            fn_name: fn_name.clone(),
                            type_args: type_args
                                .iter()
                                .map(type_to_key)
                                .collect::<Result<Vec<_>, _>>()?,
                        };
                        if !self.seen.contains(&key) {
                            self.seen.insert(key.clone());
                            self.queue.push_back(InstRequest {
                                key,
                                span,
                                concrete_types: type_args.clone(),
                            });
                        }
                    }
                    // If type_args contain TypeParam, this call is inside a
                    // generic function body. It will be
                    // resolved transitively when that
                    // template is instantiated.
                }

                // Also recurse into params
                for param in params.iter() {
                    self.collect_from_expr(param)?;
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.collect_from_expr(lhs)?;
                self.collect_from_expr(rhs)?;
            }
            Expr::Unary { expr, .. } => {
                self.collect_from_expr(expr)?;
            }
            Expr::Paren { inner, .. } => {
                self.collect_from_expr(inner)?;
            }
            Expr::Array { elems, .. } => {
                for elem in elems.iter() {
                    self.collect_from_expr(elem)?;
                }
            }
            Expr::ArrayIndexing { lhs, index, .. } => {
                self.collect_from_expr(lhs)?;
                self.collect_from_expr(index)?;
            }
            Expr::Swizzle { lhs, params, .. } => {
                self.collect_from_expr(lhs)?;
                if let Some(ps) = params {
                    for p in ps.iter() {
                        self.collect_from_expr(p)?;
                    }
                }
            }
            Expr::Cast { lhs, .. } => {
                self.collect_from_expr(lhs)?;
            }
            Expr::Struct { fields, .. } => {
                for field in fields.iter() {
                    self.collect_from_expr(&field.expr)?;
                }
            }
            Expr::FieldAccess { base, .. } => {
                self.collect_from_expr(base)?;
            }
            Expr::Reference { expr, .. } => {
                self.collect_from_expr(expr)?;
            }
            Expr::ZeroValueArray { len, .. } => {
                self.collect_from_expr(len)?;
            }
            Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
        }
        Ok(())
    }

    /// Process the instantiation queue, generating monomorphized functions.
    fn process_queue(&mut self) -> Result<(), crate::parse::Error> {
        while let Some(request) = self.queue.pop_front() {
            if self.generated.len() + self.queue.len() > MAX_INSTANTIATIONS {
                return Err(crate::parse::Error::unsupported(
                    request.span,
                    format!("exceeded maximum of {MAX_INSTANTIATIONS} generic instantiations"),
                ));
            }
            self.instantiate(request)?;
        }
        Ok(())
    }

    /// Instantiate a single generic function with concrete types.
    fn instantiate(&mut self, request: InstRequest) -> Result<(), crate::parse::Error> {
        let template = self.templates[&request.key.fn_name].clone();
        let mangled_name = mangle_name(&request.key.fn_name, &request.concrete_types)?;

        // Check for name collisions
        if self.reserved_names.contains(&mangled_name) {
            return Err(crate::parse::Error::unsupported(
                request.span,
                format!(
                    "generated monomorphized name '{mangled_name}' collides with an existing \
                     function"
                ),
            ));
        }
        self.reserved_names.insert(mangled_name.clone());

        // Build substitution map: type_param_name -> concrete Type
        let subst: BTreeMap<String, Type> = template
            .type_params
            .iter()
            .zip(request.concrete_types.iter())
            .map(|(param, ty)| (param.to_string(), ty.clone()))
            .collect();

        // Clone and substitute
        let mut mono_fn = template.clone();
        mono_fn.type_params.clear();
        mono_fn.ident = Ident::new(&mangled_name, template.ident.span());

        // Substitute types in inputs
        for pair in mono_fn.inputs.iter_mut() {
            substitute_type(&mut pair.ty, &subst);
        }

        // Substitute in return type
        if let ReturnType::Type { ty, .. } = &mut mono_fn.return_type {
            substitute_type(ty, &subst);
        }

        // Substitute in block (types and call sites)
        substitute_block(&mut mono_fn.block, &subst);

        // Discover any new instantiations from the monomorphized body
        self.collect_from_fn(&mono_fn)?;

        self.generated.push(Item::Fn(Box::new(mono_fn)));
        Ok(())
    }

    /// Apply the results: remove generic templates, add monomorphized fns,
    /// rewrite call sites.
    fn apply(&self, module: &mut ItemMod) -> Result<(), crate::parse::Error> {
        // Remove generic template functions
        module.content.retain(|item| {
            if let Item::Fn(f) = item {
                f.type_params.is_empty()
            } else {
                true
            }
        });

        // Add generated monomorphized functions
        module.content.extend(self.generated.clone());

        // Rewrite all call sites in all items
        for item in &mut module.content {
            rewrite_calls_in_item(item, &self.templates);
        }

        Ok(())
    }
}

// ===== Substitution helpers =====

fn substitute_type(ty: &mut Type, subst: &BTreeMap<String, Type>) {
    match ty {
        Type::TypeParam { ident } => {
            if let Some(concrete) = subst.get(&ident.to_string()) {
                *ty = concrete.clone();
            }
        }
        Type::Array { elem, len, .. } => {
            substitute_type(elem, subst);
            substitute_expr(len, subst);
        }
        Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } | Type::Ptr { elem, .. } => {
            substitute_type(elem, subst);
        }
        Type::Scalar { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::Struct { .. }
        | Type::Sampler { .. }
        | Type::SamplerComparison { .. }
        | Type::Texture { .. }
        | Type::TextureDepth { .. } => {}
    }
}

fn substitute_block(block: &mut Block, subst: &BTreeMap<String, Type>) {
    for stmt in &mut block.stmt {
        substitute_stmt(stmt, subst);
    }
}

fn substitute_stmt(stmt: &mut Stmt, subst: &BTreeMap<String, Type>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some((_, ty)) = &mut local.ty {
                substitute_type(ty, subst);
            }
            if let Some(init) = &mut local.init {
                substitute_expr(&mut init.expr, subst);
            }
        }
        Stmt::Const(c) => {
            substitute_type(&mut c.ty, subst);
            substitute_expr(&mut c.expr, subst);
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            substitute_expr(lhs, subst);
            substitute_expr(rhs, subst);
        }
        Stmt::While {
            condition, body, ..
        } => {
            substitute_expr(condition, subst);
            substitute_block(body, subst);
        }
        Stmt::Loop { body, .. } => {
            substitute_block(body, subst);
        }
        Stmt::Expr { expr, .. } => {
            substitute_expr(expr, subst);
        }
        Stmt::If(if_stmt) => {
            substitute_if(if_stmt, subst);
        }
        Stmt::For(for_loop) => {
            if let Some((_, ty)) = &mut for_loop.ty {
                substitute_type(ty, subst);
            }
            substitute_expr(&mut for_loop.from, subst);
            substitute_expr(&mut for_loop.to, subst);
            substitute_block(&mut for_loop.body, subst);
        }
        Stmt::Switch(switch) => {
            substitute_expr(&mut switch.selector, subst);
            for arm in &mut switch.arms {
                for sel in &mut arm.selectors {
                    if let CaseSelector::Expr(e) = sel {
                        substitute_expr(e, subst);
                    }
                }
                substitute_block(&mut arm.body, subst);
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                substitute_expr(e, subst);
            }
        }
        Stmt::Block(block) => {
            substitute_block(block, subst);
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            substitute_expr(slab, subst);
            substitute_expr(offset, subst);
            substitute_expr(dest, subst);
            substitute_expr(size, subst);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            substitute_expr(slab, subst);
            substitute_expr(offset, subst);
            substitute_expr(src, subst);
            if let Some(s) = size {
                substitute_expr(s, subst);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
}

fn substitute_if(if_stmt: &mut crate::parse::StmtIf, subst: &BTreeMap<String, Type>) {
    substitute_expr(&mut if_stmt.condition, subst);
    substitute_block(&mut if_stmt.then_block, subst);
    if let Some(else_branch) = &mut if_stmt.else_branch {
        match &mut else_branch.body {
            ElseBody::Block(block) => substitute_block(block, subst),
            ElseBody::If(nested_if) => substitute_if(nested_if, subst),
        }
    }
}

fn substitute_expr(expr: &mut Expr, subst: &BTreeMap<String, Type>) {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            // Substitute type params in TypeMethod paths (e.g., T::method -> f32::method)
            if let FnPath::TypeMethod { ty, .. } = path {
                if let Some(concrete) = subst.get(&ty.to_string()) {
                    *ty = type_to_ident(concrete, ty.span());
                }
            }
            for ta in type_args.iter_mut() {
                substitute_type(ta, subst);
            }
            for param in params.iter_mut() {
                substitute_expr(param, subst);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            substitute_expr(lhs, subst);
            substitute_expr(rhs, subst);
        }
        Expr::Unary { expr, .. } => {
            substitute_expr(expr, subst);
        }
        Expr::Paren { inner, .. } => {
            substitute_expr(inner, subst);
        }
        Expr::Array { elems, .. } => {
            for elem in elems.iter_mut() {
                substitute_expr(elem, subst);
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            substitute_expr(lhs, subst);
            substitute_expr(index, subst);
        }
        Expr::Swizzle { lhs, params, .. } => {
            substitute_expr(lhs, subst);
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    substitute_expr(p, subst);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            substitute_expr(lhs, subst);
            substitute_type(ty, subst);
        }
        Expr::Struct { fields, .. } => {
            for field in fields.iter_mut() {
                substitute_expr(&mut field.expr, subst);
            }
        }
        Expr::FieldAccess { base, .. } => {
            substitute_expr(base, subst);
        }
        Expr::Reference { expr, .. } => {
            substitute_expr(expr, subst);
        }
        Expr::ZeroValueArray { elem_type, len, .. } => {
            substitute_type(elem_type, subst);
            substitute_expr(len, subst);
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
}

// ===== Call site rewriting =====

fn rewrite_calls_in_item(item: &mut Item, templates: &BTreeMap<String, ItemFn>) {
    match item {
        Item::Fn(f) => rewrite_calls_in_fn(f, templates),
        Item::Impl(impl_item) => {
            for ii in &mut impl_item.items {
                if let crate::parse::ImplItem::Fn(f) = ii {
                    rewrite_calls_in_fn(f, templates);
                }
            }
        }
        _ => {}
    }
}

fn rewrite_calls_in_fn(f: &mut ItemFn, templates: &BTreeMap<String, ItemFn>) {
    rewrite_calls_in_block(&mut f.block, templates);
}

fn rewrite_calls_in_block(block: &mut Block, templates: &BTreeMap<String, ItemFn>) {
    for stmt in &mut block.stmt {
        rewrite_calls_in_stmt(stmt, templates);
    }
}

fn rewrite_calls_in_stmt(stmt: &mut Stmt, templates: &BTreeMap<String, ItemFn>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &mut local.init {
                rewrite_calls_in_expr(&mut init.expr, templates);
            }
        }
        Stmt::Const(c) => {
            rewrite_calls_in_expr(&mut c.expr, templates);
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            rewrite_calls_in_expr(lhs, templates);
            rewrite_calls_in_expr(rhs, templates);
        }
        Stmt::While {
            condition, body, ..
        } => {
            rewrite_calls_in_expr(condition, templates);
            rewrite_calls_in_block(body, templates);
        }
        Stmt::Loop { body, .. } => {
            rewrite_calls_in_block(body, templates);
        }
        Stmt::Expr { expr, .. } => {
            rewrite_calls_in_expr(expr, templates);
        }
        Stmt::If(if_stmt) => {
            rewrite_calls_in_if(if_stmt, templates);
        }
        Stmt::For(for_loop) => {
            rewrite_calls_in_expr(&mut for_loop.from, templates);
            rewrite_calls_in_expr(&mut for_loop.to, templates);
            rewrite_calls_in_block(&mut for_loop.body, templates);
        }
        Stmt::Switch(switch) => {
            rewrite_calls_in_expr(&mut switch.selector, templates);
            for arm in &mut switch.arms {
                rewrite_calls_in_block(&mut arm.body, templates);
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                rewrite_calls_in_expr(e, templates);
            }
        }
        Stmt::Block(block) => {
            rewrite_calls_in_block(block, templates);
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            rewrite_calls_in_expr(slab, templates);
            rewrite_calls_in_expr(offset, templates);
            rewrite_calls_in_expr(dest, templates);
            rewrite_calls_in_expr(size, templates);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            rewrite_calls_in_expr(slab, templates);
            rewrite_calls_in_expr(offset, templates);
            rewrite_calls_in_expr(src, templates);
            if let Some(s) = size {
                rewrite_calls_in_expr(s, templates);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
}

fn rewrite_calls_in_if(if_stmt: &mut crate::parse::StmtIf, templates: &BTreeMap<String, ItemFn>) {
    rewrite_calls_in_expr(&mut if_stmt.condition, templates);
    rewrite_calls_in_block(&mut if_stmt.then_block, templates);
    if let Some(else_branch) = &mut if_stmt.else_branch {
        match &mut else_branch.body {
            ElseBody::Block(block) => rewrite_calls_in_block(block, templates),
            ElseBody::If(nested_if) => rewrite_calls_in_if(nested_if, templates),
        }
    }
}

fn rewrite_calls_in_expr(expr: &mut Expr, templates: &BTreeMap<String, ItemFn>) {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            if !type_args.is_empty() {
                let fn_name = match &*path {
                    FnPath::Ident(id) => id.to_string(),
                    FnPath::TypeMethod { ty, method, .. } => {
                        format!("{}_{}", ty, method)
                    }
                };

                if templates.contains_key(&fn_name) {
                    // Mangle the name and clear type_args
                    let mangled = mangle_name(&fn_name, type_args)
                        .expect("mangle_name should not fail for concrete types");
                    let span = match &*path {
                        FnPath::Ident(id) => id.span(),
                        FnPath::TypeMethod { ty, .. } => ty.span(),
                    };
                    *path = FnPath::Ident(Ident::new(&mangled, span));
                    type_args.clear();
                }
            }

            for param in params.iter_mut() {
                rewrite_calls_in_expr(param, templates);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            rewrite_calls_in_expr(lhs, templates);
            rewrite_calls_in_expr(rhs, templates);
        }
        Expr::Unary { expr, .. } => {
            rewrite_calls_in_expr(expr, templates);
        }
        Expr::Paren { inner, .. } => {
            rewrite_calls_in_expr(inner, templates);
        }
        Expr::Array { elems, .. } => {
            for elem in elems.iter_mut() {
                rewrite_calls_in_expr(elem, templates);
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            rewrite_calls_in_expr(lhs, templates);
            rewrite_calls_in_expr(index, templates);
        }
        Expr::Swizzle { lhs, params, .. } => {
            rewrite_calls_in_expr(lhs, templates);
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    rewrite_calls_in_expr(p, templates);
                }
            }
        }
        Expr::Cast { lhs, .. } => {
            rewrite_calls_in_expr(lhs, templates);
        }
        Expr::Struct { fields, .. } => {
            for field in fields.iter_mut() {
                rewrite_calls_in_expr(&mut field.expr, templates);
            }
        }
        Expr::FieldAccess { base, .. } => {
            rewrite_calls_in_expr(base, templates);
        }
        Expr::Reference { expr, .. } => {
            rewrite_calls_in_expr(expr, templates);
        }
        Expr::ZeroValueArray { len, .. } => {
            rewrite_calls_in_expr(len, templates);
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
}

// ===== Missing turbofish detection =====

/// Check that no calls to generic template functions remain without turbofish.
///
/// After monomorphization, all generic templates are removed. Any remaining
/// call to a template name with empty `type_args` means the user forgot
/// the turbofish annotation.
fn check_unresolved_generic_calls(
    item: &Item,
    templates: &BTreeMap<String, ItemFn>,
) -> Result<(), crate::parse::Error> {
    match item {
        Item::Fn(f) => check_unresolved_in_block(&f.block, templates),
        Item::Impl(impl_item) => {
            for ii in &impl_item.items {
                if let crate::parse::ImplItem::Fn(f) = ii {
                    check_unresolved_in_block(&f.block, templates)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_unresolved_in_block(
    block: &Block,
    templates: &BTreeMap<String, ItemFn>,
) -> Result<(), crate::parse::Error> {
    for stmt in &block.stmt {
        check_unresolved_in_stmt(stmt, templates)?;
    }
    Ok(())
}

fn check_unresolved_in_stmt(
    stmt: &Stmt,
    templates: &BTreeMap<String, ItemFn>,
) -> Result<(), crate::parse::Error> {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                check_unresolved_in_expr(&init.expr, templates)?;
            }
        }
        Stmt::Const(c) => check_unresolved_in_expr(&c.expr, templates)?,
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            check_unresolved_in_expr(lhs, templates)?;
            check_unresolved_in_expr(rhs, templates)?;
        }
        Stmt::While {
            condition, body, ..
        } => {
            check_unresolved_in_expr(condition, templates)?;
            check_unresolved_in_block(body, templates)?;
        }
        Stmt::Loop { body, .. } => check_unresolved_in_block(body, templates)?,
        Stmt::Expr { expr, .. } => check_unresolved_in_expr(expr, templates)?,
        Stmt::If(s) => {
            check_unresolved_in_expr(&s.condition, templates)?;
            check_unresolved_in_block(&s.then_block, templates)?;
            if let Some(eb) = &s.else_branch {
                match &eb.body {
                    ElseBody::Block(b) => check_unresolved_in_block(b, templates)?,
                    ElseBody::If(nested) => {
                        check_unresolved_in_expr(&nested.condition, templates)?;
                        check_unresolved_in_block(&nested.then_block, templates)?;
                    }
                }
            }
        }
        Stmt::For(f) => {
            check_unresolved_in_expr(&f.from, templates)?;
            check_unresolved_in_expr(&f.to, templates)?;
            check_unresolved_in_block(&f.body, templates)?;
        }
        Stmt::Switch(sw) => {
            check_unresolved_in_expr(&sw.selector, templates)?;
            for arm in &sw.arms {
                check_unresolved_in_block(&arm.body, templates)?;
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                check_unresolved_in_expr(e, templates)?;
            }
        }
        Stmt::Block(b) => check_unresolved_in_block(b, templates)?,
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            check_unresolved_in_expr(slab, templates)?;
            check_unresolved_in_expr(offset, templates)?;
            check_unresolved_in_expr(dest, templates)?;
            check_unresolved_in_expr(size, templates)?;
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            check_unresolved_in_expr(slab, templates)?;
            check_unresolved_in_expr(offset, templates)?;
            check_unresolved_in_expr(src, templates)?;
            if let Some(s) = size {
                check_unresolved_in_expr(s, templates)?;
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
    Ok(())
}

fn check_unresolved_in_expr(
    expr: &Expr,
    templates: &BTreeMap<String, ItemFn>,
) -> Result<(), crate::parse::Error> {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            if type_args.is_empty() {
                if let FnPath::Ident(id) = path {
                    let name = id.to_string();
                    if let Some(template) = templates.get(&name) {
                        let param_names: Vec<_> =
                            template.type_params.iter().map(|p| p.to_string()).collect();
                        return Err(crate::parse::Error::unsupported(
                            id.span(),
                            format!(
                                "calling generic function '{name}' requires type arguments, e.g., \
                                 {name}::<{}>(...)",
                                param_names.join(", ")
                            ),
                        ));
                    }
                }
            }
            for param in params.iter() {
                check_unresolved_in_expr(param, templates)?;
            }
            Ok(())
        }
        Expr::Binary { lhs, rhs, .. } => {
            check_unresolved_in_expr(lhs, templates)?;
            check_unresolved_in_expr(rhs, templates)
        }
        Expr::Unary { expr, .. } | Expr::Paren { inner: expr, .. } => {
            check_unresolved_in_expr(expr, templates)
        }
        Expr::Array { elems, .. } => {
            for e in elems.iter() {
                check_unresolved_in_expr(e, templates)?;
            }
            Ok(())
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            check_unresolved_in_expr(lhs, templates)?;
            check_unresolved_in_expr(index, templates)
        }
        Expr::Swizzle { lhs, params, .. } => {
            check_unresolved_in_expr(lhs, templates)?;
            if let Some(ps) = params {
                for p in ps.iter() {
                    check_unresolved_in_expr(p, templates)?;
                }
            }
            Ok(())
        }
        Expr::Cast { lhs, .. } | Expr::FieldAccess { base: lhs, .. } => {
            check_unresolved_in_expr(lhs, templates)
        }
        Expr::Struct { fields, .. } => {
            for f in fields.iter() {
                check_unresolved_in_expr(&f.expr, templates)?;
            }
            Ok(())
        }
        Expr::Reference { expr, .. } => check_unresolved_in_expr(expr, templates),
        Expr::ZeroValueArray { len, .. } => check_unresolved_in_expr(len, templates),
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => Ok(()),
    }
}

// ===== Cross-module instantiation collection =====

/// Collect turbofish calls that reference generic functions from imported
/// modules. These become `CrossModuleInstantiation`s that the consuming
/// module uses to generate macro invocations.
///
/// Also rewrites the call sites to use the mangled concrete function name.
fn collect_cross_module_instantiations(
    module: &mut ItemMod,
    local_templates: &BTreeMap<String, ItemFn>,
) -> Result<Vec<CrossModuleInstantiation>, crate::parse::Error> {
    // Build the import path lookup: for each use statement, record the
    // paths so we can attribute unresolved calls to an imported module.
    let import_paths: Vec<syn::Path> = module
        .content
        .iter()
        .filter_map(|item| {
            if let Item::Use(use_item) = item {
                Some(use_item.modules.clone())
            } else {
                None
            }
        })
        .flatten()
        .collect();

    let mut instantiations = Vec::new();
    let mut seen_mangled: BTreeSet<String> = BTreeSet::new();

    for item in &mut module.content {
        collect_cross_module_from_item(
            item,
            local_templates,
            &import_paths,
            &mut instantiations,
            &mut seen_mangled,
        )?;
    }

    Ok(instantiations)
}

fn collect_cross_module_from_item(
    item: &mut Item,
    local_templates: &BTreeMap<String, ItemFn>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    match item {
        Item::Fn(f) => {
            collect_cross_module_from_block(&mut f.block, local_templates, import_paths, out, seen)
        }
        Item::Impl(impl_item) => {
            for ii in &mut impl_item.items {
                if let crate::parse::ImplItem::Fn(f) = ii {
                    collect_cross_module_from_block(
                        &mut f.block,
                        local_templates,
                        import_paths,
                        out,
                        seen,
                    )?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn collect_cross_module_from_block(
    block: &mut Block,
    local_templates: &BTreeMap<String, ItemFn>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    for stmt in &mut block.stmt {
        collect_cross_module_from_stmt(stmt, local_templates, import_paths, out, seen)?;
    }
    Ok(())
}

fn collect_cross_module_from_stmt(
    stmt: &mut Stmt,
    local_templates: &BTreeMap<String, ItemFn>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    match stmt {
        Stmt::Local(local) => {
            if let Some(init) = &mut local.init {
                collect_cross_module_from_expr(
                    &mut init.expr,
                    local_templates,
                    import_paths,
                    out,
                    seen,
                )?;
            }
        }
        Stmt::Const(c) => {
            collect_cross_module_from_expr(&mut c.expr, local_templates, import_paths, out, seen)?;
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            collect_cross_module_from_expr(lhs, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(rhs, local_templates, import_paths, out, seen)?;
        }
        Stmt::While {
            condition, body, ..
        } => {
            collect_cross_module_from_expr(condition, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_block(body, local_templates, import_paths, out, seen)?;
        }
        Stmt::Loop { body, .. } => {
            collect_cross_module_from_block(body, local_templates, import_paths, out, seen)?;
        }
        Stmt::Expr { expr, .. } => {
            collect_cross_module_from_expr(expr, local_templates, import_paths, out, seen)?;
        }
        Stmt::If(s) => {
            collect_cross_module_from_expr(
                &mut s.condition,
                local_templates,
                import_paths,
                out,
                seen,
            )?;
            collect_cross_module_from_block(
                &mut s.then_block,
                local_templates,
                import_paths,
                out,
                seen,
            )?;
            if let Some(eb) = &mut s.else_branch {
                match &mut eb.body {
                    ElseBody::Block(b) => collect_cross_module_from_block(
                        b,
                        local_templates,
                        import_paths,
                        out,
                        seen,
                    )?,
                    ElseBody::If(nested) => {
                        collect_cross_module_from_expr(
                            &mut nested.condition,
                            local_templates,
                            import_paths,
                            out,
                            seen,
                        )?;
                        collect_cross_module_from_block(
                            &mut nested.then_block,
                            local_templates,
                            import_paths,
                            out,
                            seen,
                        )?;
                    }
                }
            }
        }
        Stmt::For(f) => {
            collect_cross_module_from_expr(&mut f.from, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(&mut f.to, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_block(&mut f.body, local_templates, import_paths, out, seen)?;
        }
        Stmt::Switch(sw) => {
            collect_cross_module_from_expr(
                &mut sw.selector,
                local_templates,
                import_paths,
                out,
                seen,
            )?;
            for arm in &mut sw.arms {
                collect_cross_module_from_block(
                    &mut arm.body,
                    local_templates,
                    import_paths,
                    out,
                    seen,
                )?;
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                collect_cross_module_from_expr(e, local_templates, import_paths, out, seen)?;
            }
        }
        Stmt::Block(b) => {
            collect_cross_module_from_block(b, local_templates, import_paths, out, seen)?;
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            collect_cross_module_from_expr(slab, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(offset, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(dest, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(size, local_templates, import_paths, out, seen)?;
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            collect_cross_module_from_expr(slab, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(offset, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(src, local_templates, import_paths, out, seen)?;
            if let Some(s) = size {
                collect_cross_module_from_expr(s, local_templates, import_paths, out, seen)?;
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
    Ok(())
}

fn collect_cross_module_from_expr(
    expr: &mut Expr,
    local_templates: &BTreeMap<String, ItemFn>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            if !type_args.is_empty() {
                let fn_name = match &*path {
                    FnPath::Ident(id) => id.to_string(),
                    FnPath::TypeMethod { ty, method, .. } => {
                        format!("{}_{}", ty, method)
                    }
                };

                // Skip if this is a local template (already handled by same-module mono)
                if !local_templates.contains_key(&fn_name) {
                    // This is a cross-module generic call. We need an import path.
                    if import_paths.is_empty() {
                        let span = match &*path {
                            FnPath::Ident(id) => id.span(),
                            FnPath::TypeMethod { ty, .. } => ty.span(),
                        };
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "generic function '{fn_name}' is not defined in this module and \
                                 no imports are available. Define it in this module or import the \
                                 module that defines it."
                            ),
                        ));
                    }

                    // Compute mangled type arg names
                    let mangled_type_args: Vec<String> = type_args
                        .iter()
                        .map(|ty| mangle_type(ty))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mangled_name = mangle_name(&fn_name, type_args)?;

                    // Determine the import path — use the last segment as the module name.
                    // For `use super::renderer_specialization::*;`, the path is
                    // `super::renderer_specialization`, and the module name is
                    // `renderer_specialization`.
                    // We use the first import path that exists. In practice there's
                    // usually just one relevant import.
                    let import_path = import_paths[0].clone();
                    let module_name = import_path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();

                    // Rewrite the call site to use the mangled name
                    let span = match &*path {
                        FnPath::Ident(id) => id.span(),
                        FnPath::TypeMethod { ty, .. } => ty.span(),
                    };
                    *path = FnPath::Ident(Ident::new(&mangled_name, span));
                    type_args.clear();

                    // Add instantiation if not already seen
                    if seen.insert(mangled_name) {
                        out.push(CrossModuleInstantiation {
                            import_path,
                            module_name,
                            fn_name,
                            mangled_type_args,
                        });
                    }
                }
            }

            for param in params.iter_mut() {
                collect_cross_module_from_expr(param, local_templates, import_paths, out, seen)?;
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_cross_module_from_expr(lhs, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(rhs, local_templates, import_paths, out, seen)?;
        }
        Expr::Unary { expr, .. } | Expr::Paren { inner: expr, .. } => {
            collect_cross_module_from_expr(expr, local_templates, import_paths, out, seen)?;
        }
        Expr::Array { elems, .. } => {
            for e in elems.iter_mut() {
                collect_cross_module_from_expr(e, local_templates, import_paths, out, seen)?;
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            collect_cross_module_from_expr(lhs, local_templates, import_paths, out, seen)?;
            collect_cross_module_from_expr(index, local_templates, import_paths, out, seen)?;
        }
        Expr::Swizzle { lhs, params, .. } => {
            collect_cross_module_from_expr(lhs, local_templates, import_paths, out, seen)?;
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    collect_cross_module_from_expr(p, local_templates, import_paths, out, seen)?;
                }
            }
        }
        Expr::Cast { lhs, .. } | Expr::FieldAccess { base: lhs, .. } => {
            collect_cross_module_from_expr(lhs, local_templates, import_paths, out, seen)?;
        }
        Expr::Struct { fields, .. } => {
            for f in fields.iter_mut() {
                collect_cross_module_from_expr(
                    &mut f.expr,
                    local_templates,
                    import_paths,
                    out,
                    seen,
                )?;
            }
        }
        Expr::Reference { expr, .. } => {
            collect_cross_module_from_expr(expr, local_templates, import_paths, out, seen)?;
        }
        Expr::ZeroValueArray { len, .. } => {
            collect_cross_module_from_expr(len, local_templates, import_paths, out, seen)?;
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
    Ok(())
}

// ===== Template dependency scanning =====

/// Scan a block for turbofish calls to other generic functions and build
/// `TemplateDep` entries recording the type param mappings.
fn collect_template_dependencies(
    block: &Block,
    templates: &BTreeMap<String, ItemFn>,
    caller_type_params: &[String],
) -> Vec<TemplateDep> {
    let mut deps = Vec::new();
    let mut seen = BTreeSet::new();
    scan_block_for_deps(block, templates, caller_type_params, &mut deps, &mut seen);
    deps
}

fn scan_block_for_deps(
    block: &Block,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<String>,
) {
    for stmt in &block.stmt {
        scan_stmt_for_deps(stmt, templates, caller_params, out, seen);
    }
}

fn scan_stmt_for_deps(
    stmt: &Stmt,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<String>,
) {
    match stmt {
        Stmt::Local(l) => {
            if let Some(init) = &l.init {
                scan_expr_for_deps(&init.expr, templates, caller_params, out, seen);
            }
        }
        Stmt::Const(c) => scan_expr_for_deps(&c.expr, templates, caller_params, out, seen),
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen);
            scan_expr_for_deps(rhs, templates, caller_params, out, seen);
        }
        Stmt::While {
            condition, body, ..
        } => {
            scan_expr_for_deps(condition, templates, caller_params, out, seen);
            scan_block_for_deps(body, templates, caller_params, out, seen);
        }
        Stmt::Loop { body, .. } => scan_block_for_deps(body, templates, caller_params, out, seen),
        Stmt::Expr { expr, .. } => scan_expr_for_deps(expr, templates, caller_params, out, seen),
        Stmt::If(s) => {
            scan_expr_for_deps(&s.condition, templates, caller_params, out, seen);
            scan_block_for_deps(&s.then_block, templates, caller_params, out, seen);
            if let Some(eb) = &s.else_branch {
                match &eb.body {
                    ElseBody::Block(b) => {
                        scan_block_for_deps(b, templates, caller_params, out, seen)
                    }
                    ElseBody::If(nested) => {
                        scan_expr_for_deps(&nested.condition, templates, caller_params, out, seen);
                        scan_block_for_deps(
                            &nested.then_block,
                            templates,
                            caller_params,
                            out,
                            seen,
                        );
                    }
                }
            }
        }
        Stmt::For(f) => {
            scan_expr_for_deps(&f.from, templates, caller_params, out, seen);
            scan_expr_for_deps(&f.to, templates, caller_params, out, seen);
            scan_block_for_deps(&f.body, templates, caller_params, out, seen);
        }
        Stmt::Switch(sw) => {
            scan_expr_for_deps(&sw.selector, templates, caller_params, out, seen);
            for arm in &sw.arms {
                scan_block_for_deps(&arm.body, templates, caller_params, out, seen);
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                scan_expr_for_deps(e, templates, caller_params, out, seen);
            }
        }
        Stmt::Block(b) => scan_block_for_deps(b, templates, caller_params, out, seen),
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            scan_expr_for_deps(slab, templates, caller_params, out, seen);
            scan_expr_for_deps(offset, templates, caller_params, out, seen);
            scan_expr_for_deps(dest, templates, caller_params, out, seen);
            scan_expr_for_deps(size, templates, caller_params, out, seen);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            scan_expr_for_deps(slab, templates, caller_params, out, seen);
            scan_expr_for_deps(offset, templates, caller_params, out, seen);
            scan_expr_for_deps(src, templates, caller_params, out, seen);
            if let Some(s) = size {
                scan_expr_for_deps(s, templates, caller_params, out, seen);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
}

fn scan_expr_for_deps(
    expr: &Expr,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<String>,
) {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            if !type_args.is_empty() {
                let fn_name = match path {
                    FnPath::Ident(id) => id.to_string(),
                    FnPath::TypeMethod { .. } => String::new(), /* TypeMethod deps are handled
                                                                 * differently */
                };

                if !fn_name.is_empty()
                    && templates.contains_key(&fn_name)
                    && seen.insert(fn_name.clone())
                {
                    // Build the type param mapping: for each type_arg, find
                    // which caller param it refers to.
                    let mapping: Vec<usize> = type_args
                        .iter()
                        .map(|ta| {
                            if let Type::TypeParam { ident } = ta {
                                caller_params
                                    .iter()
                                    .position(|p| p == &ident.to_string())
                                    .unwrap_or(0)
                            } else {
                                0 // Concrete type — not a param reference
                            }
                        })
                        .collect();
                    out.push(TemplateDep {
                        callee: fn_name,
                        type_param_mapping: mapping,
                    });
                }
            }
            for param in params.iter() {
                scan_expr_for_deps(param, templates, caller_params, out, seen);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen);
            scan_expr_for_deps(rhs, templates, caller_params, out, seen);
        }
        Expr::Unary { expr, .. } | Expr::Paren { inner: expr, .. } => {
            scan_expr_for_deps(expr, templates, caller_params, out, seen);
        }
        Expr::Array { elems, .. } => {
            for e in elems.iter() {
                scan_expr_for_deps(e, templates, caller_params, out, seen);
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen);
            scan_expr_for_deps(index, templates, caller_params, out, seen);
        }
        Expr::Swizzle { lhs, params, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen);
            if let Some(ps) = params {
                for p in ps.iter() {
                    scan_expr_for_deps(p, templates, caller_params, out, seen);
                }
            }
        }
        Expr::Cast { lhs, .. } | Expr::FieldAccess { base: lhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen);
        }
        Expr::Struct { fields, .. } => {
            for f in fields.iter() {
                scan_expr_for_deps(&f.expr, templates, caller_params, out, seen);
            }
        }
        Expr::Reference { expr, .. } => {
            scan_expr_for_deps(expr, templates, caller_params, out, seen)
        }
        Expr::ZeroValueArray { len, .. } => {
            scan_expr_for_deps(len, templates, caller_params, out, seen)
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
}

// ===== Type-to-ident conversion =====

/// Convert a concrete `Type` to the `Ident` used in `FnPath::TypeMethod`
/// and impl block name mangling (e.g., `Type::Scalar { ty: F32, .. }` → `f32`).
fn type_to_ident(ty: &Type, span: Span) -> Ident {
    let name = match ty {
        Type::Scalar { ty: scalar, .. } => scalar.wgsl_name().to_string(),
        Type::Vector {
            elements,
            scalar_ty,
            ..
        } => format!("Vec{}{}", elements, scalar_ty.short_name()),
        Type::Matrix { size, .. } => format!("Mat{}f", size),
        Type::Struct { ident } => return ident.clone(),
        Type::Sampler { ident } => return ident.clone(),
        Type::SamplerComparison { ident } => return ident.clone(),
        Type::Texture { ident, .. } => return ident.clone(),
        Type::TextureDepth { ident, .. } => return ident.clone(),
        // These shouldn't appear as self_ty in practice
        Type::Array { .. } => "array".to_string(),
        Type::RuntimeArray { .. } => "array".to_string(),
        Type::Atomic { .. } => "atomic".to_string(),
        Type::Ptr { .. } => "ptr".to_string(),
        Type::TypeParam { ident } => return ident.clone(),
    };
    Ident::new(&name, span)
}

// ===== Name mangling =====

fn mangle_name(base: &str, type_args: &[Type]) -> Result<String, crate::parse::Error> {
    let mut name = base.to_string();
    for ty in type_args {
        name.push('_');
        name.push_str(&mangle_type(ty)?);
    }
    Ok(name)
}

fn mangle_type(ty: &Type) -> Result<String, crate::parse::Error> {
    Ok(match ty {
        Type::Scalar { ty: scalar, .. } => scalar.wgsl_name().to_string(),
        Type::Vector {
            elements,
            scalar_ty,
            ..
        } => format!("vec{}{}", elements, scalar_ty.short_name()),
        Type::Matrix { size, .. } => format!("mat{}x{}f", size, size),
        Type::Struct { ident } => ident.to_string(),
        Type::Array { elem, len, .. } => {
            format!("array_{}_{}", mangle_type(elem)?, len_to_string(len))
        }
        Type::RuntimeArray { elem, .. } => format!("array_{}", mangle_type(elem)?),
        Type::Atomic { elem, .. } => format!("atomic_{}", mangle_type(elem)?),
        Type::Sampler { .. } => "sampler".to_string(),
        Type::SamplerComparison { .. } => "sampler_comparison".to_string(),
        Type::Texture { kind, .. } => kind.wgsl_name().replace("texture_", "tex_"),
        Type::TextureDepth { kind, .. } => kind.wgsl_name().replace("texture_", "tex_"),
        Type::Ptr { elem, .. } => format!("ptr_{}", mangle_type(elem)?),
        Type::TypeParam { ident } => {
            // This shouldn't happen for fully resolved instantiations
            ident.to_string().to_lowercase()
        }
    })
}

/// Convert an array length expression to a string for name mangling.
fn len_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Lit(lit) => format!("{lit}"),
        Expr::Ident(id) => id.to_string(),
        _ => "n".to_string(),
    }
}

// ===== Type key conversion =====

fn type_to_key(ty: &Type) -> Result<TypeKey, crate::parse::Error> {
    Ok(match ty {
        Type::Scalar { ty: scalar, .. } => TypeKey::Scalar(scalar.wgsl_name().to_string()),
        Type::Vector {
            elements,
            scalar_ty,
            ..
        } => TypeKey::Vector(
            *elements,
            Box::new(TypeKey::Scalar(scalar_ty.wgsl_name().to_string())),
        ),
        Type::Matrix { size, .. } => TypeKey::Matrix(*size),
        Type::Struct { ident } => TypeKey::Struct(ident.to_string()),
        Type::Array { elem, len, .. } => {
            TypeKey::Array(Box::new(type_to_key(elem)?), len_to_string(len))
        }
        Type::RuntimeArray { elem, .. } => TypeKey::RuntimeArray(Box::new(type_to_key(elem)?)),
        Type::Atomic { elem, .. } => TypeKey::Atomic(Box::new(type_to_key(elem)?)),
        Type::Sampler { .. } => TypeKey::Sampler,
        Type::SamplerComparison { .. } => TypeKey::SamplerComparison,
        Type::Texture { kind, .. } => TypeKey::Texture(kind.wgsl_name().to_string()),
        Type::TextureDepth { kind, .. } => TypeKey::TextureDepth(kind.wgsl_name().to_string()),
        Type::Ptr {
            address_space,
            elem,
            ..
        } => TypeKey::Ptr(format!("{:?}", address_space), Box::new(type_to_key(elem)?)),
        Type::TypeParam { ident } => TypeKey::TypeParam(ident.to_string()),
    })
}

/// Returns true if the type contains any unresolved `TypeParam`.
fn contains_type_param(ty: &Type) -> bool {
    match ty {
        Type::TypeParam { .. } => true,
        Type::Array { elem, .. }
        | Type::RuntimeArray { elem, .. }
        | Type::Atomic { elem, .. }
        | Type::Ptr { elem, .. } => contains_type_param(elem),
        Type::Scalar { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
        | Type::Struct { .. }
        | Type::Sampler { .. }
        | Type::SamplerComparison { .. }
        | Type::Texture { .. }
        | Type::TextureDepth { .. } => false,
    }
}

#[cfg(test)]
mod test {
    use crate::{
        code_gen,
        parse::{Item, ItemMod},
    };

    /// Helper: parse a module, run monomorphization, return the WGSL string.
    fn mono_wgsl(input: syn::ItemMod) -> String {
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        super::run(&mut wgsl_module).unwrap();
        let code = code_gen::generate_wgsl(&wgsl_module);
        code.source_lines().join("\n")
    }

    /// Helper: parse a module, run monomorphization, expect an error.
    fn mono_err(input: syn::ItemMod) -> String {
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        let err = super::run(&mut wgsl_module).unwrap_err();
        format!("{err}")
    }

    #[test]
    fn mono_simple() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn caller() -> f32 {
                    identity::<f32>(1.0)
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn identity_f32("),
            "Expected monomorphized fn identity_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("identity_f32(1.0"),
            "Expected rewritten call site, got:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("fn identity("),
            "Generic template should be removed, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_transitive() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<T>(x: T) -> T {
                    x
                }

                pub fn outer<T>(x: T) -> T {
                    inner::<T>(x)
                }

                pub fn root() -> f32 {
                    outer::<f32>(1.0)
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn outer_f32("),
            "Expected outer_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("fn inner_f32("),
            "Expected inner_f32 from transitive instantiation, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_multiple_instantiations() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn use_both() -> f32 {
                    let a = identity::<f32>(1.0);
                    let b = identity::<u32>(1u32);
                    a
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn identity_f32("),
            "Expected identity_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("fn identity_u32("),
            "Expected identity_u32, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_dedup() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn caller() -> f32 {
                    let a = identity::<f32>(1.0);
                    let b = identity::<f32>(2.0);
                    a
                }
            }
        };
        let wgsl = mono_wgsl(input);
        // Should only generate one identity_f32, not two
        let count = wgsl.matches("fn identity_f32(").count();
        assert_eq!(count, 1, "Expected exactly 1 identity_f32, got {count}");
    }

    #[test]
    fn mono_no_type_params_after() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn caller() -> f32 {
                    identity::<f32>(1.0)
                }
            }
        };
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        super::run(&mut wgsl_module).unwrap();

        // Verify no Item::Fn with non-empty type_params
        for item in &wgsl_module.content {
            if let Item::Fn(f) = item {
                assert!(
                    f.type_params.is_empty(),
                    "Found fn '{}' with unresolved type_params",
                    f.ident
                );
            }
        }
    }

    #[test]
    fn mono_name_collision() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn foo<T>(x: T) -> T {
                    x
                }

                pub fn foo_f32(x: f32) -> f32 {
                    x
                }

                pub fn caller() -> f32 {
                    foo::<f32>(1.0)
                }
            }
        };
        let err = mono_err(input);
        assert!(
            err.contains("collides"),
            "Expected collision error, got: {err}"
        );
    }

    #[test]
    fn mono_multi_type_params() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn pair<T, U>(a: T, b: U) -> T {
                    a
                }

                pub fn caller() -> f32 {
                    pair::<f32, u32>(1.0, 1u32)
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn pair_f32_u32("),
            "Expected pair_f32_u32, got:\n{wgsl}"
        );
    }

    #[test]
    fn trait_impl_generates_wgsl() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub trait Doubler {
                    fn double(x: f32) -> f32;
                }

                impl Doubler for f32 {
                    pub fn double(x: f32) -> f32 {
                        x + x
                    }
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn f32_double("),
            "Trait impl should generate f32_double, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_with_trait_impl() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub trait Doubler {
                    fn double(x: Self) -> Self;
                }

                impl Doubler for f32 {
                    pub fn double(x: f32) -> f32 {
                        x + x
                    }
                }

                pub fn apply<T: Doubler>(x: T) -> T {
                    T::double(x)
                }

                pub fn go() -> f32 {
                    apply::<f32>(1.0)
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn f32_double("),
            "Trait impl should generate f32_double, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("fn apply_f32("),
            "Generic fn should monomorphize to apply_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("f32_double("),
            "Monomorphized apply_f32 should call f32_double, got:\n{wgsl}"
        );
    }

    #[test]
    fn trait_impl_collision_detected() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Foo {}

                impl Foo {
                    pub fn bar() -> f32 {
                        1.0
                    }
                }

                pub trait Baz {
                    fn bar() -> f32;
                }

                impl Baz for Foo {
                    pub fn bar() -> f32 {
                        2.0
                    }
                }
            }
        };
        let err = mono_err(input);
        assert!(
            err.contains("duplicate"),
            "Expected collision error, got: {err}"
        );
    }

    #[test]
    fn mono_missing_turbofish_error() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn caller() -> f32 {
                    identity(1.0)
                }
            }
        };
        let err = mono_err(input);
        assert!(
            err.contains("requires type arguments"),
            "Expected missing turbofish error, got: {err}"
        );
        assert!(
            err.contains("identity"),
            "Error should name the generic function, got: {err}"
        );
    }

    #[test]
    fn template_macro_generation() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn identity<T>(x: T) -> T {
                    x
                }

                pub fn caller() -> f32 {
                    identity::<f32>(1.0)
                }
            }
        };
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        let result = super::run(&mut wgsl_module).unwrap();

        assert_eq!(result.template_macros.len(), 1, "Expected 1 template macro");
        let tmpl = &result.template_macros[0];
        assert_eq!(tmpl.fn_name, "identity");
        assert_eq!(tmpl.type_param_names, vec!["T"]);
        assert_eq!(tmpl.module_name, "test_mod");
        assert!(
            tmpl.template_wgsl.contains("__TPT__"),
            "Template WGSL should contain __TPT__ placeholder, got: {}",
            tmpl.template_wgsl
        );
    }

    #[test]
    fn template_wgsl_mangles_turbofish_calls() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<T>(x: T) -> T {
                    x
                }

                pub fn outer<T>(x: T) -> T {
                    inner::<T>(x)
                }

                pub fn caller() -> f32 {
                    outer::<f32>(1.0)
                }
            }
        };
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        let result = super::run(&mut wgsl_module).unwrap();

        // Find the template for `outer`
        let outer_tmpl = result
            .template_macros
            .iter()
            .find(|t| t.fn_name == "outer")
            .expect("Expected template for 'outer'");

        // The template WGSL should contain inner___TPT__ (mangled turbofish call)
        assert!(
            outer_tmpl.template_wgsl.contains("inner___TPT__"),
            "Template WGSL for outer should contain mangled call 'inner___TPT__', got:\n{}",
            outer_tmpl.template_wgsl
        );
    }

    #[test]
    fn template_dependency_metadata() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<T>(x: T) -> T {
                    x
                }

                pub fn outer<T>(x: T) -> T {
                    inner::<T>(x)
                }

                pub fn caller() -> f32 {
                    outer::<f32>(1.0)
                }
            }
        };
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        let result = super::run(&mut wgsl_module).unwrap();

        let outer_tmpl = result
            .template_macros
            .iter()
            .find(|t| t.fn_name == "outer")
            .expect("Expected template for 'outer'");

        // outer depends on inner, with T mapped to outer's 0th param
        assert_eq!(outer_tmpl.dependencies.len(), 1);
        assert_eq!(outer_tmpl.dependencies[0].callee, "inner");
        assert_eq!(outer_tmpl.dependencies[0].type_param_mapping, vec![0]);

        // inner has no dependencies
        let inner_tmpl = result
            .template_macros
            .iter()
            .find(|t| t.fn_name == "inner")
            .expect("Expected template for 'inner'");
        assert!(inner_tmpl.dependencies.is_empty());
    }
}
