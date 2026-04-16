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
        Block, CaseSelector, ElseBody, Expr, FnPath, Item, ItemFn, ItemImpl, ItemMod, ItemStruct,
        ReturnType, Stmt, Type,
    },
};

/// A cross-module template instantiation that needs to be included in the
/// consuming module's `WGSL_MODULE.source` as a macro invocation.
#[derive(Debug, Clone)]
pub struct CrossModuleInstantiation {
    /// Candidate imported modules that may contain the template.
    pub import_paths: Vec<syn::Path>,
    /// The generic function name.
    pub fn_name: String,
    /// Identifier-safe mangled type argument names, used for function name
    /// mangling and deduplication keys (e.g. `"array_f32_4"`,
    /// `"ptr_function_f32"`).
    ///
    /// These are NOT valid WGSL syntax — see `wgsl_type_args` for that.
    pub mangled_type_args: Vec<String>,
    /// Valid WGSL type syntax strings for each type argument (e.g.
    /// `"array<f32, 4>"`, `"ptr<function, f32>"`).
    ///
    /// These are used at runtime for placeholder substitution in template WGSL
    /// source (`__TPT__` -> `array<f32, 4>`).
    pub wgsl_type_args: Vec<String>,
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

    // Run same-module monomorphization if there are any generic templates
    let has_templates = !ctx.templates.is_empty() || !ctx.struct_templates.is_empty();
    if has_templates {
        ctx.discover_instantiations(module)?;
        ctx.process_queue()?;
        ctx.apply(module)?;
    }

    // Resolve cross-module turbofish calls (rewrites call sites and collects
    // instantiation info). This must run before the missing-turbofish check
    // because it clears type_args on cross-module calls.
    let cross_module_instantiations =
        collect_cross_module_instantiations(module, &ctx.templates, &ctx.struct_templates)?;

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
    /// Generic struct templates, keyed by struct name.
    struct_templates: BTreeMap<String, ItemStruct>,
    /// Generic impl blocks for generic structs, keyed by struct name.
    /// A single struct can have multiple impl blocks (inherent + trait impls).
    impl_templates: BTreeMap<String, Vec<ItemImpl>>,
    /// Queue of function instantiations to process.
    queue: VecDeque<InstRequest>,
    /// Queue of struct instantiations to process.
    struct_queue: VecDeque<StructInstRequest>,
    /// Set of already-processed or enqueued function keys.
    seen: BTreeSet<InstKey>,
    /// Set of already-processed or enqueued struct keys.
    struct_seen: BTreeSet<InstKey>,
    /// Generated concrete items (functions and structs) to insert into the
    /// module.
    generated: Vec<Item>,
    /// Reserved function names (from non-generic items) for collision
    /// detection.
    reserved_names: BTreeSet<String>,
}

/// A request to instantiate a generic struct with specific types.
struct StructInstRequest {
    key: InstKey,
    /// Span of the usage site, for error diagnostics.
    span: Span,
    /// The actual `Type` values to substitute (parallel to key.type_args).
    concrete_types: Vec<Type>,
}

impl MonoCtx {
    /// Partition module items into templates and concrete items, collecting
    /// reserved names.
    fn new(module: &ItemMod) -> Result<Self, crate::parse::Error> {
        let mut templates = BTreeMap::new();
        let mut struct_templates = BTreeMap::new();
        let mut impl_templates: BTreeMap<String, Vec<ItemImpl>> = BTreeMap::new();
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
                Item::Struct(s) => {
                    if s.type_params.is_empty() {
                        reserved_names.insert(s.ident.to_string());
                    } else {
                        struct_templates.insert(s.ident.to_string(), s.clone());
                    }
                }
                Item::Impl(impl_item) => {
                    if !impl_item.type_params.is_empty() {
                        // Generic impl block — store as a template
                        impl_templates
                            .entry(impl_item.self_ty.to_string())
                            .or_default()
                            .push(impl_item.clone());
                    } else {
                        // Concrete impl block — register reserved names
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
                                                "duplicate impl constant '{mangled}': another \
                                                 impl block already defines a constant with this \
                                                 name"
                                            ),
                                        ));
                                    }
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
            struct_templates,
            impl_templates,
            queue: VecDeque::new(),
            struct_queue: VecDeque::new(),
            seen: BTreeSet::new(),
            struct_seen: BTreeSet::new(),
            generated: Vec::new(),
            reserved_names,
        })
    }

    /// Generate template macros for generic functions and structs defined in
    /// this module.
    ///
    /// For each generic function or struct, produces a `TemplateMacro`
    /// containing WGSL source with `__TP{name}__` placeholders that can be
    /// used for cross-module instantiation via string substitution.
    fn generate_template_macros(
        &self,
        _module: &ItemMod,
    ) -> Result<Vec<TemplateMacro>, crate::parse::Error> {
        let mut macros = Vec::new();

        // --- Generic struct templates ---
        for template in self.struct_templates.values() {
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
                            type_args: vec![],
                        },
                    )
                })
                .collect();

            // Build the mangled struct name with placeholders
            let param_placeholders: Vec<String> = template
                .type_params
                .iter()
                .map(|id| format!("__TP{}__", id))
                .collect();
            let mangled_struct_name = std::iter::once(template.ident.to_string())
                .chain(param_placeholders.into_iter())
                .collect::<Vec<_>>()
                .join("_");

            // Clone and substitute the struct definition
            let mut mono_struct = template.clone();
            mono_struct.type_params.clear();
            mono_struct.ident = Ident::new(&mangled_struct_name, template.ident.span());
            for pair in mono_struct.fields.named.iter_mut() {
                substitute_type(&mut pair.ty, &placeholder_subst);
            }

            // Generate WGSL for the struct definition
            let mut code = code_gen::GeneratedWgslCode::default();
            mono_struct.write_code(&mut code);

            // Also generate WGSL for all associated impl block methods
            let original_name = template.ident.to_string();
            if let Some(impl_blocks) = self.impl_templates.get(&original_name) {
                for impl_block in impl_blocks {
                    for ii in &impl_block.items {
                        match ii {
                            crate::parse::ImplItem::Fn(f) => {
                                let method_name = format!("{}_{}", mangled_struct_name, f.ident);
                                let mut mono_fn = f.clone();
                                mono_fn.ident = Ident::new(&method_name, f.ident.span());
                                for pair in mono_fn.inputs.iter_mut() {
                                    substitute_type(&mut pair.ty, &placeholder_subst);
                                }
                                if let ReturnType::Type { ty, .. } = &mut mono_fn.return_type {
                                    substitute_type(ty, &placeholder_subst);
                                }
                                substitute_block(&mut mono_fn.block, &placeholder_subst);
                                // Rewrite struct type refs within the method body
                                rewrite_struct_type_placeholders(
                                    &mut mono_fn,
                                    &template.ident.to_string(),
                                    &mangled_struct_name,
                                );
                                mono_fn.write_code(&mut code);
                            }
                            crate::parse::ImplItem::Const(c) => {
                                let const_name = format!("{}_{}", mangled_struct_name, c.ident);
                                let mut mono_const = c.clone();
                                mono_const.ident = Ident::new(&const_name, c.ident.span());
                                substitute_type(&mut mono_const.ty, &placeholder_subst);
                                substitute_expr(&mut mono_const.expr, &placeholder_subst);
                                mono_const.write_code(&mut code);
                            }
                        }
                    }
                }
            }

            let template_wgsl = code.source_lines().join("\n");
            let type_param_names: Vec<String> = template
                .type_params
                .iter()
                .map(|id| id.to_string())
                .collect();

            // TODO: compute dependencies for struct templates (nested generic
            // struct or function references). For now, empty.
            let dependencies = vec![];

            macros.push(TemplateMacro {
                fn_name: template.ident.to_string(),
                type_param_names,
                template_wgsl,
                dependencies,
            });
        }

        // --- Generic function templates ---
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
                            type_args: vec![],
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
                collect_template_dependencies(&template.block, &self.templates, &type_param_names)?;

            macros.push(TemplateMacro {
                fn_name: template.ident.to_string(),
                type_param_names,
                template_wgsl,
                dependencies,
            });
        }

        Ok(macros)
    }

    /// Walk all concrete items to find turbofish call sites and generic struct
    /// usages.
    fn discover_instantiations(&mut self, module: &ItemMod) -> Result<(), crate::parse::Error> {
        for item in &module.content {
            match item {
                Item::Fn(f) if f.type_params.is_empty() => {
                    self.collect_from_fn(f)?;
                }
                Item::Impl(impl_item) if impl_item.type_params.is_empty() => {
                    for ii in &impl_item.items {
                        if let crate::parse::ImplItem::Fn(f) = ii {
                            self.collect_from_fn(f)?;
                        }
                    }
                }
                Item::Struct(s) if s.type_params.is_empty() => {
                    // Walk field types in concrete structs to find generic
                    // struct type arguments (e.g., `field: Pair<f32>`).
                    for pair in s.fields.named.iter() {
                        self.collect_struct_insts_from_type(&pair.ty, pair.ident.span())?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn collect_from_fn(&mut self, f: &ItemFn) -> Result<(), crate::parse::Error> {
        // Walk function signature types for generic struct usages
        for pair in f.inputs.iter() {
            self.collect_struct_insts_from_type(&pair.ty, pair.ident.span())?;
        }
        if let ReturnType::Type { ty, .. } = &f.return_type {
            self.collect_struct_insts_from_type(ty, f.ident.span())?;
        }
        self.collect_from_block(&f.block)
    }

    /// Recursively walk a type to find generic struct instantiations
    /// (e.g., `Pair<f32>`) and enqueue them for monomorphization.
    fn collect_struct_insts_from_type(
        &mut self,
        ty: &Type,
        span: Span,
    ) -> Result<(), crate::parse::Error> {
        match ty {
            Type::Struct { ident, type_args } if !type_args.is_empty() => {
                let struct_name = ident.to_string();

                if let Some(template) = self.struct_templates.get(&struct_name) {
                    if type_args.len() != template.type_params.len() {
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "'{}' expects {} type argument(s), but {} were provided",
                                struct_name,
                                template.type_params.len(),
                                type_args.len()
                            ),
                        ));
                    }

                    // Only enqueue if all type args are concrete
                    if type_args.iter().all(|ta| !contains_type_param(ta)) {
                        let key = InstKey {
                            fn_name: struct_name.clone(),
                            type_args: type_args
                                .iter()
                                .map(type_to_key)
                                .collect::<Result<Vec<_>, _>>()?,
                        };
                        if !self.struct_seen.contains(&key) {
                            self.struct_seen.insert(key.clone());
                            self.struct_queue.push_back(StructInstRequest {
                                key,
                                span,
                                concrete_types: type_args.clone(),
                            });
                        }
                    }
                }

                // Also recurse into the type args themselves
                for ta in type_args {
                    self.collect_struct_insts_from_type(ta, span)?;
                }
            }
            Type::Array { elem, .. } => {
                self.collect_struct_insts_from_type(elem, span)?;
            }
            Type::RuntimeArray { elem, .. }
            | Type::Atomic { elem, .. }
            | Type::Ptr { elem, .. } => {
                self.collect_struct_insts_from_type(elem, span)?;
            }
            _ => {}
        }
        Ok(())
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
                // Check the type annotation for generic struct usages
                if let Some((_, ty)) = &local.ty {
                    self.collect_struct_insts_from_type(ty, local.ident.span())?;
                }
                if let Some(init) = &local.init {
                    self.collect_from_expr(&init.expr)?;
                }
            }
            Stmt::Const(c) => {
                self.collect_struct_insts_from_type(&c.ty, c.ident.span())?;
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

                    // Check if this is a generic struct method call
                    // (Pair::<f32>::first). The type_args are struct type args.
                    if let FnPath::TypeMethod { ty, .. } = path {
                        let ty_name = ty.to_string();
                        if let Some(struct_tmpl) = self.struct_templates.get(&ty_name) {
                            if type_args.len() != struct_tmpl.type_params.len() {
                                return Err(crate::parse::Error::unsupported(
                                    span,
                                    format!(
                                        "'{}' expects {} type argument(s), but {} were provided",
                                        ty_name,
                                        struct_tmpl.type_params.len(),
                                        type_args.len()
                                    ),
                                ));
                            }
                            // Enqueue the struct instantiation
                            if type_args.iter().all(|ta| !contains_type_param(ta)) {
                                let key = InstKey {
                                    fn_name: ty_name.clone(),
                                    type_args: type_args.iter().map(type_to_key).collect::<Result<
                                        Vec<_>,
                                        _,
                                    >>(
                                    )?,
                                };
                                if !self.struct_seen.contains(&key) {
                                    self.struct_seen.insert(key.clone());
                                    self.struct_queue.push_back(StructInstRequest {
                                        key,
                                        span,
                                        concrete_types: type_args.clone(),
                                    });
                                }
                            }
                        }
                    }

                    if let Some(template) = self.templates.get(&fn_name) {
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
                    } else if self.reserved_names.contains(&fn_name) {
                        // A known local non-generic function called with turbofish.
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "'{fn_name}' is not a generic function, but was called with type \
                                 arguments"
                            ),
                        ));
                    }
                    // Otherwise, this may be a cross-module generic call.
                    // Defer validation/rewriting to the cross-module pass.
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
            Expr::Cast { lhs, ty } => {
                self.collect_from_expr(lhs)?;
                self.collect_struct_insts_from_type(ty, lhs.span())?;
            }
            Expr::Struct {
                ident,
                type_args,
                fields,
                ..
            } => {
                // If this is a generic struct construction, enqueue it
                if !type_args.is_empty() {
                    let struct_name = ident.to_string();
                    let span = ident.span();

                    if let Some(template) = self.struct_templates.get(&struct_name) {
                        if type_args.len() != template.type_params.len() {
                            return Err(crate::parse::Error::unsupported(
                                span,
                                format!(
                                    "'{}' expects {} type argument(s), but {} were provided",
                                    struct_name,
                                    template.type_params.len(),
                                    type_args.len()
                                ),
                            ));
                        }

                        if type_args.iter().all(|ta| !contains_type_param(ta)) {
                            let key = InstKey {
                                fn_name: struct_name.clone(),
                                type_args: type_args
                                    .iter()
                                    .map(type_to_key)
                                    .collect::<Result<Vec<_>, _>>()?,
                            };
                            if !self.struct_seen.contains(&key) {
                                self.struct_seen.insert(key.clone());
                                self.struct_queue.push_back(StructInstRequest {
                                    key,
                                    span,
                                    concrete_types: type_args.clone(),
                                });
                            }
                        }
                    }
                }
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

    /// Process both function and struct instantiation queues until empty.
    ///
    /// Struct instantiations may produce new function instantiations (from impl
    /// methods) and vice versa (a monomorphized function body may reference new
    /// generic struct types), so we drain both queues in a loop.
    fn process_queue(&mut self) -> Result<(), crate::parse::Error> {
        loop {
            let mut made_progress = false;

            // Process struct queue first (struct defs must come before
            // functions that use them).
            while let Some(request) = self.struct_queue.pop_front() {
                if self.generated.len() + self.struct_queue.len() + self.queue.len()
                    > MAX_INSTANTIATIONS
                {
                    return Err(crate::parse::Error::unsupported(
                        request.span,
                        format!("exceeded maximum of {MAX_INSTANTIATIONS} generic instantiations"),
                    ));
                }
                self.instantiate_struct(request)?;
                made_progress = true;
            }

            // Process function queue
            while let Some(request) = self.queue.pop_front() {
                if self.generated.len() + self.queue.len() + self.struct_queue.len()
                    > MAX_INSTANTIATIONS
                {
                    return Err(crate::parse::Error::unsupported(
                        request.span,
                        format!("exceeded maximum of {MAX_INSTANTIATIONS} generic instantiations"),
                    ));
                }
                self.instantiate(request)?;
                made_progress = true;
            }

            if !made_progress {
                break;
            }
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

    /// Instantiate a generic struct with concrete types, and also instantiate
    /// all associated impl block methods.
    fn instantiate_struct(
        &mut self,
        request: StructInstRequest,
    ) -> Result<(), crate::parse::Error> {
        let template = self.struct_templates[&request.key.fn_name].clone();
        let mangled_name = mangle_name(&request.key.fn_name, &request.concrete_types)?;

        // Check for name collisions
        if self.reserved_names.contains(&mangled_name) {
            return Err(crate::parse::Error::unsupported(
                request.span,
                format!(
                    "generated monomorphized struct name '{mangled_name}' collides with an \
                     existing item"
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

        // Clone and substitute fields
        let mut mono_struct = template.clone();
        mono_struct.type_params.clear();
        mono_struct.ident = Ident::new(&mangled_name, template.ident.span());

        for pair in mono_struct.fields.named.iter_mut() {
            substitute_type(&mut pair.ty, &subst);
        }

        // Discover any new struct instantiations from monomorphized fields
        for pair in mono_struct.fields.named.iter() {
            self.collect_struct_insts_from_type(&pair.ty, pair.ident.span())?;
        }

        self.generated.push(Item::Struct(mono_struct));

        // Also instantiate all associated generic impl blocks
        let original_name = request.key.fn_name.clone();
        if let Some(impl_blocks) = self.impl_templates.get(&original_name).cloned() {
            for impl_template in &impl_blocks {
                // Build substitution for the impl block's type params
                let impl_subst: BTreeMap<String, Type> = impl_template
                    .type_params
                    .iter()
                    .zip(request.concrete_types.iter())
                    .map(|(param, ty)| (param.to_string(), ty.clone()))
                    .collect();

                // Monomorphize each method and constant
                for ii in &impl_template.items {
                    match ii {
                        crate::parse::ImplItem::Fn(f) => {
                            let method_mangled_name = format!("{}_{}", mangled_name, f.ident);

                            if self.reserved_names.contains(&method_mangled_name) {
                                return Err(crate::parse::Error::unsupported(
                                    f.ident.span(),
                                    format!(
                                        "generated monomorphized method name \
                                         '{method_mangled_name}' collides with an existing item"
                                    ),
                                ));
                            }
                            self.reserved_names.insert(method_mangled_name.clone());

                            let mut mono_fn = f.clone();
                            mono_fn.ident = Ident::new(&method_mangled_name, f.ident.span());

                            // Substitute types in inputs
                            for pair in mono_fn.inputs.iter_mut() {
                                substitute_type(&mut pair.ty, &impl_subst);
                            }
                            // Substitute in return type
                            if let ReturnType::Type { ty, .. } = &mut mono_fn.return_type {
                                substitute_type(ty, &impl_subst);
                            }
                            // Substitute in block
                            substitute_block(&mut mono_fn.block, &impl_subst);

                            // Discover any new instantiations from monomorphized body
                            self.collect_from_fn(&mono_fn)?;

                            self.generated.push(Item::Fn(Box::new(mono_fn)));
                        }
                        crate::parse::ImplItem::Const(c) => {
                            let const_mangled_name = format!("{}_{}", mangled_name, c.ident);

                            let mut mono_const = c.clone();
                            mono_const.ident = Ident::new(&const_mangled_name, c.ident.span());
                            substitute_type(&mut mono_const.ty, &impl_subst);
                            substitute_expr(&mut mono_const.expr, &impl_subst);

                            self.generated.push(Item::Const(Box::new(mono_const)));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply the results: remove generic templates, add monomorphized items,
    /// rewrite all types and call sites.
    fn apply(&self, module: &mut ItemMod) -> Result<(), crate::parse::Error> {
        // Remove generic template functions, structs, and impl blocks
        module.content.retain(|item| match item {
            Item::Fn(f) => f.type_params.is_empty(),
            Item::Struct(s) => s.type_params.is_empty(),
            Item::Impl(i) => i.type_params.is_empty(),
            _ => true,
        });

        // Add generated monomorphized items
        module.content.extend(self.generated.clone());

        // Rewrite all call sites and struct types in all items
        for item in &mut module.content {
            rewrite_calls_in_item(item, &self.templates);
            rewrite_struct_types_in_item(item, &self.struct_templates);
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
        Type::Struct { type_args, .. } => {
            // Substitute type params inside generic struct type arguments
            // (e.g., Pair<T> where T is in the substitution map).
            for ta in type_args.iter_mut() {
                substitute_type(ta, subst);
            }
        }
        Type::Scalar { .. }
        | Type::Vector { .. }
        | Type::Matrix { .. }
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
        Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } => {
            // Substitute TypeMethod-style ident if it's a type param
            if let Some(concrete) = subst.get(&ident.to_string()) {
                *ident = type_to_ident(concrete, ident.span());
            }
            for ta in type_args.iter_mut() {
                substitute_type(ta, subst);
            }
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

// ===== Template struct placeholder rewriting =====
//
// When generating WGSL for a struct template's impl methods, references to the
// struct type (e.g., `Pair<__TPT__>`) need to be flattened to the
// placeholder-mangled name (e.g., `Pair___TPT__`). This must happen before code
// generation, which asserts that `type_args` is empty on all struct types.

/// Rewrite references to a generic struct within a method to use the
/// placeholder-mangled struct name. Converts `Type::Struct { ident:
/// orig_name, type_args: [...] }` to `Type::Struct { ident: mangled_name,
/// type_args: [] }`.
fn rewrite_struct_type_placeholders(f: &mut ItemFn, orig_name: &str, mangled_name: &str) {
    for pair in f.inputs.iter_mut() {
        flatten_struct_placeholder(&mut pair.ty, orig_name, mangled_name);
    }
    if let ReturnType::Type { ty, .. } = &mut f.return_type {
        flatten_struct_placeholder(ty, orig_name, mangled_name);
    }
    flatten_struct_placeholder_block(&mut f.block, orig_name, mangled_name);
}

fn flatten_struct_placeholder(ty: &mut Type, orig_name: &str, mangled_name: &str) {
    match ty {
        Type::Struct { ident, type_args } if ident == orig_name && !type_args.is_empty() => {
            *ident = Ident::new(mangled_name, ident.span());
            type_args.clear();
        }
        Type::Struct { type_args, .. } => {
            for ta in type_args.iter_mut() {
                flatten_struct_placeholder(ta, orig_name, mangled_name);
            }
        }
        Type::Array { elem, .. } => flatten_struct_placeholder(elem, orig_name, mangled_name),
        Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } | Type::Ptr { elem, .. } => {
            flatten_struct_placeholder(elem, orig_name, mangled_name);
        }
        _ => {}
    }
}

fn flatten_struct_placeholder_block(block: &mut Block, orig_name: &str, mangled_name: &str) {
    for stmt in &mut block.stmt {
        flatten_struct_placeholder_stmt(stmt, orig_name, mangled_name);
    }
}

fn flatten_struct_placeholder_stmt(stmt: &mut Stmt, orig_name: &str, mangled_name: &str) {
    match stmt {
        Stmt::Local(local) => {
            if let Some((_, ty)) = &mut local.ty {
                flatten_struct_placeholder(ty, orig_name, mangled_name);
            }
            if let Some(init) = &mut local.init {
                flatten_struct_placeholder_expr(&mut init.expr, orig_name, mangled_name);
            }
        }
        Stmt::Const(c) => {
            flatten_struct_placeholder(&mut c.ty, orig_name, mangled_name);
            flatten_struct_placeholder_expr(&mut c.expr, orig_name, mangled_name);
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            flatten_struct_placeholder_expr(lhs, orig_name, mangled_name);
            flatten_struct_placeholder_expr(rhs, orig_name, mangled_name);
        }
        Stmt::While {
            condition, body, ..
        } => {
            flatten_struct_placeholder_expr(condition, orig_name, mangled_name);
            flatten_struct_placeholder_block(body, orig_name, mangled_name);
        }
        Stmt::Loop { body, .. } => flatten_struct_placeholder_block(body, orig_name, mangled_name),
        Stmt::Expr { expr, .. } => {
            flatten_struct_placeholder_expr(expr, orig_name, mangled_name);
        }
        Stmt::If(if_stmt) => {
            flatten_struct_placeholder_expr(&mut if_stmt.condition, orig_name, mangled_name);
            flatten_struct_placeholder_block(&mut if_stmt.then_block, orig_name, mangled_name);
            if let Some(else_branch) = &mut if_stmt.else_branch {
                match &mut else_branch.body {
                    ElseBody::Block(block) => {
                        flatten_struct_placeholder_block(block, orig_name, mangled_name)
                    }
                    ElseBody::If(nested_if) => {
                        flatten_struct_placeholder_expr(
                            &mut nested_if.condition,
                            orig_name,
                            mangled_name,
                        );
                        flatten_struct_placeholder_block(
                            &mut nested_if.then_block,
                            orig_name,
                            mangled_name,
                        );
                    }
                }
            }
        }
        Stmt::For(for_loop) => {
            if let Some((_, ty)) = &mut for_loop.ty {
                flatten_struct_placeholder(ty, orig_name, mangled_name);
            }
            flatten_struct_placeholder_expr(&mut for_loop.from, orig_name, mangled_name);
            flatten_struct_placeholder_expr(&mut for_loop.to, orig_name, mangled_name);
            flatten_struct_placeholder_block(&mut for_loop.body, orig_name, mangled_name);
        }
        Stmt::Switch(switch) => {
            flatten_struct_placeholder_expr(&mut switch.selector, orig_name, mangled_name);
            for arm in &mut switch.arms {
                flatten_struct_placeholder_block(&mut arm.body, orig_name, mangled_name);
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                flatten_struct_placeholder_expr(e, orig_name, mangled_name);
            }
        }
        Stmt::Block(block) => flatten_struct_placeholder_block(block, orig_name, mangled_name),
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            flatten_struct_placeholder_expr(slab, orig_name, mangled_name);
            flatten_struct_placeholder_expr(offset, orig_name, mangled_name);
            flatten_struct_placeholder_expr(dest, orig_name, mangled_name);
            flatten_struct_placeholder_expr(size, orig_name, mangled_name);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            flatten_struct_placeholder_expr(slab, orig_name, mangled_name);
            flatten_struct_placeholder_expr(offset, orig_name, mangled_name);
            flatten_struct_placeholder_expr(src, orig_name, mangled_name);
            if let Some(s) = size {
                flatten_struct_placeholder_expr(s, orig_name, mangled_name);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
}

fn flatten_struct_placeholder_expr(expr: &mut Expr, orig_name: &str, mangled_name: &str) {
    match expr {
        Expr::FnCall {
            type_args, params, ..
        } => {
            for ta in type_args.iter_mut() {
                flatten_struct_placeholder(ta, orig_name, mangled_name);
            }
            for param in params.iter_mut() {
                flatten_struct_placeholder_expr(param, orig_name, mangled_name);
            }
        }
        Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } => {
            if ident == orig_name && !type_args.is_empty() {
                *ident = Ident::new(mangled_name, ident.span());
                type_args.clear();
            }
            for field in fields.iter_mut() {
                flatten_struct_placeholder_expr(&mut field.expr, orig_name, mangled_name);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            flatten_struct_placeholder_expr(lhs, orig_name, mangled_name);
            flatten_struct_placeholder_expr(rhs, orig_name, mangled_name);
        }
        Expr::Unary { expr, .. } => flatten_struct_placeholder_expr(expr, orig_name, mangled_name),
        Expr::Paren { inner, .. } => {
            flatten_struct_placeholder_expr(inner, orig_name, mangled_name)
        }
        Expr::Array { elems, .. } => {
            for elem in elems.iter_mut() {
                flatten_struct_placeholder_expr(elem, orig_name, mangled_name);
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            flatten_struct_placeholder_expr(lhs, orig_name, mangled_name);
            flatten_struct_placeholder_expr(index, orig_name, mangled_name);
        }
        Expr::Swizzle { lhs, params, .. } => {
            flatten_struct_placeholder_expr(lhs, orig_name, mangled_name);
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    flatten_struct_placeholder_expr(p, orig_name, mangled_name);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            flatten_struct_placeholder_expr(lhs, orig_name, mangled_name);
            flatten_struct_placeholder(ty, orig_name, mangled_name);
        }
        Expr::FieldAccess { base, .. } => {
            flatten_struct_placeholder_expr(base, orig_name, mangled_name)
        }
        Expr::Reference { expr, .. } => {
            flatten_struct_placeholder_expr(expr, orig_name, mangled_name)
        }
        Expr::ZeroValueArray { elem_type, len, .. } => {
            flatten_struct_placeholder(elem_type, orig_name, mangled_name);
            flatten_struct_placeholder_expr(len, orig_name, mangled_name);
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
}

// ===== Struct type rewriting =====

/// Rewrite generic struct types in an item to use mangled names.
///
/// After monomorphization, `Type::Struct { ident: "Pair", type_args: [f32] }`
/// becomes `Type::Struct { ident: "Pair_f32", type_args: [] }`.
fn rewrite_struct_types_in_item(item: &mut Item, struct_templates: &BTreeMap<String, ItemStruct>) {
    match item {
        Item::Fn(f) => rewrite_struct_types_in_fn(f, struct_templates),
        Item::Impl(impl_item) => {
            for ii in &mut impl_item.items {
                match ii {
                    crate::parse::ImplItem::Fn(f) => {
                        rewrite_struct_types_in_fn(f, struct_templates);
                    }
                    crate::parse::ImplItem::Const(c) => {
                        rewrite_struct_type(&mut c.ty, struct_templates);
                        rewrite_struct_types_in_expr(&mut c.expr, struct_templates);
                    }
                }
            }
        }
        Item::Struct(s) => {
            for pair in s.fields.named.iter_mut() {
                rewrite_struct_type(&mut pair.ty, struct_templates);
            }
        }
        Item::Const(c) => {
            rewrite_struct_type(&mut c.ty, struct_templates);
            rewrite_struct_types_in_expr(&mut c.expr, struct_templates);
        }
        _ => {}
    }
}

fn rewrite_struct_types_in_fn(f: &mut ItemFn, struct_templates: &BTreeMap<String, ItemStruct>) {
    for pair in f.inputs.iter_mut() {
        rewrite_struct_type(&mut pair.ty, struct_templates);
    }
    if let ReturnType::Type { ty, .. } = &mut f.return_type {
        rewrite_struct_type(ty, struct_templates);
    }
    rewrite_struct_types_in_block(&mut f.block, struct_templates);
}

fn rewrite_struct_type(ty: &mut Type, struct_templates: &BTreeMap<String, ItemStruct>) {
    match ty {
        Type::Struct { ident, type_args } if !type_args.is_empty() => {
            if struct_templates.contains_key(&ident.to_string()) {
                // Mangle the name and clear type_args
                let mangled = mangle_name(&ident.to_string(), type_args)
                    .expect("mangle_name should not fail for concrete types");
                *ident = Ident::new(&mangled, ident.span());
                type_args.clear();
            }
        }
        Type::Array { elem, .. } => {
            rewrite_struct_type(elem, struct_templates);
        }
        Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } | Type::Ptr { elem, .. } => {
            rewrite_struct_type(elem, struct_templates);
        }
        Type::Struct { type_args, .. } => {
            // Non-generic struct or struct with type_args already cleared
            for ta in type_args.iter_mut() {
                rewrite_struct_type(ta, struct_templates);
            }
        }
        _ => {}
    }
}

fn rewrite_struct_types_in_block(
    block: &mut Block,
    struct_templates: &BTreeMap<String, ItemStruct>,
) {
    for stmt in &mut block.stmt {
        rewrite_struct_types_in_stmt(stmt, struct_templates);
    }
}

fn rewrite_struct_types_in_stmt(stmt: &mut Stmt, struct_templates: &BTreeMap<String, ItemStruct>) {
    match stmt {
        Stmt::Local(local) => {
            if let Some((_, ty)) = &mut local.ty {
                rewrite_struct_type(ty, struct_templates);
            }
            if let Some(init) = &mut local.init {
                rewrite_struct_types_in_expr(&mut init.expr, struct_templates);
            }
        }
        Stmt::Const(c) => {
            rewrite_struct_type(&mut c.ty, struct_templates);
            rewrite_struct_types_in_expr(&mut c.expr, struct_templates);
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            rewrite_struct_types_in_expr(lhs, struct_templates);
            rewrite_struct_types_in_expr(rhs, struct_templates);
        }
        Stmt::While {
            condition, body, ..
        } => {
            rewrite_struct_types_in_expr(condition, struct_templates);
            rewrite_struct_types_in_block(body, struct_templates);
        }
        Stmt::Loop { body, .. } => {
            rewrite_struct_types_in_block(body, struct_templates);
        }
        Stmt::Expr { expr, .. } => {
            rewrite_struct_types_in_expr(expr, struct_templates);
        }
        Stmt::If(if_stmt) => {
            rewrite_struct_types_in_if(if_stmt, struct_templates);
        }
        Stmt::For(for_loop) => {
            if let Some((_, ty)) = &mut for_loop.ty {
                rewrite_struct_type(ty, struct_templates);
            }
            rewrite_struct_types_in_expr(&mut for_loop.from, struct_templates);
            rewrite_struct_types_in_expr(&mut for_loop.to, struct_templates);
            rewrite_struct_types_in_block(&mut for_loop.body, struct_templates);
        }
        Stmt::Switch(switch) => {
            rewrite_struct_types_in_expr(&mut switch.selector, struct_templates);
            for arm in &mut switch.arms {
                rewrite_struct_types_in_block(&mut arm.body, struct_templates);
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                rewrite_struct_types_in_expr(e, struct_templates);
            }
        }
        Stmt::Block(block) => {
            rewrite_struct_types_in_block(block, struct_templates);
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            rewrite_struct_types_in_expr(slab, struct_templates);
            rewrite_struct_types_in_expr(offset, struct_templates);
            rewrite_struct_types_in_expr(dest, struct_templates);
            rewrite_struct_types_in_expr(size, struct_templates);
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            rewrite_struct_types_in_expr(slab, struct_templates);
            rewrite_struct_types_in_expr(offset, struct_templates);
            rewrite_struct_types_in_expr(src, struct_templates);
            if let Some(s) = size {
                rewrite_struct_types_in_expr(s, struct_templates);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
}

fn rewrite_struct_types_in_if(
    if_stmt: &mut crate::parse::StmtIf,
    struct_templates: &BTreeMap<String, ItemStruct>,
) {
    rewrite_struct_types_in_expr(&mut if_stmt.condition, struct_templates);
    rewrite_struct_types_in_block(&mut if_stmt.then_block, struct_templates);
    if let Some(else_branch) = &mut if_stmt.else_branch {
        match &mut else_branch.body {
            ElseBody::Block(block) => rewrite_struct_types_in_block(block, struct_templates),
            ElseBody::If(nested_if) => rewrite_struct_types_in_if(nested_if, struct_templates),
        }
    }
}

fn rewrite_struct_types_in_expr(expr: &mut Expr, struct_templates: &BTreeMap<String, ItemStruct>) {
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            // Rewrite TypeMethod paths for generic struct methods.
            // For `Pair::<f32>::first(p)`, the type_args hold [f32] and the
            // path is TypeMethod { ty: "Pair", method: "first" }.
            // We mangle this to a simple Ident: "Pair_f32_first".
            if let FnPath::TypeMethod { ty, method, .. } = path {
                let ty_name = ty.to_string();
                if struct_templates.contains_key(&ty_name) && !type_args.is_empty() {
                    let mangled_struct = mangle_name(&ty_name, type_args)
                        .expect("mangle_name should not fail for concrete types");
                    let mangled_fn = format!("{}_{}", mangled_struct, method);
                    let span = ty.span();
                    *path = FnPath::Ident(Ident::new(&mangled_fn, span));
                    type_args.clear();
                }
            }
            for ta in type_args.iter_mut() {
                rewrite_struct_type(ta, struct_templates);
            }
            for param in params.iter_mut() {
                rewrite_struct_types_in_expr(param, struct_templates);
            }
        }
        Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } => {
            if !type_args.is_empty() && struct_templates.contains_key(&ident.to_string()) {
                let mangled = mangle_name(&ident.to_string(), type_args)
                    .expect("mangle_name should not fail for concrete types");
                *ident = Ident::new(&mangled, ident.span());
                type_args.clear();
            }
            for field in fields.iter_mut() {
                rewrite_struct_types_in_expr(&mut field.expr, struct_templates);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            rewrite_struct_types_in_expr(lhs, struct_templates);
            rewrite_struct_types_in_expr(rhs, struct_templates);
        }
        Expr::Unary { expr, .. } => {
            rewrite_struct_types_in_expr(expr, struct_templates);
        }
        Expr::Paren { inner, .. } => {
            rewrite_struct_types_in_expr(inner, struct_templates);
        }
        Expr::Array { elems, .. } => {
            for elem in elems.iter_mut() {
                rewrite_struct_types_in_expr(elem, struct_templates);
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            rewrite_struct_types_in_expr(lhs, struct_templates);
            rewrite_struct_types_in_expr(index, struct_templates);
        }
        Expr::Swizzle { lhs, params, .. } => {
            rewrite_struct_types_in_expr(lhs, struct_templates);
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    rewrite_struct_types_in_expr(p, struct_templates);
                }
            }
        }
        Expr::Cast { lhs, ty } => {
            rewrite_struct_types_in_expr(lhs, struct_templates);
            rewrite_struct_type(ty, struct_templates);
        }
        Expr::FieldAccess { base, .. } => {
            rewrite_struct_types_in_expr(base, struct_templates);
        }
        Expr::Reference { expr, .. } => {
            rewrite_struct_types_in_expr(expr, struct_templates);
        }
        Expr::ZeroValueArray { elem_type, len, .. } => {
            rewrite_struct_type(elem_type, struct_templates);
            rewrite_struct_types_in_expr(len, struct_templates);
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
    local_struct_templates: &BTreeMap<String, ItemStruct>,
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
            local_struct_templates,
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
    local_struct_templates: &BTreeMap<String, ItemStruct>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    match item {
        Item::Fn(f) => {
            // Scan function signature types for cross-module struct usages
            for pair in f.inputs.iter_mut() {
                collect_cross_module_from_type(
                    &mut pair.ty,
                    local_struct_templates,
                    import_paths,
                    out,
                    seen,
                )?;
            }
            if let ReturnType::Type { ty, .. } = &mut f.return_type {
                collect_cross_module_from_type(
                    ty,
                    local_struct_templates,
                    import_paths,
                    out,
                    seen,
                )?;
            }
            collect_cross_module_from_block(
                &mut f.block,
                local_templates,
                local_struct_templates,
                import_paths,
                out,
                seen,
            )
        }
        Item::Impl(impl_item) => {
            for ii in &mut impl_item.items {
                if let crate::parse::ImplItem::Fn(f) = ii {
                    for pair in f.inputs.iter_mut() {
                        collect_cross_module_from_type(
                            &mut pair.ty,
                            local_struct_templates,
                            import_paths,
                            out,
                            seen,
                        )?;
                    }
                    if let ReturnType::Type { ty, .. } = &mut f.return_type {
                        collect_cross_module_from_type(
                            ty,
                            local_struct_templates,
                            import_paths,
                            out,
                            seen,
                        )?;
                    }
                    collect_cross_module_from_block(
                        &mut f.block,
                        local_templates,
                        local_struct_templates,
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

/// Scan a type for cross-module generic struct usages and rewrite them.
fn collect_cross_module_from_type(
    ty: &mut Type,
    local_struct_templates: &BTreeMap<String, ItemStruct>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    match ty {
        Type::Struct { ident, type_args } if !type_args.is_empty() => {
            let struct_name = ident.to_string();
            // Only handle non-local struct templates (local ones are already
            // monomorphized by the same-module pass).
            if !local_struct_templates.contains_key(&struct_name) {
                let mangled_type_args: Vec<String> = type_args
                    .iter()
                    .map(mangle_type)
                    .collect::<Result<_, _>>()?;
                let wgsl_type_args: Vec<String> = type_args
                    .iter()
                    .map(type_to_wgsl)
                    .collect::<Result<_, _>>()?;
                let mangled_name = mangle_name(&struct_name, type_args)?;

                // Rewrite the type to use the mangled name
                *ident = Ident::new(&mangled_name, ident.span());
                let cleared_args = std::mem::take(type_args);

                if seen.insert(mangled_name) {
                    out.push(CrossModuleInstantiation {
                        import_paths: import_paths.to_vec(),
                        fn_name: struct_name,
                        mangled_type_args,
                        wgsl_type_args,
                    });
                }

                // Recurse into the (now-cleared) original type args
                let _ = cleared_args;
            } else {
                // Local struct — recurse into type_args
                for ta in type_args.iter_mut() {
                    collect_cross_module_from_type(
                        ta,
                        local_struct_templates,
                        import_paths,
                        out,
                        seen,
                    )?;
                }
            }
        }
        Type::Array { elem, .. } => {
            collect_cross_module_from_type(elem, local_struct_templates, import_paths, out, seen)?;
        }
        Type::RuntimeArray { elem, .. } | Type::Atomic { elem, .. } | Type::Ptr { elem, .. } => {
            collect_cross_module_from_type(elem, local_struct_templates, import_paths, out, seen)?;
        }
        _ => {}
    }
    Ok(())
}

fn collect_cross_module_from_block(
    block: &mut Block,
    local_templates: &BTreeMap<String, ItemFn>,
    local_struct_templates: &BTreeMap<String, ItemStruct>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    for stmt in &mut block.stmt {
        collect_cross_module_from_stmt(
            stmt,
            local_templates,
            local_struct_templates,
            import_paths,
            out,
            seen,
        )?;
    }
    Ok(())
}

fn collect_cross_module_from_stmt(
    stmt: &mut Stmt,
    local_templates: &BTreeMap<String, ItemFn>,
    local_struct_templates: &BTreeMap<String, ItemStruct>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    let lt = local_templates;
    let lst = local_struct_templates;
    let ip = import_paths;
    match stmt {
        Stmt::Local(local) => {
            if let Some((_, ty)) = &mut local.ty {
                collect_cross_module_from_type(ty, lst, ip, out, seen)?;
            }
            if let Some(init) = &mut local.init {
                collect_cross_module_from_expr(&mut init.expr, lt, lst, ip, out, seen)?;
            }
        }
        Stmt::Const(c) => {
            collect_cross_module_from_type(&mut c.ty, lst, ip, out, seen)?;
            collect_cross_module_from_expr(&mut c.expr, lt, lst, ip, out, seen)?;
        }
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            collect_cross_module_from_expr(lhs, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(rhs, lt, lst, ip, out, seen)?;
        }
        Stmt::While {
            condition, body, ..
        } => {
            collect_cross_module_from_expr(condition, lt, lst, ip, out, seen)?;
            collect_cross_module_from_block(body, lt, lst, ip, out, seen)?;
        }
        Stmt::Loop { body, .. } => {
            collect_cross_module_from_block(body, lt, lst, ip, out, seen)?;
        }
        Stmt::Expr { expr, .. } => {
            collect_cross_module_from_expr(expr, lt, lst, ip, out, seen)?;
        }
        Stmt::If(s) => {
            collect_cross_module_from_expr(&mut s.condition, lt, lst, ip, out, seen)?;
            collect_cross_module_from_block(&mut s.then_block, lt, lst, ip, out, seen)?;
            if let Some(eb) = &mut s.else_branch {
                match &mut eb.body {
                    ElseBody::Block(b) => {
                        collect_cross_module_from_block(b, lt, lst, ip, out, seen)?
                    }
                    ElseBody::If(nested) => {
                        collect_cross_module_from_expr(
                            &mut nested.condition,
                            lt,
                            lst,
                            ip,
                            out,
                            seen,
                        )?;
                        collect_cross_module_from_block(
                            &mut nested.then_block,
                            lt,
                            lst,
                            ip,
                            out,
                            seen,
                        )?;
                    }
                }
            }
        }
        Stmt::For(f) => {
            if let Some((_, ty)) = &mut f.ty {
                collect_cross_module_from_type(ty, lst, ip, out, seen)?;
            }
            collect_cross_module_from_expr(&mut f.from, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(&mut f.to, lt, lst, ip, out, seen)?;
            collect_cross_module_from_block(&mut f.body, lt, lst, ip, out, seen)?;
        }
        Stmt::Switch(sw) => {
            collect_cross_module_from_expr(&mut sw.selector, lt, lst, ip, out, seen)?;
            for arm in &mut sw.arms {
                collect_cross_module_from_block(&mut arm.body, lt, lst, ip, out, seen)?;
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                collect_cross_module_from_expr(e, lt, lst, ip, out, seen)?;
            }
        }
        Stmt::Block(b) => {
            collect_cross_module_from_block(b, lt, lst, ip, out, seen)?;
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            collect_cross_module_from_expr(slab, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(offset, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(dest, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(size, lt, lst, ip, out, seen)?;
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            collect_cross_module_from_expr(slab, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(offset, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(src, lt, lst, ip, out, seen)?;
            if let Some(s) = size {
                collect_cross_module_from_expr(s, lt, lst, ip, out, seen)?;
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
    Ok(())
}

fn collect_cross_module_from_expr(
    expr: &mut Expr,
    local_templates: &BTreeMap<String, ItemFn>,
    local_struct_templates: &BTreeMap<String, ItemStruct>,
    import_paths: &[syn::Path],
    out: &mut Vec<CrossModuleInstantiation>,
    seen: &mut BTreeSet<String>,
) -> Result<(), crate::parse::Error> {
    let lt = local_templates;
    let lst = local_struct_templates;
    let ip = import_paths;
    match expr {
        Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } => {
            if !type_args.is_empty() {
                // Check if this is a TypeMethod call to a generic struct
                // from an imported module (e.g., Pair::<f32>::first(p)).
                if let FnPath::TypeMethod { ty, method, .. } = &*path {
                    let ty_name = ty.to_string();
                    // If it's not a local struct template, it must be
                    // cross-module.
                    if !lst.contains_key(&ty_name)
                        && !lt.contains_key(&format!("{ty_name}_{method}"))
                    {
                        let mangled_type_args: Vec<String> = type_args
                            .iter()
                            .map(mangle_type)
                            .collect::<Result<_, _>>()?;
                        let wgsl_type_args: Vec<String> = type_args
                            .iter()
                            .map(type_to_wgsl)
                            .collect::<Result<_, _>>()?;
                        let mangled_struct = mangle_name(&ty_name, type_args)?;
                        let mangled_fn = format!("{}_{}", mangled_struct, method);
                        let span = ty.span();
                        *path = FnPath::Ident(Ident::new(&mangled_fn, span));
                        type_args.clear();

                        // Record the struct template instantiation
                        if seen.insert(mangled_struct) {
                            out.push(CrossModuleInstantiation {
                                import_paths: ip.to_vec(),
                                fn_name: ty_name,
                                mangled_type_args,
                                wgsl_type_args,
                            });
                        }
                    }
                }

                // Handle regular function templates
                if !type_args.is_empty() {
                    let fn_name = match &*path {
                        FnPath::Ident(id) => id.to_string(),
                        FnPath::TypeMethod { ty, method, .. } => {
                            format!("{}_{}", ty, method)
                        }
                    };

                    // Skip if this is a local template (already handled)
                    if !lt.contains_key(&fn_name) {
                        if ip.is_empty() {
                            let span = match &*path {
                                FnPath::Ident(id) => id.span(),
                                FnPath::TypeMethod { ty, .. } => ty.span(),
                            };
                            return Err(crate::parse::Error::unsupported(
                                span,
                                format!(
                                    "generic function '{fn_name}' is not defined in this module \
                                     and no imports are available. Define it in this module or \
                                     import the module that defines it."
                                ),
                            ));
                        }

                        let mangled_type_args: Vec<String> = type_args
                            .iter()
                            .map(mangle_type)
                            .collect::<Result<Vec<_>, _>>()?;
                        let wgsl_type_args: Vec<String> = type_args
                            .iter()
                            .map(type_to_wgsl)
                            .collect::<Result<Vec<_>, _>>()?;
                        let mangled_name = mangle_name(&fn_name, type_args)?;

                        let span = match &*path {
                            FnPath::Ident(id) => id.span(),
                            FnPath::TypeMethod { ty, .. } => ty.span(),
                        };
                        *path = FnPath::Ident(Ident::new(&mangled_name, span));
                        type_args.clear();

                        if seen.insert(mangled_name) {
                            out.push(CrossModuleInstantiation {
                                import_paths: ip.to_vec(),
                                fn_name,
                                mangled_type_args,
                                wgsl_type_args,
                            });
                        }
                    }
                }
            }

            for param in params.iter_mut() {
                collect_cross_module_from_expr(param, lt, lst, ip, out, seen)?;
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_cross_module_from_expr(lhs, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(rhs, lt, lst, ip, out, seen)?;
        }
        Expr::Unary { expr, .. } | Expr::Paren { inner: expr, .. } => {
            collect_cross_module_from_expr(expr, lt, lst, ip, out, seen)?;
        }
        Expr::Array { elems, .. } => {
            for e in elems.iter_mut() {
                collect_cross_module_from_expr(e, lt, lst, ip, out, seen)?;
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            collect_cross_module_from_expr(lhs, lt, lst, ip, out, seen)?;
            collect_cross_module_from_expr(index, lt, lst, ip, out, seen)?;
        }
        Expr::Swizzle { lhs, params, .. } => {
            collect_cross_module_from_expr(lhs, lt, lst, ip, out, seen)?;
            if let Some(ps) = params {
                for p in ps.iter_mut() {
                    collect_cross_module_from_expr(p, lt, lst, ip, out, seen)?;
                }
            }
        }
        Expr::Cast { lhs, ty, .. } => {
            collect_cross_module_from_expr(lhs, lt, lst, ip, out, seen)?;
            collect_cross_module_from_type(ty, lst, ip, out, seen)?;
        }
        Expr::FieldAccess { base, .. } => {
            collect_cross_module_from_expr(base, lt, lst, ip, out, seen)?;
        }
        Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } => {
            // Handle cross-module generic struct construction
            if !type_args.is_empty() {
                let struct_name = ident.to_string();
                if !lst.contains_key(&struct_name) {
                    let mangled_type_args: Vec<String> = type_args
                        .iter()
                        .map(mangle_type)
                        .collect::<Result<_, _>>()?;
                    let wgsl_type_args: Vec<String> = type_args
                        .iter()
                        .map(type_to_wgsl)
                        .collect::<Result<_, _>>()?;
                    let mangled_name = mangle_name(&struct_name, type_args)?;

                    *ident = Ident::new(&mangled_name, ident.span());
                    type_args.clear();

                    if seen.insert(mangled_name) {
                        out.push(CrossModuleInstantiation {
                            import_paths: ip.to_vec(),
                            fn_name: struct_name,
                            mangled_type_args,
                            wgsl_type_args,
                        });
                    }
                }
            }
            for f in fields.iter_mut() {
                collect_cross_module_from_expr(&mut f.expr, lt, lst, ip, out, seen)?;
            }
        }
        Expr::Reference { expr, .. } => {
            collect_cross_module_from_expr(expr, lt, lst, ip, out, seen)?;
        }
        Expr::ZeroValueArray { elem_type, len, .. } => {
            collect_cross_module_from_type(elem_type, lst, ip, out, seen)?;
            collect_cross_module_from_expr(len, lt, lst, ip, out, seen)?;
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
) -> Result<Vec<TemplateDep>, crate::parse::Error> {
    let mut deps = Vec::new();
    let mut seen: BTreeSet<(String, Vec<usize>)> = BTreeSet::new();
    scan_block_for_deps(block, templates, caller_type_params, &mut deps, &mut seen)?;
    Ok(deps)
}

fn scan_block_for_deps(
    block: &Block,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<(String, Vec<usize>)>,
) -> Result<(), crate::parse::Error> {
    for stmt in &block.stmt {
        scan_stmt_for_deps(stmt, templates, caller_params, out, seen)?;
    }
    Ok(())
}

fn scan_stmt_for_deps(
    stmt: &Stmt,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<(String, Vec<usize>)>,
) -> Result<(), crate::parse::Error> {
    match stmt {
        Stmt::Local(l) => {
            if let Some(init) = &l.init {
                scan_expr_for_deps(&init.expr, templates, caller_params, out, seen)?;
            }
        }
        Stmt::Const(c) => scan_expr_for_deps(&c.expr, templates, caller_params, out, seen)?,
        Stmt::Assignment { lhs, rhs, .. } | Stmt::CompoundAssignment { lhs, rhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen)?;
            scan_expr_for_deps(rhs, templates, caller_params, out, seen)?;
        }
        Stmt::While {
            condition, body, ..
        } => {
            scan_expr_for_deps(condition, templates, caller_params, out, seen)?;
            scan_block_for_deps(body, templates, caller_params, out, seen)?;
        }
        Stmt::Loop { body, .. } => scan_block_for_deps(body, templates, caller_params, out, seen)?,
        Stmt::Expr { expr, .. } => scan_expr_for_deps(expr, templates, caller_params, out, seen)?,
        Stmt::If(s) => {
            scan_expr_for_deps(&s.condition, templates, caller_params, out, seen)?;
            scan_block_for_deps(&s.then_block, templates, caller_params, out, seen)?;
            if let Some(eb) = &s.else_branch {
                match &eb.body {
                    ElseBody::Block(b) => {
                        scan_block_for_deps(b, templates, caller_params, out, seen)?
                    }
                    ElseBody::If(nested) => {
                        scan_expr_for_deps(&nested.condition, templates, caller_params, out, seen)?;
                        scan_block_for_deps(
                            &nested.then_block,
                            templates,
                            caller_params,
                            out,
                            seen,
                        )?;
                    }
                }
            }
        }
        Stmt::For(f) => {
            scan_expr_for_deps(&f.from, templates, caller_params, out, seen)?;
            scan_expr_for_deps(&f.to, templates, caller_params, out, seen)?;
            scan_block_for_deps(&f.body, templates, caller_params, out, seen)?;
        }
        Stmt::Switch(sw) => {
            scan_expr_for_deps(&sw.selector, templates, caller_params, out, seen)?;
            for arm in &sw.arms {
                scan_block_for_deps(&arm.body, templates, caller_params, out, seen)?;
            }
        }
        Stmt::Return { expr, .. } => {
            if let Some(e) = expr {
                scan_expr_for_deps(e, templates, caller_params, out, seen)?;
            }
        }
        Stmt::Block(b) => scan_block_for_deps(b, templates, caller_params, out, seen)?,
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
            ..
        } => {
            scan_expr_for_deps(slab, templates, caller_params, out, seen)?;
            scan_expr_for_deps(offset, templates, caller_params, out, seen)?;
            scan_expr_for_deps(dest, templates, caller_params, out, seen)?;
            scan_expr_for_deps(size, templates, caller_params, out, seen)?;
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
            ..
        } => {
            scan_expr_for_deps(slab, templates, caller_params, out, seen)?;
            scan_expr_for_deps(offset, templates, caller_params, out, seen)?;
            scan_expr_for_deps(src, templates, caller_params, out, seen)?;
            if let Some(s) = size {
                scan_expr_for_deps(s, templates, caller_params, out, seen)?;
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
    }
    Ok(())
}

fn scan_expr_for_deps(
    expr: &Expr,
    templates: &BTreeMap<String, ItemFn>,
    caller_params: &[String],
    out: &mut Vec<TemplateDep>,
    seen: &mut BTreeSet<(String, Vec<usize>)>,
) -> Result<(), crate::parse::Error> {
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

                if !fn_name.is_empty() && templates.contains_key(&fn_name) {
                    let span = match path {
                        FnPath::Ident(id) => id.span(),
                        FnPath::TypeMethod { ty, .. } => ty.span(),
                    };

                    // Build the type param mapping: for each type_arg, find
                    // which caller param it refers to.
                    let mut mapping = Vec::with_capacity(type_args.len());
                    for ta in type_args {
                        let Type::TypeParam { ident } = ta else {
                            return Err(crate::parse::Error::unsupported(
                                span,
                                format!(
                                    "template dependency '{fn_name}' must use caller type \
                                     parameters directly; concrete dependency type arguments are \
                                     not supported yet"
                                ),
                            ));
                        };

                        let Some(idx) = caller_params.iter().position(|p| p == &ident.to_string())
                        else {
                            return Err(crate::parse::Error::unsupported(
                                span,
                                format!(
                                    "template dependency '{fn_name}' uses unknown type parameter \
                                     '{ident}'"
                                ),
                            ));
                        };
                        mapping.push(idx);
                    }
                    if seen.insert((fn_name.clone(), mapping.clone())) {
                        out.push(TemplateDep {
                            callee: fn_name,
                            type_param_mapping: mapping,
                        });
                    }
                }
            }
            for param in params.iter() {
                scan_expr_for_deps(param, templates, caller_params, out, seen)?;
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen)?;
            scan_expr_for_deps(rhs, templates, caller_params, out, seen)?;
        }
        Expr::Unary { expr, .. } | Expr::Paren { inner: expr, .. } => {
            scan_expr_for_deps(expr, templates, caller_params, out, seen)?;
        }
        Expr::Array { elems, .. } => {
            for e in elems.iter() {
                scan_expr_for_deps(e, templates, caller_params, out, seen)?;
            }
        }
        Expr::ArrayIndexing { lhs, index, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen)?;
            scan_expr_for_deps(index, templates, caller_params, out, seen)?;
        }
        Expr::Swizzle { lhs, params, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen)?;
            if let Some(ps) = params {
                for p in ps.iter() {
                    scan_expr_for_deps(p, templates, caller_params, out, seen)?;
                }
            }
        }
        Expr::Cast { lhs, .. } | Expr::FieldAccess { base: lhs, .. } => {
            scan_expr_for_deps(lhs, templates, caller_params, out, seen)?;
        }
        Expr::Struct { fields, .. } => {
            for f in fields.iter() {
                scan_expr_for_deps(&f.expr, templates, caller_params, out, seen)?;
            }
        }
        Expr::Reference { expr, .. } => {
            scan_expr_for_deps(expr, templates, caller_params, out, seen)?;
        }
        Expr::ZeroValueArray { len, .. } => {
            scan_expr_for_deps(len, templates, caller_params, out, seen)?;
        }
        Expr::Lit(_) | Expr::Ident(_) | Expr::TypePath { .. } => {}
    }
    Ok(())
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
        Type::Struct { ident, .. } => return ident.clone(),
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
        Type::Struct { ident, type_args } => {
            if type_args.is_empty() {
                ident.to_string()
            } else {
                let mangled_args: Vec<String> = type_args
                    .iter()
                    .map(mangle_type)
                    .collect::<Result<_, _>>()?;
                format!("{}_{}", ident, mangled_args.join("_"))
            }
        }
        Type::Array { elem, len, .. } => {
            format!("array_{}_{}", mangle_type(elem)?, len_to_string(len))
        }
        Type::RuntimeArray { elem, .. } => format!("array_{}", mangle_type(elem)?),
        Type::Atomic { elem, .. } => format!("atomic_{}", mangle_type(elem)?),
        Type::Sampler { .. } => "sampler".to_string(),
        Type::SamplerComparison { .. } => "sampler_comparison".to_string(),
        Type::Texture { kind, .. } => kind.wgsl_name().replace("texture_", "tex_"),
        Type::TextureDepth { kind, .. } => kind.wgsl_name().replace("texture_", "tex_"),
        // Include address space so that e.g. `ptr<function, f32>` and
        // `ptr<private, f32>` produce distinct mangled names and don't
        // falsely collide.
        Type::Ptr {
            address_space,
            elem,
            ..
        } => {
            let space = match address_space {
                crate::parse::AddressSpace::Function => "function",
                crate::parse::AddressSpace::Private => "private",
                crate::parse::AddressSpace::Workgroup => "workgroup",
            };
            format!("ptr_{}_{}", space, mangle_type(elem)?)
        }
        Type::TypeParam { ident } => {
            // This shouldn't happen for fully resolved instantiations
            ident.to_string().to_lowercase()
        }
    })
}

/// Convert a concrete `Type` to a valid WGSL type syntax string.
///
/// Unlike [`mangle_type`] (which produces identifier-safe fragments for
/// function name mangling), this function produces the actual WGSL type
/// syntax used in generated shader code. For example:
///
/// | Type                    | `mangle_type`       | `type_to_wgsl`           |
/// |------------------------|---------------------|--------------------------|
/// | `f32`                  | `"f32"`             | `"f32"`                  |
/// | `array<f32, 4>`        | `"array_f32_4"`     | `"array<f32, 4>"`        |
/// | `atomic<i32>`          | `"atomic_i32"`      | `"atomic<i32>"`          |
/// | `ptr<function, f32>`   | `"ptr_function_f32"`| `"ptr<function, f32>"`   |
///
/// Scalars and vectors happen to produce identical output from both functions,
/// but composite types diverge.
fn type_to_wgsl(ty: &Type) -> Result<String, crate::parse::Error> {
    Ok(match ty {
        Type::Scalar { ty: scalar, .. } => scalar.wgsl_name().to_string(),
        Type::Vector {
            elements,
            scalar_ty,
            ..
        } => format!("vec{}{}", elements, scalar_ty.short_name()),
        Type::Matrix { size, .. } => format!("mat{}x{}f", size, size),
        Type::Struct { ident, type_args } => {
            if type_args.is_empty() {
                ident.to_string()
            } else {
                let wgsl_args: Vec<String> = type_args
                    .iter()
                    .map(type_to_wgsl)
                    .collect::<Result<_, _>>()?;
                // Generic struct types in WGSL are monomorphized, so use the
                // mangled name.
                let mangled_args: Vec<String> = type_args
                    .iter()
                    .map(mangle_type)
                    .collect::<Result<_, _>>()?;
                let _ = wgsl_args; // future: may use for template placeholders
                format!("{}_{}", ident, mangled_args.join("_"))
            }
        }
        Type::Array { elem, len, .. } => {
            format!("array<{}, {}>", type_to_wgsl(elem)?, len_to_string(len))
        }
        Type::RuntimeArray { elem, .. } => format!("array<{}>", type_to_wgsl(elem)?),
        Type::Atomic { elem, .. } => format!("atomic<{}>", type_to_wgsl(elem)?),
        Type::Sampler { .. } => "sampler".to_string(),
        Type::SamplerComparison { .. } => "sampler_comparison".to_string(),
        Type::Texture {
            kind, sampled_type, ..
        } => format!("{}<{}>", kind.wgsl_name(), sampled_type.wgsl_name()),
        Type::TextureDepth { kind, .. } => kind.wgsl_name().to_string(),
        Type::Ptr {
            address_space,
            elem,
            ..
        } => {
            let space = match address_space {
                crate::parse::AddressSpace::Function => "function",
                crate::parse::AddressSpace::Private => "private",
                crate::parse::AddressSpace::Workgroup => "workgroup",
            };
            format!("ptr<{}, {}>", space, type_to_wgsl(elem)?)
        }
        Type::TypeParam { ident } => {
            return Err(crate::parse::Error::unsupported(
                ident.span(),
                format!("cannot produce WGSL type string for unresolved type parameter '{ident}'"),
            ));
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
        Type::Struct { ident, type_args } => {
            if type_args.is_empty() {
                TypeKey::Struct(ident.to_string())
            } else {
                // For generic struct instantiations, include the mangled name
                // so different instantiations get different keys.
                let mangled_args: Vec<String> = type_args
                    .iter()
                    .map(mangle_type)
                    .collect::<Result<_, _>>()?;
                TypeKey::Struct(format!("{}_{}", ident, mangled_args.join("_")))
            }
        }
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

    #[test]
    fn template_dependency_concrete_type_arg_rejected() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<U>(x: U) -> U {
                    x
                }

                pub fn outer<T>(x: T) -> T {
                    let y: f32 = inner::<f32>(1.0);
                    let _keep = y;
                    x
                }

                pub fn caller() -> u32 {
                    outer::<u32>(1u32)
                }
            }
        };
        let err = mono_err(input);
        assert!(
            err.contains("concrete dependency type arguments are not supported yet"),
            "Expected dependency mapping error, got: {err}"
        );
    }

    #[test]
    fn template_dependency_keeps_distinct_mappings_per_callee() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<U>(x: U) -> U {
                    x
                }

                pub fn outer<T, U>(a: T, b: U) -> U {
                    let _a = inner::<T>(a);
                    inner::<U>(b)
                }

                pub fn caller() -> f32 {
                    outer::<i32, f32>(1, 2.0)
                }
            }
        };

        let mut module = crate::parse::ItemMod::try_from(&input).unwrap();
        let result = super::run(&mut module).unwrap();
        let tmpl = result
            .template_macros
            .iter()
            .find(|t| t.fn_name == "outer")
            .expect("Expected template for 'outer'");

        assert_eq!(
            tmpl.dependencies.len(),
            2,
            "Expected both inner::<T> and inner::<U> mappings to be preserved"
        );
        assert!(
            tmpl.dependencies
                .iter()
                .any(|d| d.callee == "inner" && d.type_param_mapping == vec![0]),
            "Missing dependency mapping inner::<T> -> [0]"
        );
        assert!(
            tmpl.dependencies
                .iter()
                .any(|d| d.callee == "inner" && d.type_param_mapping == vec![1]),
            "Missing dependency mapping inner::<U> -> [1]"
        );
    }

    #[test]
    fn mono_generic_struct_simple() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Pair<T> {
                    pub a: T,
                    pub b: T,
                }

                pub fn make_pair() -> f32 {
                    let p: Pair<f32> = Pair::<f32> { a: 1.0, b: 2.0 };
                    return p.a;
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("struct Pair_f32"),
            "Expected monomorphized struct Pair_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("a:") && wgsl.contains("f32"),
            "Expected field 'a: f32' in monomorphized struct, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("b:") && wgsl.contains("f32"),
            "Expected field 'b: f32' in monomorphized struct, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("Pair_f32("),
            "Expected rewritten struct constructor, got:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("struct Pair {"),
            "Generic struct template should be removed, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_generic_struct_multiple_instantiations() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Wrapper<T> {
                    pub value: T,
                }

                pub fn use_both() -> f32 {
                    let a: Wrapper<f32> = Wrapper::<f32> { value: 1.0 };
                    let b: Wrapper<i32> = Wrapper::<i32> { value: 1 };
                    return a.value;
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("struct Wrapper_f32"),
            "Expected Wrapper_f32, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("struct Wrapper_i32"),
            "Expected Wrapper_i32, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_generic_struct_with_impl() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Pair<T> {
                    pub a: T,
                    pub b: T,
                }

                impl<T> Pair<T> {
                    pub fn first(p: Pair<T>) -> T {
                        return p.a;
                    }
                }

                pub fn main_fn() -> f32 {
                    let p: Pair<f32> = Pair::<f32> { a: 1.0, b: 2.0 };
                    return Pair::<f32>::first(p);
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("struct Pair_f32"),
            "Expected monomorphized struct, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("fn Pair_f32_first("),
            "Expected monomorphized method, got:\n{wgsl}"
        );
        // The method should take Pair_f32 and return f32
        assert!(
            wgsl.contains("p:Pair_f32") || wgsl.contains("p: Pair_f32"),
            "Expected monomorphized param type, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_generic_struct_in_function_param() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Container<T> {
                    pub val: T,
                }

                pub fn extract(c: Container<f32>) -> f32 {
                    return c.val;
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("struct Container_f32"),
            "Expected monomorphized struct, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("c:Container_f32") || wgsl.contains("c: Container_f32"),
            "Expected monomorphized param type, got:\n{wgsl}"
        );
    }

    #[test]
    fn mono_generic_struct_template_removed() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub struct Pair<T> {
                    pub a: T,
                    pub b: T,
                }

                impl<T> Pair<T> {
                    pub fn first(p: Pair<T>) -> T {
                        return p.a;
                    }
                }

                pub fn use_it() -> f32 {
                    let p: Pair<f32> = Pair::<f32> { a: 1.0, b: 2.0 };
                    return Pair::<f32>::first(p);
                }
            }
        };
        let wgsl = mono_wgsl(input);
        // The generic impl block should be removed
        assert!(
            !wgsl.contains("fn first("),
            "Generic impl methods should be removed from WGSL, got:\n{wgsl}"
        );
    }
}
