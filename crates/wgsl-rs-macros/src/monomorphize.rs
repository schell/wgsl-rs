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
    parse::{Block, Expr, FnPath, Item, ItemFn, ItemImpl, ItemMod, ItemStruct, ReturnType, Type},
    parse_visitor::{self, ParseVisitorMut},
};

/// A cross-module template instantiation that needs to be included in the
/// consuming module's `WGSL_MODULE`.
#[derive(Clone)]
pub struct CrossModuleInstantiation {
    /// Candidate imported modules that may contain the template.
    pub import_paths: Vec<syn::Path>,
    /// The generic function name.
    pub fn_name: String,
    /// Identifier-safe mangled type argument names, used for function name
    /// mangling and deduplication keys (e.g. `"array_f32_4"`,
    /// `"ptr_function_f32"`).
    pub mangled_type_args: Vec<String>,
    /// The original parse-side type arguments. These are converted to IR
    /// at emission time and shipped as a `fn() -> Vec<ir::Type>`
    /// constructor on the [`crate::Module`] so runtime substitution can
    /// produce a concrete shader source.
    pub type_args: Vec<Type>,
}

/// Result of the monomorphization pass.
pub struct MonoResult {
    /// Cross-module template instantiations that need macro invocations in
    /// the consuming module's `WGSL_MODULE.source`.
    pub cross_module_instantiations: Vec<CrossModuleInstantiation>,
    /// Template macros to generate for generic functions defined in this
    /// module.
    pub template_macros: Vec<TemplateMacro>,
}

/// Information needed to emit a generic template's IR constructor.
///
/// The `items` carry parse-side AST with [`Type::TypeParam`] still
/// present — substitution happens at runtime via
/// `wgsl_rs_ir::substitute_items` after parse → IR conversion.
pub struct TemplateMacro {
    /// The generic function or struct name.
    pub fn_name: String,
    /// Type parameter names (e.g., `["M", "L", "N"]`).
    pub type_param_names: Vec<String>,
    /// The template's items, with `Type::TypeParam` nodes still in place.
    /// Converted to IR at emission time. For function templates this is
    /// a single `Item::Fn`; for struct templates it's a `Item::Struct`
    /// followed by all the rewritten impl methods/consts.
    pub items: Vec<Item>,
    /// Transitive calls to other generic functions within this template.
    pub dependencies: Vec<TemplateDep>,
}

/// A dependency from one template to another generic function.
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
    // on calls to local generic functions). The visitor takes `&mut` for
    // uniformity but doesn't actually mutate anything.
    for item in &mut module.content {
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
    Matrix(u8, u8),
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
                    // Generic entry points (`#[vertex]`, `#[fragment]`,
                    // `#[compute]`) are not template-instantiated within the
                    // module. They remain in the WGSL source with
                    // `__TP{name}__` placeholders, and the module-level
                    // template machinery substitutes them at instantiation
                    // time via `Module::instantiate`.
                    let is_generic_entry_point = !f.type_params.is_empty()
                        && !matches!(f.fn_attrs, crate::parse::FnAttrs::None);
                    if f.type_params.is_empty() || is_generic_entry_point {
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
    /// holding the parse-side AST items with `Type::TypeParam` nodes still
    /// in place. At cross-module instantiation time these items are
    /// converted to IR and the type params are replaced with concrete types
    /// via `wgsl_rs_ir::substitute_items`.
    fn generate_template_macros(
        &self,
        _module: &ItemMod,
    ) -> Result<Vec<TemplateMacro>, crate::parse::Error> {
        let mut macros = Vec::new();

        // --- Generic struct templates ---
        for template in self.struct_templates.values() {
            // Clone the struct template; clear `type_params` so it parses
            // as a concrete struct definition. The `Type::TypeParam`
            // nodes inside field types remain in place — runtime
            // substitution will replace them.
            let mut struct_clone = template.clone();
            struct_clone.type_params.clear();

            let mut items: Vec<Item> = vec![Item::Struct(struct_clone)];

            // Add all impl-block items, with un-mangled method names.
            // Runtime renaming will produce e.g. `Pair_f32_sum` from
            // `Pair_sum`.
            let original_name = template.ident.to_string();
            if let Some(impl_blocks) = self.impl_templates.get(&original_name) {
                for impl_block in impl_blocks {
                    let mut impl_clone = impl_block.clone();
                    impl_clone.type_params.clear();
                    items.push(Item::Impl(impl_clone));
                }
            }

            let type_param_names: Vec<String> = template
                .type_params
                .iter()
                .map(|id| id.to_string())
                .collect();

            // TODO: compute dependencies for struct templates (nested
            // generic struct or function references). For now, empty.
            let dependencies = vec![];

            macros.push(TemplateMacro {
                fn_name: original_name,
                type_param_names,
                items,
                dependencies,
            });
        }

        // --- Generic function templates ---
        for template in self.templates.values() {
            // Clone and clear `type_params` so the IR conversion sees a
            // concrete-looking function. `Type::TypeParam` nodes stay in
            // place; runtime substitution + renaming handle the rest.
            let mut fn_clone = template.clone();
            fn_clone.type_params.clear();

            let type_param_names: Vec<String> = template
                .type_params
                .iter()
                .map(|id| id.to_string())
                .collect();

            // Compute transitive dependencies by scanning the original
            // (pre-substitution) template body for turbofish calls to
            // other generic functions.
            let dependencies =
                collect_template_dependencies(&template.block, &self.templates, &type_param_names)?;

            macros.push(TemplateMacro {
                fn_name: template.ident.to_string(),
                type_param_names,
                items: vec![Item::Fn(Box::new(fn_clone))],
                dependencies,
            });
        }

        Ok(macros)
    }

    /// Walk all concrete items to find turbofish call sites and generic
    /// struct usages. The implementation drives [`ParseVisitorMut`] over
    /// the items that aren't themselves generic templates.
    ///
    /// `MonoCtx` itself implements `ParseVisitorMut` — `visit_expr` and
    /// `visit_type` enqueue any concrete generic instantiations they
    /// encounter; the structural recursion is handled by the default
    /// `walk_*` methods.
    fn discover_instantiations(&mut self, module: &mut ItemMod) -> Result<(), crate::parse::Error> {
        for item in &mut module.content {
            // Skip items that are themselves generic templates — those
            // get walked transitively when each instantiation is
            // monomorphized in `process_queue()`.
            let is_template = match item {
                Item::Fn(f) => !f.type_params.is_empty(),
                Item::Impl(i) => !i.type_params.is_empty(),
                Item::Struct(s) => !s.type_params.is_empty(),
                _ => false,
            };
            if is_template {
                continue;
            }
            self.visit_item(item)?;
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

        // Discover any new instantiations from the monomorphized body.
        self.visit_fn(&mut mono_fn)?;

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

        // Discover any new struct instantiations from monomorphized fields.
        for pair in mono_struct.fields.named.iter_mut() {
            self.visit_type(&mut pair.ty)?;
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

                            let mut mono_fn = (**f).clone();
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

                            // Discover any new instantiations from monomorphized body.
                            self.visit_fn(&mut mono_fn)?;

                            self.generated.push(Item::Fn(Box::new(mono_fn)));
                        }
                        crate::parse::ImplItem::Const(c) => {
                            let const_mangled_name = format!("{}_{}", mangled_name, c.ident);

                            let mut mono_const = (**c).clone();
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
        // Remove generic template functions, structs, and impl blocks.
        // Generic entry points are kept (they aren't templates — they remain
        // in the module's IR with `Type::TypeParam` nodes for module-level
        // instantiation).
        module.content.retain(|item| match item {
            Item::Fn(f) => {
                let is_generic_entry_point =
                    !f.type_params.is_empty() && !matches!(f.fn_attrs, crate::parse::FnAttrs::None);
                f.type_params.is_empty() || is_generic_entry_point
            }
            Item::Struct(s) => s.type_params.is_empty(),
            Item::Impl(i) => i.type_params.is_empty(),
            _ => true,
        });

        // Add generated monomorphized items
        module.content.extend(self.generated.clone());

        // Rewrite all call sites and struct types in all items
        for item in &mut module.content {
            rewrite_names_in_item(item, &self.templates, &self.struct_templates);
        }

        Ok(())
    }
}

// ===== Instantiation discovery (MonoCtx as visitor) =====
//
// `MonoCtx` implements [`ParseVisitorMut`] so the structural recursion
// is handled by the default `walk_*` methods. The two overrides below
// pick out concrete generic instantiations (turbofish calls, generic
// struct usages, struct method calls, struct constructors) and enqueue
// them onto `self.queue` / `self.struct_queue`.

impl ParseVisitorMut for MonoCtx {
    fn visit_type(&mut self, ty: &mut Type) -> Result<(), crate::parse::Error> {
        if let Type::Struct { ident, type_args } = ty
            && !type_args.is_empty()
        {
            let struct_name = ident.to_string();
            // We use the ident's span as the diagnostic site — the
            // original implementation passed the surrounding `pair.ident`
            // / `f.ident` / `local.ident` span instead. The struct
            // ident's span is sharper for the user, since it points at
            // the actual `Pair<f32>` text.
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
        parse_visitor::walk_type(self, ty)
    }

    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        if let Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } = expr
            && !type_args.is_empty()
        {
            let fn_name = match path {
                FnPath::Ident(id) => id.to_string(),
                FnPath::TypeMethod { ty, method, .. } => format!("{}_{}", ty, method),
            };
            let span = match path {
                FnPath::Ident(id) => id.span(),
                FnPath::TypeMethod { ty, .. } => ty.span(),
            };

            // `Pair::<f32>::first(p)` — enqueue the struct instantiation
            // (the impl-method instantiation falls out of struct
            // monomorphization).
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
                    if type_args.iter().all(|ta| !contains_type_param(ta)) {
                        let key = InstKey {
                            fn_name: ty_name.clone(),
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
                // If type_args contain a TypeParam, this call is inside a
                // generic function body. It will be resolved transitively
                // when that template itself is instantiated.
            } else if self.reserved_names.contains(&fn_name) {
                return Err(crate::parse::Error::unsupported(
                    span,
                    format!(
                        "'{fn_name}' is not a generic function, but was called with type arguments"
                    ),
                ));
            }
            // Otherwise, this may be a cross-module generic call — defer
            // validation/rewriting to the cross-module pass.

            // Recurse into params (the default walk would do this too,
            // but we've consumed the early-return path above).
            for param in params.iter_mut() {
                self.visit_expr(param)?;
            }
            return Ok(());
        }

        if let Expr::Struct {
            ident,
            type_args,
            fields,
            ..
        } = expr
            && !type_args.is_empty()
        {
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
            for field in fields.iter_mut() {
                self.visit_expr(&mut field.expr)?;
            }
            return Ok(());
        }

        parse_visitor::walk_expr(self, expr)
    }
}

// ===== Substitution helpers =====
//
// Replaces every `Type::TypeParam` (and the equivalent ident references in
// `FnPath::TypeMethod` and `Expr::Struct`) with its concrete type from a
// caller-supplied substitution map. Implemented as a [`ParseVisitorMut`].

struct SubstituteVisitor<'a> {
    subst: &'a BTreeMap<String, Type>,
}

impl ParseVisitorMut for SubstituteVisitor<'_> {
    fn visit_type(&mut self, ty: &mut Type) -> Result<(), crate::parse::Error> {
        if let Type::TypeParam { ident } = ty
            && let Some(concrete) = self.subst.get(&ident.to_string())
        {
            *ty = concrete.clone();
            return Ok(());
        }
        parse_visitor::walk_type(self, ty)
    }

    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        // Two extra rewrites that the default walker doesn't perform:
        //
        // * `T::method(...)` becomes `f32::method(...)` when `T` is in the substitution
        //   map. The default `walk_expr` only descends into `type_args` and `params`;
        //   the path itself is invisible to it.
        //
        // * `T { fields }` becomes `f32 { fields }` for the same reason (the struct
        //   ident is just an `Ident`, not a `Type`).
        match expr {
            Expr::FnCall { path, .. } => {
                if let FnPath::TypeMethod { ty, .. } = path
                    && let Some(concrete) = self.subst.get(&ty.to_string())
                {
                    *ty = type_to_ident(concrete, ty.span());
                }
            }
            Expr::Struct { ident, .. } => {
                if let Some(concrete) = self.subst.get(&ident.to_string()) {
                    *ident = type_to_ident(concrete, ident.span());
                }
            }
            _ => {}
        }
        parse_visitor::walk_expr(self, expr)
    }
}

fn substitute_type(ty: &mut Type, subst: &BTreeMap<String, Type>) {
    let mut v = SubstituteVisitor { subst };
    let _ = v.visit_type(ty);
}

fn substitute_block(block: &mut Block, subst: &BTreeMap<String, Type>) {
    let mut v = SubstituteVisitor { subst };
    let _ = v.visit_block(block);
}

fn substitute_expr(expr: &mut Expr, subst: &BTreeMap<String, Type>) {
    let mut v = SubstituteVisitor { subst };
    let _ = v.visit_expr(expr);
}

// ===== Name rewriting (calls + struct types) =====
//
// After monomorphization, every concrete instantiation of a generic
// function or struct gets a mangled name (e.g. `id<f32>` -> `id_f32`,
// `Pair<f32>` -> `Pair_f32`). This visitor walks all items and
// rewrites every reference (call sites, type references, struct
// constructors, impl methods on generic structs) to use the mangled
// names. Both fn-template and struct-template rewrites happen in a
// single pass.

struct RewriteNamesVisitor<'a> {
    fn_templates: &'a BTreeMap<String, ItemFn>,
    struct_templates: &'a BTreeMap<String, ItemStruct>,
}

impl ParseVisitorMut for RewriteNamesVisitor<'_> {
    fn visit_type(&mut self, ty: &mut Type) -> Result<(), crate::parse::Error> {
        if let Type::Struct { ident, type_args } = ty
            && !type_args.is_empty()
            && self.struct_templates.contains_key(&ident.to_string())
        {
            let mangled = mangle_name(&ident.to_string(), type_args)
                .expect("mangle_name should not fail for concrete types");
            *ident = Ident::new(&mangled, ident.span());
            type_args.clear();
        }
        parse_visitor::walk_type(self, ty)
    }

    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        match expr {
            Expr::FnCall {
                path, type_args, ..
            } if !type_args.is_empty() => {
                // First try fn-template name mangling (covers both
                // free generic functions and `Type::method` calls
                // where `Type::method` is a known template fn).
                let fn_name = match &*path {
                    FnPath::Ident(id) => id.to_string(),
                    FnPath::TypeMethod { ty, method, .. } => format!("{}_{}", ty, method),
                };
                if self.fn_templates.contains_key(&fn_name) {
                    let mangled = mangle_name(&fn_name, type_args)
                        .expect("mangle_name should not fail for concrete types");
                    let span = match &*path {
                        FnPath::Ident(id) => id.span(),
                        FnPath::TypeMethod { ty, .. } => ty.span(),
                    };
                    *path = FnPath::Ident(Ident::new(&mangled, span));
                    type_args.clear();
                } else if let FnPath::TypeMethod { ty, method, .. } = path {
                    // Then check for `StructTemplate::method(...)` calls
                    // where `StructTemplate` is a known generic struct.
                    let ty_name = ty.to_string();
                    if self.struct_templates.contains_key(&ty_name) && !type_args.is_empty() {
                        let mangled_struct = mangle_name(&ty_name, type_args)
                            .expect("mangle_name should not fail for concrete types");
                        let mangled_fn = format!("{}_{}", mangled_struct, method);
                        let span = ty.span();
                        *path = FnPath::Ident(Ident::new(&mangled_fn, span));
                        type_args.clear();
                    }
                }
            }
            Expr::Struct {
                ident, type_args, ..
            } if !type_args.is_empty()
                && self.struct_templates.contains_key(&ident.to_string()) =>
            {
                let mangled = mangle_name(&ident.to_string(), type_args)
                    .expect("mangle_name should not fail for concrete types");
                *ident = Ident::new(&mangled, ident.span());
                type_args.clear();
            }
            _ => {}
        }
        parse_visitor::walk_expr(self, expr)
    }
}

fn rewrite_names_in_item(
    item: &mut Item,
    fn_templates: &BTreeMap<String, ItemFn>,
    struct_templates: &BTreeMap<String, ItemStruct>,
) {
    let mut v = RewriteNamesVisitor {
        fn_templates,
        struct_templates,
    };
    let _ = v.visit_item(item);
}


// ===== Missing turbofish detection =====
//
// Errors if any call to a known generic template function lacks turbofish
// type arguments. Implemented as a [`ParseVisitorMut`] that doesn't actually
// mutate anything.

struct CheckUnresolvedVisitor<'a> {
    templates: &'a BTreeMap<String, ItemFn>,
}

impl ParseVisitorMut for CheckUnresolvedVisitor<'_> {
    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        if let Expr::FnCall {
            path, type_args, ..
        } = expr
            && type_args.is_empty()
            && let FnPath::Ident(id) = path
        {
            let name = id.to_string();
            if let Some(template) = self.templates.get(&name) {
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
        parse_visitor::walk_expr(self, expr)
    }
}

/// Check that no calls to generic template functions remain without turbofish.
///
/// After monomorphization, all generic templates are removed. Any remaining
/// call to a template name with empty `type_args` means the user forgot
/// the turbofish annotation.
fn check_unresolved_generic_calls(
    item: &mut Item,
    templates: &BTreeMap<String, ItemFn>,
) -> Result<(), crate::parse::Error> {
    let mut v = CheckUnresolvedVisitor { templates };
    v.visit_item(item)
}

// ===== Cross-module instantiation collection =====
//
// Scans the module's items for turbofish calls/struct constructions that
// reference generic functions or structs defined in other modules. Each
// such reference is rewritten to use a mangled concrete name, and a
// [`CrossModuleInstantiation`] entry is recorded for the consuming module
// to ship as IR constructors.

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
        let mut visitor = CrossModuleVisitor {
            local_templates,
            local_struct_templates,
            import_paths: &import_paths,
            out: &mut instantiations,
            seen: &mut seen_mangled,
        };
        visitor.visit_item(item)?;
    }

    Ok(instantiations)
}

struct CrossModuleVisitor<'a> {
    local_templates: &'a BTreeMap<String, ItemFn>,
    local_struct_templates: &'a BTreeMap<String, ItemStruct>,
    import_paths: &'a [syn::Path],
    out: &'a mut Vec<CrossModuleInstantiation>,
    seen: &'a mut BTreeSet<String>,
}

impl ParseVisitorMut for CrossModuleVisitor<'_> {
    fn visit_type(&mut self, ty: &mut Type) -> Result<(), crate::parse::Error> {
        // Cross-module generic struct usage: `OtherMod::Pair<f32>`.
        // We detect a `Type::Struct` that has type args but is *not* a
        // local struct template — it must come from an imported module.
        if let Type::Struct { ident, type_args } = ty
            && !type_args.is_empty()
        {
            let struct_name = ident.to_string();
            if !self.local_struct_templates.contains_key(&struct_name) {
                let mangled_type_args: Vec<String> = type_args
                    .iter()
                    .map(mangle_type)
                    .collect::<Result<_, _>>()?;
                let mangled_name = mangle_name(&struct_name, type_args)?;
                let parse_type_args: Vec<Type> = type_args.clone();

                *ident = Ident::new(&mangled_name, ident.span());
                type_args.clear();

                if self.seen.insert(mangled_name) {
                    self.out.push(CrossModuleInstantiation {
                        import_paths: self.import_paths.to_vec(),
                        fn_name: struct_name,
                        mangled_type_args,
                        type_args: parse_type_args,
                    });
                }
                // type_args has been cleared; nothing left to recurse into.
                return Ok(());
            }
        }
        parse_visitor::walk_type(self, ty)
    }

    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        match expr {
            Expr::FnCall {
                path,
                type_args,
                params,
                ..
            } if !type_args.is_empty() => {
                // First check for `OtherStruct::method(...)` calls where
                // `OtherStruct` is a cross-module generic struct.
                if let FnPath::TypeMethod { ty, method, .. } = &*path {
                    let ty_name = ty.to_string();
                    let combined = format!("{ty_name}_{method}");
                    if !self.local_struct_templates.contains_key(&ty_name)
                        && !self.local_templates.contains_key(&combined)
                    {
                        let mangled_type_args: Vec<String> = type_args
                            .iter()
                            .map(mangle_type)
                            .collect::<Result<_, _>>()?;
                        let parse_type_args: Vec<Type> = type_args.clone();
                        let mangled_struct = mangle_name(&ty_name, type_args)?;
                        let mangled_fn = format!("{}_{}", mangled_struct, method);
                        let span = ty.span();
                        *path = FnPath::Ident(Ident::new(&mangled_fn, span));
                        type_args.clear();

                        if self.seen.insert(mangled_struct) {
                            self.out.push(CrossModuleInstantiation {
                                import_paths: self.import_paths.to_vec(),
                                fn_name: ty_name,
                                mangled_type_args,
                                type_args: parse_type_args,
                            });
                        }
                    }
                }

                // Now handle cross-module free function templates. After
                // the `TypeMethod` path above, the path may have been
                // rewritten to an `Ident`; double-check `type_args` still
                // has entries before continuing.
                if !type_args.is_empty() {
                    let fn_name = match &*path {
                        FnPath::Ident(id) => id.to_string(),
                        FnPath::TypeMethod { ty, method, .. } => format!("{}_{}", ty, method),
                    };
                    if !self.local_templates.contains_key(&fn_name) {
                        if self.import_paths.is_empty() {
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
                        let parse_type_args: Vec<Type> = type_args.clone();
                        let mangled_name = mangle_name(&fn_name, type_args)?;

                        let span = match &*path {
                            FnPath::Ident(id) => id.span(),
                            FnPath::TypeMethod { ty, .. } => ty.span(),
                        };
                        *path = FnPath::Ident(Ident::new(&mangled_name, span));
                        type_args.clear();

                        if self.seen.insert(mangled_name) {
                            self.out.push(CrossModuleInstantiation {
                                import_paths: self.import_paths.to_vec(),
                                fn_name,
                                mangled_type_args,
                                type_args: parse_type_args,
                            });
                        }
                    }
                }
                // Recurse into the call arguments so nested
                // turbofish/struct constructions are also collected.
                for param in params.iter_mut() {
                    self.visit_expr(param)?;
                }
                return Ok(());
            }
            Expr::Struct {
                ident,
                type_args,
                fields,
                ..
            } if !type_args.is_empty() => {
                let struct_name = ident.to_string();
                if !self.local_struct_templates.contains_key(&struct_name) {
                    let mangled_type_args: Vec<String> = type_args
                        .iter()
                        .map(mangle_type)
                        .collect::<Result<_, _>>()?;
                    let parse_type_args: Vec<Type> = type_args.clone();
                    let mangled_name = mangle_name(&struct_name, type_args)?;

                    *ident = Ident::new(&mangled_name, ident.span());
                    type_args.clear();

                    if self.seen.insert(mangled_name) {
                        self.out.push(CrossModuleInstantiation {
                            import_paths: self.import_paths.to_vec(),
                            fn_name: struct_name,
                            mangled_type_args,
                            type_args: parse_type_args,
                        });
                    }
                }
                for f in fields.iter_mut() {
                    self.visit_expr(&mut f.expr)?;
                }
                return Ok(());
            }
            _ => {}
        }
        parse_visitor::walk_expr(self, expr)
    }
}


// ===== Template dependency scanning =====
//
// Scans a generic function's body for turbofish calls to other generic
// functions, building [`TemplateDep`] entries that map each callee's type
// param to one of the caller's type params by index.

struct ScanDepsVisitor<'a> {
    templates: &'a BTreeMap<String, ItemFn>,
    caller_params: &'a [String],
    out: &'a mut Vec<TemplateDep>,
    seen: &'a mut BTreeSet<(String, Vec<usize>)>,
}

impl ParseVisitorMut for ScanDepsVisitor<'_> {
    fn visit_expr(&mut self, expr: &mut Expr) -> Result<(), crate::parse::Error> {
        if let Expr::FnCall {
            path,
            type_args,
            params,
            ..
        } = expr
            && !type_args.is_empty()
        {
            // Only `FnPath::Ident(...)` calls are template-to-template
            // dependencies; `TypeMethod` paths are handled by the
            // monomorphizer's struct-method machinery.
            let fn_name = match path {
                FnPath::Ident(id) => id.to_string(),
                FnPath::TypeMethod { .. } => String::new(),
            };
            if !fn_name.is_empty() && self.templates.contains_key(&fn_name) {
                let span = match path {
                    FnPath::Ident(id) => id.span(),
                    FnPath::TypeMethod { ty, .. } => ty.span(),
                };

                // Build the type param mapping: each callee type_arg must
                // be a `Type::TypeParam` whose name appears in
                // `caller_params`.
                let mut mapping = Vec::with_capacity(type_args.len());
                for ta in type_args.iter() {
                    let Type::TypeParam { ident } = ta else {
                        return Err(crate::parse::Error::unsupported(
                            span,
                            format!(
                                "template dependency '{fn_name}' must use caller type parameters \
                                 directly; concrete dependency type arguments are not supported \
                                 yet"
                            ),
                        ));
                    };
                    let Some(idx) = self
                        .caller_params
                        .iter()
                        .position(|p| p == &ident.to_string())
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
                if self.seen.insert((fn_name.clone(), mapping.clone())) {
                    self.out.push(TemplateDep {
                        callee: fn_name,
                        type_param_mapping: mapping,
                    });
                }
            }
            // Recurse into the call arguments.
            for param in params.iter_mut() {
                self.visit_expr(param)?;
            }
            return Ok(());
        }
        parse_visitor::walk_expr(self, expr)
    }
}

/// Scan a block for turbofish calls to other generic functions and build
/// `TemplateDep` entries recording the type param mappings.
fn collect_template_dependencies(
    block: &Block,
    templates: &BTreeMap<String, ItemFn>,
    caller_type_params: &[String],
) -> Result<Vec<TemplateDep>, crate::parse::Error> {
    let mut deps = Vec::new();
    let mut seen: BTreeSet<(String, Vec<usize>)> = BTreeSet::new();
    // The visitor doesn't actually mutate; we clone to satisfy the
    // `&mut Block` signature without imposing `&mut` on our callers.
    // Templates are small (single-function bodies), so this clone is
    // negligible.
    let mut block_clone = block.clone();
    let mut v = ScanDepsVisitor {
        templates,
        caller_params: caller_type_params,
        out: &mut deps,
        seen: &mut seen,
    };
    v.visit_block(&mut block_clone)?;
    Ok(deps)
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
        Type::Matrix { columns, rows, .. } => {
            if columns == rows {
                format!("Mat{}f", columns)
            } else {
                format!("Mat{}x{}f", columns, rows)
            }
        }
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
        Type::Matrix { columns, rows, .. } => format!("mat{}x{}f", columns, rows),
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
        Type::Matrix { columns, rows, .. } => TypeKey::Matrix(*columns, *rows),
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
        ir_convert,
        parse::{Item, ItemMod},
    };

    /// Helper: parse a module, run monomorphization, return the WGSL
    /// string produced by the IR pipeline.
    fn mono_wgsl(input: syn::ItemMod) -> String {
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        super::run(&mut wgsl_module).unwrap();
        let ir_items = ir_convert::items_from_parse(&wgsl_module.content)
            .expect("parse -> IR conversion should succeed for monomorphized output");
        wgsl_rs_ir::render_items(&ir_items)
    }

    /// Helper: parse a module, run monomorphization, expect an error.
    fn mono_err(input: syn::ItemMod) -> String {
        let mut wgsl_module = ItemMod::try_from(&input).unwrap();
        match super::run(&mut wgsl_module) {
            Ok(_) => panic!("expected an error from monomorphization, got Ok"),
            Err(e) => format!("{e}"),
        }
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
        // The template's items should contain a single function whose
        // body still references the type parameter `T` directly via
        // Type::TypeParam (no string placeholders).
        assert_eq!(tmpl.items.len(), 1);
        let Item::Fn(f) = &tmpl.items[0] else {
            panic!(
                "expected Item::Fn for identity template, got: {:?}",
                tmpl.items.len()
            );
        };
        assert_eq!(f.ident.to_string(), "identity");
        // The argument type should be `Type::TypeParam { ident: "T" }`.
        let arg = f.inputs.first().expect("identity has 1 arg");
        assert!(matches!(
            &arg.ty,
            crate::parse::Type::TypeParam { ident } if ident == "T"
        ));
    }

    #[test]
    fn template_records_turbofish_dependency() {
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

        // Find the template for `outer`.
        let outer_tmpl = result
            .template_macros
            .iter()
            .find(|t| t.fn_name == "outer")
            .expect("Expected template for 'outer'");

        // The template should record the turbofish call via the
        // `dependencies` list (a single dependency `inner` mapped from
        // outer's 0th type param).
        assert_eq!(outer_tmpl.dependencies.len(), 1);
        assert_eq!(outer_tmpl.dependencies[0].callee, "inner");
        assert_eq!(outer_tmpl.dependencies[0].type_param_mapping, vec![0]);
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

    /// Regression: walkers that handle `Stmt::If` must recurse through the
    /// full `else if` chain, not stop at the first `else if`.
    ///
    /// Before the fix, a turbofish call buried in a deeply nested `else
    /// if` branch would be invisible to:
    ///   * `check_unresolved_in_stmt` (would silently miss missing turbofish)
    ///   * `collect_cross_module_from_stmt` (would miss the cross-module call)
    ///   * `scan_stmt_for_deps` (would miss the dependency edge)
    ///
    /// This test exercises all three by placing the only call to a generic
    /// function `id::<f32>(...)` inside the third arm of an else-if chain.
    /// If any walker fails to recurse, the generic template won't be
    /// monomorphized and `mono_wgsl` will not contain `fn id_f32(...)`.
    #[test]
    fn nested_else_if_recursion_finds_calls_in_deep_branches() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn id<T>(x: T) -> T {
                    x
                }

                pub fn caller(n: u32) -> f32 {
                    if n == 0u32 {
                        return 0.0;
                    } else if n == 1u32 {
                        return 1.0;
                    } else if n == 2u32 {
                        // The only turbofish call lives here, three levels
                        // deep into the else-if chain.
                        return id::<f32>(2.0);
                    } else {
                        return 3.0;
                    }
                }
            }
        };
        let wgsl = mono_wgsl(input);
        assert!(
            wgsl.contains("fn id_f32("),
            "Walker missed turbofish in third else-if arm, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("id_f32(2.0"),
            "Call site in third else-if arm not rewritten, got:\n{wgsl}"
        );
    }

    /// Regression: `scan_stmt_for_deps` must recurse through the full
    /// `else if` chain when collecting transitive template dependencies.
    ///
    /// Here `outer<T>` calls `inner::<T>(x)` only inside the third arm of
    /// an else-if chain in `outer`'s body. The dependency must still be
    /// recorded so that calling `outer::<f32>(...)` triggers
    /// instantiation of `inner_f32` too.
    #[test]
    fn nested_else_if_recursion_finds_template_deps_in_deep_branches() {
        let input: syn::ItemMod = syn::parse_quote! {
            mod test_mod {
                pub fn inner<T>(x: T) -> T {
                    x
                }

                pub fn outer<T>(x: T, n: u32) -> T {
                    if n == 0u32 {
                        return x;
                    } else if n == 1u32 {
                        return x;
                    } else if n == 2u32 {
                        return inner::<T>(x);
                    } else {
                        return x;
                    }
                }

                pub fn caller() -> f32 {
                    outer::<f32>(1.0, 2u32)
                }
            }
        };
        let wgsl = mono_wgsl(input);
        // If `scan_if_for_deps` recurses correctly, instantiating
        // `outer<f32>` cascades to instantiate `inner<f32>` too.
        assert!(
            wgsl.contains("fn outer_f32("),
            "outer was not monomorphized, got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("fn inner_f32("),
            "inner_f32 should have been instantiated as a transitive dep, got:\n{wgsl}"
        );
    }
}
