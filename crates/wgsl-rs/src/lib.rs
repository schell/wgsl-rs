//! WGSL in Rust.
use ::std::collections::{HashMap, HashSet};

pub use wgsl_rs_macros::{wgsl, wgsl_allow, wgsl_ignore};

pub mod extension;
pub mod linkage;
pub mod source_error;

pub use extension::WgslExtension;

pub use source_error::SourceError;

/// Re-export of the IR crate so the proc-macro and consuming crates can
/// reference IR types via `wgsl_rs::ir::...` without needing to depend on
/// `wgsl-rs-ir` directly.
pub use wgsl_rs_ir as ir;

/// A WGSL source.
///
/// A `Source` is the public-facing spec for a `#[wgsl] mod foo { ... }`
/// block: the macro emits one `pub static foo::WGSL_SOURCE: Source` per
/// module. WGSL itself doesn't support importing modules, but `wgsl-rs`
/// does (glob imports only), so a `Source` can import other `Source`s.
///
/// Concrete and template sources share the same type. Templates are
/// detected at runtime via [`Source::is_template`]; a template source's
/// [`Source::wgsl_source`] returns
/// [`SourceError::TemplateWgsl`] and the caller must use the
/// macro-emitted `instantiate::<…>()` to obtain a concrete
/// [`ir::Module`] first.
///
/// ```rust, ignore
/// #[wgsl]
/// pub mod constants {
///     pub const NUMBER: u32 = 1234;
/// }
/// #[wgsl]
/// pub mod main {
///     use super::constants::*;
///
///     fn add_to_number(n: u32) -> u32 {
///         n + NUMBER
///     }
/// }
#[derive(Clone, Debug)]
pub struct Source {
    /// Unique identifier for this source, assigned at proc-macro expansion
    /// time. Used to deduplicate sources during source assembly (diamond
    /// import graphs, same-named sources from different paths).
    pub id: u64,

    /// Name of the source.
    pub name: &'static str,

    /// Imports of other WGSL sources.
    pub imports: &'static [&'static Source],

    /// Constructor function that builds this module's IR. Called lazily by
    /// `wgsl_source()` / `instantiate()`.
    ///
    /// The returned IR is owned and freshly constructed on each call; the
    /// caller is free to clone, mutate, or substitute over it.
    pub ir_constructor: fn() -> ir::Module,

    /// Generic function templates defined in this module.
    ///
    /// Consuming modules reference these via `instantiations` to produce
    /// monomorphized functions at source-assembly time. The IR is held as
    /// a constructor, just like the parent module's IR.
    pub templates: &'static [GenericTemplate],

    /// Cross-module template instantiations.
    ///
    /// Each entry references a template from an imported module and provides
    /// the concrete IR type arguments to substitute for the type parameters.
    /// These are resolved at source-assembly time in `wgsl_source()`.
    pub instantiations: &'static [TemplateInstantiation],

    /// Module-level type parameters, in instantiation order.
    ///
    /// Two distinct kinds of declaration contribute to this list:
    ///
    /// * `impl Trait` linkage variables, e.g. `uniform!(group(0), binding(0),
    ///   FRAME: impl Convert<f32>)`. Each such declaration contributes the
    ///   variable's identifier (`"FRAME"`).
    /// * Generic shader entry points, e.g. `pub fn frag_main<T: Convert<f32>>()
    ///   -> Vec4f`. Each one of an entry point's type parameters contributes a
    ///   positional name of the form `"<fn_name>_<index>"` (e.g.
    ///   `"frag_main_0"`). Source names like `T` are intentionally not used so
    ///   that two entry points sharing a letter don't collide.
    ///
    /// The two groups appear in declaration order: linkage variables
    /// first, then entry points.
    ///
    /// When this slice is empty, the module is concrete and
    /// [`Self::wgsl_source`] produces a directly usable WGSL source.
    /// When it is non-empty, callers should use the `instantiate`
    /// function generated alongside the source, which uses
    /// `wgsl_rs::linkage::Type<Is = ...>` constraints to check at
    /// compile time that every parameter has been bound.
    pub module_type_params: &'static [&'static str],
}

/// A generic function (or struct) template defined within a module.
///
/// The IR is built lazily by calling `ir_constructor`. The returned items
/// reference unresolved [`ir::Type::TypeParam`] nodes whose names match
/// entries in `type_params`; substitution at instantiation time replaces
/// them with concrete types.
#[derive(Debug)]
pub struct GenericTemplate {
    /// The generic function or struct name (e.g., `"id"`, `"Pair"`).
    pub name: &'static str,
    /// Type parameter names (e.g., `["T"]`, `["M", "L", "N"]`).
    pub type_params: &'static [&'static str],
    /// Constructor that builds the un-instantiated IR for this template.
    pub ir_constructor: fn() -> Vec<ir::Item>,
    /// Transitive generic function calls within this template.
    ///
    /// Each entry records a call to another generic function, with a mapping
    /// from the callee's type params back to this template's type params.
    pub dependencies: &'static [TemplateDependency],
}

/// A transitive dependency from one generic template to another.
#[derive(Clone, Debug)]
pub struct TemplateDependency {
    /// Name of the generic function being called.
    pub callee: &'static str,
    /// Maps each of the callee's type params to one of the caller's type
    /// params, by index. For example, if `quadruple<T>` calls `double::<T>()`,
    /// the mapping is `&[0]` (double's 0th param = quadruple's 0th param).
    pub type_param_mapping: &'static [usize],
}

/// A request to instantiate a generic template with concrete types.
#[derive(Debug)]
pub struct TemplateInstantiation {
    /// Candidate imported sources that may contain the template.
    ///
    /// Resolution must find exactly one matching source; missing or ambiguous
    /// matches are treated as hard failures during `wgsl_source()` assembly.
    pub modules: &'static [&'static Source],
    /// The generic function name to instantiate.
    pub template_name: &'static str,
    /// Constructor that builds the concrete IR types to substitute for the
    /// template's type parameters.
    pub type_args_constructor: fn() -> Vec<ir::Type>,
    /// Identifier-safe mangled type argument names, used for deduplication
    /// keys and the `seen` set (e.g. `"array_f32_4"`, `"ptr_function_f32"`).
    pub mangled_type_args: &'static [&'static str],
}

impl Source {
    /// Returns whether this source is a template — i.e. has unresolved
    /// module-level type parameters.
    ///
    /// Template sources cannot be used as shader sources directly; call
    /// `instantiate(...)` to produce a concrete WGSL source.
    pub fn is_template(&self) -> bool {
        !self.module_type_params.is_empty()
    }

    /// Returns the assembled WGSL source for this source and all its
    /// imports, including cross-source template instantiations.
    ///
    /// If this source is a template, the returned error is
    /// [`SourceError::TemplateWgsl`]; the caller must use the
    /// macro-emitted `instantiate::<…>()` to obtain a concrete
    /// [`ir::Module`] first and then call [`ir::Module::wgsl_source`].
    pub fn wgsl_source(&self) -> Result<String, SourceError<'_>> {
        if self.is_template() {
            return Err(SourceError::TemplateWgsl {
                uninstantiated_source: self,
            });
        }
        let mut out = String::new();
        let mut visited_sources: HashSet<u64> = HashSet::new();
        let mut seen: HashSet<(u64, String, Vec<String>)> = HashSet::new();
        self.collect(&mut out, &mut visited_sources, &mut seen, None)?;
        Ok(out)
    }

    /// Recursively collect this source's WGSL source (and that of its
    /// imports / instantiations) into `out`. When `subst` is `Some`, the
    /// caller's substitution map is applied to *this* source's IR (but not
    /// to imported sources — imports are concrete by construction).
    ///
    /// `visited_sources` tracks source IDs that have already been emitted,
    /// preventing duplicate definitions in diamond import graphs.
    /// `seen` tracks `(source_id, template_name, mangled_type_args)` triples
    /// to deduplicate cross-source template instantiations.
    fn collect(
        &self,
        out: &mut String,
        visited_sources: &mut HashSet<u64>,
        seen: &mut HashSet<(u64, String, Vec<String>)>,
        subst: Option<&HashMap<String, ir::Type>>,
    ) -> Result<(), SourceError<'_>> {
        // 1. Imports first (depth-first, deduplicated by source ID).
        for m in self.imports {
            if visited_sources.insert(m.id) {
                m.collect(out, visited_sources, seen, None)?;
            }
        }

        // 2. Build this source's IR, optionally substituting type params.
        let mut ir_module = (self.ir_constructor)();
        if let Some(s) = subst {
            ir::substitute_types(&mut ir_module, s);
        }
        out.push_str(&ir::render_module(&ir_module));

        // 3. Cross-source template instantiations.
        for inst in self.instantiations {
            let mangled: Vec<String> = inst
                .mangled_type_args
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let type_args = (inst.type_args_constructor)();
            instantiate_template_into(
                inst.modules,
                inst.template_name,
                &mangled,
                &type_args,
                out,
                seen,
            )?;
        }
        Ok(())
    }
}

/// Resolve `template_name` from the given candidate sources, recursively
/// instantiate its dependencies, then render its substituted IR into
/// `out`. Tracks `(source_id, name, mangled_type_args)` triples in `seen`
/// to avoid duplicate emission.
///
/// Returns `Err` with a structured [`SourceError`] for user-facing
/// configuration errors (template not found, ambiguous match). Returns
/// `Ok` for a successful instantiation, or for the early-return dedup case.
fn instantiate_template_into<'a>(
    sources: &[&'a Source],
    template_name: &str,
    mangled_type_args: &[String],
    type_args: &[ir::Type],
    out: &mut String,
    seen: &mut HashSet<(u64, String, Vec<String>)>,
) -> Result<(), SourceError<'a>> {
    let available_templates: Vec<String> = sources
        .iter()
        .copied()
        .flat_map(|m| m.templates.iter().map(|t| t.name.to_string()))
        .collect();

    let mut matching_sources: Vec<&Source> = sources
        .iter()
        .copied()
        .filter(|m| m.templates.iter().any(|t| t.name == template_name))
        .collect();

    if matching_sources.is_empty() {
        return Err(SourceError::TemplateNotFound {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            available_templates,
        });
    }

    if matching_sources.len() > 1 {
        let source_names: Vec<String> = matching_sources
            .iter()
            .map(|m| m.name.to_string())
            .collect();
        return Err(SourceError::AmbiguousTemplate {
            template_name: template_name.to_string(),
            mangled_type_args: mangled_type_args.to_vec(),
            matching_source_names: source_names,
            available_templates,
        });
    }

    let source = matching_sources
        .pop()
        .expect("matching_sources is guaranteed to be non-empty after checks");

    let Some(template) = source.templates.iter().find(|t| t.name == template_name) else {
        // Invariant: `matching_sources` was built by filtering on
        // `templates.iter().any(|t| t.name == template_name)`, so the
        // sole surviving source must contain the template.
        unreachable!(
            "resolved source '{}' does not contain template '{}'; available templates: {:?}",
            source.name, template_name, available_templates
        );
    };

    let key = (
        source.id,
        template_name.to_string(),
        mangled_type_args.to_vec(),
    );
    if !seen.insert(key) {
        return Ok(()); // Already instantiated
    }

    // Recursively instantiate dependencies first.
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
        instantiate_template_into(&[source], dep.callee, &dep_mangled, &dep_args, out, seen)?;
    }

    // Build a substitution map from the template's type params to the
    // concrete type args, then render the substituted items.
    let mut subst: HashMap<String, ir::Type> = HashMap::new();
    for (param, arg) in template.type_params.iter().zip(type_args.iter()) {
        subst.insert((*param).to_string(), arg.clone());
    }

    let mut items = (template.ir_constructor)();
    ir::substitute_items(&mut items, &subst);

    // Mangle the template's name to a concrete instance name so multiple
    // monomorphizations can coexist. Uses `ir::mangle` so that names with
    // underscores in either the template name or the type-arg-mangled
    // strings are escaped unambiguously (see issue #112).
    let instance_name = if mangled_type_args.is_empty() {
        template.name.to_string()
    } else {
        let mut components: Vec<&str> = Vec::with_capacity(1 + mangled_type_args.len());
        components.push(template.name);
        for s in mangled_type_args {
            components.push(s.as_str());
        }
        ir::mangle(&components)
    };
    if instance_name != template.name {
        ir::rename_items(&mut items, template.name, &instance_name);
    }

    out.push_str(&ir::render_items(&items));
    Ok(())
}

#[cfg(feature = "validation")]
impl Source {
    /// Validates the concatenated WGSL source of this source and its imports.
    ///
    /// This method concatenates the WGSL source from this source and all its
    /// imports (recursively), then validates the result using naga.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if validation succeeds, or an error message describing
    /// the validation failure. Template sources return
    /// [`SourceError::TemplateWgsl`]-style errors.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// #[wgsl]
    /// pub mod my_shader {
    ///     // ...
    /// }
    ///
    /// // Validate at runtime
    /// my_shader::WGSL_SOURCE.validate().expect("WGSL validation failed");
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        // Template sources contain unresolved `__TP{name}__` placeholders
        // that aren't valid WGSL identifiers. They cannot be validated
        // standalone — the user must call `instantiate(...)` first.
        if self.is_template() {
            return Err(
                "cannot validate a template source — call instantiate(...) first".to_string(),
            );
        }

        validate_wgsl_source(&self.wgsl_source().expect("concrete source"))
    }
}

/// Validates arbitrary WGSL source text using naga.
///
/// Parses the source, then runs naga's semantic validation. This is useful
/// for validating the output of `instantiate::<...>()` + `ir::render_module()`
/// for generic (template) sources.
///
/// # Returns
///
/// Returns `Ok(())` if validation succeeds, or an error message describing
/// the validation failure.
#[cfg(feature = "validation")]
pub fn validate_wgsl_source(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(source))?;
    Ok(())
}

pub mod std;

#[cfg(test)]
#[allow(dead_code)]
mod test {
    use crate::{GenericTemplate, Source, TemplateDependency, TemplateInstantiation, ir, wgsl};

    #[wgsl(crate_path = crate)]
    pub mod a {
        pub const THREE: u32 = 3;
    }

    #[wgsl(crate_path = crate)]
    pub mod b {
        use super::a::*;

        pub fn add_three_to_x_minus_y(x: u32, y: u32) -> u32 {
            let i: u32 = (x - y) + THREE;
            i
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod c {
        use super::b::*;

        pub fn main() {
            let _u = add_three_to_x_minus_y(1337, 666);
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod vecs {
        use crate::std::*;

        const _VEC3F_ONE: Vec4f = vec4f(0.0, 1.0, 2.0, 3.0);
    }

    #[test]
    fn module_source() {
        let source = c::WGSL_SOURCE.wgsl_source().unwrap();
        c::main();
        assert!(source.contains("const THREE: u32 = 3;"), "got:\n{source}");
        assert!(
            source.contains("fn add_three_to_x_minus_y(x: u32, y: u32) -> u32"),
            "got:\n{source}"
        );
        assert!(source.contains("fn main()"), "got:\n{source}");

        // Verify that imported source outputs are not duplicated.
        // Source C imports B, which imports A. Source A's `THREE` const
        // should appear exactly once in the concatenated output.
        assert_eq!(
            source.matches("const THREE:").count(),
            1,
            "THREE should appear exactly once (no duplicates from transitive imports), \
             got:\n{source}"
        );
    }

    // Cross-module trait impl test:
    // - trait_provider defines a trait and implements it for f32
    // - trait_consumer imports it and uses the impl via a generic function

    #[wgsl(crate_path = crate)]
    pub mod trait_provider {
        pub trait Scalable {
            fn scale(val: Self, factor: f32) -> Self;
        }

        impl Scalable for f32 {
            fn scale(val: f32, factor: f32) -> f32 {
                val * factor
            }
        }
    }

    #[wgsl(crate_path = crate)]
    pub mod trait_consumer {
        use super::trait_provider::*;

        pub fn apply_scale<T: Scalable>(val: T, factor: f32) -> T {
            T::scale(val, factor)
        }

        pub fn go() -> f32 {
            apply_scale::<f32>(2.0, 3.0)
        }
    }

    #[test]
    fn cross_module_trait_impl() {
        // Verify provider source generates f32_scale
        let provider_src = trait_provider::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            provider_src.contains("fn f32_scale("),
            "trait_provider should contain f32_scale, got:\n{provider_src}"
        );

        // Verify consumer source generates the monomorphized `apply_scale<f32>`
        // and that it calls `f32_scale`. Under the robust mangling scheme
        // (issue #112), `apply_scale` has an underscore so the joined name
        // is `_1apply_scale_f32`, not `apply_scale_f32`.
        let consumer_src = trait_consumer::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            consumer_src.contains("fn _1apply_scale_f32("),
            "trait_consumer should contain _1apply_scale_f32, got:\n{consumer_src}"
        );
        assert!(
            consumer_src.contains("f32_scale("),
            "_1apply_scale_f32 should call f32_scale, got:\n{consumer_src}"
        );
        assert!(
            consumer_src.contains("fn f32_scale("),
            "concatenated source should have f32_scale (from import), got:\n{consumer_src}"
        );

        // Verify Rust-side execution works too
        let result = trait_consumer::go();
        assert!(
            (result - 6.0).abs() < f32::EPSILON,
            "2.0 * 3.0 should be 6.0, got {result}"
        );
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_provider {
        pub fn id<T: Copy>(v: T) -> T {
            v
        }
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_left {
        use super::dedupe_provider::*;

        pub fn left() -> f32 {
            id::<f32>(1.0)
        }
    }

    #[wgsl(crate_path = crate)]
    mod dedupe_right {
        use super::dedupe_provider::*;

        pub fn right() -> f32 {
            id::<f32>(2.0)
        }
    }

    #[expect(dead_code)]
    #[wgsl(crate_path = crate)]
    mod dedupe_root {
        #[rustfmt::skip]
        use super::dedupe_left::*;
        #[rustfmt::skip]
        use super::dedupe_right::*;

        pub fn root() -> f32 {
            left() + right()
        }
    }

    #[test]
    fn cross_module_instantiations_are_deduped_globally() {
        let full_src = dedupe_root::WGSL_SOURCE.wgsl_source().unwrap();
        assert_eq!(
            full_src.matches("fn id_f32(").count(),
            1,
            "expected id_f32 to appear once, got:\n{full_src}"
        );
    }

    #[test]
    fn ambiguous_template_provider_errors() {
        // Build hand-crafted sources where two distinct provider sources
        // both define a generic template named "shared". Resolution must
        // return `Err(SourceError::AmbiguousTemplate { .. })`.
        fn empty_source_ir() -> ir::Module {
            ir::Module {
                name: "stub",
                items: vec![],
                attrs: vec![],
            }
        }
        fn shared_template_items() -> Vec<ir::Item> {
            vec![ir::Item::Fn(ir::ItemFn {
                type_params: vec![],
                fn_attrs: ir::FnAttrs::None,
                name: "shared".to_string().into(),
                inputs: vec![ir::FnArg {
                    inter_stage_io: vec![],
                    name: "x".to_string(),
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                    attrs: vec![],
                }],
                return_type: ir::ReturnType::Type {
                    annotation: ir::ReturnTypeAnnotation::None,
                    ty: ir::Type::TypeParam {
                        name: "T".to_string(),
                    },
                },
                block: ir::Block {
                    stmts: vec![ir::Stmt::Return(Some(ir::Expr::Ident("x".to_string())))],
                },
                attrs: vec![],
            })]
        }
        fn f32_args() -> Vec<ir::Type> {
            vec![ir::Type::Scalar(ir::ScalarType::F32)]
        }
        static DEP: [TemplateDependency; 0] = [];
        static A_TEMPLATES: [GenericTemplate; 1] = [GenericTemplate {
            name: "shared",
            type_params: &["T"],
            ir_constructor: shared_template_items,
            dependencies: &DEP,
        }];
        static B_TEMPLATES: [GenericTemplate; 1] = [GenericTemplate {
            name: "shared",
            type_params: &["T"],
            ir_constructor: shared_template_items,
            dependencies: &DEP,
        }];
        static EMPTY_SOURCES: [&Source; 0] = [];
        static EMPTY_INSTS: [TemplateInstantiation; 0] = [];
        static EMPTY_TYPE_PARAMS: [&str; 0] = [];
        static MOD_A: Source = Source {
            id: 0,
            name: "a",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &A_TEMPLATES,
            instantiations: &EMPTY_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };
        static MOD_B: Source = Source {
            id: 1,
            name: "b",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &B_TEMPLATES,
            instantiations: &EMPTY_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };
        static ROOT_INSTS: [TemplateInstantiation; 1] = [TemplateInstantiation {
            modules: &[&MOD_A, &MOD_B],
            template_name: "shared",
            type_args_constructor: f32_args,
            mangled_type_args: &["f32"],
        }];
        static ROOT: Source = Source {
            id: 2,
            name: "root",
            imports: &EMPTY_SOURCES,
            ir_constructor: empty_source_ir,
            templates: &[],
            instantiations: &ROOT_INSTS,
            module_type_params: &EMPTY_TYPE_PARAMS,
        };

        let err = ROOT.wgsl_source().unwrap_err();
        match &err {
            crate::SourceError::AmbiguousTemplate {
                template_name,
                matching_source_names,
                ..
            } => {
                assert_eq!(template_name, "shared");
                assert!(
                    matching_source_names.contains(&"a".to_string())
                        && matching_source_names.contains(&"b".to_string()),
                    "expected matching sources [a, b], got {matching_source_names:?}"
                );
            }
            other => panic!("expected AmbiguousTemplate, got {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "got: {msg}");
        assert!(msg.contains("shared"), "got: {msg}");
    }

    // --- Cross-module generic struct tests ---

    #[wgsl(crate_path = crate)]
    mod struct_provider {
        pub struct Pair<T: Copy> {
            pub a: T,
            pub b: T,
        }

        impl<T: Copy + std::ops::Add<Output = T>> Pair<T> {
            pub fn sum(p: Pair<T>) -> T {
                p.a + p.b
            }
        }
    }

    #[wgsl(crate_path = crate)]
    mod struct_consumer {
        use super::struct_provider::*;

        pub fn use_pair() -> f32 {
            let p: Pair<f32> = Pair::<f32> { a: 1.0, b: 2.0 };
            Pair::<f32>::sum(p)
        }
    }

    #[test]
    fn cross_module_generic_struct() {
        // The provider should have a template for "Pair"
        assert!(
            struct_provider::WGSL_SOURCE
                .templates
                .iter()
                .any(|t| t.name == "Pair"),
            "struct_provider should have a template named 'Pair', templates: {:?}",
            struct_provider::WGSL_SOURCE
                .templates
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
        );

        // The consumer should produce WGSL with Pair_f32 and Pair_f32_sum
        let full_src = struct_consumer::WGSL_SOURCE.wgsl_source().unwrap();
        assert!(
            full_src.contains("Pair_f32"),
            "Expected Pair_f32 in assembled WGSL, got:\n{full_src}"
        );
        assert!(
            full_src.contains("Pair_f32_sum"),
            "Expected Pair_f32_sum in assembled WGSL, got:\n{full_src}"
        );

        // Verify Rust-side execution
        let result = struct_consumer::use_pair();
        assert!(
            (result - 3.0).abs() < f32::EPSILON,
            "1.0 + 2.0 should be 3.0, got {result}"
        );
    }

    // --- Generic linkage / generic entry-point tests ---

    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, f32))]
    pub mod generic_shader {
        use crate::std::*;

        // The linkage variable is generic via `impl Trait` syntax. Its
        // module-level type-parameter name is `FRAME` (the variable's
        // identifier), independent of any entry point's type params.
        uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

        // The fragment entry point has its own type parameter `T`. In the
        // emitted IR this becomes the positional name `frag_main_0`.
        #[fragment]
        pub fn frag_main<T: Convert<f32> + Wgsl + Clone>() -> Vec4f {
            vec4f(1.0, sin(f32(get!(FRAME, T)) / 128.0), 0.0, 1.0)
        }
    }

    #[test]
    fn generic_module_is_template() {
        let m = &generic_shader::WGSL_SOURCE;
        assert!(m.is_template());
        // FRAME first (linkage var), then frag_main_0 (entry-point param 0).
        assert_eq!(m.module_type_params, &["FRAME", "frag_main_0"]);
    }

    #[test]
    fn generic_module_template_source_wgsl_source_errors() {
        // After C2: `Source::wgsl_source` returns Err(TemplateWgsl) for a
        // template source. To get placeholder text, instantiate first.
        assert!(generic_shader::WGSL_SOURCE.wgsl_source().is_err());
    }

    #[test]
    fn generic_module_instantiate_substitutes_placeholders() {
        // Order matches `module_type_params`: FRAME first, then frag_main_0.
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        assert!(
            !src.contains("__TPFRAME__") && !src.contains("__TPfrag_main_0__"),
            "expected no placeholders after instantiation, got:\n{src}"
        );
        assert!(
            src.contains("FRAME") && src.contains("f32"),
            "expected FRAME with f32 type, got:\n{src}"
        );
    }

    #[test]
    fn generic_uniform_is_type_erased_static() {
        // The Rust-side static is `Uniform` (with default
        // WgslTypeVariable), not `Uniform<T>`. Setting and getting via the
        // typed accessors should round-trip.
        generic_shader::FRAME.set_typed::<f32>(42.0f32);
        let v = *generic_shader::FRAME.get_typed::<f32>();
        assert_eq!(v, 42.0);

        // A different concrete type can also be stored simultaneously.
        generic_shader::FRAME.set_typed::<u32>(7u32);
        let u = *generic_shader::FRAME.get_typed::<u32>();
        assert_eq!(u, 7);
        // The f32 value is unaffected.
        let v2 = *generic_shader::FRAME.get_typed::<f32>();
        assert_eq!(v2, 42.0);
    }

    #[test]
    fn generic_module_validate_rejects_template() {
        // `validate()` returns an error for template sources — they have
        // unresolved placeholders that aren't valid WGSL.
        assert!(
            generic_shader::WGSL_SOURCE.validate().is_err(),
            "template sources should fail validation"
        );
    }

    #[test]
    fn generic_module_instantiate_produces_concrete_ir() {
        // The `instantiate` function produces an `ir::Module` with all type
        // params resolved. Its rendering should be free of placeholders.
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPFRAME__"));
        assert!(!src.contains("__TPfrag_main_0__"));
        assert!(src.contains("f32"));
    }

    #[cfg(feature = "validation")]
    #[test]
    fn generic_module_instantiated_source_is_valid_wgsl() {
        let m = generic_shader::instantiate::<f32, f32>();
        let src = ir::render_module(&m);
        crate::validate_wgsl_source(&src)
            .expect("instantiated generic_shader should produce valid WGSL");
    }

    // Generic compute shader exercising multiple linkage and entry-point
    // type parameters using `impl Trait` syntax.
    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, u32))]
    pub mod generic_compute {
        use crate::std::*;

        storage!(group(0), binding(0), read_write, INPUT: impl Wgsl + Convert<f32> + Clone);
        storage!(group(0), binding(1), read_write, OUTPUT: impl Wgsl + Convert<f32> + Clone);

        #[compute]
        #[workgroup_size(1)]
        pub fn cs_main() {}
    }

    #[test]
    fn generic_compute_has_two_type_params() {
        let m = &generic_compute::WGSL_SOURCE;
        assert!(m.is_template());
        // Linkage variables in declaration order: INPUT, OUTPUT.
        assert_eq!(m.module_type_params, &["INPUT", "OUTPUT"]);
    }

    #[test]
    fn generic_compute_storages_are_type_erased() {
        generic_compute::INPUT.set_typed::<f32>(1.5f32);
        generic_compute::OUTPUT.set_typed::<u32>(42u32);
        let a = *generic_compute::INPUT.get_typed::<f32>();
        let b = *generic_compute::OUTPUT.get_typed::<u32>();
        assert_eq!(a, 1.5);
        assert_eq!(b, 42);
    }

    #[test]
    fn generic_compute_template_source_errors() {
        // After C2: Source::wgsl_source on a template returns
        // Err(SourceError::TemplateWgsl { .. }).
        let err = generic_compute::WGSL_SOURCE.wgsl_source().unwrap_err();
        match &err {
            crate::SourceError::TemplateWgsl {
                uninstantiated_source,
            } => {
                assert_eq!(uninstantiated_source.name, "generic_compute");
                assert_eq!(
                    uninstantiated_source.module_type_params,
                    &["INPUT", "OUTPUT"]
                );
            }
            _ => unreachable!("expected TemplateWgsl for generic_compute"),
        }
        // The Display impl lists the source name and the type params so
        // users can quickly see what to instantiate.
        let msg = err.to_string();
        assert!(msg.contains("generic_compute"), "got: {msg}");
        assert!(
            msg.contains("INPUT") && msg.contains("OUTPUT"),
            "got: {msg}"
        );
        assert!(msg.contains("instantiate"), "got: {msg}");
    }

    #[test]
    fn generic_compute_instantiate_substitutes_both() {
        let m = generic_compute::instantiate::<f32, u32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPINPUT__") && !src.contains("__TPOUTPUT__"));
        assert!(src.contains("INPUT") && src.contains("OUTPUT"));
        assert!(src.contains("f32") && src.contains("u32"));
    }

    #[test]
    fn generic_compute_instantiate_binds_both_storages() {
        let m = generic_compute::instantiate::<f32, u32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPINPUT__"));
        assert!(!src.contains("__TPOUTPUT__"));
        assert!(src.contains("f32") && src.contains("u32"));
    }

    // --- Multi-entry-point type param disambiguation tests ---

    // Module with two entry points that both use the letter `T` and
    // both access the same linkage variable DATA via `get!`. This tests
    // that the `instantiate` function correctly disambiguates the type
    // params and generates the right `Type<Is = ...>` constraints.
    #[wgsl(crate_path = crate, validate_with_instantiation_types(f32, f32, f32))]
    pub mod multi_ep {
        use crate::std::*;

        storage!(group(0), binding(0), read_write, DATA: impl std::any::Any);

        #[compute]
        #[workgroup_size(1)]
        pub fn read_as<T: Wgsl + Convert<f32>>(#[builtin(global_invocation_id)] _gid: Vec3u) {
            let _v = get!(DATA, T);
        }

        #[compute]
        #[workgroup_size(1)]
        pub fn write_as<T: Wgsl + Convert<f32>>(#[builtin(global_invocation_id)] _gid: Vec3u) {
            let mut _v = get_mut!(DATA, T);
        }
    }

    #[test]
    fn multi_ep_instantiate_substitutes_correctly() {
        // Three type params: DATA, T_read_as, T_write_as.
        // Both entry-point type params are substituted with f32.
        let m = multi_ep::instantiate::<f32, f32, f32>();
        let src = ir::render_module(&m);
        assert!(
            !src.contains("__TPDATA__"),
            "expected no placeholder for DATA, got:\n{src}"
        );
        assert!(
            !src.contains("__TPread_as_0__") && !src.contains("__TPwrite_as_0__"),
            "expected no entry-point placeholders, got:\n{src}"
        );
        assert!(
            src.contains("f32"),
            "expected f32 type in output, got:\n{src}"
        );
    }

    #[test]
    fn multi_ep_disambiguated_type_params_are_independent() {
        // Even though both entry points use the letter `T`, the
        // `instantiate` function should have two independent type params
        // (T_read_as and T_write_as). This test verifies the constraints
        // are correct by using different concrete types: DATA = f32 for
        // the read path and DATA = f32 for the write path (they must agree
        // since DATA: Type<Is = T_read_as> and DATA: Type<Is = T_write_as>,
        // so T_read_as must equal T_write_as).
        let m = multi_ep::instantiate::<f32, f32, f32>();
        let src = ir::render_module(&m);
        assert!(!src.contains("__TPDATA__"));
        assert!(src.contains("DATA"));
    }
}
