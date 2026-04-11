//! WGSL in Rust.
pub use wgsl_rs_macros::{wgsl, wgsl_allow};

/// A WGSL "module".
///
/// WGSL doesn't support importing modules, but `wgsl-rs` does,
/// with limitations. Specifically, `wgsl-rs` only supports glob
/// importing other modules.
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
pub struct Module {
    /// Name of the module.
    pub name: &'static str,

    /// Imports of other WGSL modules.
    pub imports: &'static [&'static Module],

    /// Source of the module, by line.
    pub source: &'static [&'static str],

    /// Generic function templates defined in this module.
    ///
    /// These are WGSL function sources with `__TP{name}__` placeholders for
    /// type parameters. Consuming modules reference these via
    /// `instantiations` to produce monomorphized functions at source-assembly
    /// time.
    pub templates: &'static [GenericTemplate],

    /// Cross-module template instantiations.
    ///
    /// Each entry references a template from an imported module and provides
    /// the concrete type names to substitute for the type parameters. These
    /// are resolved at source-assembly time in `wgsl_source()`.
    pub instantiations: &'static [TemplateInstantiation],
}

/// A generic function template with placeholder type parameters.
pub struct GenericTemplate {
    /// The generic function name (e.g., `"shade_fragment"`).
    pub name: &'static str,
    /// Type parameter names (e.g., `["M", "L", "N"]`).
    pub type_params: &'static [&'static str],
    /// WGSL source with `__TP{name}__` placeholders for each type parameter.
    pub wgsl_source: &'static str,
    /// Transitive generic function calls within this template.
    ///
    /// Each entry records a call to another generic function, with a mapping
    /// from the callee's type params back to this template's type params.
    pub dependencies: &'static [TemplateDependency],
}

/// A transitive dependency from one generic template to another.
pub struct TemplateDependency {
    /// Name of the generic function being called.
    pub callee: &'static str,
    /// Maps each of the callee's type params to one of the caller's type
    /// params, by index. For example, if `quadruple<T>` calls `double::<T>()`,
    /// the mapping is `&[0]` (double's 0th param = quadruple's 0th param).
    pub type_param_mapping: &'static [usize],
}

/// A request to instantiate a generic template with concrete types.
pub struct TemplateInstantiation {
    /// Candidate imported modules that may contain the template.
    pub modules: &'static [&'static Module],
    /// The generic function name to instantiate.
    pub template_name: &'static str,
    /// Concrete type names to substitute for each type parameter.
    pub type_args: &'static [&'static str],
}

impl Module {
    /// Returns the concatenated WGSL source of this module and all its imports,
    /// including any cross-module template instantiations.
    ///
    /// This recursively collects source lines from all imported modules first,
    /// then appends this module's source lines, then appends any instantiated
    /// template functions (including transitive dependencies).
    pub fn wgsl_source(&self) -> Vec<String> {
        let mut instantiated: ::std::collections::HashSet<(String, String, Vec<String>)> =
            ::std::collections::HashSet::new();
        let mut src = vec![];
        self.collect_wgsl_source(&mut src, &mut instantiated);
        src
    }

    fn collect_wgsl_source(
        &self,
        out: &mut Vec<String>,
        seen: &mut ::std::collections::HashSet<(String, String, Vec<String>)>,
    ) {
        for module in self.imports.iter() {
            module.collect_wgsl_source(out, seen);
        }
        for line in self.source {
            out.push(line.to_string());
        }
        for inst in self.instantiations {
            let type_args: Vec<String> = inst.type_args.iter().map(|s| s.to_string()).collect();
            Self::instantiate_template(inst.modules, inst.template_name, &type_args, out, seen);
        }
    }

    /// Instantiate a single template and recursively instantiate its
    /// dependencies. Tracks already-instantiated `(name, type_args)` pairs
    /// to avoid duplicates.
    fn instantiate_template(
        modules: &[&Module],
        template_name: &str,
        type_args: &[String],
        out: &mut Vec<String>,
        seen: &mut ::std::collections::HashSet<(String, String, Vec<String>)>,
    ) {
        let Some(module) = modules
            .iter()
            .copied()
            .find(|m| m.templates.iter().any(|t| t.name == template_name))
        else {
            return; // Template not found in imported modules
        };

        let key = (
            module.name.to_string(),
            template_name.to_string(),
            type_args.to_vec(),
        );
        if !seen.insert(key) {
            return; // Already instantiated
        }

        let Some(template) = module.templates.iter().find(|t| t.name == template_name) else {
            return;
        };

        // Recursively instantiate dependencies first (so callees appear before callers)
        for dep in template.dependencies {
            // Map the dependency's type params through the current type_args
            let dep_type_args: Vec<String> = dep
                .type_param_mapping
                .iter()
                .map(|&idx| type_args[idx].clone())
                .collect();
            Self::instantiate_template(&[module], dep.callee, &dep_type_args, out, seen);
        }

        // Substitute placeholders in the template WGSL
        let mut wgsl = template.wgsl_source.to_string();
        for (param_name, arg) in template.type_params.iter().zip(type_args.iter()) {
            let placeholder = format!("__TP{param_name}__");
            wgsl = wgsl.replace(&placeholder, arg);
        }
        out.push(wgsl);
    }
}

#[cfg(feature = "validation")]
impl Module {
    /// Validates the concatenated WGSL source of this module and its imports.
    ///
    /// This method concatenates the WGSL source from this module and all its
    /// imports (recursively), then validates the result using naga.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if validation succeeds, or an error message describing
    /// the validation failure.
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
    /// my_shader::WGSL_MODULE.validate().expect("WGSL validation failed");
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        let source = self.wgsl_source().join("\n");

        // Parse the WGSL source
        let module =
            naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;

        // Run semantic validation
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| e.emit_to_string(&source))?;

        Ok(())
    }
}

pub mod std;

#[cfg(test)]
mod test {
    use crate::wgsl;

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
        let source = c::WGSL_MODULE.wgsl_source();
        c::main();
        let expected: Vec<String> = vec![
            "const THREE: u32 = 3;",
            "fn add_three_to_x_minus_y(x:u32, y:u32) -> u32 {",
            "    let i: u32 = (x-y)+THREE;",
            "    return i;",
            "}",
            "",
            "fn main() {",
            "    let _u = add_three_to_x_minus_y(1337, 666);",
            "}",
            "",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(&expected, &source);

        // Verify that imported module sources are not duplicated.
        // Module C imports B, which imports A. Module A's source should
        // appear exactly once in the concatenated output.
        let a_count = source
            .iter()
            .filter(|line| a::WGSL_MODULE.source.contains(&line.as_str()))
            .count();
        assert_eq!(
            a::WGSL_MODULE.source.len(),
            a_count,
            "module A's source lines should appear exactly once (no duplicates from transitive \
             imports)"
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
        // Verify provider module generates f32_scale
        let provider_src = trait_provider::WGSL_MODULE
            .source
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            provider_src.contains("fn f32_scale("),
            "trait_provider should contain f32_scale, got:\n{provider_src}"
        );

        // Verify consumer module generates apply_scale_f32 that calls f32_scale
        let consumer_src = trait_consumer::WGSL_MODULE
            .source
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            consumer_src.contains("fn apply_scale_f32("),
            "trait_consumer should contain apply_scale_f32, got:\n{consumer_src}"
        );
        assert!(
            consumer_src.contains("f32_scale("),
            "apply_scale_f32 should call f32_scale, got:\n{consumer_src}"
        );

        // Verify the concatenated source has both
        let full_src = trait_consumer::WGSL_MODULE.wgsl_source().join("\n");
        assert!(
            full_src.contains("fn f32_scale(") && full_src.contains("fn apply_scale_f32("),
            "concatenated source should have both functions, got:\n{full_src}"
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
        let full_src = dedupe_root::WGSL_MODULE.wgsl_source().join("\n");
        assert_eq!(
            full_src.matches("fn id_f32(").count(),
            1,
            "expected id_f32 to appear once, got:\n{full_src}"
        );
    }
}
