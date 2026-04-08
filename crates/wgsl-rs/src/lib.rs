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
}

impl Module {
    /// Returns the concatenated WGSL source of this module and all its imports.
    ///
    /// This recursively collects source lines from all imported modules first,
    /// then appends this module's source lines.
    pub fn wgsl_source(&self) -> Vec<&'static str> {
        let mut src = vec![];
        for module in self.imports.iter() {
            src.extend(module.wgsl_source());
        }
        src.extend(self.source);
        src
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
        let expected = vec![
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
        ];
        assert_eq!(&expected, &source);

        // Verify that imported module sources are not duplicated.
        // Module C imports B, which imports A. Module A's source should
        // appear exactly once in the concatenated output.
        let a_count = source
            .iter()
            .filter(|line| a::WGSL_MODULE.source.contains(line))
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
}
