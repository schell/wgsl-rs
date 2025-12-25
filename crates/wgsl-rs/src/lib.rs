//! WGSL in Rust.
pub use wgsl_rs_macros::wgsl;

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

        const VEC3F_ONE: Vec4f = vec4f(0.0, 1.0, 2.0, 3.0);
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
    }
}
