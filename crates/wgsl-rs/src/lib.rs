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
    pub imports: &'static [&'static Module],
    pub source: &'static str,
}

impl Module {
    pub fn wgsl_source(&self) -> String {
        let mut src = String::new();
        for module in self.imports.iter() {
            src.push_str(&module.wgsl_source());
        }
        src.push_str(self.source);
        src
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
        assert_eq!(
            "const THREE : u32 = 3;fn add_three_to_x_minus_y(x : u32, y : u32) -> u32
{ let i : u32 = (x - y) + THREE; return i; }fn main() { let _u = add_three_to_x_minus_y(1337, 666); }",
            &source
        );
    }
}
