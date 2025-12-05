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

pub mod std {
    //! The WGSL standard library.
    //!
    //! This module is only imported into Rust code.
    //!
    //! The glob-imported import statement `use wgsl_rs::std::*` pulls in
    //! WGSL types and functions into Rust that are part of the global scope
    //! in WGSL, but don't already exist in Rust's global scope.
    //!
    //! These types include (but are not limited to) vector types like `Vec2f`, `Vec3f`, etc.
    //! and constructors like `vec2f` and `vec3f`.

    pub trait IsScalar: Sized {
        type Vector2;
        type Vector3;
        type Vector4;

        fn vec2(x: Self, y: Self) -> Vec2<Self>;
        fn vec3(x: Self, y: Self, z: Self) -> Vec3<Self>;
        fn vec4(x: Self, y: Self, z: Self, w: Self) -> Vec4<Self>;
    }

    pub struct Vec2<T: IsScalar> {
        inner: T::Vector2,
    }

    pub struct Vec3<T: IsScalar> {
        inner: T::Vector3,
    }

    pub struct Vec4<T: IsScalar> {
        inner: T::Vector4,
    }
    pub type Vec2f = Vec2<f32>;
    pub type Vec3f = Vec3<f32>;
    pub type Vec4f = Vec4<f32>;
    pub const fn vec4f(x: f32, y: f32, z: f32, w: f32) -> Vec4f {
        Vec4 {
            inner: glam::Vec4::new(x, y, z, w),
        }
    }

    macro_rules! impl_is_scalar {
        ($ty:ty, $vec2:ty, $vec3:ty, $vec4:ty) => {
            impl IsScalar for $ty {
                type Vector2 = $vec2;
                type Vector3 = $vec3;
                type Vector4 = $vec4;

                fn vec2(x: Self, y: Self) -> Vec2<Self> {
                    Vec2 {
                        inner: <$vec2>::new(x, y),
                    }
                }

                fn vec3(x: Self, y: Self, z: Self) -> Vec3<Self> {
                    Vec3 {
                        inner: <$vec3>::new(x, y, z),
                    }
                }

                fn vec4(x: Self, y: Self, z: Self, w: Self) -> Vec4<Self> {
                    Vec4 {
                        inner: <$vec4>::new(x, y, z, w),
                    }
                }
            }
        };
    }

    impl_is_scalar!(f32, glam::Vec2, glam::Vec3, glam::Vec4);
    impl_is_scalar!(i32, glam::IVec2, glam::IVec3, glam::IVec4);
    impl_is_scalar!(u32, glam::UVec2, glam::UVec3, glam::UVec4);
    impl_is_scalar!(bool, glam::BVec2, glam::BVec3, glam::BVec4);
}

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
