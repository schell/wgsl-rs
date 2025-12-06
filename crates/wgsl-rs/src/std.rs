//! The WGSL standard library, for Rust.
//!
//! When using the module-level `wgsl` proc-macro, this module must only be glob-imported.
//!
//! The glob-imported import statement `use wgsl_rs::std::*` pulls in all the
//! WGSL types and functions into Rust that are part of the global scope
//! in WGSL, but don't already exist in Rust's global scope.
//!
//! These types include (but are not limited to) vector types like `Vec2f`, `Vec3f`, etc.
//! and constructors like `vec2`, `vec2f` and `vec3i`, etc.

pub trait IsScalar: Sized {
    type Vector2;
    type Vector3;
    type Vector4;

    fn vec2(x: Self, y: Self) -> Vec2<Self>;
    fn vec3(x: Self, y: Self, z: Self) -> Vec3<Self>;
    fn vec4(x: Self, y: Self, z: Self, w: Self) -> Vec4<Self>;
}

// #[repr(transparent)]
// pub struct Vec2<T: IsScalar> {
//     inner: T::Vector2,
// }
// pub type Vec2f = Vec2<f32>;
// pub const fn vec2f(x: f32, y: f32) -> Vec2f {
//     Vec2 {
//         inner: glam::Vec2::new(x, y),
//     }
// }
// wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [r, g]);
// wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [x, y]);

macro_rules! vector {
    // Example: vector!(2, f, ty:f32, [x, y], [r, g]);
    ($n:literal, $t:ident, $ty:ty, [$($field:ident),+], [$($swizzle:ident),+]) => {
        /// A vector in N dimensions.
        #[repr(transparent)]
        pub struct Vec$n<T: IsScalar> {
            inner: T::Vector$n,
        }
        pub type Vec$n$t = Vec$n<$ty>;
        /// Vector constructor
        pub const fn vec$n$t($($field: $ty),+) -> Vec$n$t {
            Vec$n {
                inner: glam::Vec$n::new($($field),+),
            }
        }
        wgsl_rs_macros::swizzle!(Vec$n$t, [$ty, Vec$n$t], [vec$n$t], [$($field),+], [$($swizzle),+]);
        wgsl_rs_macros::swizzle!(Vec$n$t, [$ty, Vec$n$t], [vec$n$t], [$($field),+], [$($field),+]);
    }
}

vector!(2, f, f32, [x, y], [r, g]);

#[repr(transparent)]
pub struct Vec3<T: IsScalar> {
    inner: T::Vector3,
}
pub type Vec3f = Vec3<f32>;
pub const fn vec3f(x: f32, y: f32, z: f32) -> Vec3f {
    Vec3 {
        inner: glam::Vec3::new(x, y, z),
    }
}

#[repr(transparent)]
pub struct Vec4<T: IsScalar> {
    inner: T::Vector4,
}
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

pub fn vec4<T: IsScalar>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    T::vec4(x, y, z, w)
}
