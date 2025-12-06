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

macro_rules! impl_swizzles {
    // Usage: impl_swizzles!(Vec2, x y);
    ($VecN:ident, $($c:ident)+) => {
        // Collect the identifiers into a vector we can iterate over
        let idents = [$(stringify!($c)),+];
    };
}

#[repr(transparent)]
pub struct Vec2<T: IsScalar> {
    inner: T::Vector2,
}

#[repr(transparent)]
pub struct Vec3<T: IsScalar> {
    inner: T::Vector3,
}

#[repr(transparent)]
pub struct Vec4<T: IsScalar> {
    inner: T::Vector4,
}

pub type Vec2f = Vec2<f32>;
pub type Vec3f = Vec3<f32>;
pub type Vec4f = Vec4<f32>;

pub const fn vec2f(x: f32, y: f32) -> Vec2f {
    Vec2 {
        inner: glam::Vec2::new(x, y),
    }
}

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
