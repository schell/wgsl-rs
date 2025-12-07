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

/// vector! generates the N-vector type and constructors
macro_rules! vector {
    // Example: vector!(2, [x, y]);
    ($n:literal, [$($field:ident),+]) => {
        paste::paste! {
            /// A vector.
            #[repr(transparent)]
            #[derive(Clone, Copy)]
            pub struct [<Vec $n>]<T: IsScalar> {
                inner: T::[<Vector $n>],
            }

            /// Alias for `Vec{N}::vec{n}`, since that syntax cannot produce valid WGSL.
            pub fn [<vec $n>]<T: IsScalar>($($field: T),+) -> [<Vec$n>]<T> {
                T::[<vec $n>]($($field),+)
            }
        }
    }
}

/// aliases! generates aliases for the different component types.
macro_rules! aliases {
    ($n: literal, $t:ident, $ty:ty, $glam_ty:ty, [$($field:ident),+]) => {
        paste::paste! {
            pub type [<Vec $n $t>] = [<Vec $n>]<$ty>;
            /// Vector constructor
            pub const fn [<vec $n $t>]($($field: $ty),+) -> [<Vec $n $t>] {
                [<Vec $n>] {
                    inner: glam::[<$glam_ty $n>]::new($($field),+),
                }
            }
        }
    }
}

vector!(2, [x, y]);
aliases!(2, f, f32, Vec, [x, y]);
aliases!(2, i, i32, IVec, [x, y]);
aliases!(2, u, u32, UVec, [x, y]);
aliases!(2, b, bool, BVec, [x, y]);
wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [x, y]);

vector!(3, [x, y, z]);
aliases!(3, f, f32, Vec, [x, y, z]);
aliases!(3, i, i32, IVec, [x, y, z]);
aliases!(3, u, u32, UVec, [x, y, z]);
aliases!(3, b, bool, BVec, [x, y, z]);
wgsl_rs_macros::swizzle!(
    Vec3f,
    [f32, Vec2f, Vec3f],
    [vec2f, vec3f],
    [x, y, z],
    [r, g, b]
);
wgsl_rs_macros::swizzle!(
    Vec3f,
    [f32, Vec2f, Vec3f],
    [vec2f, vec3f],
    [x, y, z],
    [x, y, z]
);
wgsl_rs_macros::swizzle!(
    Vec3i,
    [i32, Vec2i, Vec3i],
    [vec2i, vec3i],
    [x, y, z],
    [r, g, b]
);
wgsl_rs_macros::swizzle!(
    Vec3i,
    [i32, Vec2i, Vec3i],
    [vec2i, vec3i],
    [x, y, z],
    [x, y, z]
);
wgsl_rs_macros::swizzle!(
    Vec3u,
    [u32, Vec2u, Vec3u],
    [vec2u, vec3u],
    [x, y, z],
    [r, g, b]
);
wgsl_rs_macros::swizzle!(
    Vec3u,
    [u32, Vec2u, Vec3u],
    [vec2u, vec3u],
    [x, y, z],
    [x, y, z]
);
wgsl_rs_macros::swizzle!(
    Vec3b,
    [bool, Vec2b, Vec3b],
    [vec2b, vec3b],
    [x, y, z],
    [r, g, b]
);
wgsl_rs_macros::swizzle!(
    Vec3b,
    [bool, Vec2b, Vec3b],
    [vec2b, vec3b],
    [x, y, z],
    [x, y, z]
);

vector!(4, [x, y, z, w]);
aliases!(4, f, f32, Vec, [x, y, z, w]);
aliases!(4, i, i32, IVec, [x, y, z, w]);
aliases!(4, u, u32, UVec, [x, y, z, w]);
aliases!(4, b, bool, BVec, [x, y, z, w]);
wgsl_rs_macros::swizzle!(
    Vec4f,
    [f32, Vec2f, Vec3f, Vec4f],
    [vec2f, vec3f, vec4f],
    [x, y, z, w],
    [r, g, b, a]
);
wgsl_rs_macros::swizzle!(
    Vec4f,
    [f32, Vec2f, Vec3f, Vec4f],
    [vec2f, vec3f, vec4f],
    [x, y, z, w],
    [x, y, z, w]
);
wgsl_rs_macros::swizzle!(
    Vec4i,
    [i32, Vec2i, Vec3i, Vec4i],
    [vec2i, vec3i, vec4i],
    [x, y, z, w],
    [r, g, b, a]
);
wgsl_rs_macros::swizzle!(
    Vec4i,
    [i32, Vec2i, Vec3i, Vec4i],
    [vec2i, vec3i, vec4i],
    [x, y, z, w],
    [x, y, z, w]
);
wgsl_rs_macros::swizzle!(
    Vec4u,
    [u32, Vec2u, Vec3u, Vec4u],
    [vec2u, vec3u, vec4u],
    [x, y, z, w],
    [r, g, b, a]
);
wgsl_rs_macros::swizzle!(
    Vec4u,
    [u32, Vec2u, Vec3u, Vec4u],
    [vec2u, vec3u, vec4u],
    [x, y, z, w],
    [x, y, z, w]
);
wgsl_rs_macros::swizzle!(
    Vec4b,
    [bool, Vec2b, Vec3b, Vec4b],
    [vec2b, vec3b, vec4b],
    [x, y, z, w],
    [r, g, b, a]
);
wgsl_rs_macros::swizzle!(
    Vec4b,
    [bool, Vec2b, Vec3b, Vec4b],
    [vec2b, vec3b, vec4b],
    [x, y, z, w],
    [x, y, z, w]
);

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
/// In Rust, use `const` for compile-time constant values that are inlined wherever used.
/// Use `static` for global variables with a fixed memory address, which may have interior mutability.
/// 
/// - Prefer `const` for simple values, especially primitives and when you want zero runtime overhead.
/// - Use `static` for values that need a stable address, or for global state (including with `Mutex`, `Atomic*`, etc).
/// - `static mut` is unsafe and should be avoided; prefer safe wrappers like `Mutex` or `Atomic*`.
/// - `const` can be used in patterns, types, and generics; `static` cannot.
/// - Both can be used for sharing data between threads, but only `static` can be mutable (with synchronization).
