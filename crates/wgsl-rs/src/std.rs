//! The WGSL standard library, for Rust.
//!
//! When using the module-level `wgsl` proc-macro, this module must only be
//! glob-imported.
//!
//! The glob-imported import statement `use wgsl_rs::std::*` pulls in all the
//! WGSL types and functions into Rust that are part of the global scope
//! in WGSL, but don't already exist in Rust's global scope.
//!
//! These types include (but are not limited to) vector types like `Vec2f`,
//! `Vec3f`, etc. and constructors like `vec2`, `vec2f` and `vec3i`, etc.

use std::sync::{Arc, LazyLock, RwLock};

pub use wgsl_rs_macros::{
    builtin, compute, fragment, input, output, storage, uniform, vertex, workgroup_size,
};

mod numeric_builtin_functions;
pub use numeric_builtin_functions::*;

/// Trait identifying WGSL's concrete scalar types.
pub trait IsScalar: Sized {
    type Vector2;
    type Vector3;
    type Vector4;
    type Matrix2;
    type Matrix3;
    type Matrix4;

    /// Constructs a two dimensional vector.
    fn vec2(x: Self, y: Self) -> Vec2<Self>;

    /// Constructs a three dimensional vector.
    fn vec3(x: Self, y: Self, z: Self) -> Vec3<Self>;

    /// Constructs a four dimensional vector.
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
    ($ty:ty, $vec2:ty, $vec3:ty, $vec4:ty, $mat2:ty, $mat3:ty, $mat4:ty) => {
        impl IsScalar for $ty {
            type Vector2 = $vec2;
            type Vector3 = $vec3;
            type Vector4 = $vec4;
            type Matrix2 = $mat2;
            type Matrix3 = $mat3;
            type Matrix4 = $mat4;

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
impl_is_scalar!(
    f32,
    glam::Vec2,
    glam::Vec3,
    glam::Vec4,
    glam::Mat2,
    glam::Mat3,
    glam::Mat4
);
impl_is_scalar!(i32, glam::IVec2, glam::IVec3, glam::IVec4, (), (), ());
impl_is_scalar!(u32, glam::UVec2, glam::UVec3, glam::UVec4, (), (), ());
impl_is_scalar!(bool, glam::BVec2, glam::BVec3, glam::BVec4, (), (), ());

/// matrix! generates the N×N matrix type
macro_rules! matrix {
    ($n:literal) => {
        paste::paste! {
            /// A square matrix.
            #[repr(transparent)]
            #[derive(Clone, Copy)]
            pub struct [<Mat $n>]<T: IsScalar> {
                inner: T::[<Matrix $n>],
            }
        }
    };
}

/// matrix_aliases! generates aliases for the f32 matrix types and column-wise
/// constructors.
macro_rules! matrix_aliases {
    ($n:literal, $glam_ty:ty, $col_ty:ident, [$($col:ident),+]) => {
        paste::paste! {
            pub type [<Mat $n f>] = [<Mat $n>]<f32>;

            /// Column-wise matrix constructor
            pub const fn [<mat $n x $n f>]($($col: $col_ty),+) -> [<Mat $n f>] {
                [<Mat $n>] {
                    inner: <$glam_ty>::from_cols($($col.inner),+),
                }
            }
        }
    };
}

matrix!(2);
matrix!(3);
matrix!(4);
matrix_aliases!(2, glam::Mat2, Vec2f, [col0, col1]);
matrix_aliases!(3, glam::Mat3, Vec3f, [col0, col1, col2]);
matrix_aliases!(4, glam::Mat4, Vec4f, [col0, col1, col2, col3]);

/// From<glam::MatN> for MatNf...
macro_rules! impl_from_mat {
    ($from:ty, $to:ident) => {
        impl From<$from> for $to {
            fn from(value: $from) -> Self {
                $to { inner: value }
            }
        }
    };
}
impl_from_mat!(glam::Mat2, Mat2f);
impl_from_mat!(glam::Mat3, Mat3f);
impl_from_mat!(glam::Mat4, Mat4f);

/// From<glam::VecN> for VecN<T>...
macro_rules! impl_from_vec {
    ($from:ty, $to:ident) => {
        impl From<$from> for $to {
            fn from(value: $from) -> Self {
                $to { inner: value }
            }
        }
    };
}
impl_from_vec!(glam::Vec2, Vec2f);
impl_from_vec!(glam::Vec3, Vec3f);
impl_from_vec!(glam::Vec4, Vec4f);
impl_from_vec!(glam::IVec2, Vec2i);
impl_from_vec!(glam::IVec3, Vec3i);
impl_from_vec!(glam::IVec4, Vec4i);
impl_from_vec!(glam::UVec2, Vec2u);
impl_from_vec!(glam::UVec3, Vec3u);
impl_from_vec!(glam::UVec4, Vec4u);
impl_from_vec!(glam::BVec2, Vec2b);
impl_from_vec!(glam::BVec3, Vec3b);
impl_from_vec!(glam::BVec4, Vec4b);

/// A shader uniform, backed by a storage buffer on the CPU.
pub struct UniformVariable<T> {
    pub group: u32,
    pub binding: u32,
    pub value: Arc<RwLock<Option<T>>>,
}

pub type Uniform<T> = LazyLock<UniformVariable<T>>;

/// A shader storage buffer, backed by a storage buffer on the CPU.
pub struct StorageVariable<T> {
    pub group: u32,
    pub binding: u32,
    pub read_write: bool,
    pub value: Arc<RwLock<Option<T>>>,
}

pub type Storage<T> = LazyLock<StorageVariable<T>>;

/// Used to provide WGSL type conversion functions like `f32(...)`, etc.
pub trait Convert<T> {
    fn convert(self) -> T;
}

impl<A: Clone + Convert<B>, B> Convert<B> for &Uniform<A> {
    fn convert(self) -> B {
        let guard = self.value.read().expect("could not read value");
        let maybe = guard.as_ref();
        let a = maybe.cloned().expect("uniform value has not been set");
        a.convert()
    }
}

macro_rules! impl_convert_as {
    ($from:ty, $to:ty) => {
        impl Convert<$to> for $from {
            fn convert(self) -> $to {
                self as $to
            }
        }
    };
}
impl_convert_as!(f32, u32);
impl_convert_as!(f32, i32);
impl_convert_as!(i32, f32);
impl_convert_as!(i32, u32);
impl_convert_as!(u32, f32);
impl_convert_as!(u32, i32);

/// Returns the input cast to f32.
pub fn f32(t: impl Convert<f32>) -> f32 {
    t.convert()
}

/// Returns the input cast to u32.
pub fn u32(t: impl Convert<u32>) -> u32 {
    t.convert()
}

/// Returns the input cast to i32.
pub fn i32(t: impl Convert<i32>) -> i32 {
    t.convert()
}
