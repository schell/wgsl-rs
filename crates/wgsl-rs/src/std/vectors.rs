//! Vector implementations.

/// Trait for accessing the underlying type of a vector.
pub trait ScalarCompOfVec<const N: usize>: Sized {
    type Output: Copy + Clone;

    fn vec_from_array(array: [Self; N]) -> Vec<N, Self>;
    fn vec_to_array(vec: Vec<N, Self>) -> [Self; N];
}

/// An `N` dimensional vector.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Vec<const N: usize, T: ScalarCompOfVec<N>> {
    pub(crate) inner: <T as ScalarCompOfVec<N>>::Output,
}

pub type Vec2<T> = Vec<2, T>;
pub type Vec3<T> = Vec<3, T>;
pub type Vec4<T> = Vec<4, T>;

macro_rules! vec_constructor {
    ($n:literal, [$($comps:ident),+]) => {
        impl<T: ScalarCompOfVec<$n>> Vec<$n, T> {
            paste::paste! {
                pub fn [<vec $n>]($($comps: T),+) -> Vec<$n, T> {
                    T::vec_from_array([$($comps),+])
                }
            }
        }
    }
}

vec_constructor!(2, [x, y]);
vec_constructor!(3, [x, y, z]);
vec_constructor!(4, [x, y, z, w]);

/// vector! generates type aliases and an N-vector const constructor.
macro_rules! vector {
    // Example: vector!(2, [x, y]);
    ($n:literal, $ty:ty, $ty_suffix:ident, $glam_ty:ty, [$($fields:ident),+]) => {
        paste::paste! {
            impl ScalarCompOfVec<$n> for $ty {
                type Output = $glam_ty;

                fn vec_from_array(array: [Self; $n]) -> Vec<$n, $ty> {
                    Vec {
                        inner: $glam_ty::from_array(array)
                    }
                }

                fn vec_to_array(vec: Vec<$n, $ty>) -> [$ty; $n] {
                    vec.inner.into()
                }
            }

            #[doc = concat!("Concrete type alias for a ", $n, "dimensional vector of ", stringify!($ty), "scalar components.")]
            pub type [<Vec $n $ty_suffix>] = Vec<$n, $ty>;

            #[doc = concat!("Constructor for a ", $n, " dimensional vector of ", stringify!($ty), " scalar components.")]
            pub const fn [<vec $n $ty_suffix>]($($fields: $ty),+) -> Vec<$n, $ty> {
                Vec {
                    inner: $glam_ty::new($($fields),+)
                }
            }
        }
    }
}

vector!(2, f32, f, glam::Vec2, [x, y]);
vector!(2, i32, i, glam::IVec2, [x, y]);
vector!(2, u32, u, glam::UVec2, [x, y]);
vector!(2, bool, b, glam::BVec2, [x, y]);

wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [x, y]);

vector!(3, f32, f, glam::Vec3, [x, y, z]);
vector!(3, i32, i, glam::IVec3, [x, y, z]);
vector!(3, u32, u, glam::UVec3, [x, y, z]);
vector!(3, bool, b, glam::BVec3, [x, y, z]);
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

vector!(4, f32, f, glam::Vec4, [x, y, z, w]);
vector!(4, i32, i, glam::IVec4, [x, y, z, w]);
vector!(4, u32, u, glam::UVec4, [x, y, z, w]);
vector!(4, bool, b, glam::BVec4, [x, y, z, w]);
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

/// Creates:
/// * From<glam::VecN> for VecN<T>
macro_rules! impl_from_vec {
    ($from:ty, $to:ident, $ty:ty, $n:literal) => {
        impl From<$from> for $to {
            fn from(value: $from) -> Self {
                $to { inner: value }
            }
        }

        impl From<[$ty; $n]> for $to {
            fn from(array: [$ty; $n]) -> Self {
                <$ty>::vec_from_array(array)
            }
        }

        impl From<$to> for [$ty; $n] {
            fn from(vec: $to) -> [$ty; $n] {
                <$ty>::vec_to_array(vec)
            }
        }
    };
}
impl_from_vec!(glam::Vec2, Vec2f, f32, 2);
impl_from_vec!(glam::Vec3, Vec3f, f32, 3);
impl_from_vec!(glam::Vec4, Vec4f, f32, 4);
impl_from_vec!(glam::IVec2, Vec2i, i32, 2);
impl_from_vec!(glam::IVec3, Vec3i, i32, 3);
impl_from_vec!(glam::IVec4, Vec4i, i32, 4);
impl_from_vec!(glam::UVec2, Vec2u, u32, 2);
impl_from_vec!(glam::UVec3, Vec3u, u32, 3);
impl_from_vec!(glam::UVec4, Vec4u, u32, 4);
impl_from_vec!(glam::BVec2, Vec2b, bool, 2);
impl_from_vec!(glam::BVec3, Vec3b, bool, 3);
impl_from_vec!(glam::BVec4, Vec4b, bool, 4);
