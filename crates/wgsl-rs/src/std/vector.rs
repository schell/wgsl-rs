//! Vector implementations.
//!
//! Plain structs with public fields that mirror WGSL's vector types.
//! Components are accessed directly via `.x`, `.y`, `.z`, `.w` fields,
//! or by index with `v[0]`, `v[1]`, etc.
//!
//! Vectors support swizzling by method-calling. For example, [`Vec3::xxx`]
//! and [`Vec4::bgra`].
#![expect(
    clippy::self_named_constructors,
    reason = "WGSL uses self named constructors"
)]

/// A 2-dimensional vector.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

/// A 3-dimensional vector.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A 4-dimensional vector.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

/// Concrete type alias for a 2-dimensional vector of `f32` scalar components.
pub type Vec2f = Vec2<f32>;
/// Concrete type alias for a 2-dimensional vector of `i32` scalar components.
pub type Vec2i = Vec2<i32>;
/// Concrete type alias for a 2-dimensional vector of `u32` scalar components.
pub type Vec2u = Vec2<u32>;
/// Concrete type alias for a 2-dimensional vector of `bool` scalar components.
pub type Vec2b = Vec2<bool>;

/// Concrete type alias for a 3-dimensional vector of `f32` scalar components.
pub type Vec3f = Vec3<f32>;
/// Concrete type alias for a 3-dimensional vector of `i32` scalar components.
pub type Vec3i = Vec3<i32>;
/// Concrete type alias for a 3-dimensional vector of `u32` scalar components.
pub type Vec3u = Vec3<u32>;
/// Concrete type alias for a 3-dimensional vector of `bool` scalar components.
pub type Vec3b = Vec3<bool>;

/// Concrete type alias for a 4-dimensional vector of `f32` scalar components.
pub type Vec4f = Vec4<f32>;
/// Concrete type alias for a 4-dimensional vector of `i32` scalar components.
pub type Vec4i = Vec4<i32>;
/// Concrete type alias for a 4-dimensional vector of `u32` scalar components.
pub type Vec4u = Vec4<u32>;
/// Concrete type alias for a 4-dimensional vector of `bool` scalar components.
pub type Vec4b = Vec4<bool>;

impl<T> Vec2<T> {
    /// Construct a 2-dimensional vector from components.
    pub const fn vec2(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Convert to an array of components.
    pub fn to_array(self) -> [T; 2] {
        [self.x, self.y]
    }
}

impl<T: Copy> Vec2<T> {
    /// Construct from an array of components.
    pub fn from_array(arr: [T; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }
}

impl<T> Vec3<T> {
    /// Construct a 3-dimensional vector from components.
    pub const fn vec3(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Convert to an array of components.
    pub fn to_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }
}

impl<T: Copy> Vec3<T> {
    /// Construct from an array of components.
    pub fn from_array(arr: [T; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl<T> Vec4<T> {
    /// Construct a 4-dimensional vector from components.
    pub const fn vec4(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    /// Convert to an array of components.
    pub fn to_array(self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl<T: Copy> Vec4<T> {
    /// Construct from an array of components.
    pub fn from_array(arr: [T; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

// Const constructor functions matching WGSL naming conventions.

/// Constructor for a 2-dimensional vector of `f32` scalar components.
pub const fn vec2f(x: f32, y: f32) -> Vec2<f32> {
    Vec2 { x, y }
}

/// Constructor for a 2-dimensional vector of `i32` scalar components.
pub const fn vec2i(x: i32, y: i32) -> Vec2<i32> {
    Vec2 { x, y }
}

/// Constructor for a 2-dimensional vector of `u32` scalar components.
pub const fn vec2u(x: u32, y: u32) -> Vec2<u32> {
    Vec2 { x, y }
}

/// Constructor for a 2-dimensional vector of `bool` scalar components.
pub const fn vec2b(x: bool, y: bool) -> Vec2<bool> {
    Vec2 { x, y }
}

/// Constructor for a 3-dimensional vector of `f32` scalar components.
pub const fn vec3f(x: f32, y: f32, z: f32) -> Vec3<f32> {
    Vec3 { x, y, z }
}

/// Constructor for a 3-dimensional vector of `i32` scalar components.
pub const fn vec3i(x: i32, y: i32, z: i32) -> Vec3<i32> {
    Vec3 { x, y, z }
}

/// Constructor for a 3-dimensional vector of `u32` scalar components.
pub const fn vec3u(x: u32, y: u32, z: u32) -> Vec3<u32> {
    Vec3 { x, y, z }
}

/// Constructor for a 3-dimensional vector of `bool` scalar components.
pub const fn vec3b(x: bool, y: bool, z: bool) -> Vec3<bool> {
    Vec3 { x, y, z }
}

/// Constructor for a 4-dimensional vector of `f32` scalar components.
pub const fn vec4f(x: f32, y: f32, z: f32, w: f32) -> Vec4<f32> {
    Vec4 { x, y, z, w }
}

/// Constructor for a 4-dimensional vector of `i32` scalar components.
pub const fn vec4i(x: i32, y: i32, z: i32, w: i32) -> Vec4<i32> {
    Vec4 { x, y, z, w }
}

/// Constructor for a 4-dimensional vector of `u32` scalar components.
pub const fn vec4u(x: u32, y: u32, z: u32, w: u32) -> Vec4<u32> {
    Vec4 { x, y, z, w }
}

/// Constructor for a 4-dimensional vector of `bool` scalar components.
pub const fn vec4b(x: bool, y: bool, z: bool, w: bool) -> Vec4<bool> {
    Vec4 { x, y, z, w }
}

// Index impls for `usize` and `u32`.

impl<T> std::ops::Index<usize> for Vec2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds: Vec2 has 2 components but index is {index}"),
        }
    }
}

impl<T> std::ops::IndexMut<usize> for Vec2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds: Vec2 has 2 components but index is {index}"),
        }
    }
}

impl<T> std::ops::Index<u32> for Vec2<T> {
    type Output = T;

    fn index(&self, index: u32) -> &T {
        &self[index as usize]
    }
}

impl<T> std::ops::IndexMut<u32> for Vec2<T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        &mut self[index as usize]
    }
}

impl<T> std::ops::Index<usize> for Vec3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds: Vec3 has 3 components but index is {index}"),
        }
    }
}

impl<T> std::ops::IndexMut<usize> for Vec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds: Vec3 has 3 components but index is {index}"),
        }
    }
}

impl<T> std::ops::Index<u32> for Vec3<T> {
    type Output = T;

    fn index(&self, index: u32) -> &T {
        &self[index as usize]
    }
}

impl<T> std::ops::IndexMut<u32> for Vec3<T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        &mut self[index as usize]
    }
}

impl<T> std::ops::Index<usize> for Vec4<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds: Vec4 has 4 components but index is {index}"),
        }
    }
}

impl<T> std::ops::IndexMut<usize> for Vec4<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds: Vec4 has 4 components but index is {index}"),
        }
    }
}

impl<T> std::ops::Index<u32> for Vec4<T> {
    type Output = T;

    fn index(&self, index: u32) -> &T {
        &self[index as usize]
    }
}

impl<T> std::ops::IndexMut<u32> for Vec4<T> {
    fn index_mut(&mut self, index: u32) -> &mut T {
        &mut self[index as usize]
    }
}

// Swizzle methods (generated by proc macro).
// These provide multi-component and single-component access using both
// the positional (x, y, z, w) and color (r, g, b, a) naming conventions.

wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2f, [f32, Vec2f], [vec2f], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2i, [i32, Vec2i], [vec2i], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2u, [u32, Vec2u], [vec2u], [x, y], [x, y]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [r, g]);
wgsl_rs_macros::swizzle!(Vec2b, [bool, Vec2b], [vec2b], [x, y], [x, y]);

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

// From/Into conversions for glam types and arrays.

macro_rules! impl_from_vec2 {
    ($glam_ty:ty, $scalar:ty) => {
        impl From<$glam_ty> for Vec2<$scalar> {
            fn from(v: $glam_ty) -> Self {
                Vec2 { x: v.x, y: v.y }
            }
        }

        impl From<Vec2<$scalar>> for $glam_ty {
            fn from(v: Vec2<$scalar>) -> Self {
                <$glam_ty>::new(v.x, v.y)
            }
        }

        impl From<[$scalar; 2]> for Vec2<$scalar> {
            fn from(arr: [$scalar; 2]) -> Self {
                Vec2 {
                    x: arr[0],
                    y: arr[1],
                }
            }
        }

        impl From<Vec2<$scalar>> for [$scalar; 2] {
            fn from(v: Vec2<$scalar>) -> [$scalar; 2] {
                [v.x, v.y]
            }
        }
    };
}

macro_rules! impl_from_vec3 {
    ($glam_ty:ty, $scalar:ty) => {
        impl From<$glam_ty> for Vec3<$scalar> {
            fn from(v: $glam_ty) -> Self {
                Vec3 {
                    x: v.x,
                    y: v.y,
                    z: v.z,
                }
            }
        }

        impl From<Vec3<$scalar>> for $glam_ty {
            fn from(v: Vec3<$scalar>) -> Self {
                <$glam_ty>::new(v.x, v.y, v.z)
            }
        }

        impl From<[$scalar; 3]> for Vec3<$scalar> {
            fn from(arr: [$scalar; 3]) -> Self {
                Vec3 {
                    x: arr[0],
                    y: arr[1],
                    z: arr[2],
                }
            }
        }

        impl From<Vec3<$scalar>> for [$scalar; 3] {
            fn from(v: Vec3<$scalar>) -> [$scalar; 3] {
                [v.x, v.y, v.z]
            }
        }
    };
}

macro_rules! impl_from_vec4 {
    ($glam_ty:ty, $scalar:ty) => {
        impl From<$glam_ty> for Vec4<$scalar> {
            fn from(v: $glam_ty) -> Self {
                Vec4 {
                    x: v.x,
                    y: v.y,
                    z: v.z,
                    w: v.w,
                }
            }
        }

        impl From<Vec4<$scalar>> for $glam_ty {
            fn from(v: Vec4<$scalar>) -> Self {
                <$glam_ty>::new(v.x, v.y, v.z, v.w)
            }
        }

        impl From<[$scalar; 4]> for Vec4<$scalar> {
            fn from(arr: [$scalar; 4]) -> Self {
                Vec4 {
                    x: arr[0],
                    y: arr[1],
                    z: arr[2],
                    w: arr[3],
                }
            }
        }

        impl From<Vec4<$scalar>> for [$scalar; 4] {
            fn from(v: Vec4<$scalar>) -> [$scalar; 4] {
                [v.x, v.y, v.z, v.w]
            }
        }
    };
}

impl_from_vec2!(glam::Vec2, f32);
impl_from_vec2!(glam::IVec2, i32);
impl_from_vec2!(glam::UVec2, u32);
impl_from_vec2!(glam::BVec2, bool);

impl_from_vec3!(glam::Vec3, f32);
impl_from_vec3!(glam::IVec3, i32);
impl_from_vec3!(glam::UVec3, u32);
impl_from_vec3!(glam::BVec3, bool);

impl_from_vec4!(glam::Vec4, f32);
impl_from_vec4!(glam::IVec4, i32);
impl_from_vec4!(glam::UVec4, u32);
impl_from_vec4!(glam::BVec4, bool);

// Arithmetic operations.
//
// For f32 vectors, we delegate to glam for potential SIMD optimization.
// For integer vectors, we operate directly on fields.

/// Implements vector-vector binary operations (Add, Sub, Mul, Div) for Vec2.
macro_rules! impl_vec2_ops {
    ($scalar:ty) => {
        impl std::ops::Add for Vec2<$scalar> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Vec2 {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                }
            }
        }
        impl std::ops::Sub for Vec2<$scalar> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Vec2 {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                }
            }
        }
        impl std::ops::Mul for Vec2<$scalar> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Vec2 {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                }
            }
        }
        impl std::ops::Div for Vec2<$scalar> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                Vec2 {
                    x: self.x / rhs.x,
                    y: self.y / rhs.y,
                }
            }
        }
    };
}

/// Implements vector-vector binary operations (Add, Sub, Mul, Div) for Vec3.
macro_rules! impl_vec3_ops {
    ($scalar:ty) => {
        impl std::ops::Add for Vec3<$scalar> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Vec3 {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                }
            }
        }
        impl std::ops::Sub for Vec3<$scalar> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Vec3 {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                }
            }
        }
        impl std::ops::Mul for Vec3<$scalar> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Vec3 {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                    z: self.z * rhs.z,
                }
            }
        }
        impl std::ops::Div for Vec3<$scalar> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                Vec3 {
                    x: self.x / rhs.x,
                    y: self.y / rhs.y,
                    z: self.z / rhs.z,
                }
            }
        }
    };
}

/// Implements vector-vector binary operations (Add, Sub, Mul, Div) for Vec4.
macro_rules! impl_vec4_ops {
    ($scalar:ty) => {
        impl std::ops::Add for Vec4<$scalar> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Vec4 {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                    z: self.z + rhs.z,
                    w: self.w + rhs.w,
                }
            }
        }
        impl std::ops::Sub for Vec4<$scalar> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Vec4 {
                    x: self.x - rhs.x,
                    y: self.y - rhs.y,
                    z: self.z - rhs.z,
                    w: self.w - rhs.w,
                }
            }
        }
        impl std::ops::Mul for Vec4<$scalar> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Vec4 {
                    x: self.x * rhs.x,
                    y: self.y * rhs.y,
                    z: self.z * rhs.z,
                    w: self.w * rhs.w,
                }
            }
        }
        impl std::ops::Div for Vec4<$scalar> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                Vec4 {
                    x: self.x / rhs.x,
                    y: self.y / rhs.y,
                    z: self.z / rhs.z,
                    w: self.w / rhs.w,
                }
            }
        }
    };
}

/// Implements vector-vector Rem for Vec2.
macro_rules! impl_vec2_rem {
    ($scalar:ty) => {
        impl std::ops::Rem for Vec2<$scalar> {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self {
                Vec2 {
                    x: self.x % rhs.x,
                    y: self.y % rhs.y,
                }
            }
        }
    };
}

/// Implements vector-vector Rem for Vec3.
macro_rules! impl_vec3_rem {
    ($scalar:ty) => {
        impl std::ops::Rem for Vec3<$scalar> {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self {
                Vec3 {
                    x: self.x % rhs.x,
                    y: self.y % rhs.y,
                    z: self.z % rhs.z,
                }
            }
        }
    };
}

/// Implements vector-vector Rem for Vec4.
macro_rules! impl_vec4_rem {
    ($scalar:ty) => {
        impl std::ops::Rem for Vec4<$scalar> {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self {
                Vec4 {
                    x: self.x % rhs.x,
                    y: self.y % rhs.y,
                    z: self.z % rhs.z,
                    w: self.w % rhs.w,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector binary operations for Vec2.
macro_rules! impl_vec2_scalar_ops {
    ($scalar:ty) => {
        impl std::ops::Add<$scalar> for Vec2<$scalar> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                Vec2 {
                    x: self.x + rhs,
                    y: self.y + rhs,
                }
            }
        }
        impl std::ops::Sub<$scalar> for Vec2<$scalar> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                Vec2 {
                    x: self.x - rhs,
                    y: self.y - rhs,
                }
            }
        }
        impl std::ops::Mul<$scalar> for Vec2<$scalar> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Vec2 {
                    x: self.x * rhs,
                    y: self.y * rhs,
                }
            }
        }
        impl std::ops::Div<$scalar> for Vec2<$scalar> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                Vec2 {
                    x: self.x / rhs,
                    y: self.y / rhs,
                }
            }
        }
        impl std::ops::Add<Vec2<$scalar>> for $scalar {
            type Output = Vec2<$scalar>;
            fn add(self, rhs: Vec2<$scalar>) -> Vec2<$scalar> {
                Vec2 {
                    x: self + rhs.x,
                    y: self + rhs.y,
                }
            }
        }
        impl std::ops::Sub<Vec2<$scalar>> for $scalar {
            type Output = Vec2<$scalar>;
            fn sub(self, rhs: Vec2<$scalar>) -> Vec2<$scalar> {
                Vec2 {
                    x: self - rhs.x,
                    y: self - rhs.y,
                }
            }
        }
        impl std::ops::Mul<Vec2<$scalar>> for $scalar {
            type Output = Vec2<$scalar>;
            fn mul(self, rhs: Vec2<$scalar>) -> Vec2<$scalar> {
                Vec2 {
                    x: self * rhs.x,
                    y: self * rhs.y,
                }
            }
        }
        impl std::ops::Div<Vec2<$scalar>> for $scalar {
            type Output = Vec2<$scalar>;
            fn div(self, rhs: Vec2<$scalar>) -> Vec2<$scalar> {
                Vec2 {
                    x: self / rhs.x,
                    y: self / rhs.y,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector binary operations for Vec3.
macro_rules! impl_vec3_scalar_ops {
    ($scalar:ty) => {
        impl std::ops::Add<$scalar> for Vec3<$scalar> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                Vec3 {
                    x: self.x + rhs,
                    y: self.y + rhs,
                    z: self.z + rhs,
                }
            }
        }
        impl std::ops::Sub<$scalar> for Vec3<$scalar> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                Vec3 {
                    x: self.x - rhs,
                    y: self.y - rhs,
                    z: self.z - rhs,
                }
            }
        }
        impl std::ops::Mul<$scalar> for Vec3<$scalar> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Vec3 {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                }
            }
        }
        impl std::ops::Div<$scalar> for Vec3<$scalar> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                Vec3 {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                }
            }
        }
        impl std::ops::Add<Vec3<$scalar>> for $scalar {
            type Output = Vec3<$scalar>;
            fn add(self, rhs: Vec3<$scalar>) -> Vec3<$scalar> {
                Vec3 {
                    x: self + rhs.x,
                    y: self + rhs.y,
                    z: self + rhs.z,
                }
            }
        }
        impl std::ops::Sub<Vec3<$scalar>> for $scalar {
            type Output = Vec3<$scalar>;
            fn sub(self, rhs: Vec3<$scalar>) -> Vec3<$scalar> {
                Vec3 {
                    x: self - rhs.x,
                    y: self - rhs.y,
                    z: self - rhs.z,
                }
            }
        }
        impl std::ops::Mul<Vec3<$scalar>> for $scalar {
            type Output = Vec3<$scalar>;
            fn mul(self, rhs: Vec3<$scalar>) -> Vec3<$scalar> {
                Vec3 {
                    x: self * rhs.x,
                    y: self * rhs.y,
                    z: self * rhs.z,
                }
            }
        }
        impl std::ops::Div<Vec3<$scalar>> for $scalar {
            type Output = Vec3<$scalar>;
            fn div(self, rhs: Vec3<$scalar>) -> Vec3<$scalar> {
                Vec3 {
                    x: self / rhs.x,
                    y: self / rhs.y,
                    z: self / rhs.z,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector binary operations for Vec4.
macro_rules! impl_vec4_scalar_ops {
    ($scalar:ty) => {
        impl std::ops::Add<$scalar> for Vec4<$scalar> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self {
                Vec4 {
                    x: self.x + rhs,
                    y: self.y + rhs,
                    z: self.z + rhs,
                    w: self.w + rhs,
                }
            }
        }
        impl std::ops::Sub<$scalar> for Vec4<$scalar> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self {
                Vec4 {
                    x: self.x - rhs,
                    y: self.y - rhs,
                    z: self.z - rhs,
                    w: self.w - rhs,
                }
            }
        }
        impl std::ops::Mul<$scalar> for Vec4<$scalar> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Vec4 {
                    x: self.x * rhs,
                    y: self.y * rhs,
                    z: self.z * rhs,
                    w: self.w * rhs,
                }
            }
        }
        impl std::ops::Div<$scalar> for Vec4<$scalar> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                Vec4 {
                    x: self.x / rhs,
                    y: self.y / rhs,
                    z: self.z / rhs,
                    w: self.w / rhs,
                }
            }
        }
        impl std::ops::Add<Vec4<$scalar>> for $scalar {
            type Output = Vec4<$scalar>;
            fn add(self, rhs: Vec4<$scalar>) -> Vec4<$scalar> {
                Vec4 {
                    x: self + rhs.x,
                    y: self + rhs.y,
                    z: self + rhs.z,
                    w: self + rhs.w,
                }
            }
        }
        impl std::ops::Sub<Vec4<$scalar>> for $scalar {
            type Output = Vec4<$scalar>;
            fn sub(self, rhs: Vec4<$scalar>) -> Vec4<$scalar> {
                Vec4 {
                    x: self - rhs.x,
                    y: self - rhs.y,
                    z: self - rhs.z,
                    w: self - rhs.w,
                }
            }
        }
        impl std::ops::Mul<Vec4<$scalar>> for $scalar {
            type Output = Vec4<$scalar>;
            fn mul(self, rhs: Vec4<$scalar>) -> Vec4<$scalar> {
                Vec4 {
                    x: self * rhs.x,
                    y: self * rhs.y,
                    z: self * rhs.z,
                    w: self * rhs.w,
                }
            }
        }
        impl std::ops::Div<Vec4<$scalar>> for $scalar {
            type Output = Vec4<$scalar>;
            fn div(self, rhs: Vec4<$scalar>) -> Vec4<$scalar> {
                Vec4 {
                    x: self / rhs.x,
                    y: self / rhs.y,
                    z: self / rhs.z,
                    w: self / rhs.w,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector Rem for Vec2.
macro_rules! impl_vec2_scalar_rem {
    ($scalar:ty) => {
        impl std::ops::Rem<$scalar> for Vec2<$scalar> {
            type Output = Self;
            fn rem(self, rhs: $scalar) -> Self {
                Vec2 {
                    x: self.x % rhs,
                    y: self.y % rhs,
                }
            }
        }
        impl std::ops::Rem<Vec2<$scalar>> for $scalar {
            type Output = Vec2<$scalar>;
            fn rem(self, rhs: Vec2<$scalar>) -> Vec2<$scalar> {
                Vec2 {
                    x: self % rhs.x,
                    y: self % rhs.y,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector Rem for Vec3.
macro_rules! impl_vec3_scalar_rem {
    ($scalar:ty) => {
        impl std::ops::Rem<$scalar> for Vec3<$scalar> {
            type Output = Self;
            fn rem(self, rhs: $scalar) -> Self {
                Vec3 {
                    x: self.x % rhs,
                    y: self.y % rhs,
                    z: self.z % rhs,
                }
            }
        }
        impl std::ops::Rem<Vec3<$scalar>> for $scalar {
            type Output = Vec3<$scalar>;
            fn rem(self, rhs: Vec3<$scalar>) -> Vec3<$scalar> {
                Vec3 {
                    x: self % rhs.x,
                    y: self % rhs.y,
                    z: self % rhs.z,
                }
            }
        }
    };
}

/// Implements vector-scalar and scalar-vector Rem for Vec4.
macro_rules! impl_vec4_scalar_rem {
    ($scalar:ty) => {
        impl std::ops::Rem<$scalar> for Vec4<$scalar> {
            type Output = Self;
            fn rem(self, rhs: $scalar) -> Self {
                Vec4 {
                    x: self.x % rhs,
                    y: self.y % rhs,
                    z: self.z % rhs,
                    w: self.w % rhs,
                }
            }
        }
        impl std::ops::Rem<Vec4<$scalar>> for $scalar {
            type Output = Vec4<$scalar>;
            fn rem(self, rhs: Vec4<$scalar>) -> Vec4<$scalar> {
                Vec4 {
                    x: self % rhs.x,
                    y: self % rhs.y,
                    z: self % rhs.z,
                    w: self % rhs.w,
                }
            }
        }
    };
}

// Float vectors: Add, Sub, Mul, Div, Rem
impl_vec2_ops!(f32);
impl_vec3_ops!(f32);
impl_vec4_ops!(f32);
impl_vec2_rem!(f32);
impl_vec3_rem!(f32);
impl_vec4_rem!(f32);
impl_vec2_scalar_ops!(f32);
impl_vec3_scalar_ops!(f32);
impl_vec4_scalar_ops!(f32);
impl_vec2_scalar_rem!(f32);
impl_vec3_scalar_rem!(f32);
impl_vec4_scalar_rem!(f32);

// Signed integer vectors: Add, Sub, Mul, Div (no Rem)
impl_vec2_ops!(i32);
impl_vec3_ops!(i32);
impl_vec4_ops!(i32);
impl_vec2_scalar_ops!(i32);
impl_vec3_scalar_ops!(i32);
impl_vec4_scalar_ops!(i32);

// Unsigned integer vectors: Add, Sub, Mul, Div (no Rem)
impl_vec2_ops!(u32);
impl_vec3_ops!(u32);
impl_vec4_ops!(u32);
impl_vec2_scalar_ops!(u32);
impl_vec3_scalar_ops!(u32);
impl_vec4_scalar_ops!(u32);

// Neg impls for signed types.

impl std::ops::Neg for Vec2<f32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Neg for Vec3<f32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Neg for Vec4<f32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec4 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl std::ops::Neg for Vec2<i32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Neg for Vec3<i32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Neg for Vec4<i32> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec4 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}
