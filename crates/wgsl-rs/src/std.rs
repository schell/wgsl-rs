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

use std::sync::{Arc, LazyLock, RwLock};

pub use wgsl_rs_macros::{fragment, uniform, vertex};

/// Trait identifying WGSL's concrete scalar types.
pub trait IsScalar: Sized {
    type Vector2;
    type Vector3;
    type Vector4;

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

/// Trait used to "read" a value out of a type.
///
/// This is used for conversion, so far.
/// Eg, `f32(value)`.
pub trait CanRead<T> {
    fn read_value(&self) -> T;
}

pub fn f32<T: CanRead<f32>>(t: T) -> f32 {
    t.read_value()
}

/// A shader uniform, backed by a storage buffer on the CPU.
pub struct UniformVariable<T> {
    pub group: u32,
    pub binding: u32,
    pub value: Arc<RwLock<Option<T>>>,
}

pub type Uniform<T> = LazyLock<UniformVariable<T>>;

impl<T: Clone> CanRead<T> for &'static Uniform<T> {
    fn read_value(&self) -> T {
        let guard = self.value.read().expect("could not read value");
        let maybe = guard.as_ref();
        maybe.cloned().expect("uniform value has not been set")
    }
}

/// Trait that provides numeric builtin functions.
///
/// See <https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions>.
///
/// ## NOTE: This is a WIP.
/// Here is a table of the numeric builtin functions from the WGSL spec:
///
/// | Function         | Description                                                                 |
/// |------------------|-----------------------------------------------------------------------------|
/// | abs              | Returns the absolute value of each component.                               |
/// | acos             | Returns the arccosine of each component.                                    |
/// | acosh            | Returns the inverse hyperbolic cosine of each component.                    |
/// | asin             | Returns the arcsine of each component.                                      |
/// | asinh            | Returns the inverse hyperbolic sine of each component.                      |
/// | atan             | Returns the arctangent of each component.                                   |
/// | atanh            | Returns the inverse hyperbolic tangent of each component.                   |
/// | atan2            | Returns the arctangent of y/x for each component.                           |
/// | ceil             | Returns the smallest integer greater than or equal to each component.        |
/// | clamp            | Returns each component clamped between min and max.                         |
/// | cos              | Returns the cosine of each component.                                       |
/// | cosh             | Returns the hyperbolic cosine of each component.                            |
/// | countLeadingZeros| Returns the number of leading zero bits in each component.                  |
/// | countOneBits     | Returns the number of one bits in each component.                           |
/// | countTrailingZeros| Returns the number of trailing zero bits in each component.                |
/// | degrees          | Converts radians to degrees for each component.                             |
/// | distance         | Returns the distance between two points.                                    |
/// | exp              | Returns the natural exponent of each component.                             |
/// | exp2             | Returns 2 raised to the power of each component.                            |
/// | extractBits      | Extracts bits from each component.                                          |
/// | floor            | Returns the largest integer less than or equal to each component.           |
/// | fma              | Returns the fused multiply-add of the components.                           |
/// | fract            | Returns the fractional part of each component.                              |
/// | frexp            | Returns the mantissa and exponent of each component.                        |
/// | insertBits       | Inserts bits into each component.                                           |
/// | length           | Returns the length of a vector.                                             |
/// | log              | Returns the natural logarithm of each component.                            |
/// | log2             | Returns the base-2 logarithm of each component.                             |
/// | max              | Returns the maximum of each component and the corresponding value in y.     |
/// | min              | Returns the minimum of each component and the corresponding value in y.     |
/// | mix              | Returns the linear interpolation of x and y by a.                           |
/// | modf             | Returns the integer and fractional parts of each component.                 |
/// | normalize        | Returns the normalized vector.                                              |
/// | pow              | Returns x raised to the power of y for each component.                      |
/// | quantizeToF16    | Quantizes each component to f16 precision.                                  |
/// | radians          | Converts degrees to radians for each component.                             |
/// | reflect          | Returns the reflection direction for an incident vector.                    |
/// | reverseBits      | Returns the bits of each component reversed.                                |
/// | round            | Returns the nearest integer to each component.                              |
/// | sign             | Returns the sign of each component.                                         |
/// | sin              | Returns the sine of each component.                                         |
/// | sinh             | Returns the hyperbolic sine of each component.                              |
/// | smoothstep       | Returns smooth Hermite interpolation between 0 and 1.                       |
/// | sqrt             | Returns the square root of each component.                                  |
/// | step             | Returns 0.0 or 1.0 based on comparison with edge.                           |
/// | tan              | Returns the tangent of each component.                                      |
/// | tanh             | Returns the hyperbolic tangent of each component.                           |
/// | trunc            | Returns the integer part of each component.                                 |
///
/// See <https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions>.
pub trait IsNumericBuiltin {
    /// Returns the sine of `e`, where `e` is in radians. Component-wise when type `T` is a vector.
    fn sin(self) -> Self;
}

/// Returns the sine of `e`, where `e` is in radians. Component-wise when type `T` is a vector.
pub fn sin<T: IsNumericBuiltin>(e: T) -> T {
    T::sin(e)
}

impl IsNumericBuiltin for f32 {
    fn sin(self) -> Self {
        self.sin()
    }
}
