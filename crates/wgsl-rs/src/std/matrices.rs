use super::*;

/// Trait identifying the standard library's inner matrix types.
///
/// WGSL supports matrices consisting of `f32` and `f16` elements.
pub trait ScalarCompOfMatrix<const N: usize, const M: usize>:
    ScalarCompOfVec<N> + ScalarCompOfVec<M>
{
    type Output: Clone + Copy;
}

/// An `NxM` dimensional column-major matrix.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Mat<const N: usize, const M: usize, T: ScalarCompOfMatrix<N, M>> {
    pub(crate) inner: <T as ScalarCompOfMatrix<N, M>>::Output,
}

/// matrix! generates the N×N matrix:
/// * ScalarCompOfMatrix impl
/// * concretized type alias
/// * const constructor
macro_rules! matrix {
    ($n:literal, $m: literal, $ty:ty, $inner:ty, $construct_inner:path, $col_ty:ty, [$($cols:ident),+]) => {
        paste::paste! {
            impl ScalarCompOfMatrix<$n, $m> for $ty {
                type Output = $inner;
            }

            #[doc = concat!("A ", $n, "x", $m, " column-major matrix of ", stringify!($ty), " components.")]
            pub type [<Mat $n x $m f>] = Mat<$n, $m, $ty>;

            #[doc = concat!("Constructs a ", $n, "x", $m, " column-major matrix of ", stringify!($ty), " components.")]
            pub const fn [<mat $n x $m f>]($($cols: $col_ty<$ty>),+) -> Mat<$n, $m, $ty> {
                let inner = $construct_inner($($cols.inner),+);
                Mat { inner }
            }
        }
    };
}

// Mat<2, 2, f32>
matrix!(
    2,
    2,
    f32,
    glam::Mat2,
    glam::Mat2::from_cols,
    Vec2,
    [x_axis, y_axis]
);

const fn mat32_inner(x_axis: glam::Vec3, y_axis: glam::Vec3) -> [glam::Vec3; 2] {
    [x_axis, y_axis]
}
matrix!(
    2,
    3,
    f32,
    [glam::Vec3; 2],
    mat32_inner,
    Vec3,
    [x_axis, y_axis]
);

// Mat<3, 3, f32>
matrix!(
    3,
    3,
    f32,
    glam::Mat3,
    glam::Mat3::from_cols,
    Vec3,
    [x_axis, y_axis, z_axis]
);

// Mat<3, 2, f32>
matrix!(
    3,
    2,
    f32,
    glam::Affine2,
    glam::Affine2::from_cols,
    Vec2,
    [x_axis, y_axis, z_axis]
);


// Mat<4, 4, f32>
matrix!(
    4,
    4,
    f32,
    glam::Mat4,
    glam::Mat4::from_cols,
    Vec4,
    [x_axis, y_axis, z_axis, w_axis]
);

// Mat<4, 3, f32>
const fn mat43_inner(
    x_axis: glam::Vec3,
    y_axis: glam::Vec3,
    z_axis: glam::Vec3,
    w_axis: glam::Vec3,
) -> [glam::Vec3; 4] {
    [x_axis, y_axis, z_axis, w_axis]
}
matrix!(
    4,
    3,
    f32,
    [glam::Vec3; 4],
    mat43_inner,
    Vec3,
    [x_axis, y_axis, z_axis, w_axis]
);

// Mat<4, 2, f32>
const fn mat42_inner(
    x_axis: glam::Vec2,
    y_axis: glam::Vec2,
    z_axis: glam::Vec2,
    w_axis: glam::Vec2,
) -> [glam::Vec2; 4] {
    [x_axis, y_axis, z_axis, w_axis]
}
matrix!(
    4,
    2,
    f32,
    [glam::Vec2; 4],
    mat42_inner,
    Vec2,
    [x_axis, y_axis, z_axis, w_axis]
);

/// Alias for Mat2x2f.
pub type Mat2f = Mat2x2f;
/// Alias for Mat3x3f.
pub type Mat3f = Mat3x3f;
/// Alias for Mat4x4f.
pub type Mat4f = Mat4x4f;

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

/// Implements matrix * matrix multiplication for square matrices.
macro_rules! impl_mat_mul_mat {
    ($mat:ty) => {
        impl std::ops::Mul<$mat> for $mat {
            type Output = $mat;
            fn mul(self, rhs: $mat) -> Self::Output {
                Self {
                    inner: self.inner * rhs.inner,
                }
            }
        }
    };
}
impl_mat_mul_mat!(Mat2f);
impl_mat_mul_mat!(Mat3f);
impl_mat_mul_mat!(Mat4f);

/// Implements matrix * vector multiplication.
macro_rules! impl_mat_mul_vec {
    ($mat:ty, $vec:ty) => {
        impl std::ops::Mul<$vec> for $mat {
            type Output = $vec;
            fn mul(self, rhs: $vec) -> Self::Output {
                <$vec>::from(self.inner * rhs.inner)
            }
        }
    };
}
impl_mat_mul_vec!(Mat2f, Vec2f);
impl_mat_mul_vec!(Mat3f, Vec3f);
impl_mat_mul_vec!(Mat4f, Vec4f);

/// Implements matrix * scalar and scalar * matrix multiplication.
macro_rules! impl_mat_mul_scalar {
    ($mat:ty) => {
        impl std::ops::Mul<f32> for $mat {
            type Output = $mat;
            fn mul(self, rhs: f32) -> Self::Output {
                Self {
                    inner: self.inner * rhs,
                }
            }
        }

        impl std::ops::Mul<$mat> for f32 {
            type Output = $mat;
            fn mul(self, rhs: $mat) -> Self::Output {
                <$mat>::from(self * rhs.inner)
            }
        }
    };
}
impl_mat_mul_scalar!(Mat2f);
impl_mat_mul_scalar!(Mat3f);
impl_mat_mul_scalar!(Mat4f);
