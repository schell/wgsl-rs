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

// Mat<2, 4, f32>
const fn mat24_inner(x_axis: glam::Vec4, y_axis: glam::Vec4) -> [glam::Vec4; 2] {
    [x_axis, y_axis]
}
matrix!(
    2,
    4,
    f32,
    [glam::Vec4; 2],
    mat24_inner,
    Vec4,
    [x_axis, y_axis]
);

// Mat<3, 4, f32>
const fn mat34_inner(
    x_axis: glam::Vec4,
    y_axis: glam::Vec4,
    z_axis: glam::Vec4,
) -> [glam::Vec4; 3] {
    [x_axis, y_axis, z_axis]
}
matrix!(
    3,
    4,
    f32,
    [glam::Vec4; 3],
    mat34_inner,
    Vec4,
    [x_axis, y_axis, z_axis]
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

/// Provides the numeric built-in function `determinant`.
pub trait NumericBuiltinDeterminant {
    /// The scalar type of the matrix elements.
    type Scalar;

    /// Returns the determinant of a square matrix.
    fn determinant(self) -> Self::Scalar;
}

/// Returns the determinant of a square matrix.
///
/// Only defined for square matrices (`matCxC`).
pub fn determinant<T: NumericBuiltinDeterminant>(e: T) -> T::Scalar {
    <T as NumericBuiltinDeterminant>::determinant(e)
}

impl NumericBuiltinDeterminant for Mat2f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        self.inner.determinant()
    }
}

impl NumericBuiltinDeterminant for Mat3f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        self.inner.determinant()
    }
}

impl NumericBuiltinDeterminant for Mat4f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        self.inner.determinant()
    }
}

/// Provides the numeric built-in function `transpose`.
pub trait NumericBuiltinTranspose {
    /// The transposed matrix type (columns and rows swapped).
    type Output;

    /// Returns the transpose of the matrix.
    fn transpose(self) -> Self::Output;
}

/// Returns the transpose of a matrix.
///
/// For a `matRxC` input, returns a `matCxR` output.
pub fn transpose<T: NumericBuiltinTranspose>(e: T) -> T::Output {
    <T as NumericBuiltinTranspose>::transpose(e)
}

impl NumericBuiltinTranspose for Mat2f {
    type Output = Mat2f;

    fn transpose(self) -> Mat2f {
        Mat2f {
            inner: self.inner.transpose(),
        }
    }
}

impl NumericBuiltinTranspose for Mat3f {
    type Output = Mat3f;

    fn transpose(self) -> Mat3f {
        Mat3f {
            inner: self.inner.transpose(),
        }
    }
}

impl NumericBuiltinTranspose for Mat4f {
    type Output = Mat4f;

    fn transpose(self) -> Mat4f {
        Mat4f {
            inner: self.inner.transpose(),
        }
    }
}

// Non-square matrix transpose implementations.
//
// For non-square matrices backed by `[glam::VecM; N]`, we need manual
// transpose logic. A matrix with N columns of M-component vectors transposes
// to M columns of N-component vectors.

impl NumericBuiltinTranspose for Mat2x3f {
    type Output = Mat3x2f;

    fn transpose(self) -> Mat3x2f {
        // self is 2 columns of Vec3: [[x0,y0,z0], [x1,y1,z1]]
        // transpose is 3 columns of Vec2: [[x0,x1], [y0,y1], [z0,z1]]
        let [c0, c1] = self.inner;
        let c0a = c0.to_array();
        let c1a = c1.to_array();
        Mat3x2f {
            inner: glam::Affine2::from_cols(
                glam::Vec2::new(c0a[0], c1a[0]),
                glam::Vec2::new(c0a[1], c1a[1]),
                glam::Vec2::new(c0a[2], c1a[2]),
            ),
        }
    }
}

impl NumericBuiltinTranspose for Mat3x2f {
    type Output = Mat2x3f;

    fn transpose(self) -> Mat2x3f {
        // self is 3 columns of Vec2: Affine2 { matrix2: Mat2(x_axis, y_axis),
        // translation } transpose is 2 columns of Vec3
        let c0 = self.inner.matrix2.x_axis.to_array();
        let c1 = self.inner.matrix2.y_axis.to_array();
        let c2 = self.inner.translation.to_array();
        Mat2x3f {
            inner: [
                glam::Vec3::new(c0[0], c1[0], c2[0]),
                glam::Vec3::new(c0[1], c1[1], c2[1]),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat4x3f {
    type Output = Mat3x4f;

    fn transpose(self) -> Mat3x4f {
        // self is 4 columns of Vec3
        // transpose is 3 columns of Vec4
        let [c0, c1, c2, c3] = self.inner;
        let c0a = c0.to_array();
        let c1a = c1.to_array();
        let c2a = c2.to_array();
        let c3a = c3.to_array();
        Mat3x4f {
            inner: [
                glam::Vec4::new(c0a[0], c1a[0], c2a[0], c3a[0]),
                glam::Vec4::new(c0a[1], c1a[1], c2a[1], c3a[1]),
                glam::Vec4::new(c0a[2], c1a[2], c2a[2], c3a[2]),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat3x4f {
    type Output = Mat4x3f;

    fn transpose(self) -> Mat4x3f {
        // self is 3 columns of Vec4
        // transpose is 4 columns of Vec3
        let [c0, c1, c2] = self.inner;
        let c0a = c0.to_array();
        let c1a = c1.to_array();
        let c2a = c2.to_array();
        Mat4x3f {
            inner: [
                glam::Vec3::new(c0a[0], c1a[0], c2a[0]),
                glam::Vec3::new(c0a[1], c1a[1], c2a[1]),
                glam::Vec3::new(c0a[2], c1a[2], c2a[2]),
                glam::Vec3::new(c0a[3], c1a[3], c2a[3]),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat4x2f {
    type Output = Mat2x4f;

    fn transpose(self) -> Mat2x4f {
        // self is 4 columns of Vec2
        // transpose is 2 columns of Vec4
        let [c0, c1, c2, c3] = self.inner;
        let c0a = c0.to_array();
        let c1a = c1.to_array();
        let c2a = c2.to_array();
        let c3a = c3.to_array();
        Mat2x4f {
            inner: [
                glam::Vec4::new(c0a[0], c1a[0], c2a[0], c3a[0]),
                glam::Vec4::new(c0a[1], c1a[1], c2a[1], c3a[1]),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat2x4f {
    type Output = Mat4x2f;

    fn transpose(self) -> Mat4x2f {
        // self is 2 columns of Vec4
        // transpose is 4 columns of Vec2
        let [c0, c1] = self.inner;
        let c0a = c0.to_array();
        let c1a = c1.to_array();
        Mat4x2f {
            inner: [
                glam::Vec2::new(c0a[0], c1a[0]),
                glam::Vec2::new(c0a[1], c1a[1]),
                glam::Vec2::new(c0a[2], c1a[2]),
                glam::Vec2::new(c0a[3], c1a[3]),
            ],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity_determinant_mat2() {
        let m = mat2x2f(vec2f(1.0, 0.0), vec2f(0.0, 1.0));
        assert_eq!(determinant(m), 1.0);

        let m2 = mat2x2f(vec2f(2.0, 1.0), vec2f(1.0, 3.0));
        assert!((determinant(m2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sanity_determinant_mat3() {
        let m = mat3x3f(
            vec3f(1.0, 0.0, 0.0),
            vec3f(0.0, 1.0, 0.0),
            vec3f(0.0, 0.0, 1.0),
        );
        assert_eq!(determinant(m), 1.0);
    }

    #[test]
    fn sanity_determinant_mat4() {
        let m = mat4x4f(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        );
        assert_eq!(determinant(m), 1.0);
    }

    #[test]
    fn sanity_transpose_mat2() {
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        let t = transpose(m);
        // Column 0 of transpose = row 0 of original
        let t_c0 = f32::vec_to_array(Vec2f {
            inner: t.inner.x_axis,
        });
        let t_c1 = f32::vec_to_array(Vec2f {
            inner: t.inner.y_axis,
        });
        assert_eq!(t_c0, [1.0, 3.0]);
        assert_eq!(t_c1, [2.0, 4.0]);
    }

    #[test]
    fn sanity_transpose_mat3() {
        let m = mat3x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
        );
        let t = transpose(m);
        let t_c0 = f32::vec_to_array(Vec3f {
            inner: t.inner.x_axis,
        });
        let t_c1 = f32::vec_to_array(Vec3f {
            inner: t.inner.y_axis,
        });
        let t_c2 = f32::vec_to_array(Vec3f {
            inner: t.inner.z_axis,
        });
        assert_eq!(t_c0, [1.0, 4.0, 7.0]);
        assert_eq!(t_c1, [2.0, 5.0, 8.0]);
        assert_eq!(t_c2, [3.0, 6.0, 9.0]);
    }

    #[test]
    fn sanity_transpose_mat4() {
        let m = mat4x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
            vec4f(13.0, 14.0, 15.0, 16.0),
        );
        let t = transpose(m);
        let t_c0 = f32::vec_to_array(Vec4f {
            inner: t.inner.x_axis,
        });
        assert_eq!(t_c0, [1.0, 5.0, 9.0, 13.0]);
    }

    #[test]
    fn sanity_transpose_mat2x3() {
        // 2 columns of Vec3
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let t: Mat3x2f = transpose(m);
        // 3 columns of Vec2
        let c0 = t.inner.matrix2.x_axis.to_array();
        let c1 = t.inner.matrix2.y_axis.to_array();
        let c2 = t.inner.translation.to_array();
        assert_eq!(c0, [1.0, 4.0]);
        assert_eq!(c1, [2.0, 5.0]);
        assert_eq!(c2, [3.0, 6.0]);
    }

    #[test]
    fn sanity_transpose_mat3x2() {
        // 3 columns of Vec2 (via Affine2)
        let m = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        let t: Mat2x3f = transpose(m);
        // 2 columns of Vec3
        assert_eq!(t.inner[0].to_array(), [1.0, 3.0, 5.0]);
        assert_eq!(t.inner[1].to_array(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn sanity_transpose_roundtrip() {
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let roundtrip = transpose(transpose(m));
        assert_eq!(roundtrip.inner[0].to_array(), m.inner[0].to_array());
        assert_eq!(roundtrip.inner[1].to_array(), m.inner[1].to_array());
    }

    #[test]
    fn sanity_transpose_mat4x3() {
        let m = mat4x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
            vec3f(10.0, 11.0, 12.0),
        );
        let t: Mat3x4f = transpose(m);
        assert_eq!(t.inner[0].to_array(), [1.0, 4.0, 7.0, 10.0]);
        assert_eq!(t.inner[1].to_array(), [2.0, 5.0, 8.0, 11.0]);
        assert_eq!(t.inner[2].to_array(), [3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn sanity_transpose_mat4x2() {
        let m = mat4x2f(
            vec2f(1.0, 2.0),
            vec2f(3.0, 4.0),
            vec2f(5.0, 6.0),
            vec2f(7.0, 8.0),
        );
        let t: Mat2x4f = transpose(m);
        assert_eq!(t.inner[0].to_array(), [1.0, 3.0, 5.0, 7.0]);
        assert_eq!(t.inner[1].to_array(), [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn sanity_transpose_mat2x4() {
        let m = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        let t: Mat4x2f = transpose(m);
        assert_eq!(t.inner[0].to_array(), [1.0, 5.0]);
        assert_eq!(t.inner[1].to_array(), [2.0, 6.0]);
        assert_eq!(t.inner[2].to_array(), [3.0, 7.0]);
        assert_eq!(t.inner[3].to_array(), [4.0, 8.0]);
    }

    #[test]
    fn sanity_transpose_mat3x4() {
        let m = mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
        );
        let t: Mat4x3f = transpose(m);
        assert_eq!(t.inner[0].to_array(), [1.0, 5.0, 9.0]);
        assert_eq!(t.inner[1].to_array(), [2.0, 6.0, 10.0]);
        assert_eq!(t.inner[2].to_array(), [3.0, 7.0, 11.0]);
        assert_eq!(t.inner[3].to_array(), [4.0, 8.0, 12.0]);
    }
}
