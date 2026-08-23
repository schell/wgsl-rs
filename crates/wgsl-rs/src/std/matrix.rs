//! Matrix implementations.
//!
//! Column-major matrices with public `columns` arrays, matching WGSL's
//! matrix types. Columns are accessed by index: `m[0]`, `m[1]`, etc.
//!
//! # Naming convention
//!
//! `MatCxRf` has **C columns and R rows**; each column is a `VecRf`. So
//! `Mat2x3f` holds 2 columns of `Vec3f` (6 total components). This mirrors
//! WGSL's `matCxR<T>` spelling.
//!
//! # Multiplication
//!
//! All WGSL-spec-valid `*` combinations are implemented. The operand order
//! determines the result shape, and the two orders are **not interchangeable**
//! for non-square matrices:
//!
//! | expression | input shapes | output |
//! | --- | --- | --- |
//! | `m * v` | `MatCxRf` * `VecCf` | `VecRf` |
//! | `v * m` | `VecRf` * `MatCxRf` | `VecCf` |
//! | `a * b` | `MatKxRf` * `MatCxKf` | `MatCxRf` |
//! | `s * m`, `m * s` | `f32` * `MatCxRf` | `MatCxRf` |
//!
//! For example, `Mat2x3f * Vec2f` produces a `Vec3f`, while `Vec3f *
//! Mat2x3f` produces a `Vec2f`. Scalar scaling is commutative and works on
//! either side for every matrix shape.
//!
//! See the WGSL spec's `arithmetic-expr` section for the linear-algebra
//! definitions. On the GPU these all lower to the WGSL `*` operator; the
//! impls here exist so the CPU ("two worlds") path agrees.

use super::*;

/// A 2x2 column-major matrix of `f32` components (2 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat2x2f {
    columns: [Vec2f; 2],
}

/// A 2x3 column-major matrix of `f32` components (2 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat2x3f {
    columns: [Vec3f; 2],
}

/// A 2x4 column-major matrix of `f32` components (2 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat2x4f {
    columns: [Vec4f; 2],
}

/// A 3x2 column-major matrix of `f32` components (3 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat3x2f {
    columns: [Vec2f; 3],
}

/// A 3x3 column-major matrix of `f32` components (3 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat3x3f {
    columns: [Vec3f; 3],
}

/// A 3x4 column-major matrix of `f32` components (3 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat3x4f {
    columns: [Vec4f; 3],
}

/// A 4x2 column-major matrix of `f32` components (4 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat4x2f {
    columns: [Vec2f; 4],
}

/// A 4x3 column-major matrix of `f32` components (4 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat4x3f {
    columns: [Vec3f; 4],
}

/// A 4x4 column-major matrix of `f32` components (4 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Mat4x4f {
    columns: [Vec4f; 4],
}

/// Alias for `Mat2x2f`.
pub type Mat2f = Mat2x2f;
/// Alias for `Mat3x3f`.
pub type Mat3f = Mat3x3f;
/// Alias for `Mat4x4f`.
pub type Mat4f = Mat4x4f;

macro_rules! impl_matrix_wgsl {
    ($t:ty, $cols:expr, $rows:expr) => {
        impl Wgsl for $t {
            fn to_ir() -> wgsl_rs_ir::Type {
                wgsl_rs_ir::Type::Matrix {
                    columns: $cols,
                    rows: $rows,
                    scalar_ty: Some(wgsl_rs_ir::ScalarType::F32),
                }
            }
        }
    };
}

impl_matrix_wgsl!(Mat2x2f, 2, 2);
impl_matrix_wgsl!(Mat2x3f, 2, 3);
impl_matrix_wgsl!(Mat2x4f, 2, 4);
impl_matrix_wgsl!(Mat3x2f, 3, 2);
impl_matrix_wgsl!(Mat3x3f, 3, 3);
impl_matrix_wgsl!(Mat3x4f, 3, 4);
impl_matrix_wgsl!(Mat4x2f, 4, 2);
impl_matrix_wgsl!(Mat4x3f, 4, 3);
impl_matrix_wgsl!(Mat4x4f, 4, 4);

// Const constructor functions matching WGSL naming conventions.

/// Constructs a 2x2 column-major matrix of `f32` components.
pub const fn mat2x2f(x_axis: Vec2f, y_axis: Vec2f) -> Mat2x2f {
    Mat2x2f {
        columns: [x_axis, y_axis],
    }
}

/// Constructs a 2x3 column-major matrix of `f32` components.
pub const fn mat2x3f(x_axis: Vec3f, y_axis: Vec3f) -> Mat2x3f {
    Mat2x3f {
        columns: [x_axis, y_axis],
    }
}

/// Constructs a 2x4 column-major matrix of `f32` components.
pub const fn mat2x4f(x_axis: Vec4f, y_axis: Vec4f) -> Mat2x4f {
    Mat2x4f {
        columns: [x_axis, y_axis],
    }
}

/// Constructs a 3x2 column-major matrix of `f32` components.
pub const fn mat3x2f(x_axis: Vec2f, y_axis: Vec2f, z_axis: Vec2f) -> Mat3x2f {
    Mat3x2f {
        columns: [x_axis, y_axis, z_axis],
    }
}

/// Constructs a 3x3 column-major matrix of `f32` components.
pub const fn mat3x3f(x_axis: Vec3f, y_axis: Vec3f, z_axis: Vec3f) -> Mat3x3f {
    Mat3x3f {
        columns: [x_axis, y_axis, z_axis],
    }
}

/// Constructs a 3x4 column-major matrix of `f32` components.
pub const fn mat3x4f(x_axis: Vec4f, y_axis: Vec4f, z_axis: Vec4f) -> Mat3x4f {
    Mat3x4f {
        columns: [x_axis, y_axis, z_axis],
    }
}

/// Constructs a 4x2 column-major matrix of `f32` components.
pub const fn mat4x2f(x_axis: Vec2f, y_axis: Vec2f, z_axis: Vec2f, w_axis: Vec2f) -> Mat4x2f {
    Mat4x2f {
        columns: [x_axis, y_axis, z_axis, w_axis],
    }
}

/// Constructs a 4x3 column-major matrix of `f32` components.
pub const fn mat4x3f(x_axis: Vec3f, y_axis: Vec3f, z_axis: Vec3f, w_axis: Vec3f) -> Mat4x3f {
    Mat4x3f {
        columns: [x_axis, y_axis, z_axis, w_axis],
    }
}

/// Constructs a 4x4 column-major matrix of `f32` components.
pub const fn mat4x4f(x_axis: Vec4f, y_axis: Vec4f, z_axis: Vec4f, w_axis: Vec4f) -> Mat4x4f {
    Mat4x4f {
        columns: [x_axis, y_axis, z_axis, w_axis],
    }
}

// Index impls for all matrix types, for both `usize` and `u32`.

macro_rules! impl_mat_index {
    ($mat:ty, $col_ty:ty) => {
        impl std::ops::Index<usize> for $mat {
            type Output = $col_ty;
            fn index(&self, index: usize) -> &$col_ty {
                &self.columns[index]
            }
        }

        impl std::ops::IndexMut<usize> for $mat {
            fn index_mut(&mut self, index: usize) -> &mut $col_ty {
                &mut self.columns[index]
            }
        }

        impl std::ops::Index<u32> for $mat {
            type Output = $col_ty;
            fn index(&self, index: u32) -> &$col_ty {
                &self.columns[index as usize]
            }
        }

        impl std::ops::IndexMut<u32> for $mat {
            fn index_mut(&mut self, index: u32) -> &mut $col_ty {
                &mut self.columns[index as usize]
            }
        }
    };
}

impl_mat_index!(Mat2x2f, Vec2f);
impl_mat_index!(Mat2x3f, Vec3f);
impl_mat_index!(Mat2x4f, Vec4f);
impl_mat_index!(Mat3x2f, Vec2f);
impl_mat_index!(Mat3x3f, Vec3f);
impl_mat_index!(Mat3x4f, Vec4f);
impl_mat_index!(Mat4x2f, Vec2f);
impl_mat_index!(Mat4x3f, Vec3f);
impl_mat_index!(Mat4x4f, Vec4f);

// From/Into conversions for glam types.

impl From<glam::Mat2> for Mat2x2f {
    fn from(m: glam::Mat2) -> Self {
        Mat2x2f {
            columns: [m.x_axis.into(), m.y_axis.into()],
        }
    }
}

impl From<Mat2x2f> for glam::Mat2 {
    fn from(m: Mat2x2f) -> Self {
        glam::Mat2::from_cols(m.columns[0].into(), m.columns[1].into())
    }
}

impl From<glam::Mat3> for Mat3x3f {
    fn from(m: glam::Mat3) -> Self {
        Mat3x3f {
            columns: [m.x_axis.into(), m.y_axis.into(), m.z_axis.into()],
        }
    }
}

impl From<Mat3x3f> for glam::Mat3 {
    fn from(m: Mat3x3f) -> Self {
        glam::Mat3::from_cols(
            m.columns[0].into(),
            m.columns[1].into(),
            m.columns[2].into(),
        )
    }
}

impl From<glam::Mat4> for Mat4x4f {
    fn from(m: glam::Mat4) -> Self {
        Mat4x4f {
            columns: [
                m.x_axis.into(),
                m.y_axis.into(),
                m.z_axis.into(),
                m.w_axis.into(),
            ],
        }
    }
}

impl From<Mat4x4f> for glam::Mat4 {
    fn from(m: Mat4x4f) -> Self {
        glam::Mat4::from_cols(
            m.columns[0].into(),
            m.columns[1].into(),
            m.columns[2].into(),
            m.columns[3].into(),
        )
    }
}

// === Multiplication =========================================================
//
// Scalar multiplications for the square matrix types delegate to glam; all
// matrix * vector, vector * matrix, and matrix * matrix impls use direct
// component arithmetic. glam has no non-square matrix types, so non-square
// cases must be implemented manually; the square cases share the same code
// path for uniformity and to keep the linear-algebra semantics in one place.
//
// WGSL spec semantics (see `arithmetic-expr`):
//   - `m: matCxR<T>` * `v: vecC<T>`     -> `vecR<T>`  (column-vector product)
//   - `v: vecR<T>` * `m: matCxR<T>`     -> `vecC<T>`  (row-vector product)
//   - `e1: matKxR<T>` * `e2: matCxK<T>` -> `matCxR<T>` (matrix product)
//
// The result of `v * m` is `transpose(transpose(m) * transpose(v))`, which
// expands to component `i` of the output being `dot(v, m.columns[i])`.

// === Matrix * vector ========================================================
//
// `m * v` is the linear combination `sum_j v[j] * m.columns[j]`. Since each
// column is a `VecRf` with `Add` and `Mul<f32>`, the sum yields a `VecRf`
// directly. Dispatch is on C (the number of columns / the input vector's
// length).

macro_rules! impl_mat_vec_mul {
    ($mat:ty, $vec_in:ty, $vec_out:ty, $c:tt) => {
        impl std::ops::Mul<$vec_in> for $mat {
            type Output = $vec_out;
            fn mul(self, rhs: $vec_in) -> $vec_out {
                let cols = self.columns;
                impl_mat_vec_mul!(@combine $c, cols, rhs)
            }
        }
    };
    (@combine 2, $cols:ident, $v:ident) => {
        $cols[0] * $v.x + $cols[1] * $v.y
    };
    (@combine 3, $cols:ident, $v:ident) => {
        $cols[0] * $v.x + $cols[1] * $v.y + $cols[2] * $v.z
    };
    (@combine 4, $cols:ident, $v:ident) => {
        $cols[0] * $v.x + $cols[1] * $v.y + $cols[2] * $v.z + $cols[3] * $v.w
    };
}

impl_mat_vec_mul!(Mat2x2f, Vec2f, Vec2f, 2);
impl_mat_vec_mul!(Mat2x3f, Vec2f, Vec3f, 2);
impl_mat_vec_mul!(Mat2x4f, Vec2f, Vec4f, 2);
impl_mat_vec_mul!(Mat3x2f, Vec3f, Vec2f, 3);
impl_mat_vec_mul!(Mat3x3f, Vec3f, Vec3f, 3);
impl_mat_vec_mul!(Mat3x4f, Vec3f, Vec4f, 3);
impl_mat_vec_mul!(Mat4x2f, Vec4f, Vec2f, 4);
impl_mat_vec_mul!(Mat4x3f, Vec4f, Vec3f, 4);
impl_mat_vec_mul!(Mat4x4f, Vec4f, Vec4f, 4);

// === Vector * matrix ========================================================
//
// `v * m` is per-component dots against the columns: output component `i` is
// `dot(v, m.columns[i])`. Dispatch is on C (the number of output components,
// i.e. the number of columns) and R (the input vector's length / each
// column's length).

macro_rules! impl_vec_mat_mul {
    ($vec:ty, $mat:ty, $vec_out:ty, $c:tt, $r:tt) => {
        impl std::ops::Mul<$mat> for $vec {
            type Output = $vec_out;
            fn mul(self, rhs: $mat) -> $vec_out {
                let cols = rhs.columns;
                let s = self;
                impl_vec_mat_mul!(@col $c, $r, cols, s)
            }
        }
    };
    // Output has 2 components (C=2).
    (@col 2, $r:tt, $cols:ident, $s:ident) => {
        Vec2 {
            x: impl_vec_mat_mul!(@dot $r, $s, $cols[0]),
            y: impl_vec_mat_mul!(@dot $r, $s, $cols[1]),
        }
    };
    // Output has 3 components (C=3).
    (@col 3, $r:tt, $cols:ident, $s:ident) => {
        Vec3 {
            x: impl_vec_mat_mul!(@dot $r, $s, $cols[0]),
            y: impl_vec_mat_mul!(@dot $r, $s, $cols[1]),
            z: impl_vec_mat_mul!(@dot $r, $s, $cols[2]),
        }
    };
    // Output has 4 components (C=4).
    (@col 4, $r:tt, $cols:ident, $s:ident) => {
        Vec4 {
            x: impl_vec_mat_mul!(@dot $r, $s, $cols[0]),
            y: impl_vec_mat_mul!(@dot $r, $s, $cols[1]),
            z: impl_vec_mat_mul!(@dot $r, $s, $cols[2]),
            w: impl_vec_mat_mul!(@dot $r, $s, $cols[3]),
        }
    };
    (@dot 2, $s:ident, $col:expr) => {
        $s.x * $col.x + $s.y * $col.y
    };
    (@dot 3, $s:ident, $col:expr) => {
        $s.x * $col.x + $s.y * $col.y + $s.z * $col.z
    };
    (@dot 4, $s:ident, $col:expr) => {
        $s.x * $col.x + $s.y * $col.y + $s.z * $col.z + $s.w * $col.w
    };
}

impl_vec_mat_mul!(Vec2f, Mat2x2f, Vec2f, 2, 2);
impl_vec_mat_mul!(Vec2f, Mat3x2f, Vec3f, 3, 2);
impl_vec_mat_mul!(Vec2f, Mat4x2f, Vec4f, 4, 2);
impl_vec_mat_mul!(Vec3f, Mat2x3f, Vec2f, 2, 3);
impl_vec_mat_mul!(Vec3f, Mat3x3f, Vec3f, 3, 3);
impl_vec_mat_mul!(Vec3f, Mat4x3f, Vec4f, 4, 3);
impl_vec_mat_mul!(Vec4f, Mat2x4f, Vec2f, 2, 4);
impl_vec_mat_mul!(Vec4f, Mat3x4f, Vec3f, 3, 4);
impl_vec_mat_mul!(Vec4f, Mat4x4f, Vec4f, 4, 4);

// === Matrix * matrix ========================================================
//
// `a * b` is column-wise: column `j` of the result is `a * b.columns[j]`,
// dispatched to the `Mat * Vec` impl above. This covers all 27 valid WGSL
// combinations (K, C, R each in {2, 3, 4}); square cases use the same code
// path for uniformity and to keep the implementation self-contained.

macro_rules! impl_mat_mat_mul {
    ($a:ty, $b:ty, $out:ty, $c:tt) => {
        impl std::ops::Mul<$b> for $a {
            type Output = $out;
            fn mul(self, rhs: $b) -> $out {
                type Out = $out;
                let cols = rhs.columns;
                Out {
                    columns: impl_mat_mat_mul!(@cols $c, self, cols),
                }
            }
        }
    };
    (@cols 2, $self:ident, $cols:ident) => {
        [$self * $cols[0], $self * $cols[1]]
    };
    (@cols 3, $self:ident, $cols:ident) => {
        [$self * $cols[0], $self * $cols[1], $self * $cols[2]]
    };
    (@cols 4, $self:ident, $cols:ident) => {
        [
            $self * $cols[0],
            $self * $cols[1],
            $self * $cols[2],
            $self * $cols[3],
        ]
    };
}

// K=2: a is Mat2xR, b is MatCx2, out is MatCxR.
impl_mat_mat_mul!(Mat2x2f, Mat2x2f, Mat2x2f, 2);
impl_mat_mat_mul!(Mat2x2f, Mat3x2f, Mat3x2f, 3);
impl_mat_mat_mul!(Mat2x2f, Mat4x2f, Mat4x2f, 4);
impl_mat_mat_mul!(Mat2x3f, Mat2x2f, Mat2x3f, 2);
impl_mat_mat_mul!(Mat2x3f, Mat3x2f, Mat3x3f, 3);
impl_mat_mat_mul!(Mat2x3f, Mat4x2f, Mat4x3f, 4);
impl_mat_mat_mul!(Mat2x4f, Mat2x2f, Mat2x4f, 2);
impl_mat_mat_mul!(Mat2x4f, Mat3x2f, Mat3x4f, 3);
impl_mat_mat_mul!(Mat2x4f, Mat4x2f, Mat4x4f, 4);

// K=3: a is Mat3xR, b is MatCx3, out is MatCxR.
impl_mat_mat_mul!(Mat3x2f, Mat2x3f, Mat2x2f, 2);
impl_mat_mat_mul!(Mat3x2f, Mat3x3f, Mat3x2f, 3);
impl_mat_mat_mul!(Mat3x2f, Mat4x3f, Mat4x2f, 4);
impl_mat_mat_mul!(Mat3x3f, Mat2x3f, Mat2x3f, 2);
impl_mat_mat_mul!(Mat3x3f, Mat3x3f, Mat3x3f, 3);
impl_mat_mat_mul!(Mat3x3f, Mat4x3f, Mat4x3f, 4);
impl_mat_mat_mul!(Mat3x4f, Mat2x3f, Mat2x4f, 2);
impl_mat_mat_mul!(Mat3x4f, Mat3x3f, Mat3x4f, 3);
impl_mat_mat_mul!(Mat3x4f, Mat4x3f, Mat4x4f, 4);

// K=4: a is Mat4xR, b is MatCx4, out is MatCxR.
impl_mat_mat_mul!(Mat4x2f, Mat2x4f, Mat2x2f, 2);
impl_mat_mat_mul!(Mat4x2f, Mat3x4f, Mat3x2f, 3);
impl_mat_mat_mul!(Mat4x2f, Mat4x4f, Mat4x2f, 4);
impl_mat_mat_mul!(Mat4x3f, Mat2x4f, Mat2x3f, 2);
impl_mat_mat_mul!(Mat4x3f, Mat3x4f, Mat3x3f, 3);
impl_mat_mat_mul!(Mat4x3f, Mat4x4f, Mat4x3f, 4);
impl_mat_mat_mul!(Mat4x4f, Mat2x4f, Mat2x4f, 2);
impl_mat_mat_mul!(Mat4x4f, Mat3x4f, Mat3x4f, 3);
impl_mat_mat_mul!(Mat4x4f, Mat4x4f, Mat4x4f, 4);

// === Scalar multiplication (delegated to glam for square types) =============

impl std::ops::Mul<f32> for Mat2x2f {
    type Output = Mat2x2f;
    fn mul(self, rhs: f32) -> Mat2x2f {
        let g: glam::Mat2 = self.into();
        (g * rhs).into()
    }
}

impl std::ops::Mul<Mat2x2f> for f32 {
    type Output = Mat2x2f;
    fn mul(self, rhs: Mat2x2f) -> Mat2x2f {
        let g: glam::Mat2 = rhs.into();
        (self * g).into()
    }
}

impl std::ops::Mul<f32> for Mat3x3f {
    type Output = Mat3x3f;
    fn mul(self, rhs: f32) -> Mat3x3f {
        let g: glam::Mat3 = self.into();
        (g * rhs).into()
    }
}

impl std::ops::Mul<Mat3x3f> for f32 {
    type Output = Mat3x3f;
    fn mul(self, rhs: Mat3x3f) -> Mat3x3f {
        let g: glam::Mat3 = rhs.into();
        (self * g).into()
    }
}

impl std::ops::Mul<f32> for Mat4x4f {
    type Output = Mat4x4f;
    fn mul(self, rhs: f32) -> Mat4x4f {
        let g: glam::Mat4 = self.into();
        (g * rhs).into()
    }
}

impl std::ops::Mul<Mat4x4f> for f32 {
    type Output = Mat4x4f;
    fn mul(self, rhs: Mat4x4f) -> Mat4x4f {
        let g: glam::Mat4 = rhs.into();
        (self * g).into()
    }
}

/// Multiplies each component of a non-square matrix by a scalar.
macro_rules! impl_mat_scalar_mul {
    ($mat:ty) => {
        impl std::ops::Mul<f32> for $mat {
            type Output = $mat;
            fn mul(self, rhs: f32) -> $mat {
                type Out = $mat;
                let cols = self.columns;
                Out {
                    columns: cols.map(|c| c * rhs),
                }
            }
        }

        impl std::ops::Mul<$mat> for f32 {
            type Output = $mat;
            fn mul(self, rhs: $mat) -> $mat {
                rhs * self
            }
        }
    };
}

impl_mat_scalar_mul!(Mat2x3f);
impl_mat_scalar_mul!(Mat2x4f);
impl_mat_scalar_mul!(Mat3x2f);
impl_mat_scalar_mul!(Mat3x4f);
impl_mat_scalar_mul!(Mat4x2f);
impl_mat_scalar_mul!(Mat4x3f);

// Determinant.

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

impl NumericBuiltinDeterminant for Mat2x2f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        let g: glam::Mat2 = self.into();
        g.determinant()
    }
}

impl NumericBuiltinDeterminant for Mat3x3f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        let g: glam::Mat3 = self.into();
        g.determinant()
    }
}

impl NumericBuiltinDeterminant for Mat4x4f {
    type Scalar = f32;

    fn determinant(self) -> f32 {
        let g: glam::Mat4 = self.into();
        g.determinant()
    }
}

// Transpose.

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

impl NumericBuiltinTranspose for Mat2x2f {
    type Output = Mat2x2f;

    fn transpose(self) -> Mat2x2f {
        let g: glam::Mat2 = self.into();
        g.transpose().into()
    }
}

impl NumericBuiltinTranspose for Mat3x3f {
    type Output = Mat3x3f;

    fn transpose(self) -> Mat3x3f {
        let g: glam::Mat3 = self.into();
        g.transpose().into()
    }
}

impl NumericBuiltinTranspose for Mat4x4f {
    type Output = Mat4x4f;

    fn transpose(self) -> Mat4x4f {
        let g: glam::Mat4 = self.into();
        g.transpose().into()
    }
}

// Non-square matrix transpose implementations.

impl NumericBuiltinTranspose for Mat2x3f {
    type Output = Mat3x2f;

    fn transpose(self) -> Mat3x2f {
        let [c0, c1] = self.columns;
        Mat3x2f {
            columns: [vec2f(c0.x, c1.x), vec2f(c0.y, c1.y), vec2f(c0.z, c1.z)],
        }
    }
}

impl NumericBuiltinTranspose for Mat3x2f {
    type Output = Mat2x3f;

    fn transpose(self) -> Mat2x3f {
        let [c0, c1, c2] = self.columns;
        Mat2x3f {
            columns: [vec3f(c0.x, c1.x, c2.x), vec3f(c0.y, c1.y, c2.y)],
        }
    }
}

impl NumericBuiltinTranspose for Mat4x3f {
    type Output = Mat3x4f;

    fn transpose(self) -> Mat3x4f {
        let [c0, c1, c2, c3] = self.columns;
        Mat3x4f {
            columns: [
                vec4f(c0.x, c1.x, c2.x, c3.x),
                vec4f(c0.y, c1.y, c2.y, c3.y),
                vec4f(c0.z, c1.z, c2.z, c3.z),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat3x4f {
    type Output = Mat4x3f;

    fn transpose(self) -> Mat4x3f {
        let [c0, c1, c2] = self.columns;
        Mat4x3f {
            columns: [
                vec3f(c0.x, c1.x, c2.x),
                vec3f(c0.y, c1.y, c2.y),
                vec3f(c0.z, c1.z, c2.z),
                vec3f(c0.w, c1.w, c2.w),
            ],
        }
    }
}

impl NumericBuiltinTranspose for Mat4x2f {
    type Output = Mat2x4f;

    fn transpose(self) -> Mat2x4f {
        let [c0, c1, c2, c3] = self.columns;
        Mat2x4f {
            columns: [vec4f(c0.x, c1.x, c2.x, c3.x), vec4f(c0.y, c1.y, c2.y, c3.y)],
        }
    }
}

impl NumericBuiltinTranspose for Mat2x4f {
    type Output = Mat4x2f;

    fn transpose(self) -> Mat4x2f {
        let [c0, c1] = self.columns;
        Mat4x2f {
            columns: [
                vec2f(c0.x, c1.x),
                vec2f(c0.y, c1.y),
                vec2f(c0.z, c1.z),
                vec2f(c0.w, c1.w),
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
        assert_eq!(t.columns[0].to_array(), [1.0, 3.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 4.0]);
    }

    #[test]
    fn sanity_transpose_mat3() {
        let m = mat3x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
        );
        let t = transpose(m);
        assert_eq!(t.columns[0].to_array(), [1.0, 4.0, 7.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 5.0, 8.0]);
        assert_eq!(t.columns[2].to_array(), [3.0, 6.0, 9.0]);
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
        assert_eq!(t.columns[0].to_array(), [1.0, 5.0, 9.0, 13.0]);
    }

    #[test]
    fn sanity_transpose_mat2x3() {
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let t: Mat3x2f = transpose(m);
        assert_eq!(t.columns[0].to_array(), [1.0, 4.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 5.0]);
        assert_eq!(t.columns[2].to_array(), [3.0, 6.0]);
    }

    #[test]
    fn sanity_transpose_mat3x2() {
        let m = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        let t: Mat2x3f = transpose(m);
        assert_eq!(t.columns[0].to_array(), [1.0, 3.0, 5.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn sanity_transpose_roundtrip() {
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let roundtrip = transpose(transpose(m));
        assert_eq!(roundtrip.columns[0].to_array(), m.columns[0].to_array());
        assert_eq!(roundtrip.columns[1].to_array(), m.columns[1].to_array());
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
        assert_eq!(t.columns[0].to_array(), [1.0, 4.0, 7.0, 10.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 5.0, 8.0, 11.0]);
        assert_eq!(t.columns[2].to_array(), [3.0, 6.0, 9.0, 12.0]);
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
        assert_eq!(t.columns[0].to_array(), [1.0, 3.0, 5.0, 7.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn sanity_transpose_mat2x4() {
        let m = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        let t: Mat4x2f = transpose(m);
        assert_eq!(t.columns[0].to_array(), [1.0, 5.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 6.0]);
        assert_eq!(t.columns[2].to_array(), [3.0, 7.0]);
        assert_eq!(t.columns[3].to_array(), [4.0, 8.0]);
    }

    #[test]
    fn sanity_transpose_mat3x4() {
        let m = mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
        );
        let t: Mat4x3f = transpose(m);
        assert_eq!(t.columns[0].to_array(), [1.0, 5.0, 9.0]);
        assert_eq!(t.columns[1].to_array(), [2.0, 6.0, 10.0]);
        assert_eq!(t.columns[2].to_array(), [3.0, 7.0, 11.0]);
        assert_eq!(t.columns[3].to_array(), [4.0, 8.0, 12.0]);
    }

    #[test]
    fn sanity_index_usize() {
        let m = mat3x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
        );
        assert_eq!(m[0usize].x, 1.0);
        assert_eq!(m[1usize].y, 5.0);
        assert_eq!(m[2usize].z, 9.0);
    }

    #[test]
    fn sanity_index_u32() {
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        assert_eq!(m[0u32].x, 1.0);
        assert_eq!(m[1u32].y, 4.0);
    }

    #[test]
    fn sanity_index_mut() {
        let mut m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        m[0usize].x = 10.0;
        assert_eq!(m[0usize].x, 10.0);
    }

    #[test]
    fn module_modify_mat4() {
        #[crate::wgsl(crate_path = crate)]
        pub mod mat {
            #![allow(dead_code)]

            use crate::std::*;

            #[derive(Wgsl)]
            #[wgsl_path(crate)]
            pub struct Uniforms {
                pub projection: Mat4f,
                pub modelview: Mat4f,
            }

            uniform!(group(0), binding(0), UNIFORMS: Uniforms);

            pub struct VertexInput {
                #[location(0)]
                pub position: Vec3f,
            }

            pub struct VertexOutput {
                #[builtin(position)]
                pub clip_position: Vec4f,
            }

            #[vertex]
            pub fn vs_main(input: VertexInput) -> VertexOutput {
                let projection = get!(UNIFORMS).projection;
                let mut modelview = get!(UNIFORMS).modelview;
                // Just for access sake, mess with the modelview matrix
                modelview[0u32].y += 10.0;
                VertexOutput {
                    clip_position: projection
                        * modelview
                        * vec4f(input.position.x, input.position.y, input.position.z, 1.0),
                }
            }
        }
    }

    // === Matrix * vector tests ================================================

    #[test]
    fn mat2x2_mul_vec2() {
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        let v = vec2f(5.0, 6.0);
        // [1 3][5] = [1*5 + 3*6, 2*5 + 4*6] = [23, 34]
        // [2 4][6]
        let out: Vec2f = m * v;
        assert_eq!(out.to_array(), [23.0, 34.0]);
    }

    #[test]
    fn mat2x3_mul_vec2() {
        // mat2x3f: 2 columns of vec3f. m * vec2 -> vec3.
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let v = vec2f(7.0, 8.0);
        // Row 0: 1*7 + 4*8 = 39
        // Row 1: 2*7 + 5*8 = 54
        // Row 2: 3*7 + 6*8 = 69
        let out: Vec3f = m * v;
        assert_eq!(out.to_array(), [39.0, 54.0, 69.0]);
    }

    #[test]
    fn mat3x2_mul_vec3() {
        let m = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        let v = vec3f(7.0, 8.0, 9.0);
        // Row 0: 1*7 + 3*8 + 5*9 = 7+24+45 = 76
        // Row 1: 2*7 + 4*8 + 6*9 = 14+32+54 = 100
        let out: Vec2f = m * v;
        assert_eq!(out.to_array(), [76.0, 100.0]);
    }

    #[test]
    fn mat2x4_mul_vec2() {
        let m = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        let v = vec2f(9.0, 10.0);
        // Row 0: 1*9 + 5*10 = 59
        // Row 1: 2*9 + 6*10 = 78
        // Row 2: 3*9 + 7*10 = 97
        // Row 3: 4*9 + 8*10 = 116
        let out: Vec4f = m * v;
        assert_eq!(out.to_array(), [59.0, 78.0, 97.0, 116.0]);
    }

    #[test]
    fn mat4x2_mul_vec4() {
        let m = mat4x2f(
            vec2f(1.0, 2.0),
            vec2f(3.0, 4.0),
            vec2f(5.0, 6.0),
            vec2f(7.0, 8.0),
        );
        let v = vec4f(9.0, 10.0, 11.0, 12.0);
        // Row 0: 1*9 + 3*10 + 5*11 + 7*12 = 9+30+55+84 = 178
        // Row 1: 2*9 + 4*10 + 6*11 + 8*12 = 18+40+66+96 = 220
        let out: Vec2f = m * v;
        assert_eq!(out.to_array(), [178.0, 220.0]);
    }

    #[test]
    fn mat3x4_mul_vec3() {
        let m = mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
        );
        let v = vec3f(13.0, 14.0, 15.0);
        // Row 0: 1*13 + 5*14 + 9*15 = 13+70+135 = 218
        // Row 1: 2*13 + 6*14 + 10*15 = 26+84+150 = 260
        // Row 2: 3*13 + 7*14 + 11*15 = 39+98+165 = 302
        // Row 3: 4*13 + 8*14 + 12*15 = 52+112+180 = 344
        let out: Vec4f = m * v;
        assert_eq!(out.to_array(), [218.0, 260.0, 302.0, 344.0]);
    }

    #[test]
    fn mat4x3_mul_vec4() {
        let m = mat4x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
            vec3f(10.0, 11.0, 12.0),
        );
        let v = vec4f(13.0, 14.0, 15.0, 16.0);
        // Row 0: 1*13 + 4*14 + 7*15 + 10*16 = 13+56+105+160 = 334
        // Row 1: 2*13 + 5*14 + 8*15 + 11*16 = 26+70+120+176 = 392
        // Row 2: 3*13 + 6*14 + 9*15 + 12*16 = 39+84+135+192 = 450
        let out: Vec3f = m * v;
        assert_eq!(out.to_array(), [334.0, 392.0, 450.0]);
    }

    // === Vector * matrix tests =================================================
    //
    // For `v * m`, output component i = dot(v, m.columns[i]).
    // Per the WGSL spec this is `transpose(transpose(m) * transpose(v))`.

    #[test]
    fn vec2_mul_mat2x2() {
        let v = vec2f(5.0, 6.0);
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        // Out[0] = dot(v, m[0]) = 5*1 + 6*2 = 17
        // Out[1] = dot(v, m[1]) = 5*3 + 6*4 = 39
        let out: Vec2f = v * m;
        assert_eq!(out.to_array(), [17.0, 39.0]);
    }

    #[test]
    fn vec3_mul_mat2x3() {
        // The example from issue #152.
        let v = vec3f(9.0, 8.0, 7.0);
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        // Out[0] = 9*1 + 8*2 + 7*3 = 9+16+21 = 46
        // Out[1] = 9*4 + 8*5 + 7*6 = 36+40+42 = 118
        let out: Vec2f = v * m;
        assert_eq!(out.to_array(), [46.0, 118.0]);
    }

    #[test]
    fn vec2_mul_mat3x2() {
        let v = vec2f(7.0, 8.0);
        let m = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        // Out[0] = 7*1 + 8*2 = 23
        // Out[1] = 7*3 + 8*4 = 53
        // Out[2] = 7*5 + 8*6 = 83
        let out: Vec3f = v * m;
        assert_eq!(out.to_array(), [23.0, 53.0, 83.0]);
    }

    #[test]
    fn vec4_mul_mat2x4() {
        let v = vec4f(9.0, 10.0, 11.0, 12.0);
        let m = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        // Out[0] = 9*1 + 10*2 + 11*3 + 12*4 = 9+20+33+48 = 110
        // Out[1] = 9*5 + 10*6 + 11*7 + 12*8 = 45+60+77+96 = 278
        let out: Vec2f = v * m;
        assert_eq!(out.to_array(), [110.0, 278.0]);
    }

    #[test]
    fn vec3_mul_mat4x3() {
        let v = vec3f(13.0, 14.0, 15.0);
        let m = mat4x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
            vec3f(10.0, 11.0, 12.0),
        );
        // Out[0] = 13*1 + 14*2 + 15*3 = 13+28+45 = 86
        // Out[1] = 13*4 + 14*5 + 15*6 = 52+70+90 = 212
        // Out[2] = 13*7 + 14*8 + 15*9 = 91+112+135 = 338
        // Out[3] = 13*10 + 14*11 + 15*12 = 130+154+180 = 464
        let out: Vec4f = v * m;
        assert_eq!(out.to_array(), [86.0, 212.0, 338.0, 464.0]);
    }

    /// Sanity check: `v * m` equals `transpose(m) * v` when interpreted as the
    /// column-vector-matrix product of the transpose. This verifies the two
    /// implementations agree with each other across all R sizes.
    #[test]
    fn vec_mul_mat_matches_transpose_mat_mul_vec() {
        // R=2
        {
            let v = vec2f(1.5, -2.5);
            let m = mat3x2f(vec2f(0.5, 1.0), vec2f(-1.5, 2.5), vec2f(3.0, -0.5));
            let lhs: Vec3f = v * m;
            let mt: Mat2x3f = transpose(m);
            let rhs: Vec3f = mt * v;
            assert_eq!(lhs.to_array(), rhs.to_array());
        }
        // R=3
        {
            let v = vec3f(1.0, 2.0, 3.0);
            let m = mat2x3f(vec3f(4.0, 5.0, 6.0), vec3f(7.0, 8.0, 9.0));
            let lhs: Vec2f = v * m;
            let mt: Mat3x2f = transpose(m);
            let rhs: Vec2f = mt * v;
            assert_eq!(lhs.to_array(), rhs.to_array());
        }
        // R=4
        {
            let v = vec4f(1.0, 2.0, 3.0, 4.0);
            let m = mat2x4f(vec4f(5.0, 6.0, 7.0, 8.0), vec4f(9.0, 10.0, 11.0, 12.0));
            let lhs: Vec2f = v * m;
            let mt: Mat4x2f = transpose(m);
            let rhs: Vec2f = mt * v;
            assert_eq!(lhs.to_array(), rhs.to_array());
        }
    }

    // === Matrix * matrix tests =================================================

    #[test]
    fn mat2x2_mul_mat2x2() {
        let a = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        let b = mat2x2f(vec2f(5.0, 6.0), vec2f(7.0, 8.0));
        // Column 0 of out = a * b[0] = a * vec2f(5,6) = (1*5+3*6, 2*5+4*6) = (23,34)
        // Column 1 of out = a * b[1] = a * vec2f(7,8) = (1*7+3*8, 2*7+4*8) = (31,46)
        let out: Mat2x2f = a * b;
        assert_eq!(out.columns[0].to_array(), [23.0, 34.0]);
        assert_eq!(out.columns[1].to_array(), [31.0, 46.0]);
    }

    #[test]
    fn mat2x3_mul_mat3x2() {
        // a is mat2x3 (K=2 cols, R=3 rows), b is mat3x2 (C=3 cols, K=2 rows),
        // out is mat3x3 (3 cols, 3 rows).
        let a = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let b = mat3x2f(vec2f(7.0, 8.0), vec2f(9.0, 10.0), vec2f(11.0, 12.0));
        // Column j of out = a * b[j].
        // out[0] = a * vec2f(7,8)  = (1*7+4*8, 2*7+5*8, 3*7+6*8)   = (39, 54, 69)
        // out[1] = a * vec2f(9,10) = (1*9+4*10, 2*9+5*10, 3*9+6*10) = (49, 68, 87)
        // out[2] = a * vec2f(11,12)= (1*11+4*12, 2*11+5*12, 3*11+6*12)=(59,82,105)
        let out: Mat3x3f = a * b;
        assert_eq!(out.columns[0].to_array(), [39.0, 54.0, 69.0]);
        assert_eq!(out.columns[1].to_array(), [49.0, 68.0, 87.0]);
        assert_eq!(out.columns[2].to_array(), [59.0, 82.0, 105.0]);
    }

    #[test]
    fn mat3x2_mul_mat2x3() {
        // a is mat3x2 (K=3 cols, R=2 rows), b is mat2x3 (C=2 cols, K=3 rows),
        // out is mat2x2 (2 cols, 2 rows).
        let a = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        let b = mat2x3f(vec3f(7.0, 8.0, 9.0), vec3f(10.0, 11.0, 12.0));
        // out[0] = a * b[0] = a * vec3f(7,8,9) = (1*7+3*8+5*9, 2*7+4*8+6*9) = (76, 100)
        // out[1] = a * b[1] = a * vec3f(10,11,12) = (1*10+3*11+5*12, 2*10+4*11+6*12) =
        // (103, 136)
        let out: Mat2x2f = a * b;
        assert_eq!(out.columns[0].to_array(), [76.0, 100.0]);
        assert_eq!(out.columns[1].to_array(), [103.0, 136.0]);
    }

    #[test]
    fn mat2x3_mul_mat4x2() {
        // a is mat2x3 (K=2, R=3), b is mat4x2 (C=4, K=2), out is mat4x3.
        let a = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let b = mat4x2f(
            vec2f(7.0, 8.0),
            vec2f(9.0, 10.0),
            vec2f(11.0, 12.0),
            vec2f(13.0, 14.0),
        );
        // out[j] = a * b[j]:
        // [0] = a * (7,8)   = (39, 54, 69)
        // [1] = a * (9,10)  = (49, 68, 87)
        // [2] = a * (11,12) = (59, 82, 105)
        // [3] = a * (13,14) = (69, 96, 123)
        let out: Mat4x3f = a * b;
        assert_eq!(out.columns[0].to_array(), [39.0, 54.0, 69.0]);
        assert_eq!(out.columns[1].to_array(), [49.0, 68.0, 87.0]);
        assert_eq!(out.columns[2].to_array(), [59.0, 82.0, 105.0]);
        assert_eq!(out.columns[3].to_array(), [69.0, 96.0, 123.0]);
    }

    #[test]
    fn mat3x4_mul_mat2x3() {
        // a is mat3x4 (K=3, R=4), b is mat2x3 (C=2, K=3), out is mat2x4.
        let a = mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
        );
        let b = mat2x3f(vec3f(13.0, 14.0, 15.0), vec3f(16.0, 17.0, 18.0));
        // out[0] = a * (13,14,15) =
        //   (1*13+5*14+9*15, 2*13+6*14+10*15, 3*13+7*14+11*15, 4*13+8*14+12*15)
        //  = (218, 260, 302, 344)
        // out[1] = a * (16,17,18) =
        //   (1*16+5*17+9*18, 2*16+6*17+10*18, 3*16+7*17+11*18, 4*16+8*17+12*18)
        //  = (263, 314, 365, 416)
        let out: Mat2x4f = a * b;
        assert_eq!(out.columns[0].to_array(), [218.0, 260.0, 302.0, 344.0]);
        assert_eq!(out.columns[1].to_array(), [263.0, 314.0, 365.0, 416.0]);
    }

    #[test]
    fn mat4x4_mul_mat4x4() {
        let a = mat4x4f(
            vec4f(1.0, 0.0, 0.0, 0.0),
            vec4f(0.0, 1.0, 0.0, 0.0),
            vec4f(0.0, 0.0, 1.0, 0.0),
            vec4f(0.0, 0.0, 0.0, 1.0),
        );
        let b = mat4x4f(
            vec4f(2.0, 3.0, 4.0, 5.0),
            vec4f(6.0, 7.0, 8.0, 9.0),
            vec4f(10.0, 11.0, 12.0, 13.0),
            vec4f(14.0, 15.0, 16.0, 17.0),
        );
        // Identity * b = b.
        let out: Mat4x4f = a * b;
        assert_eq!(out.columns[0].to_array(), [2.0, 3.0, 4.0, 5.0]);
        assert_eq!(out.columns[1].to_array(), [6.0, 7.0, 8.0, 9.0]);
        assert_eq!(out.columns[2].to_array(), [10.0, 11.0, 12.0, 13.0]);
        assert_eq!(out.columns[3].to_array(), [14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn mat2x4_mul_mat4x2() {
        // a is mat2x4 (K=2, R=4), b is mat4x2 (C=4, K=2), out is mat4x4.
        let a = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        let b = mat4x2f(
            vec2f(9.0, 10.0),
            vec2f(11.0, 12.0),
            vec2f(13.0, 14.0),
            vec2f(15.0, 16.0),
        );
        // out[j] = a * b[j]:
        // [0] = a * (9,10)  = (1*9+5*10, 2*9+6*10, 3*9+7*10, 4*9+8*10) = (59,78,97,116)
        // [1] = a * (11,12) = (1*11+5*12, 2*11+6*12, 3*11+7*12, 4*11+8*12) =
        // (71,94,117,140) [2] = a * (13,14) = (83,110,137,164)
        // [3] = a * (15,16) = (95,126,157,188)
        let out: Mat4x4f = a * b;
        assert_eq!(out.columns[0].to_array(), [59.0, 78.0, 97.0, 116.0]);
        assert_eq!(out.columns[1].to_array(), [71.0, 94.0, 117.0, 140.0]);
        assert_eq!(out.columns[2].to_array(), [83.0, 110.0, 137.0, 164.0]);
        assert_eq!(out.columns[3].to_array(), [95.0, 126.0, 157.0, 188.0]);
    }

    // === Scalar * matrix tests =================================================

    #[test]
    fn mat2x3_scalar_mul() {
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let out: Mat2x3f = m * 2.0;
        assert_eq!(out.columns[0].to_array(), [2.0, 4.0, 6.0]);
        assert_eq!(out.columns[1].to_array(), [8.0, 10.0, 12.0]);
        // Commutative.
        let out2: Mat2x3f = 2.0 * m;
        assert_eq!(out2.columns[0].to_array(), [2.0, 4.0, 6.0]);
        assert_eq!(out2.columns[1].to_array(), [8.0, 10.0, 12.0]);
    }

    #[test]
    fn mat4x2_scalar_mul() {
        let m = mat4x2f(
            vec2f(1.0, 2.0),
            vec2f(3.0, 4.0),
            vec2f(5.0, 6.0),
            vec2f(7.0, 8.0),
        );
        let out: Mat4x2f = m * 3.0;
        assert_eq!(out.columns[0].to_array(), [3.0, 6.0]);
        assert_eq!(out.columns[3].to_array(), [21.0, 24.0]);
    }

    // === GPU-side `#[wgsl]` compile check ======================================
    //
    // Ensures the proc-macro accepts `vec * mat` and `mat * vec` expressions
    // and renders them to WGSL using the `*` operator (the GPU side handles
    // the linear algebra natively).

    #[test]
    fn module_vec_mat_mul_compiles() {
        #[crate::wgsl(crate_path = crate)]
        pub mod vec_mat {
            #![allow(dead_code)]

            use crate::std::*;

            pub fn vec_times_mat(v: Vec3f, m: Mat2x3f) -> Vec2f {
                v * m
            }

            pub fn mat_times_vec(m: Mat2x3f, v: Vec2f) -> Vec3f {
                m * v
            }

            pub fn mat_times_mat(a: Mat2x3f, b: Mat3x2f) -> Mat3x3f {
                a * b
            }
        }
    }
}
