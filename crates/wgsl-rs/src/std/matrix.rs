//! Matrix implementations.
//!
//! Column-major matrices with public `columns` arrays, matching WGSL's
//! matrix types. Columns are accessed by index: `m[0]`, `m[1]`, etc.

use super::*;

/// A 2x2 column-major matrix of `f32` components (2 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat2x2f {
    columns: [Vec2f; 2],
}

/// A 2x3 column-major matrix of `f32` components (2 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat2x3f {
    columns: [Vec3f; 2],
}

/// A 2x4 column-major matrix of `f32` components (2 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat2x4f {
    columns: [Vec4f; 2],
}

/// A 3x2 column-major matrix of `f32` components (3 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat3x2f {
    columns: [Vec2f; 3],
}

/// A 3x3 column-major matrix of `f32` components (3 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat3x3f {
    columns: [Vec3f; 3],
}

/// A 3x4 column-major matrix of `f32` components (3 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat3x4f {
    columns: [Vec4f; 3],
}

/// A 4x2 column-major matrix of `f32` components (4 columns of `Vec2f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat4x2f {
    columns: [Vec2f; 4],
}

/// A 4x3 column-major matrix of `f32` components (4 columns of `Vec3f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat4x3f {
    columns: [Vec3f; 4],
}

/// A 4x4 column-major matrix of `f32` components (4 columns of `Vec4f`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Mat4x4f {
    columns: [Vec4f; 4],
}

/// Alias for `Mat2x2f`.
pub type Mat2f = Mat2x2f;
/// Alias for `Mat3x3f`.
pub type Mat3f = Mat3x3f;
/// Alias for `Mat4x4f`.
pub type Mat4f = Mat4x4f;

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

// Arithmetic: matrix * matrix, matrix * vector, matrix * scalar, scalar *
// matrix. Delegated to glam for the square matrix types.

impl std::ops::Mul<Mat2x2f> for Mat2x2f {
    type Output = Mat2x2f;
    fn mul(self, rhs: Mat2x2f) -> Mat2x2f {
        let g: glam::Mat2 = self.into();
        let rg: glam::Mat2 = rhs.into();
        (g * rg).into()
    }
}

impl std::ops::Mul<Vec2f> for Mat2x2f {
    type Output = Vec2f;
    fn mul(self, rhs: Vec2f) -> Vec2f {
        let g: glam::Mat2 = self.into();
        let gv: glam::Vec2 = rhs.into();
        (g * gv).into()
    }
}

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

impl std::ops::Mul<Mat3x3f> for Mat3x3f {
    type Output = Mat3x3f;
    fn mul(self, rhs: Mat3x3f) -> Mat3x3f {
        let g: glam::Mat3 = self.into();
        let rg: glam::Mat3 = rhs.into();
        (g * rg).into()
    }
}

impl std::ops::Mul<Vec3f> for Mat3x3f {
    type Output = Vec3f;
    fn mul(self, rhs: Vec3f) -> Vec3f {
        let g: glam::Mat3 = self.into();
        let gv: glam::Vec3 = rhs.into();
        (g * gv).into()
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

impl std::ops::Mul<Mat4x4f> for Mat4x4f {
    type Output = Mat4x4f;
    fn mul(self, rhs: Mat4x4f) -> Mat4x4f {
        let g: glam::Mat4 = self.into();
        let rg: glam::Mat4 = rhs.into();
        (g * rg).into()
    }
}

impl std::ops::Mul<Vec4f> for Mat4x4f {
    type Output = Vec4f;
    fn mul(self, rhs: Vec4f) -> Vec4f {
        let g: glam::Mat4 = self.into();
        let gv: glam::Vec4 = rhs.into();
        (g * gv).into()
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

            pub struct Uniforms {
                pub projection: Mat4f,
                pub modelview: Mat4f,
            }

            uniform!(group(0), binding(0), UNIFORMS: Uniforms);

            #[input]
            pub struct VertexInput {
                #[location(0)]
                pub position: Vec3f,
            }

            #[output]
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
}
