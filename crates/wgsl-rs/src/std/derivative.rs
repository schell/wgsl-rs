//! WGSL derivative builtin functions.
//!
//! Provides the 9 derivative builtins used in fragment shaders:
//! `dpdx`, `dpdxCoarse`, `dpdxFine`, `dpdy`, `dpdyCoarse`, `dpdyFine`,
//! `fwidth`, `fwidthCoarse`, `fwidthFine`.
//!
//! On the CPU, when running inside
//! [`dispatch_fragments`](super::runtime::dispatch_fragments), these functions
//! compute real derivatives using the 2×2 quad context. Outside a dispatch
//! context they return zero.
//!
//! On the GPU, the `#[wgsl]` proc-macro transpiles these to the corresponding
//! WGSL builtin function calls.

use super::{Vec2f, Vec3f, Vec4f};

// ---------------------------------------------------------------------------
// Trait definitions
// ---------------------------------------------------------------------------

/// Partial derivative of `e` with respect to window x coordinates.
///
/// On CPU, delegates to [`dpdx_fine`].
pub trait DerivativeBuiltinDpdx {
    fn dpdx(self) -> Self;
}

/// Coarse partial derivative of `e` with respect to window x coordinates.
pub trait DerivativeBuiltinDpdxCoarse {
    fn dpdx_coarse(self) -> Self;
}

/// Fine partial derivative of `e` with respect to window x coordinates.
pub trait DerivativeBuiltinDpdxFine {
    fn dpdx_fine(self) -> Self;
}

/// Partial derivative of `e` with respect to window y coordinates.
///
/// On CPU, delegates to [`dpdy_fine`].
pub trait DerivativeBuiltinDpdy {
    fn dpdy(self) -> Self;
}

/// Coarse partial derivative of `e` with respect to window y coordinates.
pub trait DerivativeBuiltinDpdyCoarse {
    fn dpdy_coarse(self) -> Self;
}

/// Fine partial derivative of `e` with respect to window y coordinates.
pub trait DerivativeBuiltinDpdyFine {
    fn dpdy_fine(self) -> Self;
}

/// Returns `abs(dpdx(e)) + abs(dpdy(e))`.
///
/// On CPU, delegates to [`fwidth_fine`].
pub trait DerivativeBuiltinFwidth {
    fn fwidth(self) -> Self;
}

/// Returns `abs(dpdx_coarse(e)) + abs(dpdy_coarse(e))`.
pub trait DerivativeBuiltinFwidthCoarse {
    fn fwidth_coarse(self) -> Self;
}

/// Returns `abs(dpdx_fine(e)) + abs(dpdy_fine(e))`.
pub trait DerivativeBuiltinFwidthFine {
    fn fwidth_fine(self) -> Self;
}

// ---------------------------------------------------------------------------
// Free functions (public API matching WGSL names in snake_case)
// ---------------------------------------------------------------------------

/// Partial derivative of `e` with respect to window x coordinates.
pub fn dpdx<T: DerivativeBuiltinDpdx>(e: T) -> T {
    <T as DerivativeBuiltinDpdx>::dpdx(e)
}

/// Coarse partial derivative of `e` with respect to window x coordinates.
pub fn dpdx_coarse<T: DerivativeBuiltinDpdxCoarse>(e: T) -> T {
    <T as DerivativeBuiltinDpdxCoarse>::dpdx_coarse(e)
}

/// Fine partial derivative of `e` with respect to window x coordinates.
pub fn dpdx_fine<T: DerivativeBuiltinDpdxFine>(e: T) -> T {
    <T as DerivativeBuiltinDpdxFine>::dpdx_fine(e)
}

/// Partial derivative of `e` with respect to window y coordinates.
pub fn dpdy<T: DerivativeBuiltinDpdy>(e: T) -> T {
    <T as DerivativeBuiltinDpdy>::dpdy(e)
}

/// Coarse partial derivative of `e` with respect to window y coordinates.
pub fn dpdy_coarse<T: DerivativeBuiltinDpdyCoarse>(e: T) -> T {
    <T as DerivativeBuiltinDpdyCoarse>::dpdy_coarse(e)
}

/// Fine partial derivative of `e` with respect to window y coordinates.
pub fn dpdy_fine<T: DerivativeBuiltinDpdyFine>(e: T) -> T {
    <T as DerivativeBuiltinDpdyFine>::dpdy_fine(e)
}

/// Returns `abs(dpdx(e)) + abs(dpdy(e))`.
pub fn fwidth<T: DerivativeBuiltinFwidth>(e: T) -> T {
    <T as DerivativeBuiltinFwidth>::fwidth(e)
}

/// Returns `abs(dpdx_coarse(e)) + abs(dpdy_coarse(e))`.
pub fn fwidth_coarse<T: DerivativeBuiltinFwidthCoarse>(e: T) -> T {
    <T as DerivativeBuiltinFwidthCoarse>::fwidth_coarse(e)
}

/// Returns `abs(dpdx_fine(e)) + abs(dpdy_fine(e))`.
pub fn fwidth_fine<T: DerivativeBuiltinFwidthFine>(e: T) -> T {
    <T as DerivativeBuiltinFwidthFine>::fwidth_fine(e)
}

// ---------------------------------------------------------------------------
// Implementations — scalar f32
// ---------------------------------------------------------------------------

/// Calls `quad_fn` if a quad context is available, otherwise returns
/// `fallback`.
///
/// **Divergence note:** The underlying `QuadContext` uses barriers that require
/// all 4 quad invocations to participate. If a shader has non-uniform control
/// flow (e.g. an early return in only some invocations), the barrier will
/// deadlock rather than returning an indeterminate value like a GPU would. This
/// is a known limitation of the dispatch runtime as a whole; a timeout-based or
/// participation-tracking fallback would be needed to handle divergence safely.
#[cfg(feature = "dispatch-runtime")]
fn with_quad_or<R>(fallback: R, quad_fn: impl FnOnce(&super::runtime::QuadContext, u8) -> R) -> R {
    super::runtime::with_quad_context(|opt| match opt {
        Some((ctx, idx)) => quad_fn(ctx, idx),
        None => fallback,
    })
}

// --- dpdx variants (f32) ---

#[cfg(feature = "dispatch-runtime")]
impl DerivativeBuiltinDpdxFine for f32 {
    fn dpdx_fine(self) -> Self {
        with_quad_or(0.0, |ctx, idx| ctx.dpdx_fine_f32(idx, self))
    }
}

#[cfg(not(feature = "dispatch-runtime"))]
impl DerivativeBuiltinDpdxFine for f32 {
    fn dpdx_fine(self) -> Self {
        0.0
    }
}

#[cfg(feature = "dispatch-runtime")]
impl DerivativeBuiltinDpdxCoarse for f32 {
    fn dpdx_coarse(self) -> Self {
        with_quad_or(0.0, |ctx, idx| ctx.dpdx_coarse_f32(idx, self))
    }
}

#[cfg(not(feature = "dispatch-runtime"))]
impl DerivativeBuiltinDpdxCoarse for f32 {
    fn dpdx_coarse(self) -> Self {
        0.0
    }
}

impl DerivativeBuiltinDpdx for f32 {
    fn dpdx(self) -> Self {
        self.dpdx_fine()
    }
}

// --- dpdy variants (f32) ---

#[cfg(feature = "dispatch-runtime")]
impl DerivativeBuiltinDpdyFine for f32 {
    fn dpdy_fine(self) -> Self {
        with_quad_or(0.0, |ctx, idx| ctx.dpdy_fine_f32(idx, self))
    }
}

#[cfg(not(feature = "dispatch-runtime"))]
impl DerivativeBuiltinDpdyFine for f32 {
    fn dpdy_fine(self) -> Self {
        0.0
    }
}

#[cfg(feature = "dispatch-runtime")]
impl DerivativeBuiltinDpdyCoarse for f32 {
    fn dpdy_coarse(self) -> Self {
        with_quad_or(0.0, |ctx, idx| ctx.dpdy_coarse_f32(idx, self))
    }
}

#[cfg(not(feature = "dispatch-runtime"))]
impl DerivativeBuiltinDpdyCoarse for f32 {
    fn dpdy_coarse(self) -> Self {
        0.0
    }
}

impl DerivativeBuiltinDpdy for f32 {
    fn dpdy(self) -> Self {
        self.dpdy_fine()
    }
}

// --- fwidth variants (f32) ---

impl DerivativeBuiltinFwidthFine for f32 {
    fn fwidth_fine(self) -> Self {
        self.dpdx_fine().abs() + self.dpdy_fine().abs()
    }
}

impl DerivativeBuiltinFwidthCoarse for f32 {
    fn fwidth_coarse(self) -> Self {
        self.dpdx_coarse().abs() + self.dpdy_coarse().abs()
    }
}

impl DerivativeBuiltinFwidth for f32 {
    fn fwidth(self) -> Self {
        self.fwidth_fine()
    }
}

// ---------------------------------------------------------------------------
// Implementations — vectors (Vec2f, Vec3f, Vec4f)
// ---------------------------------------------------------------------------

/// Implements a fine dpdx derivative for a vector type.
macro_rules! impl_dpdx_fine_vec {
    ($ty:ty, $n:expr) => {
        #[cfg(feature = "dispatch-runtime")]
        impl DerivativeBuiltinDpdxFine for $ty {
            fn dpdx_fine(self) -> Self {
                let arr = self.to_array();
                let mut buf = [0.0f32; 4];
                buf[..$n].copy_from_slice(&arr);
                let result = with_quad_or([0.0f32; 4], |ctx, idx| {
                    ctx.dpdx_fine_components(idx, &buf, $n)
                });
                let mut out = [0.0f32; $n];
                out.copy_from_slice(&result[..$n]);
                Self::from_array(out)
            }
        }

        #[cfg(not(feature = "dispatch-runtime"))]
        impl DerivativeBuiltinDpdxFine for $ty {
            fn dpdx_fine(self) -> Self {
                Self::default()
            }
        }
    };
}

/// Implements a coarse dpdx derivative for a vector type.
macro_rules! impl_dpdx_coarse_vec {
    ($ty:ty, $n:expr) => {
        #[cfg(feature = "dispatch-runtime")]
        impl DerivativeBuiltinDpdxCoarse for $ty {
            fn dpdx_coarse(self) -> Self {
                let arr = self.to_array();
                let mut buf = [0.0f32; 4];
                buf[..$n].copy_from_slice(&arr);
                let result = with_quad_or([0.0f32; 4], |ctx, idx| {
                    ctx.dpdx_coarse_components(idx, &buf, $n)
                });
                let mut out = [0.0f32; $n];
                out.copy_from_slice(&result[..$n]);
                Self::from_array(out)
            }
        }

        #[cfg(not(feature = "dispatch-runtime"))]
        impl DerivativeBuiltinDpdxCoarse for $ty {
            fn dpdx_coarse(self) -> Self {
                Self::default()
            }
        }
    };
}

/// Implements a fine dpdy derivative for a vector type.
macro_rules! impl_dpdy_fine_vec {
    ($ty:ty, $n:expr) => {
        #[cfg(feature = "dispatch-runtime")]
        impl DerivativeBuiltinDpdyFine for $ty {
            fn dpdy_fine(self) -> Self {
                let arr = self.to_array();
                let mut buf = [0.0f32; 4];
                buf[..$n].copy_from_slice(&arr);
                let result = with_quad_or([0.0f32; 4], |ctx, idx| {
                    ctx.dpdy_fine_components(idx, &buf, $n)
                });
                let mut out = [0.0f32; $n];
                out.copy_from_slice(&result[..$n]);
                Self::from_array(out)
            }
        }

        #[cfg(not(feature = "dispatch-runtime"))]
        impl DerivativeBuiltinDpdyFine for $ty {
            fn dpdy_fine(self) -> Self {
                Self::default()
            }
        }
    };
}

/// Implements a coarse dpdy derivative for a vector type.
macro_rules! impl_dpdy_coarse_vec {
    ($ty:ty, $n:expr) => {
        #[cfg(feature = "dispatch-runtime")]
        impl DerivativeBuiltinDpdyCoarse for $ty {
            fn dpdy_coarse(self) -> Self {
                let arr = self.to_array();
                let mut buf = [0.0f32; 4];
                buf[..$n].copy_from_slice(&arr);
                let result = with_quad_or([0.0f32; 4], |ctx, idx| {
                    ctx.dpdy_coarse_components(idx, &buf, $n)
                });
                let mut out = [0.0f32; $n];
                out.copy_from_slice(&result[..$n]);
                Self::from_array(out)
            }
        }

        #[cfg(not(feature = "dispatch-runtime"))]
        impl DerivativeBuiltinDpdyCoarse for $ty {
            fn dpdy_coarse(self) -> Self {
                Self::default()
            }
        }
    };
}

/// Implements the default dpdx (delegates to fine) for a vector type.
macro_rules! impl_dpdx_default_vec {
    ($ty:ty) => {
        impl DerivativeBuiltinDpdx for $ty {
            fn dpdx(self) -> Self {
                self.dpdx_fine()
            }
        }
    };
}

/// Implements the default dpdy (delegates to fine) for a vector type.
macro_rules! impl_dpdy_default_vec {
    ($ty:ty) => {
        impl DerivativeBuiltinDpdy for $ty {
            fn dpdy(self) -> Self {
                self.dpdy_fine()
            }
        }
    };
}

/// Implements fwidth_fine for a vector type.
macro_rules! impl_fwidth_fine_vec {
    ($ty:ty, $n:expr) => {
        impl DerivativeBuiltinFwidthFine for $ty {
            fn fwidth_fine(self) -> Self {
                let dx = self.dpdx_fine();
                let dy = self.dpdy_fine();
                let dx_arr = dx.to_array();
                let dy_arr = dy.to_array();
                let mut out = [0.0f32; $n];
                for i in 0..$n {
                    out[i] = dx_arr[i].abs() + dy_arr[i].abs();
                }
                Self::from_array(out)
            }
        }
    };
}

/// Implements fwidth_coarse for a vector type.
macro_rules! impl_fwidth_coarse_vec {
    ($ty:ty, $n:expr) => {
        impl DerivativeBuiltinFwidthCoarse for $ty {
            fn fwidth_coarse(self) -> Self {
                let dx = self.dpdx_coarse();
                let dy = self.dpdy_coarse();
                let dx_arr = dx.to_array();
                let dy_arr = dy.to_array();
                let mut out = [0.0f32; $n];
                for i in 0..$n {
                    out[i] = dx_arr[i].abs() + dy_arr[i].abs();
                }
                Self::from_array(out)
            }
        }
    };
}

/// Implements the default fwidth (delegates to fine) for a vector type.
macro_rules! impl_fwidth_default_vec {
    ($ty:ty) => {
        impl DerivativeBuiltinFwidth for $ty {
            fn fwidth(self) -> Self {
                self.fwidth_fine()
            }
        }
    };
}

/// Implements all 9 derivative traits for a vector type.
macro_rules! impl_all_derivative_vec {
    ($ty:ty, $n:expr) => {
        impl_dpdx_fine_vec!($ty, $n);
        impl_dpdx_coarse_vec!($ty, $n);
        impl_dpdx_default_vec!($ty);
        impl_dpdy_fine_vec!($ty, $n);
        impl_dpdy_coarse_vec!($ty, $n);
        impl_dpdy_default_vec!($ty);
        impl_fwidth_fine_vec!($ty, $n);
        impl_fwidth_coarse_vec!($ty, $n);
        impl_fwidth_default_vec!($ty);
    };
}

impl_all_derivative_vec!(Vec2f, 2);
impl_all_derivative_vec!(Vec3f, 3);
impl_all_derivative_vec!(Vec4f, 4);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpdx_outside_dispatch_returns_zero() {
        assert_eq!(dpdx(42.0f32), 0.0);
        assert_eq!(dpdx_fine(42.0f32), 0.0);
        assert_eq!(dpdx_coarse(42.0f32), 0.0);
    }

    #[test]
    fn dpdy_outside_dispatch_returns_zero() {
        assert_eq!(dpdy(42.0f32), 0.0);
        assert_eq!(dpdy_fine(42.0f32), 0.0);
        assert_eq!(dpdy_coarse(42.0f32), 0.0);
    }

    #[test]
    fn fwidth_outside_dispatch_returns_zero() {
        assert_eq!(fwidth(42.0f32), 0.0);
        assert_eq!(fwidth_fine(42.0f32), 0.0);
        assert_eq!(fwidth_coarse(42.0f32), 0.0);
    }

    #[test]
    fn vec_derivatives_outside_dispatch_return_zero() {
        let v2 = Vec2f { x: 1.0, y: 2.0 };
        let v3 = Vec3f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let v4 = Vec4f {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 4.0,
        };

        assert_eq!(dpdx(v2), Vec2f::default());
        assert_eq!(dpdy(v3), Vec3f::default());
        assert_eq!(fwidth(v4), Vec4f::default());
    }

    #[cfg(feature = "dispatch-runtime")]
    mod dispatch {
        use super::*;
        use crate::std::runtime::dispatch_fragments;

        #[test]
        fn dpdx_fine_scalar_in_dispatch() {
            // In a 2x2 grid, position.x values are 0.5, 1.5 (with pixel-center
            // offset). dpdx_fine of position.x should be 1.0 everywhere.
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| dpdx_fine(builtins.position.x),
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdx_fine(position.x) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn dpdy_fine_scalar_in_dispatch() {
            // dpdy_fine of position.y should be 1.0 everywhere.
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| dpdy_fine(builtins.position.y),
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdy_fine(position.y) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn dpdx_coarse_scalar_in_dispatch() {
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| dpdx_coarse(builtins.position.x),
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdx_coarse(position.x) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn dpdy_coarse_scalar_in_dispatch() {
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| dpdy_coarse(builtins.position.y),
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdy_coarse(position.y) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn dpdx_default_delegates_to_fine() {
            let results =
                dispatch_fragments(2, 2, |_, _| (), |builtins, _| dpdx(builtins.position.x));
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdx(position.x) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn fwidth_scalar_in_dispatch() {
            // fwidth(position.x) = abs(dpdx(position.x)) + abs(dpdy(position.x))
            // dpdx(position.x) = 1.0, dpdy(position.x) = 0.0
            // So fwidth = 1.0
            let results =
                dispatch_fragments(2, 2, |_, _| (), |builtins, _| fwidth(builtins.position.x));
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "fwidth(position.x) should be 1.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn fwidth_scalar_both_axes() {
            // For a value that varies in both axes equally:
            // v = position.x + position.y
            // dpdx(v) = 1.0, dpdy(v) = 1.0
            // fwidth(v) = 2.0
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| {
                    let v = builtins.position.x + builtins.position.y;
                    fwidth(v)
                },
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 2.0).abs() < 1e-6,
                        "fwidth(position.x + position.y) should be 2.0, got {v}"
                    );
                }
            }
        }

        #[test]
        fn dpdx_fine_vec2_in_dispatch() {
            // Derivative of (position.x, position.y) w.r.t. x:
            // dpdx(position.x) = 1.0, dpdx(position.y) = 0.0
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| {
                    dpdx_fine(Vec2f {
                        x: builtins.position.x,
                        y: builtins.position.y,
                    })
                },
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v.x - 1.0).abs() < 1e-6,
                        "dpdx_fine(position).x should be 1.0, got {}",
                        v.x
                    );
                    assert!(
                        v.y.abs() < 1e-6,
                        "dpdx_fine(position).y should be 0.0, got {}",
                        v.y
                    );
                }
            }
        }

        #[test]
        fn dpdy_fine_vec3_in_dispatch() {
            // Derivative of (position.x, position.y, const) w.r.t. y:
            // dpdy(position.x) = 0.0, dpdy(position.y) = 1.0, dpdy(5.0) = 0.0
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| {
                    dpdy_fine(Vec3f {
                        x: builtins.position.x,
                        y: builtins.position.y,
                        z: 5.0,
                    })
                },
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(v.x.abs() < 1e-6, "dpdy(position.x) should be 0.0");
                    assert!((v.y - 1.0).abs() < 1e-6, "dpdy(position.y) should be 1.0");
                    assert!(v.z.abs() < 1e-6, "dpdy(constant) should be 0.0");
                }
            }
        }

        #[test]
        fn fwidth_vec4_in_dispatch() {
            // fwidth of (position.x, position.y, 0, 0):
            // fwidth.x = |dpdx(pos.x)| + |dpdy(pos.x)| = 1 + 0 = 1
            // fwidth.y = |dpdx(pos.y)| + |dpdy(pos.y)| = 0 + 1 = 1
            // fwidth.z = 0, fwidth.w = 0
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| {
                    fwidth(Vec4f {
                        x: builtins.position.x,
                        y: builtins.position.y,
                        z: 0.0,
                        w: 0.0,
                    })
                },
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!((v.x - 1.0).abs() < 1e-6, "fwidth(position).x should be 1.0");
                    assert!((v.y - 1.0).abs() < 1e-6, "fwidth(position).y should be 1.0");
                    assert!(v.z.abs() < 1e-6, "fwidth(0).z should be 0.0");
                    assert!(v.w.abs() < 1e-6, "fwidth(0).w should be 0.0");
                }
            }
        }

        #[test]
        fn larger_grid_dpdx() {
            // Test on a larger grid (4x4) to exercise multiple quads.
            let results =
                dispatch_fragments(4, 4, |_, _| (), |builtins, _| dpdx(builtins.position.x));
            assert_eq!(results.len(), 4);
            for row in &results {
                assert_eq!(row.len(), 4);
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "dpdx(position.x) should be 1.0 on larger grid, got {v}"
                    );
                }
            }
        }

        #[test]
        fn odd_grid_derivatives() {
            // Test on a 3x3 grid (partial quads at edges).
            let results = dispatch_fragments(
                3,
                3,
                |_, _| (),
                |builtins, _| (dpdx(builtins.position.x), dpdy(builtins.position.y)),
            );
            assert_eq!(results.len(), 3);
            for row in &results {
                assert_eq!(row.len(), 3);
                for val in row {
                    let (dx, dy) = val.expect("non-helper should have output");
                    assert!(
                        (dx - 1.0).abs() < 1e-6,
                        "dpdx(position.x) should be 1.0, got {dx}"
                    );
                    assert!(
                        (dy - 1.0).abs() < 1e-6,
                        "dpdy(position.y) should be 1.0, got {dy}"
                    );
                }
            }
        }

        #[test]
        fn nonlinear_value_derivatives() {
            // Test derivative of a nonlinear function: v = position.x * position.x
            // Quad at (0,0): positions 0.5, 1.5 (top row), 0.5, 1.5 (bottom row)
            // Values: 0.25, 2.25, 0.25, 2.25
            // dpdx_fine for top row: 2.25 - 0.25 = 2.0
            // dpdx_fine for bottom row: 2.25 - 0.25 = 2.0
            let results = dispatch_fragments(
                2,
                2,
                |_, _| (),
                |builtins, _| {
                    let x = builtins.position.x;
                    dpdx_fine(x * x)
                },
            );
            for row in &results {
                for val in row {
                    let v = val.expect("non-helper should have output");
                    assert!(
                        (v - 2.0).abs() < 1e-6,
                        "dpdx_fine(x*x) on [0.5,1.5] quad should be 2.0, got {v}"
                    );
                }
            }
        }
    }
}
