use crate::WgslLayout;

// ===== Scalars =====

impl WgslLayout for f32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl WgslLayout for i32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl WgslLayout for u32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl WgslLayout for bool {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

// ===== Atomically accessible types =====

impl WgslLayout for wgsl_rs::std::Atomic<u32> {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

impl WgslLayout for wgsl_rs::std::Atomic<i32> {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

// ===== Vectors (32-bit scalar elements) =====

impl<T: wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec2<T> {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}

impl<T: wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec3<T> {
    const SIZE: usize = 12;
    const ALIGN: usize = 16;
}

impl<T: wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec4<T> {
    const SIZE: usize = 16;
    const ALIGN: usize = 16;
}

// ===== Matrices =====

macro_rules! impl_mat_layout {
    ($ty:ty, $align:expr, $size:expr) => {
        impl WgslLayout for $ty {
            const SIZE: usize = $size;
            const ALIGN: usize = $align;
        }
    };
}

impl_mat_layout!(wgsl_rs::std::Mat2x2f, 8, 16);
impl_mat_layout!(wgsl_rs::std::Mat2x3f, 16, 32);
impl_mat_layout!(wgsl_rs::std::Mat2x4f, 16, 32);
impl_mat_layout!(wgsl_rs::std::Mat3x2f, 8, 24);
impl_mat_layout!(wgsl_rs::std::Mat3x3f, 16, 48);
impl_mat_layout!(wgsl_rs::std::Mat3x4f, 16, 48);
impl_mat_layout!(wgsl_rs::std::Mat4x2f, 8, 32);
impl_mat_layout!(wgsl_rs::std::Mat4x3f, 16, 64);
impl_mat_layout!(wgsl_rs::std::Mat4x4f, 16, 64);

// ===== Fixed-size arrays =====

impl<T: WgslLayout, const N: usize> WgslLayout for [T; N] {
    const SIZE: usize = N * crate::round_up(T::ALIGN, T::SIZE);
    const ALIGN: usize = T::ALIGN;
}

// ===== Runtime-sized arrays =====

impl<T: WgslLayout> WgslLayout for wgsl_rs::std::RuntimeArray<T> {
    const SIZE: usize = 0;
    const ALIGN: usize = T::ALIGN;
}
