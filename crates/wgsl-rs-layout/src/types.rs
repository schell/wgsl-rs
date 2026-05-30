use crate::{Error, WgslLayout};

// ===== Scalars =====

impl WgslLayout for f32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        buf[..4].copy_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

impl WgslLayout for i32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        buf[..4].copy_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

impl WgslLayout for u32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        buf[..4].copy_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

impl WgslLayout for bool {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val: u32 = if *self { 1 } else { 0 };
        buf[..4].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }
}

// ===== Atomically accessible types =====

impl WgslLayout for wgsl_rs::std::Atomic<u32> {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val = wgsl_rs::std::atomic_load(self);
        buf[..4].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }
}

impl WgslLayout for wgsl_rs::std::Atomic<i32> {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val = wgsl_rs::std::atomic_load_i32(self);
        buf[..4].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }
}

// ===== Vectors (32-bit scalar elements) =====

impl<T: WgslLayout + wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec2<T> {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 8 {
            return Err(Error::BufferTooSmall {
                needed: 8,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::write_layout_bytes(&self.x, &mut buf[0..stride])?;
        T::write_layout_bytes(&self.y, &mut buf[stride..stride * 2])?;
        Ok(())
    }
}

impl<T: WgslLayout + wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec3<T> {
    const SIZE: usize = 12;
    const ALIGN: usize = 16;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 12 {
            return Err(Error::BufferTooSmall {
                needed: 12,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::write_layout_bytes(&self.x, &mut buf[0..stride])?;
        T::write_layout_bytes(&self.y, &mut buf[stride..stride * 2])?;
        T::write_layout_bytes(&self.z, &mut buf[stride * 2..stride * 3])?;
        Ok(())
    }
}

impl<T: WgslLayout + wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec4<T> {
    const SIZE: usize = 16;
    const ALIGN: usize = 16;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 16 {
            return Err(Error::BufferTooSmall {
                needed: 16,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::write_layout_bytes(&self.x, &mut buf[0..stride])?;
        T::write_layout_bytes(&self.y, &mut buf[stride..stride * 2])?;
        T::write_layout_bytes(&self.z, &mut buf[stride * 2..stride * 3])?;
        T::write_layout_bytes(&self.w, &mut buf[stride * 3..stride * 4])?;
        Ok(())
    }
}

// ===== Matrices =====

macro_rules! impl_mat_layout {
    ($ty:ty, $row_vec:ty, $cols:literal, $stride:expr, $align:expr, $size:expr) => {
        impl WgslLayout for $ty {
            const SIZE: usize = $size;
            const ALIGN: usize = $align;

            fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
                if buf.len() < $size {
                    return Err(Error::BufferTooSmall {
                        needed: $size,
                        actual: buf.len(),
                    });
                }
                for i in 0..$cols {
                    let col_offset = i * $stride;
                    let col: &$row_vec = &self[i];
                    col.write_layout_bytes(&mut buf[col_offset..])?;
                }
                Ok(())
            }
        }
    };
}

impl_mat_layout!(wgsl_rs::std::Mat2x2f, wgsl_rs::std::Vec2f, 2, 8, 8, 16);
impl_mat_layout!(wgsl_rs::std::Mat2x3f, wgsl_rs::std::Vec3f, 2, 16, 16, 32);
impl_mat_layout!(wgsl_rs::std::Mat2x4f, wgsl_rs::std::Vec4f, 2, 16, 16, 32);
impl_mat_layout!(wgsl_rs::std::Mat3x2f, wgsl_rs::std::Vec2f, 3, 8, 8, 24);
impl_mat_layout!(wgsl_rs::std::Mat3x3f, wgsl_rs::std::Vec3f, 3, 16, 16, 48);
impl_mat_layout!(wgsl_rs::std::Mat3x4f, wgsl_rs::std::Vec4f, 3, 16, 16, 48);
impl_mat_layout!(wgsl_rs::std::Mat4x2f, wgsl_rs::std::Vec2f, 4, 8, 8, 32);
impl_mat_layout!(wgsl_rs::std::Mat4x3f, wgsl_rs::std::Vec3f, 4, 16, 16, 64);
impl_mat_layout!(wgsl_rs::std::Mat4x4f, wgsl_rs::std::Vec4f, 4, 16, 16, 64);

// ===== Fixed-size arrays =====

impl<T: WgslLayout, const N: usize> WgslLayout for [T; N] {
    const SIZE: usize = N * crate::round_up(T::ALIGN, T::SIZE);
    const ALIGN: usize = T::ALIGN;

    fn write_layout_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        let stride = crate::round_up(T::ALIGN, T::SIZE);
        if buf.len() < N * stride {
            return Err(Error::BufferTooSmall {
                needed: N * stride,
                actual: buf.len(),
            });
        }
        for (i, elem) in self.iter().enumerate() {
            elem.write_layout_bytes(&mut buf[i * stride..])?;
        }
        Ok(())
    }
}

// ===== Runtime-sized arrays =====

impl<T: WgslLayout> WgslLayout for wgsl_rs::std::RuntimeArray<T> {
    const SIZE: usize = 0;
    const ALIGN: usize = T::ALIGN;

    fn write_layout_bytes(&self, _buf: &mut [u8]) -> Result<(), Error> {
        Ok(())
    }
}
