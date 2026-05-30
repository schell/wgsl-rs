use crate::{Error, WgslLayout};

// ===== Scalars =====

impl WgslLayout for f32 {
    const SIZE: usize = 4;
    const ALIGN: usize = 4;

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        Ok(f32::from_le_bytes(buf[..4].try_into().unwrap()))
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        Ok(i32::from_le_bytes(buf[..4].try_into().unwrap()))
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        Ok(u32::from_le_bytes(buf[..4].try_into().unwrap()))
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val = u32::from_le_bytes(buf[..4].try_into().unwrap());
        Ok(val != 0)
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val = u32::from_le_bytes(buf[..4].try_into().unwrap());
        let a = wgsl_rs::std::Atomic::default();
        wgsl_rs::std::atomic_store(&a, val);
        Ok(a)
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 4 {
            return Err(Error::BufferTooSmall {
                needed: 4,
                actual: buf.len(),
            });
        }
        let val = i32::from_le_bytes(buf[..4].try_into().unwrap());
        let a = wgsl_rs::std::Atomic::default();
        wgsl_rs::std::atomic_store_i32(&a, val);
        Ok(a)
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 8 {
            return Err(Error::BufferTooSmall {
                needed: 8,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        let x = T::layout_read_bytes(&buf[0..stride])?;
        let y = T::layout_read_bytes(&buf[stride..stride * 2])?;
        Ok(wgsl_rs::std::Vec2 { x, y })
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 8 {
            return Err(Error::BufferTooSmall {
                needed: 8,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::layout_write_bytes(&self.x, &mut buf[0..stride])?;
        T::layout_write_bytes(&self.y, &mut buf[stride..stride * 2])?;
        Ok(())
    }
}

impl<T: WgslLayout + wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec3<T> {
    const SIZE: usize = 12;
    const ALIGN: usize = 16;

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 12 {
            return Err(Error::BufferTooSmall {
                needed: 12,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        let x = T::layout_read_bytes(&buf[0..stride])?;
        let y = T::layout_read_bytes(&buf[stride..stride * 2])?;
        let z = T::layout_read_bytes(&buf[stride * 2..stride * 3])?;
        Ok(wgsl_rs::std::Vec3 { x, y, z })
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 12 {
            return Err(Error::BufferTooSmall {
                needed: 12,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::layout_write_bytes(&self.x, &mut buf[0..stride])?;
        T::layout_write_bytes(&self.y, &mut buf[stride..stride * 2])?;
        T::layout_write_bytes(&self.z, &mut buf[stride * 2..stride * 3])?;
        Ok(())
    }
}

impl<T: WgslLayout + wgsl_rs::std::WgslScalar> WgslLayout for wgsl_rs::std::Vec4<T> {
    const SIZE: usize = 16;
    const ALIGN: usize = 16;

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < 16 {
            return Err(Error::BufferTooSmall {
                needed: 16,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        let x = T::layout_read_bytes(&buf[0..stride])?;
        let y = T::layout_read_bytes(&buf[stride..stride * 2])?;
        let z = T::layout_read_bytes(&buf[stride * 2..stride * 3])?;
        let w = T::layout_read_bytes(&buf[stride * 3..stride * 4])?;
        Ok(wgsl_rs::std::Vec4 { x, y, z, w })
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.len() < 16 {
            return Err(Error::BufferTooSmall {
                needed: 16,
                actual: buf.len(),
            });
        }
        let stride = T::SIZE;
        T::layout_write_bytes(&self.x, &mut buf[0..stride])?;
        T::layout_write_bytes(&self.y, &mut buf[stride..stride * 2])?;
        T::layout_write_bytes(&self.z, &mut buf[stride * 2..stride * 3])?;
        T::layout_write_bytes(&self.w, &mut buf[stride * 3..stride * 4])?;
        Ok(())
    }
}

// ===== Matrices =====

macro_rules! impl_mat_layout {
    ($ty:ty, $row_vec:ty, $cols:literal, $stride:expr, $align:expr, $size:expr) => {
        impl WgslLayout for $ty {
            const SIZE: usize = $size;
            const ALIGN: usize = $align;

            fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
                if buf.len() < $size {
                    return Err(Error::BufferTooSmall {
                        needed: $size,
                        actual: buf.len(),
                    });
                }
                let mut mat = Self::default();
                for i in 0..$cols {
                    let col_offset = i * $stride;
                    mat[i] = <$row_vec>::layout_read_bytes(&buf[col_offset..])?;
                }
                Ok(mat)
            }

            fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
                if buf.len() < $size {
                    return Err(Error::BufferTooSmall {
                        needed: $size,
                        actual: buf.len(),
                    });
                }
                for i in 0..$cols {
                    let col_offset = i * $stride;
                    let col: &$row_vec = &self[i];
                    col.layout_write_bytes(&mut buf[col_offset..])?;
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

    fn layout_read_bytes(buf: &[u8]) -> Result<Self, Error> {
        let stride = crate::round_up(T::ALIGN, T::SIZE);
        if buf.len() < N * stride {
            return Err(Error::BufferTooSmall {
                needed: N * stride,
                actual: buf.len(),
            });
        }
        // Initialize with default values then overwrite
        let mut arr: [T; N] = unsafe { std::mem::zeroed() };
        for (i, slot) in arr.iter_mut().enumerate() {
            let elem = T::layout_read_bytes(&buf[i * stride..])?;
            *slot = elem;
        }
        Ok(arr)
    }

    fn layout_write_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        let stride = crate::round_up(T::ALIGN, T::SIZE);
        if buf.len() < N * stride {
            return Err(Error::BufferTooSmall {
                needed: N * stride,
                actual: buf.len(),
            });
        }
        for (i, elem) in self.iter().enumerate() {
            elem.layout_write_bytes(&mut buf[i * stride..])?;
        }
        Ok(())
    }
}

// ===== Runtime-sized arrays =====

impl<T: WgslLayout> WgslLayout for wgsl_rs::std::RuntimeArray<T> {
    const SIZE: usize = 0;
    const ALIGN: usize = T::ALIGN;

    fn layout_read_bytes(_buf: &[u8]) -> Result<Self, Error> {
        Ok(wgsl_rs::std::RuntimeArray::new())
    }

    fn layout_write_bytes(&self, _buf: &mut [u8]) -> Result<(), Error> {
        Ok(())
    }
}
