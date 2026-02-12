//! Bit manipulation primitives used in the "std" WGSL library.

use super::*;

/// Provides the numeric built-in function `countLeadingZeros`.
pub trait NumericBuiltinCountLeadingZeros {
    /// The number of consecutive 0 bits starting from the most significant
    /// bit. Also known as "clz" in some languages.
    fn count_leading_zeros(self) -> Self;
}

/// The number of consecutive 0 bits starting from the most significant bit
/// of e. Component-wise when T is a vector.
/// Also known as "clz" in some languages.
pub fn count_leading_zeros<T: NumericBuiltinCountLeadingZeros>(e: T) -> T {
    <T as NumericBuiltinCountLeadingZeros>::count_leading_zeros(e)
}

mod count_leading_zeros {
    use super::*;

    impl NumericBuiltinCountLeadingZeros for u32 {
        fn count_leading_zeros(self) -> Self {
            self.leading_zeros()
        }
    }

    impl NumericBuiltinCountLeadingZeros for i32 {
        fn count_leading_zeros(self) -> Self {
            self.leading_zeros() as i32
        }
    }

    macro_rules! impl_clz_vec {
        ($ty:ty, $scalar:ty) => {
            impl NumericBuiltinCountLeadingZeros for $ty {
                fn count_leading_zeros(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = (*elem).leading_zeros() as $scalar;
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_clz_vec!(Vec2u, u32);
    impl_clz_vec!(Vec3u, u32);
    impl_clz_vec!(Vec4u, u32);
    impl_clz_vec!(Vec2i, i32);
    impl_clz_vec!(Vec3i, i32);
    impl_clz_vec!(Vec4i, i32);
}

/// Provides the numeric built-in function `countOneBits`.
pub trait NumericBuiltinCountOneBits {
    /// The number of 1 bits in the representation of e.
    /// Also known as "population count".
    fn count_one_bits(self) -> Self;
}

/// The number of 1 bits in the representation of e.
/// Also known as "population count".
/// Component-wise when T is a vector.
pub fn count_one_bits<T: NumericBuiltinCountOneBits>(e: T) -> T {
    <T as NumericBuiltinCountOneBits>::count_one_bits(e)
}

mod count_one_bits {
    use super::*;

    impl NumericBuiltinCountOneBits for u32 {
        fn count_one_bits(self) -> Self {
            self.count_ones()
        }
    }

    impl NumericBuiltinCountOneBits for i32 {
        fn count_one_bits(self) -> Self {
            self.count_ones() as i32
        }
    }

    macro_rules! impl_popcount_vec {
        ($ty:ty, $scalar:ty) => {
            impl NumericBuiltinCountOneBits for $ty {
                fn count_one_bits(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = (*elem).count_ones() as $scalar;
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_popcount_vec!(Vec2u, u32);
    impl_popcount_vec!(Vec3u, u32);
    impl_popcount_vec!(Vec4u, u32);
    impl_popcount_vec!(Vec2i, i32);
    impl_popcount_vec!(Vec3i, i32);
    impl_popcount_vec!(Vec4i, i32);
}

/// Provides the numeric built-in function `countTrailingZeros`.
pub trait NumericBuiltinCountTrailingZeros {
    /// The number of consecutive 0 bits starting from the least significant
    /// bit. Also known as "ctz" in some languages.
    fn count_trailing_zeros(self) -> Self;
}

/// The number of consecutive 0 bits starting from the least significant bit
/// of e. Component-wise when T is a vector.
/// Also known as "ctz" in some languages.
pub fn count_trailing_zeros<T: NumericBuiltinCountTrailingZeros>(e: T) -> T {
    <T as NumericBuiltinCountTrailingZeros>::count_trailing_zeros(e)
}

mod count_trailing_zeros {
    use super::*;

    impl NumericBuiltinCountTrailingZeros for u32 {
        fn count_trailing_zeros(self) -> Self {
            self.trailing_zeros()
        }
    }

    impl NumericBuiltinCountTrailingZeros for i32 {
        fn count_trailing_zeros(self) -> Self {
            self.trailing_zeros() as i32
        }
    }

    macro_rules! impl_ctz_vec {
        ($ty:ty, $scalar:ty) => {
            impl NumericBuiltinCountTrailingZeros for $ty {
                fn count_trailing_zeros(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = (*elem).trailing_zeros() as $scalar;
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_ctz_vec!(Vec2u, u32);
    impl_ctz_vec!(Vec3u, u32);
    impl_ctz_vec!(Vec4u, u32);
    impl_ctz_vec!(Vec2i, i32);
    impl_ctz_vec!(Vec3i, i32);
    impl_ctz_vec!(Vec4i, i32);
}

/// Provides the numeric built-in function `reverseBits`.
pub trait NumericBuiltinReverseBits {
    /// Reverses the bits in e: The bit at position k of the result equals
    /// the bit at position 31-k of e.
    fn reverse_bits(self) -> Self;
}

/// Reverses the bits in e: The bit at position k of the result equals
/// the bit at position 31-k of e.
/// Component-wise when T is a vector.
pub fn reverse_bits<T: NumericBuiltinReverseBits>(e: T) -> T {
    <T as NumericBuiltinReverseBits>::reverse_bits(e)
}

mod reverse_bits {
    use super::*;

    impl NumericBuiltinReverseBits for u32 {
        fn reverse_bits(self) -> Self {
            self.reverse_bits()
        }
    }

    impl NumericBuiltinReverseBits for i32 {
        fn reverse_bits(self) -> Self {
            self.reverse_bits()
        }
    }

    macro_rules! impl_reverse_bits_vec {
        ($ty:ty) => {
            impl NumericBuiltinReverseBits for $ty {
                fn reverse_bits(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = (*elem).reverse_bits();
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_reverse_bits_vec!(Vec2u);
    impl_reverse_bits_vec!(Vec3u);
    impl_reverse_bits_vec!(Vec4u);
    impl_reverse_bits_vec!(Vec2i);
    impl_reverse_bits_vec!(Vec3i);
    impl_reverse_bits_vec!(Vec4i);
}

/// Provides the numeric built-in function `firstLeadingBit`.
pub trait NumericBuiltinFirstLeadingBit {
    /// For unsigned: Returns T(-1) if e is zero, otherwise the position of
    /// the most significant 1 bit in e.
    /// For signed: Returns -1 if e is 0 or -1, otherwise the position of
    /// the most significant bit in e that is different from e's
    /// sign bit.
    fn first_leading_bit(self) -> Self;
}

/// For scalar T, the result is:
/// - T(-1) if e is zero (unsigned) or if e is 0 or -1 (signed).
/// - Otherwise the position of the most significant 1 bit (unsigned) or the
///   position of the most significant bit different from the sign bit (signed).
///
/// Component-wise when T is a vector.
pub fn first_leading_bit<T: NumericBuiltinFirstLeadingBit>(e: T) -> T {
    <T as NumericBuiltinFirstLeadingBit>::first_leading_bit(e)
}

mod first_leading_bit {
    use super::*;

    impl NumericBuiltinFirstLeadingBit for u32 {
        fn first_leading_bit(self) -> Self {
            if self == 0 {
                u32::MAX // T(-1) for unsigned
            } else {
                31 - self.leading_zeros()
            }
        }
    }

    impl NumericBuiltinFirstLeadingBit for i32 {
        fn first_leading_bit(self) -> Self {
            if self == 0 || self == -1 {
                -1
            } else if self > 0 {
                // For positive: find position of MSB (same as unsigned)
                31 - self.leading_zeros() as i32
            } else {
                // For negative: find first bit that differs from sign bit (1)
                // This is equivalent to finding the first 0 bit from the MSB
                // We can do this by inverting and finding the first 1 bit
                31 - self.leading_ones() as i32
            }
        }
    }

    macro_rules! impl_first_leading_bit_vec_u {
        ($ty:ty) => {
            impl NumericBuiltinFirstLeadingBit for $ty {
                fn first_leading_bit(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = if *elem == 0 {
                            u32::MAX
                        } else {
                            31 - (*elem).leading_zeros()
                        };
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_first_leading_bit_vec_u!(Vec2u);
    impl_first_leading_bit_vec_u!(Vec3u);
    impl_first_leading_bit_vec_u!(Vec4u);

    macro_rules! impl_first_leading_bit_vec_i {
        ($ty:ty) => {
            impl NumericBuiltinFirstLeadingBit for $ty {
                fn first_leading_bit(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = if *elem == 0 || *elem == -1 {
                            -1
                        } else if *elem > 0 {
                            31 - (*elem).leading_zeros() as i32
                        } else {
                            31 - (*elem).leading_ones() as i32
                        };
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_first_leading_bit_vec_i!(Vec2i);
    impl_first_leading_bit_vec_i!(Vec3i);
    impl_first_leading_bit_vec_i!(Vec4i);
}

/// Provides the numeric built-in function `firstTrailingBit`.
pub trait NumericBuiltinFirstTrailingBit {
    /// For scalar T, the result is:
    /// - T(-1) if e is zero.
    /// - Otherwise the position of the least significant 1 bit in e.
    fn first_trailing_bit(self) -> Self;
}

/// For scalar T, the result is:
/// - T(-1) if e is zero.
/// - Otherwise the position of the least significant 1 bit in e.
///
/// Component-wise when T is a vector.
pub fn first_trailing_bit<T: NumericBuiltinFirstTrailingBit>(e: T) -> T {
    <T as NumericBuiltinFirstTrailingBit>::first_trailing_bit(e)
}

mod first_trailing_bit {
    use super::*;

    impl NumericBuiltinFirstTrailingBit for u32 {
        fn first_trailing_bit(self) -> Self {
            if self == 0 {
                u32::MAX // T(-1) for unsigned
            } else {
                self.trailing_zeros()
            }
        }
    }

    impl NumericBuiltinFirstTrailingBit for i32 {
        fn first_trailing_bit(self) -> Self {
            if self == 0 {
                -1
            } else {
                self.trailing_zeros() as i32
            }
        }
    }

    macro_rules! impl_first_trailing_bit_vec_u {
        ($ty:ty) => {
            impl NumericBuiltinFirstTrailingBit for $ty {
                fn first_trailing_bit(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = if *elem == 0 {
                            u32::MAX
                        } else {
                            (*elem).trailing_zeros()
                        };
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_first_trailing_bit_vec_u!(Vec2u);
    impl_first_trailing_bit_vec_u!(Vec3u);
    impl_first_trailing_bit_vec_u!(Vec4u);

    macro_rules! impl_first_trailing_bit_vec_i {
        ($ty:ty) => {
            impl NumericBuiltinFirstTrailingBit for $ty {
                fn first_trailing_bit(self) -> Self {
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = if *elem == 0 {
                            -1
                        } else {
                            (*elem).trailing_zeros() as i32
                        };
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_first_trailing_bit_vec_i!(Vec2i);
    impl_first_trailing_bit_vec_i!(Vec3i);
    impl_first_trailing_bit_vec_i!(Vec4i);
}

/// Provides the numeric built-in function `extractBits`.
pub trait NumericBuiltinExtractBits {
    /// Reads bits from an integer.
    /// For signed types: with sign extension.
    /// For unsigned types: without sign extension (zero extension).
    fn extract_bits(self, offset: u32, count: u32) -> Self;
}

/// Reads bits from an integer.
/// - `offset`: The bit position to start extracting from.
/// - `count`: The number of bits to extract.
///
/// For signed types: with sign extension from bit `count-1`.
/// For unsigned types: without sign extension (zero extension).
/// Component-wise when T is a vector (offset and count apply to all
/// components).
///
/// Note: In debug builds, asserts that `offset + count <= 32`.
pub fn extract_bits<T: NumericBuiltinExtractBits>(e: T, offset: u32, count: u32) -> T {
    debug_assert!(
        offset + count <= 32,
        "extractBits: offset ({}) + count ({}) must not exceed 32",
        offset,
        count
    );
    <T as NumericBuiltinExtractBits>::extract_bits(e, offset, count)
}

mod extract_bits {
    use super::*;

    impl NumericBuiltinExtractBits for u32 {
        fn extract_bits(self, offset: u32, count: u32) -> Self {
            let w = 32u32;
            let o = std::cmp::Ord::min(offset, w);
            let c = std::cmp::Ord::min(count, w - o);
            if c == 0 {
                return 0;
            }
            // Create mask for c bits
            let mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
            (self >> o) & mask
        }
    }

    impl NumericBuiltinExtractBits for i32 {
        fn extract_bits(self, offset: u32, count: u32) -> Self {
            let w = 32u32;
            let o = std::cmp::Ord::min(offset, w);
            let c = std::cmp::Ord::min(count, w - o);
            if c == 0 {
                return 0;
            }
            // Extract bits as unsigned first
            let mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
            let extracted = ((self as u32) >> o) & mask;
            // Sign extend from bit c-1
            // If bit c-1 is set, we need to fill the upper bits with 1s
            let sign_bit = 1u32 << (c - 1);
            if extracted & sign_bit != 0 {
                // Sign extend: set all bits above c-1 to 1
                let sign_extend_mask = !mask;
                (extracted | sign_extend_mask) as i32
            } else {
                extracted as i32
            }
        }
    }

    macro_rules! impl_extract_bits_vec_u {
        ($ty:ty) => {
            impl NumericBuiltinExtractBits for $ty {
                fn extract_bits(self, offset: u32, count: u32) -> Self {
                    let w = 32u32;
                    let o = std::cmp::Ord::min(offset, w);
                    let c = std::cmp::Ord::min(count, w - o);
                    if c == 0 {
                        return Self {
                            inner: Default::default(),
                        };
                    }
                    let mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        *elem = (*elem >> o) & mask;
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_extract_bits_vec_u!(Vec2u);
    impl_extract_bits_vec_u!(Vec3u);
    impl_extract_bits_vec_u!(Vec4u);

    macro_rules! impl_extract_bits_vec_i {
        ($ty:ty) => {
            impl NumericBuiltinExtractBits for $ty {
                fn extract_bits(self, offset: u32, count: u32) -> Self {
                    let w = 32u32;
                    let o = std::cmp::Ord::min(offset, w);
                    let c = std::cmp::Ord::min(count, w - o);
                    if c == 0 {
                        return Self {
                            inner: Default::default(),
                        };
                    }
                    let mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
                    let sign_bit = 1u32 << (c - 1);
                    let sign_extend_mask = !mask;
                    let mut array = self.inner.to_array();
                    for elem in array.iter_mut() {
                        let extracted = ((*elem as u32) >> o) & mask;
                        *elem = if extracted & sign_bit != 0 {
                            (extracted | sign_extend_mask) as i32
                        } else {
                            extracted as i32
                        };
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_extract_bits_vec_i!(Vec2i);
    impl_extract_bits_vec_i!(Vec3i);
    impl_extract_bits_vec_i!(Vec4i);
}

/// Provides the numeric built-in function `insertBits`.
pub trait NumericBuiltinInsertBits {
    /// Sets bits in an integer.
    /// Bits `offset..offset+count-1` of the result are copied from bits
    /// `0..count-1` of `newbits`. Other bits of the result are copied from
    /// `e`.
    fn insert_bits(self, newbits: Self, offset: u32, count: u32) -> Self;
}

/// Sets bits in an integer.
/// - `e`: The original value.
/// - `newbits`: The bits to insert.
/// - `offset`: The bit position to start inserting at.
/// - `count`: The number of bits to insert.
///
/// Bits `offset..offset+count-1` of the result are copied from bits
/// `0..count-1` of `newbits`. Other bits of the result are copied from `e`.
/// Component-wise when T is a vector (offset and count apply to all
/// components).
///
/// Note: In debug builds, asserts that `offset + count <= 32`.
pub fn insert_bits<T: NumericBuiltinInsertBits>(e: T, newbits: T, offset: u32, count: u32) -> T {
    debug_assert!(
        offset + count <= 32,
        "insertBits: offset ({}) + count ({}) must not exceed 32",
        offset,
        count
    );
    <T as NumericBuiltinInsertBits>::insert_bits(e, newbits, offset, count)
}

mod insert_bits {
    use super::*;

    impl NumericBuiltinInsertBits for u32 {
        fn insert_bits(self, newbits: Self, offset: u32, count: u32) -> Self {
            let w = 32u32;
            let o = std::cmp::Ord::min(offset, w);
            let c = std::cmp::Ord::min(count, w - o);
            if c == 0 {
                return self;
            }
            // Create mask for c bits at position o
            let low_mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
            let mask = low_mask << o;
            // Clear the bits at position o..o+c in self, then insert newbits
            (self & !mask) | ((newbits & low_mask) << o)
        }
    }

    impl NumericBuiltinInsertBits for i32 {
        fn insert_bits(self, newbits: Self, offset: u32, count: u32) -> Self {
            let w = 32u32;
            let o = std::cmp::Ord::min(offset, w);
            let c = std::cmp::Ord::min(count, w - o);
            if c == 0 {
                return self;
            }
            // Work with unsigned values for bit manipulation
            let e = self as u32;
            let nb = newbits as u32;
            let low_mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
            let mask = low_mask << o;
            ((e & !mask) | ((nb & low_mask) << o)) as i32
        }
    }

    macro_rules! impl_insert_bits_vec_u {
        ($ty:ty) => {
            impl NumericBuiltinInsertBits for $ty {
                fn insert_bits(self, newbits: Self, offset: u32, count: u32) -> Self {
                    let w = 32u32;
                    let o = std::cmp::Ord::min(offset, w);
                    let c = std::cmp::Ord::min(count, w - o);
                    if c == 0 {
                        return self;
                    }
                    let low_mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
                    let mask = low_mask << o;
                    let mut array = self.inner.to_array();
                    let newbits_array = newbits.inner.to_array();
                    for (elem, nb) in array.iter_mut().zip(newbits_array.iter()) {
                        *elem = (*elem & !mask) | ((*nb & low_mask) << o);
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_insert_bits_vec_u!(Vec2u);
    impl_insert_bits_vec_u!(Vec3u);
    impl_insert_bits_vec_u!(Vec4u);

    macro_rules! impl_insert_bits_vec_i {
        ($ty:ty) => {
            impl NumericBuiltinInsertBits for $ty {
                fn insert_bits(self, newbits: Self, offset: u32, count: u32) -> Self {
                    let w = 32u32;
                    let o = std::cmp::Ord::min(offset, w);
                    let c = std::cmp::Ord::min(count, w - o);
                    if c == 0 {
                        return self;
                    }
                    let low_mask = if c >= 32 { u32::MAX } else { (1u32 << c) - 1 };
                    let mask = low_mask << o;
                    let mut array = self.inner.to_array();
                    let newbits_array = newbits.inner.to_array();
                    for (elem, nb) in array.iter_mut().zip(newbits_array.iter()) {
                        let e = *elem as u32;
                        let n = *nb as u32;
                        *elem = ((e & !mask) | ((n & low_mask) << o)) as i32;
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_insert_bits_vec_i!(Vec2i);
    impl_insert_bits_vec_i!(Vec3i);
    impl_insert_bits_vec_i!(Vec4i);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity_count_leading_zeros() {
        // u32 tests
        assert_eq!(count_leading_zeros(0u32), 32);
        assert_eq!(count_leading_zeros(1u32), 31);
        assert_eq!(count_leading_zeros(0x80000000u32), 0);
        assert_eq!(count_leading_zeros(0x00010000u32), 15);

        // i32 tests
        assert_eq!(count_leading_zeros(0i32), 32);
        assert_eq!(count_leading_zeros(1i32), 31);
        assert_eq!(count_leading_zeros(-1i32), 0); // All bits set

        // Vector test
        let v = vec2u(0, 1);
        let result = count_leading_zeros(v);
        assert_eq!(result.x(), 32);
        assert_eq!(result.y(), 31);
    }

    #[test]
    fn sanity_count_one_bits() {
        // u32 tests
        assert_eq!(count_one_bits(0u32), 0);
        assert_eq!(count_one_bits(1u32), 1);
        assert_eq!(count_one_bits(0b1111u32), 4);
        assert_eq!(count_one_bits(u32::MAX), 32);

        // i32 tests
        assert_eq!(count_one_bits(0i32), 0);
        assert_eq!(count_one_bits(1i32), 1);
        assert_eq!(count_one_bits(-1i32), 32); // All bits set

        // Vector test
        let v = vec2u(0b1010, 0b1111);
        let result = count_one_bits(v);
        assert_eq!(result.x(), 2);
        assert_eq!(result.y(), 4);
    }

    #[test]
    fn sanity_count_trailing_zeros() {
        // u32 tests
        assert_eq!(count_trailing_zeros(0u32), 32);
        assert_eq!(count_trailing_zeros(1u32), 0);
        assert_eq!(count_trailing_zeros(0b1000u32), 3);
        assert_eq!(count_trailing_zeros(0x80000000u32), 31);

        // i32 tests
        assert_eq!(count_trailing_zeros(0i32), 32);
        assert_eq!(count_trailing_zeros(1i32), 0);
        assert_eq!(count_trailing_zeros(-1i32), 0);

        // Vector test
        let v = vec2u(0b1000, 0b0100);
        let result = count_trailing_zeros(v);
        assert_eq!(result.x(), 3);
        assert_eq!(result.y(), 2);
    }

    #[test]
    fn sanity_reverse_bits() {
        // u32 tests
        assert_eq!(reverse_bits(0u32), 0);
        assert_eq!(reverse_bits(1u32), 0x80000000);
        assert_eq!(reverse_bits(0x80000000u32), 1);
        assert_eq!(reverse_bits(u32::MAX), u32::MAX);

        // i32 tests
        assert_eq!(reverse_bits(0i32), 0);
        assert_eq!(reverse_bits(1i32), i32::MIN); // 0x80000000 as i32
        assert_eq!(reverse_bits(-1i32), -1);

        // Vector test
        let v = vec2u(1, 0x80000000);
        let result = reverse_bits(v);
        assert_eq!(result.x(), 0x80000000);
        assert_eq!(result.y(), 1);
    }

    #[test]
    fn sanity_first_leading_bit() {
        // u32 tests
        assert_eq!(first_leading_bit(0u32), u32::MAX); // -1 as u32
        assert_eq!(first_leading_bit(1u32), 0);
        assert_eq!(first_leading_bit(2u32), 1);
        assert_eq!(first_leading_bit(0x80000000u32), 31);
        assert_eq!(first_leading_bit(0b1010u32), 3);

        // i32 tests (signed behavior)
        assert_eq!(first_leading_bit(0i32), -1);
        assert_eq!(first_leading_bit(-1i32), -1);
        assert_eq!(first_leading_bit(1i32), 0);
        assert_eq!(first_leading_bit(0b1010i32), 3);
        // Negative number: find first bit different from sign bit
        assert_eq!(first_leading_bit(-2i32), 0); // -2 = 0xFFFFFFFE, first 0 is at bit 0

        // Vector test
        let v = vec2u(0, 8);
        let result = first_leading_bit(v);
        assert_eq!(result.x(), u32::MAX);
        assert_eq!(result.y(), 3);
    }

    #[test]
    fn sanity_first_trailing_bit() {
        // u32 tests
        assert_eq!(first_trailing_bit(0u32), u32::MAX); // -1 as u32
        assert_eq!(first_trailing_bit(1u32), 0);
        assert_eq!(first_trailing_bit(0b1000u32), 3);
        assert_eq!(first_trailing_bit(0x80000000u32), 31);

        // i32 tests
        assert_eq!(first_trailing_bit(0i32), -1);
        assert_eq!(first_trailing_bit(1i32), 0);
        assert_eq!(first_trailing_bit(0b1000i32), 3);
        assert_eq!(first_trailing_bit(-1i32), 0);

        // Vector test
        let v = vec2u(0, 0b1000);
        let result = first_trailing_bit(v);
        assert_eq!(result.x(), u32::MAX);
        assert_eq!(result.y(), 3);
    }

    #[test]
    fn sanity_extract_bits() {
        // u32 tests (zero extension)
        assert_eq!(extract_bits(0b11111111u32, 0, 4), 0b1111);
        assert_eq!(extract_bits(0b11111111u32, 4, 4), 0b1111);
        assert_eq!(extract_bits(0b11110000u32, 4, 4), 0b1111);
        assert_eq!(extract_bits(0xFFFFFFFFu32, 0, 32), 0xFFFFFFFF);
        assert_eq!(extract_bits(0x12345678u32, 0, 0), 0); // count=0 returns 0

        // i32 tests (sign extension)
        // Extract 4 bits starting at position 0 from 0b1111 (15)
        // Result: 0b1111 with sign bit at position 3 = 1, so sign extend to -1
        assert_eq!(extract_bits(0b1111i32, 0, 4), -1);
        // Extract 4 bits that represent 0b0111 (7), no sign extension needed
        assert_eq!(extract_bits(0b0111i32, 0, 4), 7);
        // Extract 8 bits from 0xFF at position 0
        assert_eq!(extract_bits(0xFFi32, 0, 8), -1); // 0xFF sign extends to -1

        // Vector test
        let v = vec2u(0xFF, 0xF0);
        let result = extract_bits(v, 4, 4);
        assert_eq!(result.x(), 0xF);
        assert_eq!(result.y(), 0xF);
    }

    #[test]
    fn sanity_insert_bits() {
        // u32 tests
        assert_eq!(insert_bits(0u32, 0b1111u32, 0, 4), 0b1111);
        assert_eq!(insert_bits(0u32, 0b1111u32, 4, 4), 0b11110000);
        assert_eq!(insert_bits(0xFFFFFFFFu32, 0u32, 0, 4), 0xFFFFFFF0);
        assert_eq!(insert_bits(0x12345678u32, 0u32, 0, 0), 0x12345678); // count=0 returns e

        // i32 tests
        assert_eq!(insert_bits(0i32, 0b1111i32, 0, 4), 0b1111);
        assert_eq!(insert_bits(0i32, 0b1111i32, 4, 4), 0b11110000);
        assert_eq!(insert_bits(-1i32, 0i32, 0, 4), -16); // Clear low 4 bits of -1

        // Vector test
        let v = vec2u(0, 0xFF);
        let newbits = vec2u(0xF, 0);
        let result = insert_bits(v, newbits, 4, 4);
        assert_eq!(result.x(), 0xF0);
        assert_eq!(result.y(), 0x0F); // bits 4-7 cleared, was 0xF0, now 0x0F
    }

    #[test]
    fn sanity_bit_manipulation_vectors() {
        // Test that all vector types work for bit manipulation
        let v3u = vec3u(1, 2, 4);
        let clz = count_leading_zeros(v3u);
        assert_eq!(clz.x(), 31);
        assert_eq!(clz.y(), 30);
        assert_eq!(clz.z(), 29);

        let v4i = vec4i(0, 1, -1, 8);
        let ftb = first_trailing_bit(v4i);
        assert_eq!(ftb.x(), -1);
        assert_eq!(ftb.y(), 0);
        assert_eq!(ftb.z(), 0);
        assert_eq!(ftb.w(), 3);
    }
}
