//! Traits that provide WGSL's numeric builtin functions.
//!
//! See <https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions>.
//!
//! | Status `x` for done or `!` for won't do | Function | Parameter Types | Description |
//! | --- | --- | --- | --- |
//! |x| fn abs(e: T ) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | The absolute value of e. Component-wise when T is a vector. <br>       If e is a floating-point type, then the result is e with a positive sign bit.<br>    If e is an unsigned integer scalar type, then the result is e.<br>    If e is a signed integer scalar type and evaluates to the largest<br>    negative value, then the result is e. |
//! |x| fn acos(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) > 1. |
//! | | fn acosh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when e < 1. |
//! |x| fn asin(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) > 1. |
//! | | fn asinh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the inverse hyperbolic sine (sinh-1) of e, as a hyperbolic angle in radians. That is, approximates x such that sinh(x) = e. <br>       Component-wise when T is a vector. |
//! |x| fn atan(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the principal value, in radians, of the inverse tangent (tan-1) of e. That is, approximates x with π/2 ≤ x ≤ π/2, such that tan(x) = e. <br>       Component-wise when T is a vector. |
//! | | fn atanh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) ≥ 1. |
//! | | fn atan2(y: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns an angle, in radians, in the interval [-π, π] whose tangent is y÷x. <br>       The quadrant selected by the result depends on the signs of y and x.<br>    For example, the function may be implemented as:<br>       <br>        <br>         atan(y/x) when x > 0<br>        <br>         atan(y/x) + π when (x < 0) and (y > 0)<br>        <br>         atan(y/x) - π when (x < 0) and (y < 0)<br>       <br>       Note: atan2 is ill-defined when y/x is ill-defined, at the origin (x,y) = (0,0), and when y is non-normal or infinite.<br>       Component-wise when T is a vector. |
//! |x| fn ceil(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the ceiling of e. Component-wise when T is a vector. |
//! |x| fn clamp(e: T, low: T, high: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Restricts the value of e within a range. <br>       If T is an integer type, then the result is min(max(e, low), high).<br>       If T is a floating-point type, then the result is either min(max(e, low), high), or the median of the three values e, low, high.<br>       Component-wise when T is a vector.<br>       If low is greater than high, then:<br>       <br>        <br>         It is a shader-creation error if low and high are const-expressions.<br>        <br>         It is a pipeline-creation error if low and high are override-expressions. |
//! |x| fn cos(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the cosine of e, where e is in radians. Component-wise when T is a vector. |
//! | | fn cosh(arg: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic cosine of arg, where arg is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function (earg + e−arg)÷2,<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector |
//! | | fn countLeadingZeros(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of consecutive 0 bits starting from the most significant bit<br>        of e, when T is a scalar type. Component-wise when T is a vector. Also known as "clz" in some languages. |
//! | | fn countOneBits(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of 1 bits in the representation of e. Also known as "population count". Component-wise when T is a vector. |
//! | | fn countTrailingZeros(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of consecutive 0 bits starting from the least significant bit<br>        of e, when T is a scalar type. Component-wise when T is a vector. Also known as "ctz" in some languages. |
//! |x| fn cross(e1: vec3<T>, e2: vec3<T>) -> vec3<T> | T is AbstractFloat, f32, or f16 | Returns the cross product of e1 and e2. |
//! |x| fn degrees(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Converts radians to degrees, approximating e1 × 180 ÷ π. Component-wise when T is a vector |
//! | | fn determinant(e: matCxC<T>) -> T | T is AbstractFloat, f32, or f16 | Returns the determinant of e. |
//! | | fn distance(e1: T, e2: T) -> S | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the distance between e1 and e2 (e.g. length(e1 - e2)). |
//! |x| fn dot(e1: vecN<T>, e2: vecN<T>) -> T | T is AbstractInt, AbstractFloat, i32, u32, f32, or f16 | Returns the dot product of e1 and e2. |
//! |x| fn exp(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the natural exponentiation of e1 (e.g. ee1). Component-wise when T is a vector. |
//! |x| fn exp2(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns 2 raised to the power e (e.g. 2e). Component-wise when T is a vector. |
//! | | fn extractBits(e: T, offset: u32, count: u32) -> T | T is i32 or vecN<i32> | Reads bits from an integer, with sign extension. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is 0 if c is 0. <br>        Otherwise, bits 0..c - 1 of the result are copied from bits o..o + c - 1 of e.<br>       Other bits of the result are the same as bit c - 1 of the result. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! | | fn extractBits(e: T, offset: u32, count: u32) -> T | T is u32 or vecN<u32> | Reads bits from an integer, without sign extension. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is 0 if c is 0. <br>        Otherwise, bits 0..c - 1 of the result are copied from bits o..o + c - 1 of e.<br>       Other bits of the result are 0. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! | | fn faceForward(e1: T, e2: T, e3: T) -> T | T is vecN<AbstractFloat>, vecN<f32>, or vecN<f16> | Returns e1 if dot(e2, e3) is negative, and -e1 otherwise. |
//! | | fn firstLeadingBit(e: T) -> T | T is i32 or vecN<i32> | Note: Since signed integers use twos-complement representation,<br>the sign bit appears in the most significant bit position. |
//! | | fn firstLeadingBit(e: T) -> T | T is u32 or vecN<u32> | For scalar T, the result is: <br>       <br>        T(-1) if e is zero. <br>        Otherwise the position of the most significant 1<br>            bit in e. <br>       <br>        Component-wise when T is a vector. |
//! | | fn firstTrailingBit(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | For scalar T, the result is: <br>       <br>        T(-1) if e is zero. <br>        Otherwise the position of the least significant 1<br>            bit in e. <br>       <br>        Component-wise when T is a vector. |
//! |x| fn floor(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the floor of e. Component-wise when T is a vector. |
//! | | fn fma(e1: T, e2: T, e3: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns e1 * e2 + e3. Component-wise when T is a vector. <br>       Note: The name fma is short for "fused multiply add".<br>       Note: The IEEE-754 fusedMultiplyAdd operation computes the intermediate results<br>    as if with unbounded range and precision, and only the final result is rounded<br>    to the destination type.<br>    However, the § 14.6.1 Floating Point Accuracy rule for fma allows an implementation<br>    which performs an ordinary multiply to the target type followed by an ordinary addition.<br>    In this case the intermediate values may overflow or lose accuracy, and the overall<br>    operation is not "fused" at all. |
//! |x| fn fract(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: Valid results are in the closed interval [0, 1.0].<br>For example, if e is a very small negative number, then fract(e) may be 1.0. |
//! | | fn insertBits(e: T, newbits: T, offset: u32, count: u32) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | Sets bits in an integer. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is e if c is 0. <br>        Otherwise,<br>       bits o..o + c - 1 of the result are copied from bits 0..c - 1 of newbits.<br>       Other bits of the result are copied from e. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! |x| fn inverseSqrt(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e ≤ 0. |
//! | | fn ldexp(e1: T, e2: I) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> I is AbstractInt, i32, vecN<AbstractInt>, or vecN<i32> I is a vector if and only if T is a vector I is concrete if and only if T is a concrete | Returns e1 * 2e2, except: <br>       <br>        <br>         The result may be zero if e2 + bias ≤ 0.<br>        <br>         If e2 > bias + 1<br>         <br>          <br>           It is a shader-creation error if e2 is a const-expression.<br>          <br>           It is a pipeline-creation error if e2 is an override-expression.<br>          <br>           Otherwise the result is an indeterminate value for T.<br>         <br>       <br>       Here, bias is the exponent bias of the floating point format:<br>       <br>        <br>         15 for f16<br>        <br>         127 for f32<br>        <br>         1023 for AbstractFloat, when AbstractFloat is IEEE-754 binary64<br>       <br>       If x is zero or a finite normal value for its type, then:<br>        x = ldexp(frexp(x).fract, frexp(x).exp) <br>       Component-wise when T is a vector.<br>       Note: A mnemonic for the name ldexp is "load exponent".<br>    The name may have been taken from the corresponding instruction in the floating point unit of<br>    the PDP-11. |
//! |x| fn length(e: T) -> S | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the length of e. Evaluates to the absolute value of e if T is scalar. Evaluates to sqrt(e[0]2 + e[1]2 + ...) if T is a vector type. <br>       Note: The scalar case may be evaluated as sqrt(e * e),<br>        which may unnecessarily overflow or lose accuracy. |
//! |x| fn log(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e < 0. |
//! |x| fn log2(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e < 0. |
//! |x| fn max(e1: T, e2: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Returns e2 if e1 is less than e2, and e1 otherwise. Component-wise when T is a vector. <br>       If e1 and e2 are floating-point values, then:<br>       <br>        <br>         If both e1 and e2 are denormalized, then the result may be either value.<br>        <br>         If one operand is a NaN, the other is returned.<br>        <br>         If both operands are NaNs, a NaN is returned. |
//! |x| fn min(e1: T, e2: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Returns e2 if e2 is less than e1, and e1 otherwise. Component-wise when T is a vector. <br>       If e1 and e2 are floating-point values, then:<br>       <br>        <br>         If both e1 and e2 are denormalized, then the result may be either value.<br>        <br>         If one operand is a NaN, the other is returned.<br>        <br>         If both operands are NaNs, a NaN is returned. |
//! |x| fn mix(e1: T, e2: T, e3: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the linear blend of e1 and e2 (e.g. e1 * (1 - e3) + e2 * e3). Component-wise when T is a vector. |
//! | | fn mix(e1: T2, e2: T2, e3: T) -> T2 | T is AbstractFloat, f32, or f16 T2 is vecN<T> | Returns the component-wise linear blend of e1 and e2,<br>        using scalar blending factor e3 for each component. Same as mix(e1, e2, T2(e3)). |
//! | | fn modf(e: T) -> __modf_result_f32 | T is f32 | Note: A value cannot be explicitly declared with the type __modf_result_f32,<br>but a value may infer the type. |
//! | | fn modf(e: T) -> __modf_result_f16 | T is f16 | Note: A value cannot be explicitly declared with the type __modf_result_f16,<br>but a value may infer the type. |
//! | | fn modf(e: T) -> __modf_result_abstract | T is AbstractFloat | Note: A value cannot be explicitly declared with the type __modf_result_abstract,<br>but a value may infer the type. |
//! | | fn modf(e: T) -> __modf_result_vecN_f32 | T is vecN<f32> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_f32,<br>but a value may infer the type. |
//! | | fn modf(e: T) -> __modf_result_vecN_f16 | T is vecN<f16> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_f16,<br>but a value may infer the type. |
//! | | fn modf(e: T) -> __modf_result_vecN_abstract | T is vecN<AbstractFloat> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_abstract,<br>but a value may infer the type. |
//! |x| fn normalize(e: vecN<T> ) -> vecN<T> | T is AbstractFloat, f32, or f16 | Returns a unit vector in the same direction as e. |
//! |x| fn pow(e1: T, e2: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns e1 raised to the power e2. Component-wise when T is a vector. |
//! | | fn quantizeToF16(e: T) -> T | T is f32 or vecN<f32> | Note: The vec2<f32> case is the same as unpack2x16float(pack2x16float(e)). |
//! |x| fn radians(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Converts degrees to radians, approximating e1 × π ÷ 180. Component-wise when T is a vector |
//! | | fn reflect(e1: T, e2: T) -> T | T is vecN<AbstractFloat>, vecN<f32>, or vecN<f16> | For the incident vector e1 and surface orientation e2, returns the reflection direction e1 - 2 * dot(e2, e1) * e2. |
//! | | fn refract(e1: T, e2: T, e3: I) -> T | T is vecN<I> I is AbstractFloat, f32, or f16 | For the incident vector e1 and surface normal e2, and the ratio of<br>    indices of refraction e3,<br>    let k = 1.0 - e3 * e3 * (1.0 - dot(e2, e1) * dot(e2, e1)).<br>    If k < 0.0, returns the refraction vector 0.0, otherwise return the refraction vector e3 * e1 - (e3 * dot(e2, e1) + sqrt(k)) * e2. |
//! | | fn reverseBits(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | Reverses the bits in e:  The bit at position k of the result equals the<br>        bit at position 31 -k of e. Component-wise when T is a vector. |
//! |x| fn round(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Result is the integer k nearest to e, as a floating point value. When e lies halfway between integers k and k + 1,<br>        the result is k when k is even, and k + 1 when k is odd. Component-wise when T is a vector. |
//! |x| fn saturate(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns clamp(e, 0.0, 1.0). Component-wise when T is a vector. |
//! |x| fn sign(e: T) -> T | S is AbstractInt, AbstractFloat, i32, f32, or f16. T is S, or vecN<S> | Result is: <br>       <br>         1 when e > 0 <br>         0 when e = 0 <br>         -1 when e < 0 <br>       <br>       Component-wise when T is a vector. |
//! |x| fn sin(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the sine of e, where e is in radians. Component-wise when T is a vector. |
//! | | fn sinh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic sine of e, where e is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function<br>    (earg − e−arg)÷2,<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector. |
//! | | fn smoothstep(low: T, high: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the smooth Hermite interpolation between 0 and 1. Component-wise when T is a vector. <br>       For scalar T, the result is t * t * (3.0 - 2.0 * t),<br>    where t = clamp((x - low) / (high - low), 0.0, 1.0). |
//! |x| fn sqrt(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the square root of e. Component-wise when T is a vector. |
//! |x| fn step(edge: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns 1.0 if edge ≤ x, and 0.0 otherwise. Component-wise when T is a vector. |
//! |x| fn tan(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the tangent of e, where e is in radians. Component-wise when T is a vector. |
//! |x| fn tanh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic tangent of e, where e is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function<br>    (earg − e−arg) ÷ (earg + e−arg)<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector. |
//! | | fn transpose(e: matRxC<T>) -> matCxR<T> | T is AbstractFloat, f32, or f16 | Returns the transpose of e. |
//! |x| fn trunc(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns truncate(e), the nearest whole number whose absolute value<br>    is less than or equal to the absolute value of e. Component-wise when T is a vector. |

use crate::std::*;

/// Provides the numeric built-in function `abs`.
pub trait NumericBuiltinAbs {
    /// The absolute value of e. Component-wise when T is a vector.
    /// * If e is a floating-point type, then the result is e with a positive
    ///   sign bit.
    /// * If e is an unsigned integer scalar type, then the result is e.
    /// * If e is a signed integer scalar type and evaluates to the largest
    ///   negative value, then the result is e.
    fn abs(self) -> Self;
}

/// The absolute value of e. Component-wise when T is a vector.
/// * If e is a floating-point type, then the result is e with a positive sign
///   bit.
/// * If e is an unsigned integer scalar type, then the result is e.
/// * If e is a signed integer scalar type and evaluates to the largest negative
///   value, then the result is e.
pub fn abs<T: NumericBuiltinAbs>(e: T) -> T {
    <T as NumericBuiltinAbs>::abs(e)
}

/// Provides the numeric built-in function `acos`.
pub trait NumericBuiltinAcos {
    /// Returns the principal value, in radians, of the inverse cosine (cos⁻¹)
    /// of e. Component-wise when T is a vector.
    /// Note: The result is not mathematically meaningful when abs(e) > 1.
    fn acos(self) -> Self;
}

/// Returns the principal value, in radians, of the inverse cosine (cos⁻¹) of e.
/// Component-wise when T is a vector.
/// Note: The result is not mathematically meaningful when abs(e) > 1.
pub fn acos<T: NumericBuiltinAcos>(e: T) -> T {
    <T as NumericBuiltinAcos>::acos(e)
}

/// Provides the numeric built-in function `sin`.
pub trait NumericBuiltinSin {
    /// Returns the sine of e, where e is in radians. Component-wise when T is a
    /// vector.
    fn sin(self) -> Self;
}

/// Returns the sine of e, where e is in radians. Component-wise when T is a
/// vector.
pub fn sin<T: NumericBuiltinSin>(e: T) -> T {
    <T as NumericBuiltinSin>::sin(e)
}

/// Provides the numeric built-in function `asin`.
pub trait NumericBuiltinAsin {
    /// Returns the principal value, in radians, of the inverse sine (sin⁻¹) of
    /// e. Component-wise when T is a vector.
    /// Note: The result is not mathematically meaningful when abs(e) > 1.
    fn asin(self) -> Self;
}

/// Returns the principal value, in radians, of the inverse sine (sin⁻¹) of e.
/// Component-wise when T is a vector.
/// Note: The result is not mathematically meaningful when abs(e) > 1.
pub fn asin<T: NumericBuiltinAsin>(e: T) -> T {
    <T as NumericBuiltinAsin>::asin(e)
}

/// Provides the numeric built-in function `atan`.
pub trait NumericBuiltinAtan {
    /// Returns the principal value, in radians, of the inverse tangent (tan⁻¹)
    /// of e. Component-wise when T is a vector.
    fn atan(self) -> Self;
}

/// Returns the principal value, in radians, of the inverse tangent (tan⁻¹) of
/// e. Component-wise when T is a vector.
pub fn atan<T: NumericBuiltinAtan>(e: T) -> T {
    <T as NumericBuiltinAtan>::atan(e)
}

/// Provides the numeric built-in function `cos`.
pub trait NumericBuiltinCos {
    /// Returns the cosine of e, where e is in radians. Component-wise when T is
    /// a vector.
    fn cos(self) -> Self;
}

/// Returns the cosine of e, where e is in radians. Component-wise when T is a
/// vector.
pub fn cos<T: NumericBuiltinCos>(e: T) -> T {
    <T as NumericBuiltinCos>::cos(e)
}

/// Provides the numeric built-in function `tan`.
pub trait NumericBuiltinTan {
    /// Returns the tangent of e, where e is in radians. Component-wise when T
    /// is a vector.
    fn tan(self) -> Self;
}

/// Returns the tangent of e, where e is in radians. Component-wise when T is a
/// vector.
pub fn tan<T: NumericBuiltinTan>(e: T) -> T {
    <T as NumericBuiltinTan>::tan(e)
}

mod abs {
    use super::*;
    macro_rules! impl_abs_scalar {
        ($ty:ty) => {
            impl NumericBuiltinAbs for $ty {
                fn abs(self) -> Self {
                    self.abs()
                }
            }
        };
    }
    impl_abs_scalar!(f32);
    impl_abs_scalar!(i32);

    macro_rules! impl_abs_uself {
        ($ty:ty) => {
            impl NumericBuiltinAbs for $ty {
                fn abs(self) -> Self {
                    self
                }
            }
        };
    }
    impl_abs_uself!(u32);
    impl_abs_uself!(Vec2u);
    impl_abs_uself!(Vec3u);
    impl_abs_uself!(Vec4u);

    macro_rules! impl_abs_vec {
        ($ty:ty) => {
            impl NumericBuiltinAbs for $ty {
                fn abs(self) -> Self {
                    Self {
                        inner: self.inner.abs(),
                    }
                }
            }
        };
    }
    impl_abs_vec!(Vec2f);
    impl_abs_vec!(Vec3f);
    impl_abs_vec!(Vec4f);
    impl_abs_vec!(Vec2i);
    impl_abs_vec!(Vec3i);
    impl_abs_vec!(Vec4i);
}

mod acos {
    use super::*;

    macro_rules! impl_acos_scalar {
        ($ty:ty) => {
            impl NumericBuiltinAcos for $ty {
                fn acos(self) -> Self {
                    self.acos()
                }
            }
        };
    }
    impl_acos_scalar!(f32);

    macro_rules! impl_acos_vec {
        ($ty:ty) => {
            impl NumericBuiltinAcos for $ty {
                fn acos(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.acos()),
                    }
                }
            }
        };
    }
    impl_acos_vec!(Vec2f);
    impl_acos_vec!(Vec3f);
    impl_acos_vec!(Vec4f);
}

mod sin {
    use super::*;

    macro_rules! impl_sin_scalar {
        ($ty:ty) => {
            impl NumericBuiltinSin for $ty {
                fn sin(self) -> Self {
                    self.sin()
                }
            }
        };
    }
    impl_sin_scalar!(f32);

    macro_rules! impl_sin_vec {
        ($ty:ty) => {
            impl NumericBuiltinSin for $ty {
                fn sin(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.sin()),
                    }
                }
            }
        };
    }
    impl_sin_vec!(Vec2f);
    impl_sin_vec!(Vec3f);
    impl_sin_vec!(Vec4f);
}

mod asin {
    use super::*;

    macro_rules! impl_asin_scalar {
        ($ty:ty) => {
            impl NumericBuiltinAsin for $ty {
                fn asin(self) -> Self {
                    self.asin()
                }
            }
        };
    }
    impl_asin_scalar!(f32);

    macro_rules! impl_asin_vec {
        ($ty:ty) => {
            impl NumericBuiltinAsin for $ty {
                fn asin(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.asin()),
                    }
                }
            }
        };
    }
    impl_asin_vec!(Vec2f);
    impl_asin_vec!(Vec3f);
    impl_asin_vec!(Vec4f);
}

mod atan {
    use super::*;

    macro_rules! impl_atan_scalar {
        ($ty:ty) => {
            impl NumericBuiltinAtan for $ty {
                fn atan(self) -> Self {
                    self.atan()
                }
            }
        };
    }
    impl_atan_scalar!(f32);

    macro_rules! impl_atan_vec {
        ($ty:ty) => {
            impl NumericBuiltinAtan for $ty {
                fn atan(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.atan()),
                    }
                }
            }
        };
    }
    impl_atan_vec!(Vec2f);
    impl_atan_vec!(Vec3f);
    impl_atan_vec!(Vec4f);
}

mod cos {
    use super::*;

    macro_rules! impl_cos_scalar {
        ($ty:ty) => {
            impl NumericBuiltinCos for $ty {
                fn cos(self) -> Self {
                    self.cos()
                }
            }
        };
    }
    impl_cos_scalar!(f32);

    macro_rules! impl_cos_vec {
        ($ty:ty) => {
            impl NumericBuiltinCos for $ty {
                fn cos(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.cos()),
                    }
                }
            }
        };
    }
    impl_cos_vec!(Vec2f);
    impl_cos_vec!(Vec3f);
    impl_cos_vec!(Vec4f);
}

mod tan {
    use super::*;

    macro_rules! impl_tan_scalar {
        ($ty:ty) => {
            impl NumericBuiltinTan for $ty {
                fn tan(self) -> Self {
                    self.tan()
                }
            }
        };
    }
    impl_tan_scalar!(f32);

    macro_rules! impl_tan_vec {
        ($ty:ty) => {
            impl NumericBuiltinTan for $ty {
                fn tan(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.tan()),
                    }
                }
            }
        };
    }
    impl_tan_vec!(Vec2f);
    impl_tan_vec!(Vec3f);
    impl_tan_vec!(Vec4f);
}

// --- Additional Numeric Builtin Implementations ---

pub trait NumericBuiltinFract {
    fn fract(self) -> Self;
}
pub fn fract<T: NumericBuiltinFract>(e: T) -> T {
    <T as NumericBuiltinFract>::fract(e)
}
mod fract {
    use super::*;
    impl NumericBuiltinFract for f32 {
        fn fract(self) -> Self {
            self.fract()
        }
    }
    macro_rules! impl_fract_vec {
        ($ty:ty) => {
            impl NumericBuiltinFract for $ty {
                fn fract(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.fract()),
                    }
                }
            }
        };
    }
    impl_fract_vec!(Vec2f);
    impl_fract_vec!(Vec3f);
    impl_fract_vec!(Vec4f);
}

pub trait NumericBuiltinInverseSqrt {
    fn inverse_sqrt(self) -> Self;
}
pub fn inverse_sqrt<T: NumericBuiltinInverseSqrt>(e: T) -> T {
    <T as NumericBuiltinInverseSqrt>::inverse_sqrt(e)
}
mod inverse_sqrt {
    use super::*;
    impl NumericBuiltinInverseSqrt for f32 {
        fn inverse_sqrt(self) -> Self {
            1.0 / self.sqrt()
        }
    }
    macro_rules! impl_inverse_sqrt_vec {
        ($ty:ty) => {
            impl NumericBuiltinInverseSqrt for $ty {
                fn inverse_sqrt(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| 1.0 / t.sqrt()),
                    }
                }
            }
        };
    }
    impl_inverse_sqrt_vec!(Vec2f);
    impl_inverse_sqrt_vec!(Vec3f);
    impl_inverse_sqrt_vec!(Vec4f);
}

pub trait NumericBuiltinLength {
    fn length(self) -> f32;
}
pub fn length<T: NumericBuiltinLength>(e: T) -> f32 {
    <T as NumericBuiltinLength>::length(e)
}
mod length {
    use super::*;
    impl NumericBuiltinLength for f32 {
        fn length(self) -> f32 {
            self.abs()
        }
    }
    impl NumericBuiltinLength for Vec2f {
        fn length(self) -> f32 {
            self.inner.length()
        }
    }
    impl NumericBuiltinLength for Vec3f {
        fn length(self) -> f32 {
            self.inner.length()
        }
    }
    impl NumericBuiltinLength for Vec4f {
        fn length(self) -> f32 {
            self.inner.length()
        }
    }
}

pub trait NumericBuiltinLog {
    fn log(self) -> Self;
}
pub fn log<T: NumericBuiltinLog>(e: T) -> T {
    <T as NumericBuiltinLog>::log(e)
}
mod log {
    use super::*;
    impl NumericBuiltinLog for f32 {
        fn log(self) -> Self {
            self.ln()
        }
    }
    macro_rules! impl_log_vec {
        ($ty:ty) => {
            impl NumericBuiltinLog for $ty {
                fn log(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.ln()),
                    }
                }
            }
        };
    }
    impl_log_vec!(Vec2f);
    impl_log_vec!(Vec3f);
    impl_log_vec!(Vec4f);
}

pub trait NumericBuiltinLog2 {
    fn log2(self) -> Self;
}
pub fn log2<T: NumericBuiltinLog2>(e: T) -> T {
    <T as NumericBuiltinLog2>::log2(e)
}
mod log2 {
    use super::*;
    impl NumericBuiltinLog2 for f32 {
        fn log2(self) -> Self {
            self.log2()
        }
    }
    macro_rules! impl_log2_vec {
        ($ty:ty) => {
            impl NumericBuiltinLog2 for $ty {
                fn log2(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.log2()),
                    }
                }
            }
        };
    }
    impl_log2_vec!(Vec2f);
    impl_log2_vec!(Vec3f);
    impl_log2_vec!(Vec4f);
}

pub trait NumericBuiltinMax {
    fn max(self, other: Self) -> Self;
}
pub fn max<T: NumericBuiltinMax>(a: T, b: T) -> T {
    <T as NumericBuiltinMax>::max(a, b)
}
mod max {
    use super::*;
    impl NumericBuiltinMax for f32 {
        fn max(self, other: Self) -> Self {
            self.max(other)
        }
    }
    impl NumericBuiltinMax for u32 {
        fn max(self, other: Self) -> Self {
            std::cmp::Ord::max(self, other)
        }
    }
    impl NumericBuiltinMax for i32 {
        fn max(self, other: Self) -> Self {
            std::cmp::Ord::max(self, other)
        }
    }
    macro_rules! impl_max_vec {
        ($ty:ty) => {
            impl NumericBuiltinMax for $ty {
                fn max(self, other: Self) -> Self {
                    self.inner.max(other.inner).into()
                }
            }
        };
    }
    impl_max_vec!(Vec2f);
    impl_max_vec!(Vec3f);
    impl_max_vec!(Vec4f);
}

pub trait NumericBuiltinMin {
    fn min(self, other: Self) -> Self;
}
pub fn min<T: NumericBuiltinMin>(a: T, b: T) -> T {
    <T as NumericBuiltinMin>::min(a, b)
}
mod min {
    use super::*;
    impl NumericBuiltinMin for f32 {
        fn min(self, other: Self) -> Self {
            self.min(other)
        }
    }
    impl NumericBuiltinMin for u32 {
        fn min(self, other: Self) -> Self {
            std::cmp::Ord::min(self, other)
        }
    }
    impl NumericBuiltinMin for i32 {
        fn min(self, other: Self) -> Self {
            std::cmp::Ord::min(self, other)
        }
    }
    macro_rules! impl_min_vec {
        ($ty:ty) => {
            impl NumericBuiltinMin for $ty {
                fn min(self, other: Self) -> Self {
                    self.inner.min(other.inner).into()
                }
            }
        };
    }
    impl_min_vec!(Vec2f);
    impl_min_vec!(Vec3f);
    impl_min_vec!(Vec4f);
}

pub trait NumericBuiltinPow {
    fn pow(self, other: Self) -> Self;
}
pub fn pow<T: NumericBuiltinPow>(a: T, b: T) -> T {
    <T as NumericBuiltinPow>::pow(a, b)
}
mod pow {
    use super::*;
    impl NumericBuiltinPow for f32 {
        fn pow(self, other: Self) -> Self {
            self.powf(other)
        }
    }
    macro_rules! impl_pow_vec {
        ($ty:ty) => {
            impl NumericBuiltinPow for $ty {
                fn pow(self, other: Self) -> Self {
                    let mut array = self.inner.to_array();
                    let exps = other.inner.to_array();
                    for i in 0..array.len() {
                        array[i] = array[i].powf(exps[i]);
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_pow_vec!(Vec2f);
    impl_pow_vec!(Vec3f);
    impl_pow_vec!(Vec4f);
}

pub trait NumericBuiltinRadians {
    fn radians(self) -> Self;
}
pub fn radians<T: NumericBuiltinRadians>(e: T) -> T {
    <T as NumericBuiltinRadians>::radians(e)
}
mod radians {
    use super::*;
    impl NumericBuiltinRadians for f32 {
        fn radians(self) -> Self {
            self * std::f32::consts::PI / 180.0
        }
    }
    macro_rules! impl_radians_vec {
        ($ty:ty) => {
            impl NumericBuiltinRadians for $ty {
                fn radians(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t * std::f32::consts::PI / 180.0),
                    }
                }
            }
        };
    }
    impl_radians_vec!(Vec2f);
    impl_radians_vec!(Vec3f);
    impl_radians_vec!(Vec4f);
}

pub trait NumericBuiltinRound {
    fn round(self) -> Self;
}
pub fn round<T: NumericBuiltinRound>(e: T) -> T {
    <T as NumericBuiltinRound>::round(e)
}
mod round {
    use super::*;
    impl NumericBuiltinRound for f32 {
        fn round(self) -> Self {
            self.round()
        }
    }
    macro_rules! impl_round_vec {
        ($ty:ty) => {
            impl NumericBuiltinRound for $ty {
                fn round(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.round()),
                    }
                }
            }
        };
    }
    impl_round_vec!(Vec2f);
    impl_round_vec!(Vec3f);
    impl_round_vec!(Vec4f);
}

pub trait NumericBuiltinSaturate {
    fn saturate(self) -> Self;
}
pub fn saturate<T: NumericBuiltinSaturate>(e: T) -> T {
    <T as NumericBuiltinSaturate>::saturate(e)
}
mod saturate {
    use super::*;
    impl NumericBuiltinSaturate for f32 {
        fn saturate(self) -> Self {
            self.clamp(0.0, 1.0)
        }
    }
    macro_rules! impl_saturate_vec {
        ($ty:ty) => {
            impl NumericBuiltinSaturate for $ty {
                fn saturate(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.clamp(0.0, 1.0)),
                    }
                }
            }
        };
    }
    impl_saturate_vec!(Vec2f);
    impl_saturate_vec!(Vec3f);
    impl_saturate_vec!(Vec4f);
}

pub trait NumericBuiltinSign {
    fn sign(self) -> Self;
}
pub fn sign<T: NumericBuiltinSign>(e: T) -> T {
    <T as NumericBuiltinSign>::sign(e)
}
mod sign {
    use super::*;
    impl NumericBuiltinSign for f32 {
        fn sign(self) -> Self {
            if self > 0.0 {
                1.0
            } else if self < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
    }
    macro_rules! impl_sign_vec {
        ($ty:ty) => {
            impl NumericBuiltinSign for $ty {
                fn sign(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| {
                            if t > 0.0 {
                                1.0
                            } else if t < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        }),
                    }
                }
            }
        };
    }
    impl_sign_vec!(Vec2f);
    impl_sign_vec!(Vec3f);
    impl_sign_vec!(Vec4f);
}

pub trait NumericBuiltinSqrt {
    fn sqrt(self) -> Self;
}
pub fn sqrt<T: NumericBuiltinSqrt>(e: T) -> T {
    <T as NumericBuiltinSqrt>::sqrt(e)
}
mod sqrt {
    use super::*;
    impl NumericBuiltinSqrt for f32 {
        fn sqrt(self) -> Self {
            self.sqrt()
        }
    }
    macro_rules! impl_sqrt_vec {
        ($ty:ty) => {
            impl NumericBuiltinSqrt for $ty {
                fn sqrt(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.sqrt()),
                    }
                }
            }
        };
    }
    impl_sqrt_vec!(Vec2f);
    impl_sqrt_vec!(Vec3f);
    impl_sqrt_vec!(Vec4f);
}

pub trait NumericBuiltinStep {
    fn step(self, x: Self) -> Self;
}
pub fn step<T: NumericBuiltinStep>(edge: T, x: T) -> T {
    <T as NumericBuiltinStep>::step(edge, x)
}
mod step {
    use super::*;
    impl NumericBuiltinStep for f32 {
        fn step(self, x: Self) -> Self {
            if x >= self { 1.0 } else { 0.0 }
        }
    }
    macro_rules! impl_step_vec {
        ($ty:ty) => {
            impl NumericBuiltinStep for $ty {
                fn step(self, other: Self) -> Self {
                    let mut array = self.inner.to_array();
                    let other = other.inner.to_array();
                    for i in 0..array.len() {
                        array[i] = array[i].step(other[i]);
                    }
                    Self {
                        inner: array.into(),
                    }
                }
            }
        };
    }
    impl_step_vec!(Vec2f);
    impl_step_vec!(Vec3f);
    impl_step_vec!(Vec4f);
}

pub trait NumericBuiltinTanh {
    fn tanh(self) -> Self;
}
pub fn tanh<T: NumericBuiltinTanh>(e: T) -> T {
    <T as NumericBuiltinTanh>::tanh(e)
}
mod tanh {
    use super::*;
    impl NumericBuiltinTanh for f32 {
        fn tanh(self) -> Self {
            self.tanh()
        }
    }
    macro_rules! impl_tanh_vec {
        ($ty:ty) => {
            impl NumericBuiltinTanh for $ty {
                fn tanh(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.tanh()),
                    }
                }
            }
        };
    }
    impl_tanh_vec!(Vec2f);
    impl_tanh_vec!(Vec3f);
    impl_tanh_vec!(Vec4f);
}

pub trait NumericBuiltinTrunc {
    fn trunc(self) -> Self;
}
pub fn trunc<T: NumericBuiltinTrunc>(e: T) -> T {
    <T as NumericBuiltinTrunc>::trunc(e)
}
mod trunc {
    use super::*;
    impl NumericBuiltinTrunc for f32 {
        fn trunc(self) -> Self {
            self.trunc()
        }
    }
    macro_rules! impl_trunc_vec {
        ($ty:ty) => {
            impl NumericBuiltinTrunc for $ty {
                fn trunc(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.trunc()),
                    }
                }
            }
        };
    }
    impl_trunc_vec!(Vec2f);
    impl_trunc_vec!(Vec3f);
    impl_trunc_vec!(Vec4f);
}

/// Provides the numeric built-in function `dot`.
pub trait NumericBuiltinDot {
    type Scalar;
    /// Returns the dot product of two vectors.
    fn dot(self, other: Self) -> Self::Scalar;
}

/// Returns the dot product of e1 and e2.
pub fn dot<T: NumericBuiltinDot>(e1: T, e2: T) -> T::Scalar {
    <T as NumericBuiltinDot>::dot(e1, e2)
}

mod dot {
    use super::*;

    macro_rules! impl_dot_vec_f {
        ($ty:ty, $scalar:ty) => {
            impl NumericBuiltinDot for $ty {
                type Scalar = $scalar;
                fn dot(self, other: Self) -> Self::Scalar {
                    self.inner.dot(other.inner)
                }
            }
        };
    }
    impl_dot_vec_f!(Vec2f, f32);
    impl_dot_vec_f!(Vec3f, f32);
    impl_dot_vec_f!(Vec4f, f32);

    macro_rules! impl_dot_vec_i {
        ($ty:ty, $scalar:ty) => {
            impl NumericBuiltinDot for $ty {
                type Scalar = $scalar;
                fn dot(self, other: Self) -> Self::Scalar {
                    let a = self.inner.to_array();
                    let b = other.inner.to_array();
                    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
                }
            }
        };
    }
    impl_dot_vec_i!(Vec2i, i32);
    impl_dot_vec_i!(Vec3i, i32);
    impl_dot_vec_i!(Vec4i, i32);
    impl_dot_vec_i!(Vec2u, u32);
    impl_dot_vec_i!(Vec3u, u32);
    impl_dot_vec_i!(Vec4u, u32);
}

/// Provides the numeric built-in function `normalize`.
pub trait NumericBuiltinNormalize {
    /// Returns a unit vector in the same direction as e.
    fn normalize(self) -> Self;
}

/// Returns a unit vector in the same direction as e.
pub fn normalize<T: NumericBuiltinNormalize>(e: T) -> T {
    <T as NumericBuiltinNormalize>::normalize(e)
}

mod normalize {
    use super::*;

    macro_rules! impl_normalize_vec {
        ($ty:ty) => {
            impl NumericBuiltinNormalize for $ty {
                fn normalize(self) -> Self {
                    Self {
                        inner: self.inner.normalize(),
                    }
                }
            }
        };
    }
    impl_normalize_vec!(Vec2f);
    impl_normalize_vec!(Vec3f);
    impl_normalize_vec!(Vec4f);
}

/// Provides the numeric built-in function `cross`.
pub trait NumericBuiltinCross {
    /// Returns the cross product of e1 and e2.
    fn cross(self, other: Self) -> Self;
}

/// Returns the cross product of e1 and e2 (vec3 only).
pub fn cross<T: NumericBuiltinCross>(e1: T, e2: T) -> T {
    <T as NumericBuiltinCross>::cross(e1, e2)
}

mod cross {
    use super::*;

    impl NumericBuiltinCross for Vec3f {
        fn cross(self, other: Self) -> Self {
            Self {
                inner: self.inner.cross(other.inner),
            }
        }
    }
}

/// Provides the numeric built-in function `clamp`.
pub trait NumericBuiltinClamp {
    /// Restricts the value of e within a range [low, high].
    fn clamp(self, low: Self, high: Self) -> Self;
}

/// Restricts the value of e within a range [low, high].
pub fn clamp<T: NumericBuiltinClamp>(e: T, low: T, high: T) -> T {
    <T as NumericBuiltinClamp>::clamp(e, low, high)
}

mod clamp {
    use super::*;

    impl NumericBuiltinClamp for f32 {
        fn clamp(self, low: Self, high: Self) -> Self {
            self.clamp(low, high)
        }
    }

    impl NumericBuiltinClamp for i32 {
        fn clamp(self, low: Self, high: Self) -> Self {
            std::cmp::Ord::clamp(self, low, high)
        }
    }

    impl NumericBuiltinClamp for u32 {
        fn clamp(self, low: Self, high: Self) -> Self {
            std::cmp::Ord::clamp(self, low, high)
        }
    }

    macro_rules! impl_clamp_vec_f {
        ($ty:ty) => {
            impl NumericBuiltinClamp for $ty {
                fn clamp(self, low: Self, high: Self) -> Self {
                    Self {
                        inner: self.inner.clamp(low.inner, high.inner),
                    }
                }
            }
        };
    }
    impl_clamp_vec_f!(Vec2f);
    impl_clamp_vec_f!(Vec3f);
    impl_clamp_vec_f!(Vec4f);

    macro_rules! impl_clamp_vec_i {
        ($ty:ty) => {
            impl NumericBuiltinClamp for $ty {
                fn clamp(self, low: Self, high: Self) -> Self {
                    Self {
                        inner: self.inner.clamp(low.inner, high.inner),
                    }
                }
            }
        };
    }
    impl_clamp_vec_i!(Vec2i);
    impl_clamp_vec_i!(Vec3i);
    impl_clamp_vec_i!(Vec4i);
    impl_clamp_vec_i!(Vec2u);
    impl_clamp_vec_i!(Vec3u);
    impl_clamp_vec_i!(Vec4u);
}

/// Provides the numeric built-in function `mix`.
pub trait NumericBuiltinMix {
    /// Returns the linear blend of e1 and e2: e1 * (1 - e3) + e2 * e3.
    fn mix(self, e2: Self, e3: Self) -> Self;
}

/// Returns the linear blend of e1 and e2: e1 * (1 - e3) + e2 * e3.
pub fn mix<T: NumericBuiltinMix>(e1: T, e2: T, e3: T) -> T {
    <T as NumericBuiltinMix>::mix(e1, e2, e3)
}

mod mix {
    use super::*;

    impl NumericBuiltinMix for f32 {
        fn mix(self, e2: Self, e3: Self) -> Self {
            self * (1.0 - e3) + e2 * e3
        }
    }

    // For vectors, we use component-wise mix
    impl NumericBuiltinMix for Vec2f {
        fn mix(self, e2: Self, e3: Self) -> Self {
            let a = self.inner.to_array();
            let b = e2.inner.to_array();
            let t = e3.inner.to_array();
            Self {
                inner: glam::Vec2::new(
                    a[0] * (1.0 - t[0]) + b[0] * t[0],
                    a[1] * (1.0 - t[1]) + b[1] * t[1],
                ),
            }
        }
    }

    impl NumericBuiltinMix for Vec3f {
        fn mix(self, e2: Self, e3: Self) -> Self {
            let a = self.inner.to_array();
            let b = e2.inner.to_array();
            let t = e3.inner.to_array();
            Self {
                inner: glam::Vec3::new(
                    a[0] * (1.0 - t[0]) + b[0] * t[0],
                    a[1] * (1.0 - t[1]) + b[1] * t[1],
                    a[2] * (1.0 - t[2]) + b[2] * t[2],
                ),
            }
        }
    }

    impl NumericBuiltinMix for Vec4f {
        fn mix(self, e2: Self, e3: Self) -> Self {
            let a = self.inner.to_array();
            let b = e2.inner.to_array();
            let t = e3.inner.to_array();
            Self {
                inner: glam::Vec4::new(
                    a[0] * (1.0 - t[0]) + b[0] * t[0],
                    a[1] * (1.0 - t[1]) + b[1] * t[1],
                    a[2] * (1.0 - t[2]) + b[2] * t[2],
                    a[3] * (1.0 - t[3]) + b[3] * t[3],
                ),
            }
        }
    }
}

/// Provides the numeric built-in function `floor`.
pub trait NumericBuiltinFloor {
    /// Returns the floor of e. Component-wise when T is a vector.
    fn floor(self) -> Self;
}

/// Returns the floor of e. Component-wise when T is a vector.
pub fn floor<T: NumericBuiltinFloor>(e: T) -> T {
    <T as NumericBuiltinFloor>::floor(e)
}

mod floor {
    use super::*;

    impl NumericBuiltinFloor for f32 {
        fn floor(self) -> Self {
            self.floor()
        }
    }

    macro_rules! impl_floor_vec {
        ($ty:ty) => {
            impl NumericBuiltinFloor for $ty {
                fn floor(self) -> Self {
                    Self {
                        inner: self.inner.floor(),
                    }
                }
            }
        };
    }
    impl_floor_vec!(Vec2f);
    impl_floor_vec!(Vec3f);
    impl_floor_vec!(Vec4f);
}

/// Provides the numeric built-in function `ceil`.
pub trait NumericBuiltinCeil {
    /// Returns the ceiling of e. Component-wise when T is a vector.
    fn ceil(self) -> Self;
}

/// Returns the ceiling of e. Component-wise when T is a vector.
pub fn ceil<T: NumericBuiltinCeil>(e: T) -> T {
    <T as NumericBuiltinCeil>::ceil(e)
}

mod ceil {
    use super::*;

    impl NumericBuiltinCeil for f32 {
        fn ceil(self) -> Self {
            self.ceil()
        }
    }

    macro_rules! impl_ceil_vec {
        ($ty:ty) => {
            impl NumericBuiltinCeil for $ty {
                fn ceil(self) -> Self {
                    Self {
                        inner: self.inner.ceil(),
                    }
                }
            }
        };
    }
    impl_ceil_vec!(Vec2f);
    impl_ceil_vec!(Vec3f);
    impl_ceil_vec!(Vec4f);
}

/// Provides the numeric built-in function `exp`.
pub trait NumericBuiltinExp {
    /// Returns the natural exponentiation of e (e^e). Component-wise when T is
    /// a vector.
    fn exp(self) -> Self;
}

/// Returns the natural exponentiation of e (e^e). Component-wise when T is a
/// vector.
pub fn exp<T: NumericBuiltinExp>(e: T) -> T {
    <T as NumericBuiltinExp>::exp(e)
}

mod exp {
    use super::*;

    impl NumericBuiltinExp for f32 {
        fn exp(self) -> Self {
            self.exp()
        }
    }

    macro_rules! impl_exp_vec {
        ($ty:ty) => {
            impl NumericBuiltinExp for $ty {
                fn exp(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.exp()),
                    }
                }
            }
        };
    }
    impl_exp_vec!(Vec2f);
    impl_exp_vec!(Vec3f);
    impl_exp_vec!(Vec4f);
}

/// Provides the numeric built-in function `exp2`.
pub trait NumericBuiltinExp2 {
    /// Returns 2 raised to the power e (2^e). Component-wise when T is a
    /// vector.
    fn exp2(self) -> Self;
}

/// Returns 2 raised to the power e (2^e). Component-wise when T is a vector.
pub fn exp2<T: NumericBuiltinExp2>(e: T) -> T {
    <T as NumericBuiltinExp2>::exp2(e)
}

mod exp2 {
    use super::*;

    impl NumericBuiltinExp2 for f32 {
        fn exp2(self) -> Self {
            self.exp2()
        }
    }

    macro_rules! impl_exp2_vec {
        ($ty:ty) => {
            impl NumericBuiltinExp2 for $ty {
                fn exp2(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t.exp2()),
                    }
                }
            }
        };
    }
    impl_exp2_vec!(Vec2f);
    impl_exp2_vec!(Vec3f);
    impl_exp2_vec!(Vec4f);
}

/// Provides the numeric built-in function `degrees`.
pub trait NumericBuiltinDegrees {
    /// Converts radians to degrees. Component-wise when T is a vector.
    fn degrees(self) -> Self;
}

/// Converts radians to degrees. Component-wise when T is a vector.
pub fn degrees<T: NumericBuiltinDegrees>(e: T) -> T {
    <T as NumericBuiltinDegrees>::degrees(e)
}

mod degrees {
    use super::*;

    impl NumericBuiltinDegrees for f32 {
        fn degrees(self) -> Self {
            self * 180.0 / std::f32::consts::PI
        }
    }

    macro_rules! impl_degrees_vec {
        ($ty:ty) => {
            impl NumericBuiltinDegrees for $ty {
                fn degrees(self) -> Self {
                    Self {
                        inner: self.inner.map(|t| t * 180.0 / std::f32::consts::PI),
                    }
                }
            }
        };
    }
    impl_degrees_vec!(Vec2f);
    impl_degrees_vec!(Vec3f);
    impl_degrees_vec!(Vec4f);
}

/// Provides the logical built-in function `all`.
pub trait LogicalBuiltinAll {
    /// Returns true if all components are true.
    fn all(self) -> bool;
}

/// Returns true if all components of e are true.
pub fn all<T: LogicalBuiltinAll>(e: T) -> bool {
    <T as LogicalBuiltinAll>::all(e)
}

mod all {
    use super::*;

    impl LogicalBuiltinAll for Vec2b {
        fn all(self) -> bool {
            self.inner.all()
        }
    }

    impl LogicalBuiltinAll for Vec3b {
        fn all(self) -> bool {
            self.inner.all()
        }
    }

    impl LogicalBuiltinAll for Vec4b {
        fn all(self) -> bool {
            self.inner.all()
        }
    }
}

/// Provides the logical built-in function `any`.
pub trait LogicalBuiltinAny {
    /// Returns true if any component is true.
    fn any(self) -> bool;
}

/// Returns true if any component of e is true.
pub fn any<T: LogicalBuiltinAny>(e: T) -> bool {
    <T as LogicalBuiltinAny>::any(e)
}

mod any {
    use super::*;

    impl LogicalBuiltinAny for Vec2b {
        fn any(self) -> bool {
            self.inner.any()
        }
    }

    impl LogicalBuiltinAny for Vec3b {
        fn any(self) -> bool {
            self.inner.any()
        }
    }

    impl LogicalBuiltinAny for Vec4b {
        fn any(self) -> bool {
            self.inner.any()
        }
    }
}

pub enum SelectCondition<T> {
    Bool(bool),
    Vec(T),
}

/// Provides the logical built-in function `select`.
pub trait LogicalBuiltinSelect<Condition> {
    /// Returns t if cond is true, else f.
    fn select(f: Self, t: Self, cond: Condition) -> Self;
}

/// Returns t if cond is true, else f.
pub fn select<T: LogicalBuiltinSelect<C>, C>(f: T, t: T, cond: C) -> T {
    <T as LogicalBuiltinSelect<C>>::select(f, t, cond)
}

mod select {
    use super::*;

    macro_rules! impl_select_bool {
        ($ty:ty) => {
            impl LogicalBuiltinSelect<bool> for $ty {
                fn select(f: Self, t: Self, cond: bool) -> Self {
                    if cond { t } else { f }
                }
            }
        };
    }

    impl_select_bool!(f32);
    impl_select_bool!(i32);
    impl_select_bool!(u32);
    impl_select_bool!(bool);
    impl_select_bool!(Vec2f);
    impl_select_bool!(Vec3f);
    impl_select_bool!(Vec4f);
    impl_select_bool!(Vec2i);
    impl_select_bool!(Vec3i);
    impl_select_bool!(Vec4i);
    impl_select_bool!(Vec2u);
    impl_select_bool!(Vec3u);
    impl_select_bool!(Vec4u);
    impl_select_bool!(Vec2b);
    impl_select_bool!(Vec3b);
    impl_select_bool!(Vec4b);

    macro_rules! impl_select_vec {
        ($n:literal, $ty:ty) => {
            impl LogicalBuiltinSelect<Vec<$n, bool>> for Vec<$n, $ty> {
                fn select(f: Self, t: Self, cond: Vec<$n, bool>) -> Self {
                    let cond_array = bool::vec_to_array(cond);
                    let f_array = <$ty>::vec_to_array(f);
                    let mut t_array = <$ty>::vec_to_array(t);
                    for i in 0..t_array.len() {
                        t_array[i] = if cond_array[i] {
                            t_array[i]
                        } else {
                            f_array[i]
                        };
                    }
                    <$ty>::vec_from_array(t_array)
                }
            }
        };
    }

    impl_select_vec!(2, f32);
    impl_select_vec!(3, f32);
    impl_select_vec!(4, f32);

    impl_select_vec!(2, i32);
    impl_select_vec!(3, i32);
    impl_select_vec!(4, i32);

    impl_select_vec!(2, u32);
    impl_select_vec!(3, u32);
    impl_select_vec!(4, u32);

    impl_select_vec!(2, bool);
    impl_select_vec!(3, bool);
    impl_select_vec!(4, bool);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity_abs() {
        let t = -3.0f32;
        let abs_t = abs(t);
        assert_eq!(3.0, abs_t);
    }

    #[test]
    fn sanity_acos() {
        let t = 1.0f32;
        let acos_t = acos(t);
        assert_eq!(0.0, acos_t);
    }

    #[test]
    fn sanity_sin() {
        let t = 0.0f32;
        let sin_t = sin(t);
        assert_eq!(0.0, sin_t);
    }

    #[test]
    fn sanity_asin() {
        let t = 0.0f32;
        let asin_t = asin(t);
        assert_eq!(0.0, asin_t);
    }

    #[test]
    fn sanity_atan() {
        let t = 0.0f32;
        let atan_t = atan(t);
        assert_eq!(0.0, atan_t);
    }

    #[test]
    fn sanity_cos() {
        let t = 0.0f32;
        let cos_t = cos(t);
        assert_eq!(1.0, cos_t);
    }

    #[test]
    fn sanity_tan() {
        let t = 0.0f32;
        let tan_t = tan(t);
        assert_eq!(0.0, tan_t);
    }

    #[test]
    fn sanity_fract() {
        let t = 1.5f32;
        let fract_t = fract(t);
        assert_eq!(0.5, fract_t);
    }

    #[test]
    fn sanity_inverse_sqrt() {
        let t = 4.0f32;
        let inv_sqrt_t = inverse_sqrt(t);
        assert!((inv_sqrt_t - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sanity_length() {
        let t = 3.0f32;
        let len_t = length(t);
        assert_eq!(3.0, len_t);
    }

    #[test]
    fn sanity_log() {
        let t = std::f32::consts::E;
        let log_t = log(t);
        assert!((log_t - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sanity_log2() {
        let t = 8.0f32;
        let log2_t = log2(t);
        assert_eq!(3.0, log2_t);
    }

    #[test]
    fn sanity_max() {
        let a = 2.0f32;
        let b = 3.0f32;
        let max_ab = max(a, b);
        assert_eq!(3.0, max_ab);
    }

    #[test]
    fn sanity_min() {
        let a = 2.0f32;
        let b = 3.0f32;
        let min_ab = min(a, b);
        assert_eq!(2.0, min_ab);
    }

    #[test]
    fn sanity_pow() {
        let a = 2.0f32;
        let b = 3.0f32;
        let pow_ab = pow(a, b);
        assert_eq!(8.0, pow_ab);
    }

    #[test]
    fn sanity_radians() {
        let deg = 180.0f32;
        let rad = radians(deg);
        assert!((rad - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn sanity_round() {
        let t = 2.7f32;
        let round_t = round(t);
        assert_eq!(3.0, round_t);
    }

    #[test]
    fn sanity_saturate() {
        let t = 1.5f32;
        let sat_t = saturate(t);
        assert_eq!(1.0, sat_t);
    }

    #[test]
    fn sanity_sign() {
        let t = -3.0f32;
        let sign_t = sign(t);
        assert_eq!(-1.0, sign_t);
    }

    #[test]
    fn sanity_sqrt() {
        let t = 4.0f32;
        let sqrt_t = sqrt(t);
        assert_eq!(2.0, sqrt_t);
    }

    #[test]
    fn sanity_step() {
        let edge = 1.0f32;
        let x = 2.0f32;
        let step_val = step(edge, x);
        assert_eq!(1.0, step_val);
    }

    #[test]
    fn sanity_tanh() {
        let t = 0.0f32;
        let tanh_t = tanh(t);
        assert_eq!(0.0, tanh_t);
    }

    #[test]
    fn sanity_trunc() {
        let t = 2.7f32;
        let trunc_t = trunc(t);
        assert_eq!(2.0, trunc_t);
    }

    #[test]
    fn sanity_dot() {
        let a = vec3f(1.0, 2.0, 3.0);
        let b = vec3f(4.0, 5.0, 6.0);
        let dot_ab = dot(a, b);
        assert_eq!(32.0, dot_ab); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn sanity_normalize() {
        let v = vec3f(3.0, 0.0, 0.0);
        let norm_v = normalize(v);
        assert!((norm_v.inner.x - 1.0).abs() < 1e-6);
        assert!((norm_v.inner.y - 0.0).abs() < 1e-6);
        assert!((norm_v.inner.z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sanity_cross() {
        let a = vec3f(1.0, 0.0, 0.0);
        let b = vec3f(0.0, 1.0, 0.0);
        let cross_ab = cross(a, b);
        assert!((cross_ab.inner.x - 0.0).abs() < 1e-6);
        assert!((cross_ab.inner.y - 0.0).abs() < 1e-6);
        assert!((cross_ab.inner.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sanity_clamp() {
        assert_eq!(clamp(5.0f32, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0f32, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0f32, 0.0, 10.0), 10.0);
        assert_eq!(clamp(5i32, 0, 10), 5);
        assert_eq!(clamp(5u32, 0, 10), 5);
    }

    #[test]
    fn sanity_mix() {
        let result = mix(0.0f32, 10.0, 0.5);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn sanity_floor() {
        assert_eq!(floor(2.7f32), 2.0);
        assert_eq!(floor(-2.7f32), -3.0);
    }

    #[test]
    fn sanity_ceil() {
        assert_eq!(ceil(2.3f32), 3.0);
        assert_eq!(ceil(-2.3f32), -2.0);
    }

    #[test]
    fn sanity_exp() {
        let result = exp(1.0f32);
        assert!((result - std::f32::consts::E).abs() < 1e-6);
    }

    #[test]
    fn sanity_exp2() {
        assert_eq!(exp2(3.0f32), 8.0);
    }

    #[test]
    fn sanity_degrees() {
        let rad = std::f32::consts::PI;
        let deg = degrees(rad);
        assert!((deg - 180.0).abs() < 1e-5);
    }

    #[test]
    fn sanity_all() {
        assert!(all(vec2b(true, true)));
        assert!(!all(vec2b(true, false)));
        assert!(!all(vec2b(false, false)));
    }

    #[test]
    fn sanity_any() {
        assert!(any(vec2b(true, false)));
        assert!(any(vec2b(false, true)));
        assert!(!any(vec2b(false, false)));
    }

    #[test]
    fn sanity_select() {
        assert_eq!(select(1.0f32, 2.0, true), 2.0);
        assert_eq!(select(1.0f32, 2.0, false), 1.0);
        assert_eq!(select(1i32, 2, true), 2);
        assert_eq!(select(1u32, 2, false), 1);
        assert_eq!(
            select(vec2f(0.0, 1.0), vec2f(10.0, 11.0), false),
            vec2f(0.0, 1.0)
        );
        assert_eq!(
            select(vec2f(0.0, 1.0), vec2f(10.0, 11.0), vec2b(true, false)),
            vec2f(10.0, 1.0)
        );
    }
}
