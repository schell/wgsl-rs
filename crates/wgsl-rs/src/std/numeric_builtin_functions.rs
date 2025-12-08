//! Traits that provide WGSL's numeric builtin functions.
//!
//! See <https://gpuweb.github.io/gpuweb/wgsl/#numeric-builtin-functions>.
//!
//! | Status ✔️/❌ | Function | Parameter Types | Description |
//! | --- | --- | --- | --- |
//! |  ✔️ | fn abs(e: T ) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | The absolute value of e. Component-wise when T is a vector. <br>       If e is a floating-point type, then the result is e with a positive sign bit.<br>    If e is an unsigned integer scalar type, then the result is e.<br>    If e is a signed integer scalar type and evaluates to the largest<br>    negative value, then the result is e. |
//! |  ✔️ | fn acos(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) > 1. |
//! |     | fn acosh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when e < 1. |
//! |  ✔️ | fn asin(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) > 1. |
//! |     | fn asinh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the inverse hyperbolic sine (sinh-1) of e, as a hyperbolic angle in radians. That is, approximates x such that sinh(x) = e. <br>       Component-wise when T is a vector. |
//! |  ✔️ | fn atan(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the principal value, in radians, of the inverse tangent (tan-1) of e. That is, approximates x with π/2 ≤ x ≤ π/2, such that tan(x) = e. <br>       Component-wise when T is a vector. |
//! |     | fn atanh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful when abs(e) ≥ 1. |
//! |     | fn atan2(y: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns an angle, in radians, in the interval [-π, π] whose tangent is y÷x. <br>       The quadrant selected by the result depends on the signs of y and x.<br>    For example, the function may be implemented as:<br>       <br>        <br>         atan(y/x) when x > 0<br>        <br>         atan(y/x) + π when (x < 0) and (y > 0)<br>        <br>         atan(y/x) - π when (x < 0) and (y < 0)<br>       <br>       Note: atan2 is ill-defined when y/x is ill-defined, at the origin (x,y) = (0,0), and when y is non-normal or infinite.<br>       Component-wise when T is a vector. |
//! |     | fn ceil(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the ceiling of e. Component-wise when T is a vector. |
//! |     | fn clamp(e: T, low: T, high: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Restricts the value of e within a range. <br>       If T is an integer type, then the result is min(max(e, low), high).<br>       If T is a floating-point type, then the result is either min(max(e, low), high), or the median of the three values e, low, high.<br>       Component-wise when T is a vector.<br>       If low is greater than high, then:<br>       <br>        <br>         It is a shader-creation error if low and high are const-expressions.<br>        <br>         It is a pipeline-creation error if low and high are override-expressions. |
//! |  ✔️ | fn cos(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the cosine of e, where e is in radians. Component-wise when T is a vector. |
//! |     | fn cosh(arg: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic cosine of arg, where arg is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function (earg + e−arg)÷2,<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector |
//! |     | fn countLeadingZeros(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of consecutive 0 bits starting from the most significant bit<br>        of e, when T is a scalar type. Component-wise when T is a vector. Also known as "clz" in some languages. |
//! |     | fn countOneBits(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of 1 bits in the representation of e. Also known as "population count". Component-wise when T is a vector. |
//! |     | fn countTrailingZeros(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | The number of consecutive 0 bits starting from the least significant bit<br>        of e, when T is a scalar type. Component-wise when T is a vector. Also known as "ctz" in some languages. |
//! |     | fn cross(e1: vec3<T>, e2: vec3<T>) -> vec3<T> | T is AbstractFloat, f32, or f16 | Returns the cross product of e1 and e2. |
//! |     | fn degrees(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Converts radians to degrees, approximating e1 × 180 ÷ π. Component-wise when T is a vector |
//! |     | fn determinant(e: matCxC<T>) -> T | T is AbstractFloat, f32, or f16 | Returns the determinant of e. |
//! |     | fn distance(e1: T, e2: T) -> S | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the distance between e1 and e2 (e.g. length(e1 - e2)). |
//! |     | fn dot(e1: vecN<T>, e2: vecN<T>) -> T | T is AbstractInt, AbstractFloat, i32, u32, f32, or f16 | Returns the dot product of e1 and e2. |
//! |     | fn exp(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the natural exponentiation of e1 (e.g. ee1). Component-wise when T is a vector. |
//! |     | fn exp2(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns 2 raised to the power e (e.g. 2e). Component-wise when T is a vector. |
//! |     | fn extractBits(e: T, offset: u32, count: u32) -> T | T is i32 or vecN<i32> | Reads bits from an integer, with sign extension. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is 0 if c is 0. <br>        Otherwise, bits 0..c - 1 of the result are copied from bits o..o + c - 1 of e.<br>       Other bits of the result are the same as bit c - 1 of the result. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! |     | fn extractBits(e: T, offset: u32, count: u32) -> T | T is u32 or vecN<u32> | Reads bits from an integer, without sign extension. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is 0 if c is 0. <br>        Otherwise, bits 0..c - 1 of the result are copied from bits o..o + c - 1 of e.<br>       Other bits of the result are 0. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! |     | fn faceForward(e1: T, e2: T, e3: T) -> T | T is vecN<AbstractFloat>, vecN<f32>, or vecN<f16> | Returns e1 if dot(e2, e3) is negative, and -e1 otherwise. |
//! |     | fn firstLeadingBit(e: T) -> T | T is i32 or vecN<i32> | Note: Since signed integers use twos-complement representation,<br>the sign bit appears in the most significant bit position. |
//! |     | fn firstLeadingBit(e: T) -> T | T is u32 or vecN<u32> | For scalar T, the result is: <br>       <br>        T(-1) if e is zero. <br>        Otherwise the position of the most significant 1<br>            bit in e. <br>       <br>        Component-wise when T is a vector. |
//! |     | fn firstTrailingBit(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | For scalar T, the result is: <br>       <br>        T(-1) if e is zero. <br>        Otherwise the position of the least significant 1<br>            bit in e. <br>       <br>        Component-wise when T is a vector. |
//! |     | fn floor(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the floor of e. Component-wise when T is a vector. |
//! |     | fn fma(e1: T, e2: T, e3: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns e1 * e2 + e3. Component-wise when T is a vector. <br>       Note: The name fma is short for "fused multiply add".<br>       Note: The IEEE-754 fusedMultiplyAdd operation computes the intermediate results<br>    as if with unbounded range and precision, and only the final result is rounded<br>    to the destination type.<br>    However, the § 14.6.1 Floating Point Accuracy rule for fma allows an implementation<br>    which performs an ordinary multiply to the target type followed by an ordinary addition.<br>    In this case the intermediate values may overflow or lose accuracy, and the overall<br>    operation is not "fused" at all. |
//! |     | fn fract(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: Valid results are in the closed interval [0, 1.0].<br>For example, if e is a very small negative number, then fract(e) may be 1.0. |
//! |     | fn insertBits(e: T, newbits: T, offset: u32, count: u32) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | Sets bits in an integer. <br>       When T is a scalar type, then:<br>       <br>        w is the bit width of T <br>        o = min(offset, w) <br>        c = min(count, w - o) <br>        The result is e if c is 0. <br>        Otherwise,<br>       bits o..o + c - 1 of the result are copied from bits 0..c - 1 of newbits.<br>       Other bits of the result are copied from e. <br>       <br>        Component-wise when T is a vector. <br>       If count + offset is greater than w, then:<br>       <br>        <br>         It is a shader-creation error if count and offset are const-expressions.<br>        <br>         It is a pipeline-creation error if count and offset are override-expressions. |
//! |     | fn inverseSqrt(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e ≤ 0. |
//! |     | fn ldexp(e1: T, e2: I) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> I is AbstractInt, i32, vecN<AbstractInt>, or vecN<i32> I is a vector if and only if T is a vector I is concrete if and only if T is a concrete | Returns e1 * 2e2, except: <br>       <br>        <br>         The result may be zero if e2 + bias ≤ 0.<br>        <br>         If e2 > bias + 1<br>         <br>          <br>           It is a shader-creation error if e2 is a const-expression.<br>          <br>           It is a pipeline-creation error if e2 is an override-expression.<br>          <br>           Otherwise the result is an indeterminate value for T.<br>         <br>       <br>       Here, bias is the exponent bias of the floating point format:<br>       <br>        <br>         15 for f16<br>        <br>         127 for f32<br>        <br>         1023 for AbstractFloat, when AbstractFloat is IEEE-754 binary64<br>       <br>       If x is zero or a finite normal value for its type, then:<br>        x = ldexp(frexp(x).fract, frexp(x).exp) <br>       Component-wise when T is a vector.<br>       Note: A mnemonic for the name ldexp is "load exponent".<br>    The name may have been taken from the corresponding instruction in the floating point unit of<br>    the PDP-11. |
//! |     | fn length(e: T) -> S | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the length of e. Evaluates to the absolute value of e if T is scalar. Evaluates to sqrt(e[0]2 + e[1]2 + ...) if T is a vector type. <br>       Note: The scalar case may be evaluated as sqrt(e * e),<br>        which may unnecessarily overflow or lose accuracy. |
//! |     | fn log(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e < 0. |
//! |     | fn log2(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Note: The result is not mathematically meaningful if e < 0. |
//! |     | fn max(e1: T, e2: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Returns e2 if e1 is less than e2, and e1 otherwise. Component-wise when T is a vector. <br>       If e1 and e2 are floating-point values, then:<br>       <br>        <br>         If both e1 and e2 are denormalized, then the result may be either value.<br>        <br>         If one operand is a NaN, the other is returned.<br>        <br>         If both operands are NaNs, a NaN is returned. |
//! |     | fn min(e1: T, e2: T) -> T | S is AbstractInt, AbstractFloat, i32, u32, f32, or f16. T is S, or vecN<S> | Returns e2 if e2 is less than e1, and e1 otherwise. Component-wise when T is a vector. <br>       If e1 and e2 are floating-point values, then:<br>       <br>        <br>         If both e1 and e2 are denormalized, then the result may be either value.<br>        <br>         If one operand is a NaN, the other is returned.<br>        <br>         If both operands are NaNs, a NaN is returned. |
//! |     | fn mix(e1: T, e2: T, e3: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the linear blend of e1 and e2 (e.g. e1 * (1 - e3) + e2 * e3). Component-wise when T is a vector. |
//! |     | fn mix(e1: T2, e2: T2, e3: T) -> T2 | T is AbstractFloat, f32, or f16 T2 is vecN<T> | Returns the component-wise linear blend of e1 and e2,<br>        using scalar blending factor e3 for each component. Same as mix(e1, e2, T2(e3)). |
//! |     | fn modf(e: T) -> __modf_result_f32 | T is f32 | Note: A value cannot be explicitly declared with the type __modf_result_f32,<br>but a value may infer the type. |
//! |     | fn modf(e: T) -> __modf_result_f16 | T is f16 | Note: A value cannot be explicitly declared with the type __modf_result_f16,<br>but a value may infer the type. |
//! |     | fn modf(e: T) -> __modf_result_abstract | T is AbstractFloat | Note: A value cannot be explicitly declared with the type __modf_result_abstract,<br>but a value may infer the type. |
//! |     | fn modf(e: T) -> __modf_result_vecN_f32 | T is vecN<f32> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_f32,<br>but a value may infer the type. |
//! |     | fn modf(e: T) -> __modf_result_vecN_f16 | T is vecN<f16> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_f16,<br>but a value may infer the type. |
//! |     | fn modf(e: T) -> __modf_result_vecN_abstract | T is vecN<AbstractFloat> | Note: A value cannot be explicitly declared with the type __modf_result_vecN_abstract,<br>but a value may infer the type. |
//! |     | fn normalize(e: vecN<T> ) -> vecN<T> | T is AbstractFloat, f32, or f16 | Returns a unit vector in the same direction as e. |
//! |     | fn pow(e1: T, e2: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns e1 raised to the power e2. Component-wise when T is a vector. |
//! |     | fn quantizeToF16(e: T) -> T | T is f32 or vecN<f32> | Note: The vec2<f32> case is the same as unpack2x16float(pack2x16float(e)). |
//! |     | fn radians(e1: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Converts degrees to radians, approximating e1 × π ÷ 180. Component-wise when T is a vector |
//! |     | fn reflect(e1: T, e2: T) -> T | T is vecN<AbstractFloat>, vecN<f32>, or vecN<f16> | For the incident vector e1 and surface orientation e2, returns the reflection direction e1 - 2 * dot(e2, e1) * e2. |
//! |     | fn refract(e1: T, e2: T, e3: I) -> T | T is vecN<I> I is AbstractFloat, f32, or f16 | For the incident vector e1 and surface normal e2, and the ratio of<br>    indices of refraction e3,<br>    let k = 1.0 - e3 * e3 * (1.0 - dot(e2, e1) * dot(e2, e1)).<br>    If k < 0.0, returns the refraction vector 0.0, otherwise return the refraction vector e3 * e1 - (e3 * dot(e2, e1) + sqrt(k)) * e2. |
//! |     | fn reverseBits(e: T) -> T | T is i32, u32, vecN<i32>, or vecN<u32> | Reverses the bits in e:  The bit at position k of the result equals the<br>        bit at position 31 -k of e. Component-wise when T is a vector. |
//! |     | fn round(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Result is the integer k nearest to e, as a floating point value. When e lies halfway between integers k and k + 1,<br>        the result is k when k is even, and k + 1 when k is odd. Component-wise when T is a vector. |
//! |     | fn saturate(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns clamp(e, 0.0, 1.0). Component-wise when T is a vector. |
//! |     | fn sign(e: T) -> T | S is AbstractInt, AbstractFloat, i32, f32, or f16. T is S, or vecN<S> | Result is: <br>       <br>         1 when e > 0 <br>         0 when e = 0 <br>         -1 when e < 0 <br>       <br>       Component-wise when T is a vector. |
//! |  ✔️ | fn sin(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the sine of e, where e is in radians. Component-wise when T is a vector. |
//! |     | fn sinh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic sine of e, where e is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function<br>    (earg − e−arg)÷2,<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector. |
//! |     | fn smoothstep(low: T, high: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the smooth Hermite interpolation between 0 and 1. Component-wise when T is a vector. <br>       For scalar T, the result is t * t * (3.0 - 2.0 * t),<br>    where t = clamp((x - low) / (high - low), 0.0, 1.0). |
//! |     | fn sqrt(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the square root of e. Component-wise when T is a vector. |
//! |     | fn step(edge: T, x: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns 1.0 if edge ≤ x, and 0.0 otherwise. Component-wise when T is a vector. |
//! |  ✔️ | fn tan(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the tangent of e, where e is in radians. Component-wise when T is a vector. |
//! |     | fn tanh(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns the hyperbolic tangent of e, where e is a hyperbolic angle in radians.<br>    Approximates the pure mathematical function<br>    (earg − e−arg) ÷ (earg + e−arg)<br>    but not necessarily computed that way. <br>       Component-wise when T is a vector. |
//! |     | fn transpose(e: matRxC<T>) -> matCxR<T> | T is AbstractFloat, f32, or f16 | Returns the transpose of e. |
//! |     | fn trunc(e: T) -> T | S is AbstractFloat, f32, or f16. T is S or vecN<S> | Returns truncate(e), the nearest whole number whose absolute value<br>    is less than or equal to the absolute value of e. Component-wise when T is a vector. |

use crate::std::*;

/// Provides the numeric built-in function `abs`.
pub trait NumericBuiltinAbs {
    /// The absolute value of e. Component-wise when T is a vector.
    /// * If e is a floating-point type, then the result is e with a positive sign bit.
    /// * If e is an unsigned integer scalar type, then the result is e.
    /// * If e is a signed integer scalar type and evaluates to the largest negative value, then the result is e.
    fn abs(self) -> Self;
}

/// The absolute value of e. Component-wise when T is a vector.
/// * If e is a floating-point type, then the result is e with a positive sign bit.
/// * If e is an unsigned integer scalar type, then the result is e.
/// * If e is a signed integer scalar type and evaluates to the largest negative value, then the result is e.
pub fn abs<T: NumericBuiltinAbs>(e: T) -> T {
    <T as NumericBuiltinAbs>::abs(e)
}

/// Provides the numeric built-in function `acos`.
pub trait NumericBuiltinAcos {
    /// Returns the principal value, in radians, of the inverse cosine (cos⁻¹) of e.
    /// Component-wise when T is a vector.
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
    /// Returns the sine of e, where e is in radians. Component-wise when T is a vector.
    fn sin(self) -> Self;
}

/// Returns the sine of e, where e is in radians. Component-wise when T is a vector.
pub fn sin<T: NumericBuiltinSin>(e: T) -> T {
    <T as NumericBuiltinSin>::sin(e)
}

/// Provides the numeric built-in function `asin`.
pub trait NumericBuiltinAsin {
    /// Returns the principal value, in radians, of the inverse sine (sin⁻¹) of e.
    /// Component-wise when T is a vector.
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
    /// Returns the principal value, in radians, of the inverse tangent (tan⁻¹) of e.
    /// Component-wise when T is a vector.
    fn atan(self) -> Self;
}

/// Returns the principal value, in radians, of the inverse tangent (tan⁻¹) of e.
/// Component-wise when T is a vector.
pub fn atan<T: NumericBuiltinAtan>(e: T) -> T {
    <T as NumericBuiltinAtan>::atan(e)
}

/// Provides the numeric built-in function `cos`.
pub trait NumericBuiltinCos {
    /// Returns the cosine of e, where e is in radians. Component-wise when T is a vector.
    fn cos(self) -> Self;
}

/// Returns the cosine of e, where e is in radians. Component-wise when T is a vector.
pub fn cos<T: NumericBuiltinCos>(e: T) -> T {
    <T as NumericBuiltinCos>::cos(e)
}

/// Provides the numeric built-in function `tan`.
pub trait NumericBuiltinTan {
    /// Returns the tangent of e, where e is in radians. Component-wise when T is a vector.
    fn tan(self) -> Self;
}

/// Returns the tangent of e, where e is in radians. Component-wise when T is a vector.
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
}
