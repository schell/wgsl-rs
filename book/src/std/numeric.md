# Numeric Builtins

The numeric builtins mirror WGSL's numeric functions. Each is a free function
exported by `wgsl_rs::std::*`. Many are defined per concrete type via a
one-trait-per-builtin strategy: the transpiler resolves the right WGSL builtin
based on argument types.

## Trigonometric

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `sin(x)` | `sin` | Sine, radians. |
| `cos(x)` | `cos` | Cosine, radians. |
| `tan(x)` | `tan` | Tangent, radians. |
| `asin(x)` | `asin` | Arc sine, result in radians. |
| `acos(x)` | `acos` | Arc cosine, result in radians. |
| `atan(x)` | `atan` | Arc tangent, result in radians. |
| `atan2(y, x)` | `atan2` | Arc tangent of `y / x`, quadrant-aware. |
| `sinh(x)` | `sinh` | Hyperbolic sine. |
| `cosh(x)` | `cosh` | Hyperbolic cosine. |
| `tanh(x)` | `tanh` | Hyperbolic tangent. |
| `asinh(x)` | `asinh` | Arc hyperbolic sine. |
| `acosh(x)` | `acosh` | Arc hyperbolic cosine. |
| `atanh(x)` | `atanh` | Arc hyperbolic tangent. |
| `radians(x)` | `radians` | Degrees → radians. |
| `degrees(x)` | `degrees` | Radians → degrees. |

## Exponential, logarithmic, and root

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `pow(x, y)` | `pow` | `x` raised to `y`. |
| `exp(x)` | `exp` | `e^x`. |
| `exp2(x)` | `exp2` | `2^x`. |
| `log(x)` | `log` | Natural logarithm. |
| `log2(x)` | `log2` | Base-2 logarithm. |
| `sqrt(x)` | `sqrt` | Square root. |
| `inverse_sqrt(x)` | `inverseSqrt` | `1 / sqrt(x)`. |

## Rounding and floating-point decomposition

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `ceil(x)` | `ceil` | Round toward +∞. |
| `floor(x)` | `floor` | Round toward −∞. |
| `round(x)` | `round` | Round to nearest integer. |
| `trunc(x)` | `trunc` | Round toward zero. |
| `fract(x)` | `fract` | Fractional part: `x - floor(x)`. |
| `sign(x)` | `sign` | Sign of `x` as −1, 0, or +1. |
| `abs(x)` | `abs` | Absolute value. |
| `fma(a, b, c)` | `fma` | Fused multiply-add: `a*b + c` with single rounding. |
| `modf(x)` | `modf` | Split into fractional and whole parts (see below). |
| `frexp(x)` | `frexp` | Split significand and exponent (see below). |
| `ldexp(fract, exp)` | `ldexp` | `fract * 2^exp`, inverse of `frexp`. |

### `modf`

`modf(x)` returns a struct with two fields:

```rust
let r = modf(-1.5);
let frac: f32 = r.fract; //  0.5
let whole: f32 = r.whole; // -1.0
```

### `frexp`

`frexp(x)` returns a struct with:

- `.fract` — significand in `[0.5, 1.0)`
- `.exp` — exponent such that `x = fract * 2^exp`

## Interpolation and clamping

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `mix(a, b, t)` | `mix` | Linear interpolation: `a + (b - a) * t`. |
| `clamp(x, lo, hi)` | `clamp` | Clamp `x` to `[lo, hi]`. |
| `min(x, y)` | `min` | Minimum. |
| `max(x, y)` | `max` | Maximum. |
| `saturate(x)` | `saturate` (idiom) | Clamp `x` to `[0.0, 1.0]`. |
| `step(edge, x)` | `step` | `0.0` if `x < edge`, else `1.0`. |

## Geometric

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `length(v)` | `length` | Euclidean length. |
| `distance(a, b)` | `distance` | `length(a - b)`. |
| `dot(a, b)` | `dot` | Dot product. |
| `cross(a, b)` | `cross` | 3D cross product. |
| `normalize(v)` | `normalize` | Unit-length vector: `v / length(v)`. |
| `reflect(i, n)` | `reflect` | Reflection of incident `i` about normal `n`. |
| `refract(i, n, eta)` | `refract` | Refraction per Snell's law. |
| `face_forward(n, i, ng)` | `faceForward` | `n` flipped to face away from `i` relative to `ng`. |

## Per-type dispatch

Some builtins are implemented as traits with one method per concrete scalar
type (the one-trait-per-builtin strategy). This keeps the CPU implementation
type-correct and lets the transpiler emit the exact WGSL overload. You call
them as ordinary free functions; the correct specialization is inferred from
argument types.

## Example

```rust
#[wgsl]
pub mod numeric_example {
    use wgsl_rs::std::*;

    pub fn to_srgb(linear: f32) -> f32 {
        if linear <= 0.0031308 {
            linear * 12.92
        } else {
            1.055 * pow(linear, 1.0 / 2.4) - 0.055
        }
    }

    pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}
```

`smoothstep` shown above is a user-authored helper using `clamp` and `pow`.
If a WGSL `smoothStep` builtin is supported by your target, prefer it.