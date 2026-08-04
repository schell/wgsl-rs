# Matrix & Vector Functions

`wgsl_rs::std` provides vector and matrix types, constructors, and the WGSL
builtins that operate on them.

## Types

Vector and matrix types are aliases over the generic `Vec` and `Mat` structs:

| Type | WGSL | Components |
|------|------|-----------|
| `Vec2f`, `Vec3f`, `Vec4f` | `vec2f`, `vec3f`, `vec4f` | 2/3/4 × `f32` |
| `Vec2i`, `Vec3i`, `Vec4i` | `vec2i`, `vec3i`, `vec4i` | 2/3/4 × `i32` |
| `Vec2u`, `Vec3u`, `Vec4u` | `vec2u`, `vec3u`, `vec4u` | 2/3/4 × `u32` |
| `Mat2x2f`, `Mat2x3f`, `Mat2x4f` | `mat2x2f`, ... | 2 columns |
| `Mat3x2f`, `Mat3x3f`, `Mat3x4f` | `mat3x2f`, ... | 3 columns |
| `Mat4x2f`, `Mat4x3f`, `Mat4x4f` | `mat4x2f`, ... | 4 columns |
| `Mat4f` | `mat4x4f` | alias for `Mat4x4f` |

See [Scalars & Literals](../types/scalars.md), [Vectors & Swizzles](../types/vectors.md),
and [Matrices](../types/matrices.md) for full coverage.

## Constructors

Constructors are free functions named like the WGSL type:

```rust
let a = vec2f(1.0, 2.0);
let b = vec3f(1.0, 2.0, 3.0);
let c = vec4f(0.0);                 // splat
let m = mat4x4f(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
);
```

Constructors accept scalars, smaller vectors, and combinations thereof, just
like WGSL.

## Vector operations

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `dot(a, b)` | `dot` | Dot product. |
| `cross(a, b)` | `cross` | 3D cross product (`Vec3f` only). |
| `length(v)` | `length` | Euclidean length. |
| `distance(a, b)` | `distance` | `length(a - b)`. |
| `normalize(v)` | `normalize` | Unit vector. |
| `reflect(i, n)` | `reflect` | Reflect incident `i` about normal `n`. |
| `refract(i, n, eta)` | `refract` | Refraction per Snell's law. |
| `face_forward(n, i, ng)` | `faceForward` | Orient `n` to face away from `i`. |
| `step(edge, x)` | `step` | Heaviside-like step. |
| `mix(a, b, t)` | `mix` | Linear blend. |
| `clamp(x, lo, hi)` | `clamp` | Per-component clamp. |
| `min(a, b)`, `max(a, b)` | `min`, `max` | Per-component min/max. |
| `abs(v)` | `abs` | Per-component absolute value. |
| `sign(v)` | `sign` | Per-component sign. |
| `floor(v)`, `ceil(v)`, `round(v)`, `trunc(v)`, `fract(v)` | same | Per-component rounding. |
| `pow(v, e)` | `pow` | Per-component power. |
| `exp(v)`, `log(v)`, `sqrt(v)`, `inverse_sqrt(v)` | same | Per-component. |
| `sin(v)`, `cos(v)`, `tan(v)`, ... | same | Per-component trig. |

## Matrix builtins

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `transpose(m)` | `transpose` | Matrix transpose. |
| `determinant(m)` | `determinant` | Determinant of a square matrix. |

```rust
#[wgsl]
pub mod matrix_example {
    use wgsl_rs::std::*;

    pub fn normal_matrix(model: Mat4f) -> Mat3f {
        let upper = mat3x3f(
            model[0].xyz(),
            model[1].xyz(),
            model[2].xyz(),
        );
        let det = determinant(upper);
        if abs(det) < 1e-8 {
            return mat3x3f(1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0);
        }
        transpose(upper) * (1.0 / det)
    }
}
```

> `inverse` is not a WGSL builtin. Compute it from the adjugate and
> `determinant`, or use `transpose` of the cofactor matrix for the common
> 3×3 normal-matrix case.

## Component access

Vector components are accessed with `.x()`, `.y()`, `.z()`, `.w()` or via
swizzle methods like `.xyz()`, `.xy()`, `.xx()`. See
[Vectors & Swizzles](../types/vectors.md).

Matrix columns are indexed with `m[i]` (returns a vector) and individual
entries with `m[i][j]`, matching WGSL semantics.