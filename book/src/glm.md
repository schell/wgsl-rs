# The glm Module

`wgsl_rs::glm` is a small layer of GLM-lineage helpers over the WGSL types.

**Lineage:** GLSL → GLM (C++) → glam (Rust) → here. The standard library
already gives you the WGSL builtin surface as free functions (`length`,
`normalize`, `mix`, `dot`, `cross`, ...), so glm only adds what GLM added on
top of GLSL: the handful of helpers shaders keep re-writing by hand. It is
gated behind the **`glm` cargo feature, on by default**.

glm does not introduce new type names: you keep writing the wgsl-rs
spellings (`Vec3f`, `Vec2i`, ...). The value is the math, not the names,
and it keeps the transpiler completely uninvolved.

## Import

Import glm alone; it re-exports `std`, so one glob brings everything:

```rust
use wgsl_rs::glm::*;
```

## Helpers

| Function | Notes |
| -------- | ----- |
| `length_squared2/3/4(v)` | `dot(v, v)`, no square root |
| `normalize_or_zero2/3/4(v)` | never NaN for zero input, unlike `normalize` |
| `angle_between2(a, b)` | `atan2` form; `0.0` if either input is zero |
| `angle_between3(a, b)` | `atan2` form; `0.0` if either input is zero |

```rust
use wgsl_rs::glm::*;

let v = vec3f(0.0, 1.0, 0.0);
let len2 = length_squared3(v);          // dot(v, v)
let n = normalize_or_zero3(v);          // never NaN
let angle = angle_between3(v, Vec3f::X);
```

## Matrix Transforms

All matrices are column-major `Mat4x4f` (or `Mat3x3f`/`Mat2x2f` where
noted), matching glam's storage. Builders, GLM-style accumulate forms,
transform helpers, and hand-written inverses:

| Function | Notes |
| -------- | ----- |
| `translation(v)`, `scaling(v)`, `rotation_axis_angle(axis, angle)` | pure builders; `axis` must be normalized |
| `translate(m, v)`, `rotate(m, angle, axis)`, `scale(m, v)` | post-multiply: `m * builder(...)` |
| `transform_point2(m, v)`, `transform_point3(m, v)` | affine point transform, no perspective divide |
| `transform_vector2(m, v)`, `transform_vector3(m, v)` | direction transform, no translation |
| `look_at_rh(eye, center, up)`, `look_at_lh(...)` | view matrices |
| `perspective_rh/lh/rh_gl(...)`, `orthographic_rh/lh/rh_gl(...)` | projections, see conventions |
| `inverse2x2(m)`, `inverse3x3(m)`, `inverse4x4(m)` | hand-written, shared by both worlds |

`determinant` and `transpose` already live in
[`std`](./std/matrix-vector.md); glm does not wrap them. Singular input
to the inverses produces inf/NaN components, matching glam's documented
behavior.

```rust
use wgsl_rs::glm::*;

let model = translate(
    rotation_axis_angle(Vec3f::Y, 0.7),
    vec3f(1.0, 2.0, 3.0),
);
let view = look_at_rh(
    vec3f(3.0, 4.0, 5.0),
    vec3f(0.0, 0.0, 0.0),
    Vec3f::Y,
);
let clip = perspective_rh(1.2, 1.5, 0.1, 100.0) * view
    * vec4f_from_3(transform_point3(model, vec3f(0.0, 0.0, 0.0)), 1.0);
```

### Zero-Vector Semantics

WGSL's `normalize` on a zero vector is not glam's: glam returns zero,
WGSL produces NaN. Where the two worlds disagree, glm picks the defined
behavior and names it: `normalize_or_zero`, and the zero-guard inside
`angle_between`. Plain `normalize` keeps WGSL semantics.

## Conventions

Handedness and depth-range conventions follow glam's explicit suffix
style (`_rh`/`_lh`, `_gl` for -1..1 depth) rather than GLM's compile-time
defines, which Rust has no equivalent of. The unsuffixed projection
variants use the wgpu-native 0..1 depth range (right-handed is the usual
default). The projection set mirrors glam 0.30 exactly, including its
lack of `_lh_gl` variants. Quaternion helpers arrive in a later phase.