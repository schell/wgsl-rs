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

### Zero-Vector Semantics

WGSL's `normalize` on a zero vector is not glam's: glam returns zero,
WGSL produces NaN. Where the two worlds disagree, glm picks the defined
behavior and names it: `normalize_or_zero`, and the zero-guard inside
`angle_between`. Plain `normalize` keeps WGSL semantics.

## Conventions

Handedness and depth-range conventions follow glam's explicit suffix
style (`_rh`/`_lh`, `_gl` for -1..1 depth) rather than GLM's compile-time
defines, which Rust has no equivalent of. Defaults lean wgpu-native:
right-handed, 0..1 depth. Matrix and quaternion helpers arrive in later
phases.