# Matrices

wgsl-rs covers all nine WGSL matrix shapes. Square matrices have short aliases; non-square matrices use the `MatCxRf` naming convention where `C` is column count and `R` is row count.

## Square Aliases

| Alias     | Generic form     | WGSL          |
| --------- | ---------------- | ------------- |
| `Mat2f`   | `Mat2x2<f32>`    | `mat2x2<f32>` |
| `Mat3f`   | `Mat3x3<f32>`    | `mat3x3<f32>` |
| `Mat4f`   | `Mat4x4<f32>`    | `mat4x4<f32>` |

## Non-Square Aliases

| Alias       | Generic form     | WGSL          |
| ----------- | ---------------- | ------------- |
| `Mat2x3f`   | `Mat2x3<f32>`    | `mat2x3<f32>` |
| `Mat2x4f`   | `Mat2x4<f32>`    | `mat2x4<f32>` |
| `Mat3x2f`   | `Mat3x2<f32>`    | `mat3x2<f32>` |
| `Mat3x4f`   | `Mat3x4<f32>`    | `mat3x4<f32>` |
| `Mat4x2f`   | `Mat4x2<f32>`    | `mat4x2<f32>` |
| `Mat4x3f`   | `Mat4x3<f32>`    | `mat4x3<f32>` |

All matrices are `f32` only, matching the WGSL specification. The generic form (`Mat4x4<f32>`, `Mat2x3<f32>`, etc.) is accepted anywhere an alias is.

## Constructors

Matrix constructors take column vectors. The column vector width must match the row count of the matrix.

```rust
const IDENTITY: Mat4f = mat4x4f(
    vec4f(1.0, 0.0, 0.0, 0.0),
    vec4f(0.0, 1.0, 0.0, 0.0),
    vec4f(0.0, 0.0, 1.0, 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0),
);

const ROTATION_2D: Mat3f = mat3x3f(
    vec3f(0.866, 0.5,  0.0),
    vec3f(-0.5,  0.866, 0.0),
    vec3f(0.0,   0.0,  1.0),
);

const SCALE_2D: Mat2f = mat2x2f(vec2f(2.0, 0.0), vec2f(0.0, 2.0));

const M_3X2: Mat3x2f = mat3x2f(vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 0.0));
```

The constructor name mirrors WGSL: `matCxRf(col0, col1, ...)`. Each column argument must be a `VecRf` (or `VecR<i32>`/`VecR<u32>` where the matrix is integer — currently only `f32`).

## Multiplication

Matrix-times-vector and matrix-times-matrix use Rust's `*` operator and emit WGSL `*`:

```rust
let m: Mat4f = IDENTITY;
let v: Vec4f = vec4f(1.0, 2.0, 3.0, 1.0);
let transformed: Vec4f = m * v;          // -> m * v

let a: Mat4f = IDENTITY;
let b: Mat4f = IDENTITY;
let composed: Mat4f = a * b;             // -> a * b
```

The result type is inferred by Rust and verified by the transpiler against the WGSL rules: `matCxR * vecC` yields `vecR`; `matCxR * matRxC2` yields `matCx2`.