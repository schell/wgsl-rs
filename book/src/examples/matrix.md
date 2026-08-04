# Matrix

Demonstrates matrix types (`Mat2f`, `Mat3f`, `Mat4f`) and their constructors (`mat2x2f`, `mat3x3f`, `mat4x4f`) used in module-level constants.

## Rust Source

```rust
#[wgsl]
#[expect(dead_code, reason = "demonstration")]
pub mod matrix_example {
    //! Demonstrates matrix types and constructors.
    use wgsl_rs::std::*;

    // 4x4 identity matrix constant
    const IDENTITY: Mat4f = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0),
    );

    // 3x3 2D rotation matrix (30 degrees)
    // cos(30°) ≈ 0.866, sin(30°) = 0.5
    const ROTATION_2D: Mat3f = mat3x3f(
        vec3f(0.866, 0.5, 0.0),
        vec3f(-0.5, 0.866, 0.0),
        vec3f(0.0, 0.0, 1.0),
    );

    // 2x2 matrix constant
    const SCALE_2D: Mat2f = mat2x2f(vec2f(2.0, 0.0), vec2f(0.0, 2.0));

    #[vertex]
    pub fn matrix_vertex() -> Vec4f {
        vec4f(0.0, 0.0, 0.0, 1.0)
    }
}
```

## Generated WGSL

```wgsl
const IDENTITY: mat4x4f = mat4x4f(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0), vec4f(0.0, 0.0, 1.0, 0.0), vec4f(0.0, 0.0, 0.0, 1.0));
const ROTATION_2D: mat3x3f = mat3x3f(vec3f(0.866, 0.5, 0.0), vec3f(-0.5, 0.866, 0.0), vec3f(0.0, 0.0, 1.0));
const SCALE_2D: mat2x2f = mat2x2f(vec2f(2.0, 0.0), vec2f(0.0, 2.0));

@vertex fn matrix_vertex() -> @builtin(position) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
```

## Notes

- `Mat2f`/`Mat3f`/`Mat4f` alias to the WGSL `mat2x2f`/`mat3x3f`/`mat4x4f` types.
- Module-level `const` items become WGSL module-scope constants.