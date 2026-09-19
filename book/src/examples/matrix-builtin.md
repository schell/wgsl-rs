# Matrix Builtin

Demonstrates the matrix builtin functions `determinant` and `transpose` for 2x2, 3x3, and 4x4 matrices.

## Rust Source

```rust
#[wgsl]
pub mod matrix_builtin_example {
    //! Demonstrates matrix builtin functions: `determinant` and `transpose`.
    use wgsl_rs::std::*;

    pub fn demo_determinant_2x2(m: Mat2f) -> f32 {
        determinant(m)
    }

    pub fn demo_determinant_3x3(m: Mat3f) -> f32 {
        determinant(m)
    }

    pub fn demo_determinant_4x4(m: Mat4f) -> f32 {
        determinant(m)
    }

    pub fn demo_transpose_4x4(m: Mat4f) -> Mat4f {
        transpose(m)
    }
}
```

## Generated WGSL

```wgsl
fn demo_determinant_2x2(m: mat2x2f) -> f32 {
    return determinant(m);
}

fn demo_determinant_3x3(m: mat3x3f) -> f32 {
    return determinant(m);
}

fn demo_determinant_4x4(m: mat4x4f) -> f32 {
    return determinant(m);
}

fn demo_transpose_4x4(m: mat4x4f) -> mat4x4f {
    return transpose(m);
}
```

## Notes

- `determinant` and `transpose` map directly to the WGSL builtins of the same names.