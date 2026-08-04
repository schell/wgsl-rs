# Binary Operators

Demonstrates all supported binary operators in four fragment shaders: arithmetic (`+ - * / %`), comparison (`== != < <= > >=`), logical (`&& ||`), and bitwise (`& | ^ << >>`).

## Rust Source

```rust
#[wgsl]
pub mod binary_ops_example {
    //! Demonstrates all supported binary operators including:
    //! - Arithmetic: + - * / %
    //! - Comparison: == != < <= > >=
    //! - Logical: && ||
    //! - Bitwise: & | ^ << >>

    use wgsl_rs::std::*;

    // Demonstrates arithmetic operators including remainder.
    #[fragment]
    pub fn test_arithmetic() -> Vec4f {
        let a = vec3f(10.0, 11.0, 12.0);
        let b = 3.0;
        let add = a + b;
        let sub = a - b;
        let mul = a * b;
        let div = a / b;
        let rem = a % b;
        vec4f(add.x(), sub.y(), (mul * div).z(), rem.z())
    }

    // Demonstrates comparison operators.
    // All comparison operators return bool (or vecN<bool> for vectors).
    #[fragment]
    pub fn test_comparison() -> Vec4f {
        let a = 5;
        let b = 10;

        // Comparison operators
        let lt = a < b;
        let le = a <= b;
        let _gt = a > b;
        let ge = a >= b;
        let eq = a == b;
        let ne = a != b;

        // Use the booleans in a calculation
        // In WGSL, we use select() to convert bool to numeric
        let lt_val = select(0.0, 1.0, lt);
        let eq_val = select(0.0, 1.0, eq);
        let ne_val = select(0.0, 1.0, ne);
        let combined = select(0.0, 1.0, le && ge);

        vec4f(lt_val, eq_val, ne_val, combined)
    }

    // Demonstrates logical operators (short-circuit and/or).
    #[fragment]
    pub fn test_logical() -> Vec4f {
        let a = true;
        let b = false;

        // Logical operators (short-circuit evaluation)
        let and_result = a && b;
        let or_result = a || b;
        let complex = (a && b) || (!a && !b);

        let and_val = select(0.0, 1.0, and_result);
        let or_val = select(0.0, 1.0, or_result);
        let complex_val = select(0.0, 1.0, complex);

        vec4f(and_val, or_val, complex_val, 1.0)
    }

    // Demonstrates bitwise operators.
    #[fragment]
    pub fn test_bitwise() -> Vec4f {
        let a: u32 = 0xFF00;
        let b: u32 = 0x0F0F;

        // Bitwise operators
        let and_result = a & b;
        let or_result = a | b;
        let xor_result = a ^ b;

        // Shift operators
        let shl_result = a << 4u32;
        let shr_result = a >> 4u32;

        // Convert to floats for output (normalized)
        let and_f = f32(and_result) / 65535.0;
        let or_f = f32(or_result) / 65535.0;
        let xor_f = f32(xor_result) / 65535.0;
        let shift_f = f32(shl_result ^ shr_result) / 65535.0;

        vec4f(and_f, or_f, xor_f, shift_f)
    }
}
```

## Generated WGSL

```wgsl
@fragment fn test_arithmetic() -> @location(0) vec4f {
    let a = vec3f(10.0, 11.0, 12.0);
    let b = 3.0;
    let add = a + b;
    let sub = a - b;
    let mul = a * b;
    let div = a / b;
    let rem = a % b;
    return vec4f(add.x, sub.y, (mul * div).z, rem.z);
}

@fragment fn test_comparison() -> @location(0) vec4f {
    let a = 5;
    let b = 10;
    let lt = a < b;
    let le = a <= b;
    let _gt = a > b;
    let ge = a >= b;
    let eq = a == b;
    let ne = a != b;
    let lt_val = select(0.0, 1.0, lt);
    let eq_val = select(0.0, 1.0, eq);
    let ne_val = select(0.0, 1.0, ne);
    let combined = select(0.0, 1.0, le && ge);
    return vec4f(lt_val, eq_val, ne_val, combined);
}

@fragment fn test_logical() -> @location(0) vec4f {
    let a = true;
    let b = false;
    let and_result = a && b;
    let or_result = a || b;
    let complex = (a && b) || (!a && !b);
    let and_val = select(0.0, 1.0, and_result);
    let or_val = select(0.0, 1.0, or_result);
    let complex_val = select(0.0, 1.0, complex);
    return vec4f(and_val, or_val, complex_val, 1.0);
}

@fragment fn test_bitwise() -> @location(0) vec4f {
    let a: u32 = 65280;
    let b: u32 = 3855;
    let and_result = a & b;
    let or_result = a | b;
    let xor_result = a ^ b;
    let shl_result = a << 4u;
    let shr_result = a >> 4u;
    let and_f = f32(and_result) / 65535.0;
    let or_f = f32(or_result) / 65535.0;
    let xor_f = f32(xor_result) / 65535.0;
    let shift_f = f32(shl_result ^ shr_result) / 65535.0;
    return vec4f(and_f, or_f, xor_f, shift_f);
}
```

## Notes

- Hex literals like `0xFF00` are emitted as decimal WGSL literals.
- Short-circuit `&&`/`||` are preserved in the raw output; the naga-validated form lowers them to `if`/`else` because WGSL has no short-circuit logical operators.