# Control Flow

Demonstrates control-flow constructs: `if`/`else if`/`else`, `break` (including nested), explicit `return` statements, and `match`/switch with literal, or-patterns, and const patterns.

## if_example

Demonstrates `if` statements: simple `if`, `if`/`else`, `if`/`else if`/`else` chains, and nested `if`.

### Rust Source

```rust
#[wgsl]
#[allow(dead_code)]
pub mod if_example {
    //! Demonstrates if statements including:
    //! - Simple if
    //! - if/else
    //! - if/else if/else chains
    //! - Nested if statements

    use wgsl_rs::std::*;

    #[fragment]
    pub fn test_simple_if() -> Vec4f {
        let mut result = 0.0;
        if true {
            result = 1.0;
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_if_else() -> Vec4f {
        let mut result = 0.0;
        if result < 1.0 {
            result = 1.0;
        } else {
            result = 2.0;
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    #[allow(unused_assignments)]
    pub fn test_if_else_if_else() -> Vec4f {
        let x = 5;
        let mut result = 0.0;
        if x < 3 {
            result = 1.0;
        } else if x < 7 {
            result = 2.0;
        } else {
            result = 3.0;
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_nested_if() -> Vec4f {
        let x = 5;
        let y = 10;
        let mut result = 0.0;
        if x > 0 {
            if y > 5 {
                result = 1.0;
            } else {
                result = 0.5;
            }
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }
}
```

### Generated WGSL

```wgsl
@fragment fn test_simple_if() -> @location(0) vec4f {
    var result = 0.0;
    if true {
        result = 1.0;
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_if_else() -> @location(0) vec4f {
    var result = 0.0;
    if result < 1.0 {
        result = 1.0;
    } else {
        result = 2.0;
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_if_else_if_else() -> @location(0) vec4f {
    let x = 5;
    var result = 0.0;
    if x < 3 {
        result = 1.0;
    } else if x < 7 {
        result = 2.0;
    } else {
        result = 3.0;
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_nested_if() -> @location(0) vec4f {
    let x = 5;
    let y = 10;
    var result = 0.0;
    if x > 0 {
        if y > 5 {
            result = 1.0;
        } else {
            result = 0.5;
        }
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}
```

## break_example

Demonstrates `break` statements inside `while` loops, including conditional breaks and nested break.

### Rust Source

```rust
#[wgsl]
#[allow(dead_code, unused_assignments)]
pub mod break_example {
    use wgsl_rs::std::*;

    #[fragment]
    pub fn test_break_in_while() -> Vec4f {
        let mut i = 0;
        let mut sum = 0.0;

        while i < 100 {
            if i >= 10 {
                break;
            }
            sum += f32(i);
            i += 1;
        }

        vec4f(sum / 100.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_break_with_condition() -> Vec4f {
        let mut value = 1.0;
        let mut iterations = 0;

        while iterations < 100 {
            value *= 1.1;
            iterations += 1;

            if value > 50.0 {
                break;
            }
        }

        vec4f(value / 100.0, f32(iterations) / 100.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_nested_break() -> Vec4f {
        let mut i = 0;
        let mut j = 0;
        let mut found = 0;

        while i < 10 {
            j = 0;
            while j < 10 {
                if i * 10 + j == 55 {
                    found = 1;
                    break;
                }
                j += 1;
            }
            if found == 1 {
                break;
            }
            i += 1;
        }

        vec4f(f32(i) / 10.0, f32(j) / 10.0, f32(found), 1.0)
    }
}
```

### Generated WGSL

```wgsl
@fragment fn test_break_in_while() -> @location(0) vec4f {
    var i = 0;
    var sum = 0.0;
    while i < 100 {
        if i >= 10 {
            break;
        }
        sum += f32(i);
        i += 1;
    }
    return vec4f(sum / 100.0, 0.0, 0.0, 1.0);
}

@fragment fn test_break_with_condition() -> @location(0) vec4f {
    var value = 1.0;
    var iterations = 0;
    while iterations < 100 {
        value *= 1.1;
        iterations += 1;
        if value > 50.0 {
            break;
        }
    }
    return vec4f(value / 100.0, f32(iterations) / 100.0, 0.0, 1.0);
}

@fragment fn test_nested_break() -> @location(0) vec4f {
    var i = 0;
    var j = 0;
    var found = 0;
    while i < 10 {
        j = 0;
        while j < 10 {
            if i * 10 + j == 55 {
                found = 1;
                break;
            }
            j += 1;
        }
        if found == 1 {
            break;
        }
        i += 1;
    }
    return vec4f(f32(i) / 10.0, f32(j) / 10.0, f32(found), 1.0);
}
```

## return_example

Demonstrates explicit `return` statements: early returns, return with expressions, and mixed explicit/implicit returns.

### Rust Source

```rust
#[wgsl]
#[allow(dead_code, clippy::needless_return, clippy::mixed_attributes_style)]
pub mod return_example {
    //! Demonstrates explicit return statements including:
    //! - Early returns from functions
    //! - Return with expressions
    //! - Mixed explicit and implicit returns
    use wgsl_rs::std::*;

    // Helper function with early return
    pub fn clamp_positive(x: f32) -> f32 {
        if x < 0.0 {
            return 0.0;
        }
        return x;
    }

    // Function with multiple return paths
    pub fn sign(x: f32) -> f32 {
        if x > 0.0 {
            return 1.0;
        }
        if x < 0.0 {
            return -1.0;
        }
        return 0.0;
    }

    // Mixed explicit and implicit return
    pub fn abs_or_zero(x: f32, threshold: f32) -> f32 {
        if abs(x) < threshold {
            return 0.0;
        }
        abs(x)
    }

    #[fragment]
    pub fn test_explicit_returns() -> Vec4f {
        let pos = clamp_positive(-5.0); // 0.0
        let neg = clamp_positive(3.0); // 3.0
        let s1 = sign(5.0); // 1.0
        let s2 = sign(-2.0); // -1.0
        let a1 = abs_or_zero(0.1, 0.5); // 0.0
        let a2 = abs_or_zero(2.0, 0.5); // 2.0

        vec4f(pos + neg / 10.0, s1 + s2, a1 + a2 / 10.0, 1.0)
    }
}
```

### Generated WGSL

```wgsl
fn clamp_positive(x: f32) -> f32 {
    if x < 0.0 {
        return 0.0;
    }
    return x;
}

fn sign(x: f32) -> f32 {
    if x > 0.0 {
        return 1.0;
    }
    if x < 0.0 {
        return -1.0;
    }
    return 0.0;
}

fn abs_or_zero(x: f32, threshold: f32) -> f32 {
    if abs(x) < threshold {
        return 0.0;
    }
    return abs(x);
}

@fragment fn test_explicit_returns() -> @location(0) vec4f {
    let pos = clamp_positive(-5.0);
    let neg = clamp_positive(3.0);
    let s1 = sign(5.0);
    let s2 = sign(-2.0);
    let a1 = abs_or_zero(0.1, 0.5);
    let a2 = abs_or_zero(2.0, 0.5);
    return vec4f(pos + neg / 10.0, s1 + s2, a1 + a2 / 10.0, 1.0);
}
```

## switch_example

Demonstrates `match`/switch support: simple integer matching, or-patterns (multiple cases), default cases, auto-generated default when missing, and const patterns (with warning suppression).

### Rust Source

```rust
#[wgsl]
#[allow(dead_code, unused_assignments)]
pub mod switch_example {
    //! Demonstrates switch/match statement support including:
    //! - Simple integer matching
    //! - Or-patterns (multiple cases)
    //! - Default cases
    //! - Auto-generated default when missing
    //! - Const patterns (with warning suppression)

    use wgsl_rs::std::*;

    const LOW: i32 = 0;
    const MID: i32 = 1;
    const HIGH: i32 = 2;

    #[fragment]
    pub fn test_simple_switch() -> Vec4f {
        let x: i32 = 2;
        let mut result = 0.0;
        match x {
            0 => {
                result = 0.0;
            }
            1 => {
                result = 0.25;
            }
            2 => {
                result = 0.5;
            }
            _ => {
                result = 1.0;
            }
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    #[allow(clippy::manual_range_patterns)]
    pub fn test_or_patterns() -> Vec4f {
        let x: u32 = 5;
        let mut result = 0.0;
        match x {
            1 | 2 | 3 => {
                result = 0.25;
            }
            4 | 5 | 6 => {
                result = 0.5;
            }
            _ => {
                result = 1.0;
            }
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_missing_default() -> Vec4f {
        let x: i32 = 1;
        let mut result = 0.0;
        // No default arm - WGSL will get auto-generated `default: {}`
        // But Rust requires exhaustive matching, so we use a catch-all underscore
        // that will be optimized out in the test below
        match x {
            0 => {
                result = 0.0;
            }
            1 => {
                result = 1.0;
            }
            _ => {}
        }
        vec4f(result, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_const_patterns() -> Vec4f {
        let level: i32 = 1;
        let mut brightness = 0.0;
        #[wgsl_allow(non_literal_match_statement_patterns)]
        match level {
            LOW => {
                brightness = 0.0;
            }
            MID => {
                brightness = 0.5;
            }
            HIGH => {
                brightness = 1.0;
            }
            _ => {
                brightness = 0.0;
            }
        }
        vec4f(brightness, 0.0, 0.0, 1.0)
    }
}
```

### Generated WGSL

```wgsl
const LOW: i32 = 0;
const MID: i32 = 1;
const HIGH: i32 = 2;

@fragment fn test_simple_switch() -> @location(0) vec4f {
    let x: i32 = 2;
    var result = 0.0;
    switch x {
        case 0: {
            result = 0.0;
        }
        case 1: {
            result = 0.25;
        }
        case 2: {
            result = 0.5;
        }
        default: {
            result = 1.0;
        }
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_or_patterns() -> @location(0) vec4f {
    let x: u32 = 5;
    var result = 0.0;
    switch x {
        case 1, 2, 3: {
            result = 0.25;
        }
        case 4, 5, 6: {
            result = 0.5;
        }
        default: {
            result = 1.0;
        }
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_missing_default() -> @location(0) vec4f {
    let x: i32 = 1;
    var result = 0.0;
    switch x {
        case 0: {
            result = 0.0;
        }
        case 1: {
            result = 1.0;
        }
        default: {
        }
    }
    return vec4f(result, 0.0, 0.0, 1.0);
}

@fragment fn test_const_patterns() -> @location(0) vec4f {
    let level: i32 = 1;
    var brightness = 0.0;
    switch level {
        case LOW: {
            brightness = 0.0;
        }
        case MID: {
            brightness = 0.5;
        }
        case HIGH: {
            brightness = 1.0;
        }
        default: {
            brightness = 0.0;
        }
    }
    return vec4f(brightness, 0.0, 0.0, 1.0);
}
```

## Notes

- Rust `match` becomes WGSL `switch`. The `_` arm maps to `default`.
- Or-patterns (`1 | 2 | 3`) become comma-separated case selectors (`case 1, 2, 3:`).
- `#[wgsl_allow(non_literal_match_statement_patterns)]` is required when match arms reference `const` values rather than literals, since WGSL cases must be literals — the transpiler emits the const names directly (naga substitutes them).
- A missing `default` arm in Rust (with a `_ => {}` catch-all) emits an empty `default: {}` in WGSL.