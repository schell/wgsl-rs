# While & Loop

Demonstrates `while` loops and WGSL `loop` (infinite loop) statements. Includes `continue`, compound conditions, and nested loops.

## while_loop_example

### Rust Source

```rust
#[wgsl]
#[allow(dead_code, unused_assignments)]
pub mod while_loop_example {
    use wgsl_rs::std::*;

    #[fragment]
    pub fn test_simple_while() -> Vec4f {
        let mut i = 0;
        let mut sum = 0.0;

        while i < 10 {
            i += 1;
            // Skip even numbers using continue
            if i % 2 == 0 {
                continue;
            }
            sum += f32(i);
        }

        vec4f(sum / 10.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_while_with_condition() -> Vec4f {
        let mut value = 1.0;
        let mut iterations = 0;

        while value < 100.0 && iterations < 20 {
            value *= 1.5;
            iterations += 1;
        }

        vec4f(value / 100.0, f32(iterations) / 20.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_nested_while() -> Vec4f {
        let mut i = 0;
        let mut j = 0;
        let mut count = 0;

        while i < 5 {
            j = 0;
            while j < 5 {
                count += 1;
                j += 1;
            }
            i += 1;
        }

        vec4f(f32(count) / 25.0, 0.0, 0.0, 1.0)
    }
}
```

### Generated WGSL

```wgsl
@fragment fn test_simple_while() -> @location(0) vec4f {
    var i = 0;
    var sum = 0.0;
    while i < 10 {
        i += 1;
        if i % 2 == 0 {
            continue;
        }
        sum += f32(i);
    }
    return vec4f(sum / 10.0, 0.0, 0.0, 1.0);
}

@fragment fn test_while_with_condition() -> @location(0) vec4f {
    var value = 1.0;
    var iterations = 0;
    while value < 100.0 && iterations < 20 {
        value *= 1.5;
        iterations += 1;
    }
    return vec4f(value / 100.0, f32(iterations) / 20.0, 0.0, 1.0);
}

@fragment fn test_nested_while() -> @location(0) vec4f {
    var i = 0;
    var j = 0;
    var count = 0;
    while i < 5 {
        j = 0;
        while j < 5 {
            count += 1;
            j += 1;
        }
        i += 1;
    }
    return vec4f(f32(count) / 25.0, 0.0, 0.0, 1.0);
}
```

## loop_example

### Rust Source

```rust
#[wgsl]
#[allow(dead_code, unused_assignments)]
pub mod loop_example {
    //! Demonstrates WGSL loop statements (infinite loops).
    //! Note: These are demonstration examples only.

    use wgsl_rs::std::*;

    #[fragment]
    pub fn test_simple_loop() -> Vec4f {
        let mut counter: u32 = 0;
        let mut sum: f32 = 0.0;

        loop {
            sum += f32(counter);
            counter += 1;
            if counter >= 10 {
                break;
            }
        }

        vec4f(sum / 10.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_nested_loop() -> Vec4f {
        let mut i: u32 = 0;
        let mut j: u32 = 0;
        let mut result: f32 = 0.0;

        loop {
            j = 0;
            loop {
                result += 1.0;
                j += 1;
                if j >= 5 {
                    break;
                }
            }
            i += 1;
            if i >= 5 {
                break;
            }
        }

        vec4f(result / 25.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_loop_with_operations() -> Vec4f {
        let mut value: f32 = 1.0;
        let mut iterations: u32 = 0;

        loop {
            value *= 1.5;
            iterations += 1;
            if value >= 100.0 || iterations >= 20 {
                break;
            }
        }

        vec4f(value / 100.0, f32(iterations) / 20.0, 0.0, 1.0)
    }
}
```

### Generated WGSL

```wgsl
@fragment fn test_simple_loop() -> @location(0) vec4f {
    var counter: u32 = 0;
    var sum: f32 = 0.0;
    loop {
        sum += f32(counter);
        counter += 1;
        if counter >= 10 {
            break;
        }
    }
    return vec4f(sum / 10.0, 0.0, 0.0, 1.0);
}

@fragment fn test_nested_loop() -> @location(0) vec4f {
    var i: u32 = 0;
    var j: u32 = 0;
    var result: f32 = 0.0;
    loop {
        j = 0;
        loop {
            result += 1.0;
            j += 1;
            if j >= 5 {
                break;
            }
        }
        i += 1;
        if i >= 5 {
            break;
        }
    }
    return vec4f(result / 25.0, 0.0, 0.0, 1.0);
}

@fragment fn test_loop_with_operations() -> @location(0) vec4f {
    var value: f32 = 1.0;
    var iterations: u32 = 0;
    loop {
        value *= 1.5;
        iterations += 1;
        if value >= 100.0 || iterations >= 20 {
            break;
        }
    }
    return vec4f(value / 100.0, f32(iterations) / 20.0, 0.0, 1.0);
}
```

## Notes

- Rust `while` maps directly to WGSL `while`.
- Rust `loop { ... }` maps to WGSL `loop { ... }` (infinite loop with explicit `break`).
- Compound conditions and nested loops are supported.