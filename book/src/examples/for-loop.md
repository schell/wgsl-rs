# For Loop

Demonstrates for-loop support with range expressions: exclusive (`0..n`), inclusive (`0..=n`), and literal bounds. Variable bounds require `#[wgsl_allow(non_literal_loop_bounds)]`.

## Rust Source

```rust
#[wgsl]
pub mod for_loop_example {
    //! Demonstrates for-loop support with range expressions.
    //! - Exclusive ranges: `for i in 0..10 { ... }`
    //! - Inclusive ranges: `for i in 0..=9 { ... }`
    //! - Variable bounds: `for i in start..end { ... }` (requires
    //!   `#[wgsl_allow]`)
    use wgsl_rs::std::*;

    // Sum values from 0 to n-1 using exclusive range.
    // Uses #[wgsl_allow] on for-loop because `n` is a variable bound.
    pub fn sum_exclusive(n: i32) -> i32 {
        let mut total = 0;
        #[wgsl_allow(non_literal_loop_bounds)]
        for i in 0..n {
            total += i;
        }
        total
    }

    // Sum values from 0 to n (inclusive) using inclusive range.
    // Uses #[wgsl_allow] on for-loop because `n` is a variable bound.
    pub fn sum_inclusive(n: i32) -> i32 {
        let mut total = 0;
        #[wgsl_allow(non_literal_loop_bounds)]
        for i in 0..=n {
            total += i;
        }
        total
    }

    // Compute dot product of two arrays using for-loop.
    // No #[wgsl_allow] needed because bounds are literals.
    pub fn dot_product(a: [f32; 4], b: [f32; 4]) -> f32 {
        let mut result = 0.0;
        for i in 0..4 {
            result += a[i as usize] * b[i as usize];
        }
        result
    }

    // Nested for-loops: initialize a 2D-like structure.
    // No #[wgsl_allow] needed because bounds are literals.
    pub fn nested_loops() -> i32 {
        let mut sum = 0;
        for i in 0..3 {
            for j in 0..4 {
                sum += i * 4 + j;
            }
        }
        sum
    }

    #[fragment]
    pub fn for_loop_fragment() -> Vec4f {
        // Test sum_exclusive: sum of 0..10 = 0+1+2+...+9 = 45
        let exclusive_sum = sum_exclusive(10);

        // Test sum_inclusive: sum of 0..=9 = 0+1+2+...+9 = 45
        let inclusive_sum = sum_inclusive(9);

        // Test dot_product
        let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let dot = dot_product(a, b); // 1+2+3+4 = 10

        // Test nested loops: sum of (i*4+j) for i in 0..3, j in 0..4
        // = (0,1,2,3) + (4,5,6,7) + (8,9,10,11) = 6 + 22 + 38 = 66
        let nested = nested_loops();

        vec4f(
            f32(exclusive_sum) / 100.0,
            f32(inclusive_sum) / 100.0,
            dot / 10.0,
            f32(nested) / 100.0,
        )
    }
}
```

## Generated WGSL

```wgsl
fn sum_exclusive(n: i32) -> i32 {
    var total = 0;
    for (var i = 0; i < n; i++) {
        total += i;
    }
    return total;
}

fn sum_inclusive(n: i32) -> i32 {
    var total = 0;
    for (var i = 0; i <= n; i++) {
        total += i;
    }
    return total;
}

fn dot_product(a: array<f32, 4>, b: array<f32, 4>) -> f32 {
    var result = 0.0;
    for (var i = 0; i < 4; i++) {
        result += a[u32(i)] * b[u32(i)];
    }
    return result;
}

fn nested_loops() -> i32 {
    var sum = 0;
    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 4; j++) {
            sum += i * 4 + j;
        }
    }
    return sum;
}

@fragment fn for_loop_fragment() -> @location(0) vec4f {
    let exclusive_sum = sum_exclusive(10);
    let inclusive_sum = sum_inclusive(9);
    let a: array<f32, 4> = array(1.0, 2.0, 3.0, 4.0);
    let b: array<f32, 4> = array(1.0, 1.0, 1.0, 1.0);
    let dot = dot_product(a, b);
    let nested = nested_loops();
    return vec4f(f32(exclusive_sum) / 100.0, f32(inclusive_sum) / 100.0, dot / 10.0, f32(nested) / 100.0);
}
```

## Notes

- Rust `for i in 0..n` becomes a WGSL C-style `for (var i = 0; i < n; i++)`.
- Inclusive ranges (`0..=n`) translate to `i <= n`.
- `#[wgsl_allow(non_literal_loop_bounds)]` is required when loop bounds are not literal constants, because WGSL's `for` requires constant bounds — the transpiler emits them anyway and naga lowers the loop to a `loop` with a `break` condition.