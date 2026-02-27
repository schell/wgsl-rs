//! Example WGSL modules.

#![allow(dead_code)]
use wgsl_rs::wgsl;

pub const EXAMPLE_MODULES: &[&wgsl_rs::Module] = &[
    &hello_triangle::WGSL_MODULE,
    &structs::WGSL_MODULE,
    &compute_shader::WGSL_MODULE,
    &matrix_example::WGSL_MODULE,
    &impl_example::WGSL_MODULE,
    &enum_example::WGSL_MODULE,
    &binary_ops_example::WGSL_MODULE,
    &for_loop_example::WGSL_MODULE,
    &assignment_example::WGSL_MODULE,
    &zero_value_array_example::WGSL_MODULE,
    &while_loop_example::WGSL_MODULE,
    &loop_example::WGSL_MODULE,
    &if_example::WGSL_MODULE,
    &break_example::WGSL_MODULE,
    &return_example::WGSL_MODULE,
    &switch_example::WGSL_MODULE,
    &runtime_array_example::WGSL_MODULE,
    &ptr_example::WGSL_MODULE,
    &atomic_example::WGSL_MODULE,
    &texture_example::WGSL_MODULE,
    &bitcast_example::WGSL_MODULE,
    &packing_example::WGSL_MODULE,
    &advanced_numeric_example::WGSL_MODULE,
    &matrix_builtin_example::WGSL_MODULE,
    &synchronization_example::WGSL_MODULE,
    &macro_rules_definitions::WGSL_MODULE,
    &slab_read_write::WGSL_MODULE,
];

pub fn get_module_by_name(name: &str) -> Option<&'static wgsl_rs::Module> {
    EXAMPLE_MODULES
        .iter()
        .find(|&module| module.name == name)
        .map(|v| v as _)
}

#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing
    //! color. Original source is [here](https://google.github.io/tour-of-wgsl/).

    // Only glob-imports are supported, but hey, imports work!
    use wgsl_rs::std::*;

    // Define a uniform in both Rust and WGSL using the uniform! macro.
    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];

        let position = POS[vertex_index as usize];
        vec4f(position.x, position.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(get!(FRAME)) / 128.0), 0.0, 1.0)
    }
}

#[wgsl]
pub mod structs {
    use wgsl_rs::std::*;

    // Mixed builtins and user-defined inputs.
    #[input]
    pub struct MyInputs {
        #[location(0)]
        pub x: Vec4<f32>,

        #[builtin(front_facing)]
        pub y: bool,

        #[location(1)]
        #[interpolate(flat)]
        pub z: u32,

        #[location(2)]
        pub other: f32,
    }

    #[output]
    pub struct MyOutputs {
        #[location(0)]
        pub x: f32,

        #[location(1)]
        pub y: Vec4<f32>,
    }

    #[fragment]
    pub fn frag_shader(in1: MyInputs) -> MyOutputs {
        MyOutputs { x: 0.0, y: in1.x }
    }
}

#[wgsl]
pub mod compute_shader {
    //! A simple compute shader that demonstrates defining and accessing storage
    //! buffers.
    //!
    //! Storage buffers are special on the Rust side and require locking,
    //! so they are accessed with the `get!` and `get_mut!` macros, which
    //! do the heavy lifting for you. These macros are a noop in WGSL and are
    //! stripped during parsing.
    use wgsl_rs::std::*;

    // Read-only input buffer
    storage!(group(0), binding(0), INPUT: [f32; 256]);

    pub struct Output {
        pub inner: f32,
    }

    // Read-write output buffer
    storage!(group(0), binding(1), read_write, OUTPUT: Output);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Compute the index from global invocation ID
        let idx = global_id.x() as usize;
        // Use the `get!` macro to access the storage
        let input = get!(INPUT)[idx];
        // Use the `get_mut!` macro to access the storage mutably
        get_mut!(OUTPUT).inner = input;
    }
}

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

#[wgsl]
pub mod impl_example {
    //! Demonstrates struct impl blocks with explicit receiver syntax.
    //!
    //! Methods and constants are defined in impl blocks.
    //! - Methods are called using `Type::method(receiver, args)` syntax
    //! - Constants are accessed using `Type::CONSTANT` syntax
    //!
    //! Both translate to `Type_member` in WGSL output.
    use wgsl_rs::std::*;

    pub struct Light {
        pub position: Vec3f,
        pub intensity: f32,
    }

    impl Light {
        // Associated constants
        pub const DEFAULT_INTENSITY: f32 = 1.0;
        pub const DEFAULT_RANGE: f32 = 10.0;

        // Create a new light at the given position with the given intensity.
        pub fn new(position: Vec3f, intensity: f32) -> Light {
            Light {
                position,
                intensity,
            }
        }

        // Calculate light attenuation based on distance.
        // Uses inverse-square falloff.
        pub fn attenuate(light: Light, distance: f32) -> f32 {
            light.intensity / (distance * distance)
        }

        // Get the light's position.
        pub fn get_position(light: Light) -> Vec3f {
            light.position
        }
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        // Create a light using the explicit receiver syntax
        let light = Light::new(vec3f(0.0, 5.0, 0.0), Light::DEFAULT_INTENSITY);

        // Call a method using explicit path syntax: Type::method(receiver, args)
        let attenuation = Light::attenuate(light, Light::DEFAULT_RANGE / 5.0);

        // Return a color based on attenuation
        vec4f(attenuation, attenuation, attenuation, 1.0)
    }
}

#[wgsl]
pub mod enum_example {
    //! Limited support for enums.
    use wgsl_rs::std::*;

    /// Analytical lighting types.
    #[repr(u32)]
    pub enum LightType {
        Directional = 1337,
        Spot = 420,
        Point = 666,
    }

    #[repr(u32)]
    pub enum Holidays {
        // Syntax error!
        // Halloween = -23,
        AprilFoolsDay,
        WaitangiDay,
    }

    storage!(group(0), binding(0), read_write, INPUT: [Holidays; 256]);

    #[compute]
    #[workgroup_size(16)]
    pub fn compute_holidays(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let index = global_id.x();

        let holiday = &mut get_mut!(INPUT)[index as usize];

        #[wgsl_allow(non_literal_match_statement_patterns)]
        match *holiday {
            Holidays::AprilFoolsDay => {
                *holiday = Holidays::WaitangiDay;
            }
            Holidays::WaitangiDay => {
                *holiday = Holidays::AprilFoolsDay;
            }
        }
    }
}

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

#[wgsl]
#[expect(unused_assignments, reason = "demonstration")]
pub mod assignment_example {
    //! Demonstrates assignment statements including:
    //! - Simple assignment: x = expr;
    //! - Compound assignment: x += expr;, x -= expr;, etc.
    //! - Field assignment: obj.field = expr;
    //! - Array element assignment: arr[i] = expr;
    use wgsl_rs::std::*;

    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    #[fragment]
    #[expect(clippy::assign_op_pattern, reason = "demonstration")]
    pub fn test_simple_assignment() -> Vec4f {
        let mut value = 0.0;
        value = 1.0;
        value = value + 1.0; // intentionally not using += to test simple assignment
        value = value + 1.0; // value is now 3.0 and is read below
        vec4f(value, value, value, 1.0)
    }

    #[fragment]
    pub fn test_compound_assignment() -> Vec4f {
        let mut x = 10.0;
        x += 5.0; // 15.0
        x -= 3.0; // 12.0
        x *= 2.0; // 24.0
        x /= 4.0; // 6.0
        vec4f(x, x, x, 1.0)
    }

    #[fragment]
    pub fn test_bitwise_compound_assignment() -> Vec4f {
        let mut bits: u32 = 0xFF00;
        bits &= 0x0F0F;
        bits |= 0x00F0;
        bits ^= 0x000F;
        bits <<= 2u32;
        bits >>= 1u32;
        vec4f(f32(bits) / 65535.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_field_assignment() -> Vec4f {
        let mut p = Point { x: 0.0, y: 0.0 };
        p.x = 1.0;
        p.y = 2.0;
        vec4f(p.x, p.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn test_array_assignment() -> Vec4f {
        let mut arr: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        arr[0] = 1.0;
        arr[1] = 2.0;
        arr[2] = 3.0;
        arr[3] = 4.0;
        vec4f(arr[0], arr[1], arr[2], arr[3])
    }
}

/// Demonstrates zero-value array initialization using `[0u32; N]` syntax.
///
/// In WGSL, this transpiles to the zero-value constructor `array<T, N>()`.
#[wgsl]
pub mod zero_value_array_example {
    use wgsl_rs::std::*;

    #[fragment]
    pub fn test_zero_u32_array() -> Vec4f {
        let arr: [u32; 4] = [0u32; 4];
        vec4f(f32(arr[0]), f32(arr[1]), f32(arr[2]), f32(arr[3]))
    }

    #[fragment]
    pub fn test_zero_f32_array() -> Vec4f {
        let arr: [f32; 4] = [0.0f32; 4];
        vec4f(arr[0], arr[1], arr[2], arr[3])
    }

    #[fragment]
    pub fn test_zero_i32_array() -> Vec4f {
        let arr: [i32; 4] = [0i32; 4];
        vec4f(f32(arr[0]), f32(arr[1]), f32(arr[2]), f32(arr[3]))
    }

    #[fragment]
    pub fn test_zero_array_with_assignment() -> Vec4f {
        let mut arr: [f32; 4] = [0.0f32; 4];
        arr[0] = 1.0;
        arr[1] = 2.0;
        arr[2] = 3.0;
        arr[3] = 4.0;
        vec4f(arr[0], arr[1], arr[2], arr[3])
    }
}

/// Demonstrates while loop support.
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

/// Demonstrates break statement support.
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

/// Demonstrates explicit return statement support.
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

#[wgsl]
#[allow(dead_code)]
pub mod runtime_array_example {
    //! Demonstrates runtime-sized arrays (RuntimeArray<T>).
    //!
    //! Runtime-sized arrays transpile to `array<T>` in WGSL (no size
    //! parameter). They can only be used in storage buffers, typically as
    //! the last field of a struct.
    use wgsl_rs::std::*;

    pub struct Particle {
        pub position: Vec3f,
        pub velocity: Vec3f,
    }

    pub struct ParticleSystem {
        pub count: u32,
        pub particles: RuntimeArray<Particle>,
    }

    storage!(group(0), binding(0), read_write, PARTICLES: ParticleSystem);

    #[compute]
    #[workgroup_size(16, 16, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let num_particles = array_length(&get!(PARTICLES).particles);
        let index = global_id.y() * 16 + global_id.x();
        if num_particles < index {
            let velocity = get!(PARTICLES).particles[index].velocity;
            let position = &mut get_mut!(PARTICLES).particles[index].position;
            *position = *position + velocity;
        }
    }
}

#[wgsl]
#[allow(dead_code, clippy::manual_swap, clippy::assign_op_pattern)]
pub mod ptr_example {
    //! Demonstrates pointer types in function parameters.
    use wgsl_rs::std::*;

    // Increment a value through a pointer.
    pub fn increment(p: ptr!(function, i32)) {
        *p += 1;
    }

    // Swap two values through pointers.
    // Note: We use manual swap because std::mem::swap is not available in WGSL.
    pub fn swap(a: ptr!(function, f32), b: ptr!(function, f32)) {
        let tmp = *a;
        *a = *b;
        *b = tmp;
    }

    // Double a value in-place through a pointer.
    // Note: We use *p = *p * 2.0 instead of *p *= 2.0 to demonstrate dereference.
    pub fn double_value(p: ptr!(function, f32)) {
        *p = *p * 2.0;
    }

    #[fragment]
    pub fn test_ptr() -> Vec4f {
        let mut x: i32 = 5;
        increment(&mut x);
        // x is now 6

        let mut a: f32 = 1.0;
        let mut b: f32 = 2.0;
        swap(&mut a, &mut b);
        // a is now 2.0, b is now 1.0

        let mut c: f32 = 3.0;
        double_value(&mut c);
        // c is now 6.0

        vec4f(f32(x), a, b, c / 10.0)
    }
}

#[wgsl]
pub mod atomic_example {
    //! Demonstrates atomic types and workgroup variables.
    //!
    //! Atomic types provide thread-safe operations for concurrent access in
    //! compute shaders. They can only hold `i32` or `u32` values.
    //!
    //! Workgroup variables are shared between all invocations in a workgroup
    //! and can only be used in compute shaders.
    use wgsl_rs::std::*;

    // Workgroup variable with atomic counter - shared between all invocations
    workgroup!(COUNTER: Atomic<u32>);

    // Workgroup variable with atomic flags
    workgroup!(FLAGS: Atomic<i32>);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(local_invocation_index)] local_idx: u32) {
        // Each invocation can access the shared atomic counter
        // Note: atomicLoad/atomicStore builtins will be added in a future update
        // For now, this demonstrates the type parsing and code generation
        let _idx = local_idx;
    }
}

#[wgsl]
pub mod texture_example {
    //! Demonstrates using textures and texture builtin functions.
    //!
    //! WGSL provides several categories of texture operations:
    //! - **Query functions**: `textureDimensions`, `textureNumLayers`, etc.
    //! - **Load functions**: `textureLoad` - direct texel access without
    //!   filtering
    //! - **Sample functions**: `textureSample` - filtered sampling with a
    //!   sampler
    //! - **Depth comparison**: `textureSampleCompare` - for shadow mapping
    use wgsl_rs::std::*;

    // A 2D texture for color/albedo
    texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
    // A sampler for filtering the texture
    sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);

    // Fragment input with texture coordinates
    #[input]
    pub struct FragmentInput {
        #[location(0)]
        pub uv: Vec2f,
    }

    // Output struct
    #[output]
    pub struct FragmentOutput {
        #[location(0)]
        pub color: Vec4f,
    }

    // Main fragment shader demonstrating texture operations.
    #[fragment]
    pub fn frag_main(input: FragmentInput) -> FragmentOutput {
        // Sample the diffuse texture
        let albedo = texture_sample(DIFFUSE_TEX, TEX_SAMPLER, input.uv);

        FragmentOutput { color: albedo }
    }
}

#[wgsl]
#[expect(dead_code, reason = "demonstration")]
pub mod bitcast_example {
    //! Demonstrates using `bitcast` to reinterpret the bits of a value as
    //! another type.
    //!
    //! WGSL `bitcast<T>(e)` reinterprets the bit pattern of `e` as type `T`
    //! without changing any bits. This is useful for packing/unpacking data,
    //! interpreting raw buffer contents, and working with IEEE 754
    //! representations.
    //!
    //! In `wgsl-rs`, each target type has a dedicated function:
    //!   - `bitcast_f32(e)` → `bitcast<f32>(e)`
    //!   - `bitcast_u32(e)` → `bitcast<u32>(e)`
    //!   - `bitcast_i32(e)` → `bitcast<i32>(e)`
    //!   - `bitcast_vec2f(e)` → `bitcast<vec2<f32>>(e)`, etc.
    use wgsl_rs::std::*;

    // Input: raw u32 data representing packed floats
    storage!(group(0), binding(0), INPUT: [u32; 256]);

    // Output: reinterpreted as floats
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    // Reinterpret a u32 bit pattern as an f32 value.
    pub fn reinterpret_as_float(bits: u32) -> f32 {
        bitcast_f32(bits)
    }

    // Reinterpret an f32 value as its u32 bit pattern.
    pub fn float_to_bits(value: f32) -> u32 {
        bitcast_u32(value)
    }

    // Reinterpret a u32 vector as an i32 vector.
    pub fn reinterpret_vec_as_signed(v: Vec4u) -> Vec4i {
        bitcast_vec4i(v)
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x() as usize;
        // Read raw u32 bits from input and reinterpret as f32
        let raw_bits = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = bitcast_f32(raw_bits);
    }
}

#[wgsl]
pub mod packing_example {
    //! Demonstrates WGSL data packing and unpacking builtin functions.
    //!
    //! These functions convert between vector types and packed `u32`
    //! representations, useful for vertex attribute compression and storage
    //! optimization.
    use wgsl_rs::std::*;

    pub fn demo_pack4x8snorm(v: Vec4f) -> u32 {
        pack4x8snorm(v)
    }

    pub fn demo_unpack4x8snorm(e: u32) -> Vec4f {
        unpack4x8snorm(e)
    }

    pub fn demo_pack4x8unorm(v: Vec4f) -> u32 {
        pack4x8unorm(v)
    }

    pub fn demo_unpack4x8unorm(e: u32) -> Vec4f {
        unpack4x8unorm(e)
    }

    pub fn demo_pack2x16snorm(v: Vec2f) -> u32 {
        pack2x16snorm(v)
    }

    pub fn demo_unpack2x16snorm(e: u32) -> Vec2f {
        unpack2x16snorm(e)
    }

    pub fn demo_pack2x16unorm(v: Vec2f) -> u32 {
        pack2x16unorm(v)
    }

    pub fn demo_unpack2x16unorm(e: u32) -> Vec2f {
        unpack2x16unorm(e)
    }

    pub fn demo_pack2x16float(v: Vec2f) -> u32 {
        pack2x16float(v)
    }

    pub fn demo_unpack2x16float(e: u32) -> Vec2f {
        unpack2x16float(e)
    }
}

#[wgsl]
pub mod advanced_numeric_example {
    //! Demonstrates the advanced numeric builtin functions: `modf`, `frexp`,
    //! and `ldexp`.
    use wgsl_rs::std::*;

    pub fn demo_modf_fract(e: f32) -> f32 {
        let result = modf(e);
        result.fract
    }

    pub fn demo_modf_whole(e: f32) -> f32 {
        modf(e).whole
    }

    pub fn demo_frexp_fract(e: f32) -> f32 {
        frexp(e).fract
    }

    pub fn demo_frexp_exp(e: f32) -> i32 {
        frexp(e).exp
    }

    pub fn demo_ldexp(significand: f32, exponent: i32) -> f32 {
        ldexp(significand, exponent)
    }
}

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

#[wgsl]
pub mod synchronization_example {
    //! Demonstrates synchronization builtin functions for compute shaders.
    //!
    //! These functions coordinate memory visibility and execution ordering
    //! across invocations within a workgroup. They must only be called from
    //! compute shader entry points in uniform control flow.
    use wgsl_rs::std::*;

    workgroup!(SCRATCH: [u32; 64]);

    storage!(group(0), binding(0), INPUT: [u32; 64]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(local_invocation_index)] local_idx: u32) {
        // Copy input data into workgroup-shared memory.
        get_mut!(SCRATCH)[local_idx as usize] = get!(INPUT)[local_idx as usize];

        // Ensure all workgroup memory writes are visible to every invocation.
        workgroup_barrier();

        // Read from a neighbor's slot (with wrap-around) to demonstrate
        // that the barrier made all writes visible.
        let neighbor_idx: u32 = (local_idx + 1u32) % 64u32;
        let neighbor_val: u32 = get!(SCRATCH)[neighbor_idx as usize];

        // Ensure all storage writes from the workgroup are complete
        // before writing results.
        storage_barrier();

        get_mut!(OUTPUT)[local_idx as usize] = neighbor_val;
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn uniform_load_example(#[builtin(local_invocation_index)] local_idx: u32) {
        // Each invocation writes its index into shared memory.
        get_mut!(SCRATCH)[local_idx as usize] = local_idx;

        // Ensure all writes are visible before uniform load.
        workgroup_barrier();

        // Uniformly load the first element across the entire workgroup.
        // All invocations receive the same value.
        let first: [u32; 64] = workgroup_uniform_load(&SCRATCH);
        get_mut!(OUTPUT)[local_idx as usize] = first[0];
    }
}

#[wgsl]
pub mod macro_rules_definitions {
    //! It is possible to define `macro_rules!` within a WGSL module.
    //!
    //! Macros defined this way **will not generate WGSL code**, but will pass
    //! through to Rust code.
    //!
    //! Said another way - `macro_rules!` definitions will be stripped from WGSL
    //! code generation but will remain in your Rust source.

    #[expect(unused_macros)]
    macro_rules! my_macro {
        ($id:ident) => {
            id
        };
    }

    // It's also possible to use derive macros.
    //
    // Derive macros pass through without generating any extra WGSL.
    #[derive(Debug, Clone, Copy)]
    pub struct Data {
        pub inner: f32,
    }
}

#[wgsl]
pub mod slab_read_write {
    //! `wgsl-rs` includes macros for reading to and from u32 "slabs".
    //!
    //! The slab can be any indexable item such as an array, RuntimeArray,
    //! storage pointer, etc.

    use wgsl_rs::std::*;

    pub struct Data {
        pub one: f32,
        pub two: u32,
        pub three_four: Vec2f,
    }

    impl Data {
        /// `Data`'s slab size.
        ///
        /// This is the number of u32 slots it occupies in a u32 slab.
        pub const SLAB_SIZE: usize = 4;

        /// Returns an array container to hold ephemeral data read from a slab.
        pub fn array_container() -> [u32; Self::SLAB_SIZE] {
            [0, 0, 0, 0]
        }

        /// Convert an array into `Data`.
        pub fn from_array(arr: [u32; Self::SLAB_SIZE]) -> Self {
            Self {
                one: bitcast_f32(arr[0]),
                two: arr[1],
                three_four: vec2f(bitcast_f32(arr[2]), bitcast_f32(arr[3])),
            }
        }

        /// Convert `Data` into an array.
        pub fn to_array(data: Self) -> [u32; Self::SLAB_SIZE] {
            [
                bitcast_u32(data.one),
                data.two,
                bitcast_u32(data.three_four.x()),
                bitcast_u32(data.three_four.y()),
            ]
        }
    }

    storage!(group(0), binding(0), read_write, SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(8)]
    pub fn slab_example(#[builtin(local_invocation_index)] local_idx: u32) {
        let index = local_idx;

        // Create our `Data` struct from extracted data from the slab
        let mut data: Data;
        {
            // Extract the u32 data from the slab
            let mut array_data = Data::array_container();
            slab_read_array!(get!(SLAB), index, array_data, Data::SLAB_SIZE);
            data = Data::from_array(array_data);
        }

        // Modify it
        data.three_four.x = 123.0;

        // Write the modified `Data` struct back to the slab
        let out_array = Data::to_array(data);
        slab_write_array!(get_mut!(SLAB), index, out_array, Data::SLAB_SIZE);
    }
}
