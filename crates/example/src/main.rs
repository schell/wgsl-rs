use wgsl_rs::wgsl;

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
        vec4f(position.x(), position.y(), 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(FRAME) / 128.0), 0.0, 1.0)
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
    //! A simple compute shader that demonstrates storage buffers.
    use wgsl_rs::std::*;

    // Read-only input buffer
    storage!(group(0), binding(0), INPUT: [f32; 256]);

    // Read-write output buffer
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Compute the index from global invocation ID
        // Note: Storage buffer access requires additional implementation
        // This demonstrates the compute shader structure with storage buffers
        let _idx = global_id.x() as usize;
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

fn validate_and_print_source(module: &wgsl_rs::Module) {
    let source = module.wgsl_source().join("\n");
    println!("raw source:\n\n{source}\n\n");

    // Parse the source into a Module.
    let module: naga::Module = naga::front::wgsl::parse_str(&source).unwrap();

    // Validate the module.
    // Validation can be made less restrictive by changing the ValidationFlags.
    let result = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module);

    let info = match result {
        Err(e) => {
            panic!("{}", e.emit_to_string(&source));
        }
        Ok(i) => i,
    };

    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    println!("naga source:\n\n{wgsl}");
}

/// Test the linkage API when the feature is enabled.
fn print_linkage() {
    println!("\n=== Testing Linkage API ===\n");

    // Test hello_triangle linkage
    println!(
        "hello_triangle::linkage::SHADER_SOURCE length: {}",
        hello_triangle::linkage::SHADER_SOURCE.len()
    );
    println!(
        "hello_triangle bind_group_0 layout entries: {}",
        hello_triangle::linkage::bind_group_0::LAYOUT_ENTRIES.len()
    );
    println!(
        "hello_triangle vtx_main entry point: {}",
        hello_triangle::linkage::vtx_main::ENTRY_POINT
    );
    println!(
        "hello_triangle frag_main entry point: {}",
        hello_triangle::linkage::frag_main::ENTRY_POINT
    );

    // Test compute_shader linkage
    println!(
        "\ncompute_shader::linkage::SHADER_SOURCE length: {}",
        compute_shader::linkage::SHADER_SOURCE.len()
    );
    println!(
        "compute_shader bind_group_0 layout entries: {}",
        compute_shader::linkage::bind_group_0::LAYOUT_ENTRIES.len()
    );
    println!(
        "compute_shader main entry point: {}",
        compute_shader::linkage::main::ENTRY_POINT
    );
    println!(
        "compute_shader main workgroup size: {:?}",
        compute_shader::linkage::main::WORKGROUP_SIZE
    );

    // Test structs linkage
    println!(
        "\nstructs::linkage::SHADER_SOURCE length: {}",
        structs::linkage::SHADER_SOURCE.len()
    );
    println!(
        "structs frag_shader entry point: {}",
        structs::linkage::frag_shader::ENTRY_POINT
    );

    println!("\n=== Linkage API tests passed ===");
}

/// Build the linkage into a working `winit` + `wgpu` app, as a
/// dogfooding test.
fn build_linkage() {
    use std::sync::Arc;

    use futures::executor::block_on;
    use winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    };

    let event_loop = EventLoop::new().unwrap();

    struct WgpuStuff {
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        _adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl WgpuStuff {
        fn new(window: Arc<Window>) -> Self {
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window).unwrap();
            let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }))
            .expect("Failed to find an appropriate adapter");

            let (device, queue) =
                block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .expect("Failed to create device");
            let config = surface
                .get_default_config(&adapter, 800, 600)
                .expect("no default surface config");
            surface.configure(&device, &config);

            Self {
                _instance: instance,
                surface,
                _adapter: adapter,
                device,
                queue,
            }
        }
    }

    struct HelloTriangle {
        frame: u32,
        frame_uniform_buffer: wgpu::Buffer,
        bindgroup: wgpu::BindGroup,
        render_pipeline: wgpu::RenderPipeline,
    }

    impl HelloTriangle {
        fn new(wgpu_stuff: &WgpuStuff) -> Self {
            let device = &wgpu_stuff.device;
            let queue = &wgpu_stuff.queue;
            let frame = 0u32;
            let frame_uniform_buffer = hello_triangle::create_frame_buffer(device);
            queue.write_buffer(&frame_uniform_buffer, 0, &frame.to_ne_bytes());

            let bindgroup_layout = hello_triangle::linkage::bind_group_0::layout(device);
            let bindgroup = hello_triangle::linkage::bind_group_0::create(
                device,
                &bindgroup_layout,
                frame_uniform_buffer.as_entire_binding(),
            );

            let module = hello_triangle::linkage::shader_module(device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hello_triangle"),
                bind_group_layouts: &[&bindgroup_layout],
                immediate_size: 0,
            });
            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hello_triangle"),
                layout: Some(&pipeline_layout),
                vertex: hello_triangle::linkage::vtx_main::vertex_state(&module),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(hello_triangle::linkage::frag_main::fragment_state(
                    &module,
                    &[Some(wgpu::ColorTargetState {
                        format: wgpu_stuff
                            .surface
                            .get_configuration()
                            .expect("missing surface configuration")
                            .format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::all(),
                    })],
                )),
                multiview_mask: None,
                cache: None,
            });

            Self {
                frame,
                frame_uniform_buffer,
                bindgroup,
                render_pipeline,
            }
        }
    }

    struct AppInner {
        window: Arc<Window>,
        wgpu_stuff: WgpuStuff,
        hello_triangle: HelloTriangle,
    }

    impl AppInner {
        pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Self {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            let wgpu_stuff = WgpuStuff::new(window.clone());
            let hello_triangle = HelloTriangle::new(&wgpu_stuff);
            Self {
                window,
                wgpu_stuff,
                hello_triangle,
            }
        }
    }

    #[derive(Default)]
    struct App {
        inner: Option<AppInner>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            self.inner = Some(AppInner::new(event_loop));
        }

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    println!("Closing");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    if let Some(AppInner {
                        window,
                        wgpu_stuff,
                        hello_triangle,
                    }) = self.inner.as_mut()
                    {
                        let device = &wgpu_stuff.device;
                        let queue = &wgpu_stuff.queue;

                        hello_triangle.frame += 1;
                        queue.write_buffer(
                            &hello_triangle.frame_uniform_buffer,
                            0,
                            &hello_triangle.frame.to_ne_bytes(),
                        );

                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("pass"),
                            });
                        let texture = wgpu_stuff
                            .surface
                            .get_current_texture()
                            .expect("couldn't get current texture");
                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &texture
                                            .texture
                                            .create_view(&wgpu::TextureViewDescriptor::default()),
                                        depth_slice: None,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.0,
                                                g: 0.0,
                                                b: 0.0,
                                                a: 0.0,
                                            }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    ..Default::default()
                                });
                            render_pass.set_pipeline(&hello_triangle.render_pipeline);
                            render_pass.set_bind_group(0, &hello_triangle.bindgroup, &[]);
                            render_pass.draw(0..3, 0..1);
                        }

                        let index = queue.submit(Some(encoder.finish()));
                        texture.present();

                        // TODO: maybe do this elsewhere?
                        device
                            .poll(wgpu::PollType::Wait {
                                submission_index: Some(index),
                                timeout: None,
                            })
                            .unwrap();

                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

pub fn main() {
    validate_and_print_source(&hello_triangle::WGSL_MODULE);
    validate_and_print_source(&structs::WGSL_MODULE);
    validate_and_print_source(&compute_shader::WGSL_MODULE);
    validate_and_print_source(&matrix_example::WGSL_MODULE);
    validate_and_print_source(&impl_example::WGSL_MODULE);
    validate_and_print_source(&enum_example::WGSL_MODULE);
    validate_and_print_source(&binary_ops_example::WGSL_MODULE);
    validate_and_print_source(&assignment_example::WGSL_MODULE);
    validate_and_print_source(&while_loop_example::WGSL_MODULE);
    validate_and_print_source(&loop_example::WGSL_MODULE);
    validate_and_print_source(&if_example::WGSL_MODULE);
    validate_and_print_source(&break_example::WGSL_MODULE);
    validate_and_print_source(&for_loop_example::WGSL_MODULE);
    validate_and_print_source(&return_example::WGSL_MODULE);
    validate_and_print_source(&switch_example::WGSL_MODULE);

    print_linkage();
    build_linkage();
}
