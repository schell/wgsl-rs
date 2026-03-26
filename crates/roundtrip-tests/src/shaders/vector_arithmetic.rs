//! Roundtrip tests for vector arithmetic operators.
//!
//! Tests: `+`, `-`, `*`, `/`, `%` (binary operators on vectors)
//! and unary `-` (negation).
//!
//! Covers Vec4 with f32, i32, u32 scalar types, plus spot checks for Vec2/Vec3.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// vec_binary_f32: Binary operators (+, -, *, /, %) on Vec4f
///
/// Tests vector-vector operations: v1 ⊗ v2 where ⊗ ∈ {+, -, *, /, %}
#[wgsl]
pub mod vec_binary_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]); // Flattened: 64 vec4 pairs = 512 floats
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 320]); // 5 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8; // 8 floats per pair of vec4s
        let input = get!(INPUT);

        let a = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4f(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        let out_base = idx * 5;
        get_mut!(OUTPUT)[out_base + 0] = a + b;
        get_mut!(OUTPUT)[out_base + 1] = a - b;
        get_mut!(OUTPUT)[out_base + 2] = a * b; // component-wise multiplication
        get_mut!(OUTPUT)[out_base + 3] = a / b;
        get_mut!(OUTPUT)[out_base + 4] = a % b;
    }
}

/// vec_binary_i32: Binary operators (+, -, *, /) on Vec4i
#[wgsl]
pub mod vec_binary_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [i32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4i; 256]); // 4 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4i(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4i(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base + 0] = a + b;
        get_mut!(OUTPUT)[out_base + 1] = a - b;
        get_mut!(OUTPUT)[out_base + 2] = a * b;
        get_mut!(OUTPUT)[out_base + 3] = a / b;
    }
}

/// vec_binary_u32: Binary operators (+, -, *, /) on Vec4u
#[wgsl]
pub mod vec_binary_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 256]); // 4 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4u(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        let out_base = idx * 4;
        get_mut!(OUTPUT)[out_base + 0] = a + b;
        get_mut!(OUTPUT)[out_base + 1] = a - b;
        get_mut!(OUTPUT)[out_base + 2] = a * b;
        get_mut!(OUTPUT)[out_base + 3] = a / b;
    }
}

/// vec_scalar_f32: Vector-scalar operations on Vec4f
///
/// Tests: v ⊗ s and s ⊗ v where ⊗ ∈ {+, -, *, /}
#[wgsl]
pub mod vec_scalar_f32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]); // 64 vec4s + 64 scalars = 320
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4f; 512]); // 8 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 5; // 4 for vec4 + 1 for scalar
        let input = get!(INPUT);

        let v = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let s = input[base + 4];

        let out_base = idx * 8;
        // v ⊗ s
        get_mut!(OUTPUT)[out_base + 0] = v + s;
        get_mut!(OUTPUT)[out_base + 1] = v - s;
        get_mut!(OUTPUT)[out_base + 2] = v * s;
        get_mut!(OUTPUT)[out_base + 3] = v / s;
        // s ⊗ v
        get_mut!(OUTPUT)[out_base + 4] = s + v;
        get_mut!(OUTPUT)[out_base + 5] = s - v;
        get_mut!(OUTPUT)[out_base + 6] = s * v;
        get_mut!(OUTPUT)[out_base + 7] = s / v;
    }
}

/// vec_scalar_i32: Vector-scalar operations on Vec4i
#[wgsl]
pub mod vec_scalar_i32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [i32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4i; 512]); // 8 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 5;
        let input = get!(INPUT);

        let v = vec4i(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let s = input[base + 4];

        let out_base = idx * 8;
        get_mut!(OUTPUT)[out_base + 0] = v + s;
        get_mut!(OUTPUT)[out_base + 1] = v - s;
        get_mut!(OUTPUT)[out_base + 2] = v * s;
        get_mut!(OUTPUT)[out_base + 3] = v / s;
        get_mut!(OUTPUT)[out_base + 4] = s + v;
        get_mut!(OUTPUT)[out_base + 5] = s - v;
        get_mut!(OUTPUT)[out_base + 6] = s * v;
        get_mut!(OUTPUT)[out_base + 7] = s / v;
    }
}

/// vec_scalar_u32: Vector-scalar operations on Vec4u
#[wgsl]
pub mod vec_scalar_u32 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [u32; 320]);
    storage!(group(0), binding(1), read_write, OUTPUT: [Vec4u; 512]); // 8 ops * 64

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 5;
        let input = get!(INPUT);

        let v = vec4u(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let s = input[base + 4];

        let out_base = idx * 8;
        get_mut!(OUTPUT)[out_base + 0] = v + s;
        get_mut!(OUTPUT)[out_base + 1] = v - s;
        get_mut!(OUTPUT)[out_base + 2] = v * s;
        get_mut!(OUTPUT)[out_base + 3] = v / s;
        get_mut!(OUTPUT)[out_base + 4] = s + v;
        get_mut!(OUTPUT)[out_base + 5] = s - v;
        get_mut!(OUTPUT)[out_base + 6] = s * v;
        get_mut!(OUTPUT)[out_base + 7] = s / v;
    }
}

/// vec_unary: Unary negation operator (-v)
///
/// Tests negation for Vec4f and Vec4i (u32 doesn't support negation)
#[wgsl]
pub mod vec_unary {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]); // 64 vec4f + 64 vec4i (as f32 bits)
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 512]); // Same layout

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base_f = idx * 4;
        let base_i = 256 + idx * 4; // Second half of input
        let input = get!(INPUT);

        let vf = vec4f(
            input[base_f],
            input[base_f + 1],
            input[base_f + 2],
            input[base_f + 3],
        );
        let vi_bits = [
            bitcast_f32(input[base_i]),
            bitcast_f32(input[base_i + 1]),
            bitcast_f32(input[base_i + 2]),
            bitcast_f32(input[base_i + 3]),
        ];
        let vi = vec4i(
            bitcast_i32(vi_bits[0]),
            bitcast_i32(vi_bits[1]),
            bitcast_i32(vi_bits[2]),
            bitcast_i32(vi_bits[3]),
        );

        let neg_vf = -vf;
        let neg_vi = -vi;

        get_mut!(OUTPUT)[base_f] = neg_vf.x;
        get_mut!(OUTPUT)[base_f + 1] = neg_vf.y;
        get_mut!(OUTPUT)[base_f + 2] = neg_vf.z;
        get_mut!(OUTPUT)[base_f + 3] = neg_vf.w;

        get_mut!(OUTPUT)[base_i] = bitcast_f32(neg_vi.x);
        get_mut!(OUTPUT)[base_i + 1] = bitcast_f32(neg_vi.y);
        get_mut!(OUTPUT)[base_i + 2] = bitcast_f32(neg_vi.z);
        get_mut!(OUTPUT)[base_i + 3] = bitcast_f32(neg_vi.w);
    }
}

/// vec_mixed_dims: Test Vec2 and Vec3 operations for spot-checking
#[wgsl]
pub mod vec_mixed_dims {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 320]); // 64 vec2 + 64 vec3 = 320
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 384]); // 3 ops each = 384

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base2 = idx * 2;
        let base3 = 128 + idx * 3;
        let input = get!(INPUT);

        let v2 = vec2f(input[base2], input[base2 + 1]);
        let v3 = vec3f(input[base3], input[base3 + 1], input[base3 + 2]);

        let out_base2 = idx * 6;
        let out_base3 = out_base2 + 3;

        // Vec2f operations (double, scale)
        let v2_doubled = v2 + v2;
        let v2_scaled = v2 * 2.0;

        get_mut!(OUTPUT)[out_base2] = v2_doubled.x;
        get_mut!(OUTPUT)[out_base2 + 1] = v2_doubled.y;
        get_mut!(OUTPUT)[out_base2 + 2] = v2_scaled.x;

        // Vec3f operations
        let v3_doubled = v3 + v3;
        let v3_scaled = v3 * 2.0;
        let v3_neg = -v3;

        get_mut!(OUTPUT)[out_base3] = v3_doubled.x;
        get_mut!(OUTPUT)[out_base3 + 1] = v3_scaled.y;
        get_mut!(OUTPUT)[out_base3 + 2] = v3_neg.z;
    }
}

// Helper function to create flattened input for binary ops
fn flatten_vec4f_pairs(a: &[wgsl_rs::std::Vec4f; N], b: &[wgsl_rs::std::Vec4f; N]) -> [f32; 512] {
    let mut result = [0.0f32; 512];
    for i in 0..N {
        let base = i * 8;
        result[base] = a[i].x;
        result[base + 1] = a[i].y;
        result[base + 2] = a[i].z;
        result[base + 3] = a[i].w;
        result[base + 4] = b[i].x;
        result[base + 5] = b[i].y;
        result[base + 6] = b[i].z;
        result[base + 7] = b[i].w;
    }
    result
}

fn flatten_vec4i_pairs(a: &[wgsl_rs::std::Vec4i; N], b: &[wgsl_rs::std::Vec4i; N]) -> [i32; 512] {
    let mut result = [0i32; 512];
    for i in 0..N {
        let base = i * 8;
        result[base] = a[i].x;
        result[base + 1] = a[i].y;
        result[base + 2] = a[i].z;
        result[base + 3] = a[i].w;
        result[base + 4] = b[i].x;
        result[base + 5] = b[i].y;
        result[base + 6] = b[i].z;
        result[base + 7] = b[i].w;
    }
    result
}

fn flatten_vec4u_pairs(a: &[wgsl_rs::std::Vec4u; N], b: &[wgsl_rs::std::Vec4u; N]) -> [u32; 512] {
    let mut result = [0u32; 512];
    for i in 0..N {
        let base = i * 8;
        result[base] = a[i].x;
        result[base + 1] = a[i].y;
        result[base + 2] = a[i].z;
        result[base + 3] = a[i].w;
        result[base + 4] = b[i].x;
        result[base + 5] = b[i].y;
        result[base + 6] = b[i].z;
        result[base + 7] = b[i].w;
    }
    result
}

fn flatten_vec4f_scalar(v: &[wgsl_rs::std::Vec4f; N], s: &[f32; N]) -> [f32; 320] {
    let mut result = [0.0f32; 320];
    for i in 0..N {
        let base = i * 5;
        result[base] = v[i].x;
        result[base + 1] = v[i].y;
        result[base + 2] = v[i].z;
        result[base + 3] = v[i].w;
        result[base + 4] = s[i];
    }
    result
}

fn flatten_vec4i_scalar(v: &[wgsl_rs::std::Vec4i; N], s: &[i32; N]) -> [i32; 320] {
    let mut result = [0i32; 320];
    for i in 0..N {
        let base = i * 5;
        result[base] = v[i].x;
        result[base + 1] = v[i].y;
        result[base + 2] = v[i].z;
        result[base + 3] = v[i].w;
        result[base + 4] = s[i];
    }
    result
}

fn flatten_vec4u_scalar(v: &[wgsl_rs::std::Vec4u; N], s: &[u32; N]) -> [u32; 320] {
    let mut result = [0u32; 320];
    for i in 0..N {
        let base = i * 5;
        result[base] = v[i].x;
        result[base + 1] = v[i].y;
        result[base + 2] = v[i].z;
        result[base + 3] = v[i].w;
        result[base + 4] = s[i];
    }
    result
}

/// Generates test input pairs for Vec4f binary operations.
fn vec4f_binary_inputs() -> ([wgsl_rs::std::Vec4f; N], [wgsl_rs::std::Vec4f; N]) {
    use wgsl_rs::std::vec4f;

    let mut a = [vec4f(0.0, 0.0, 0.0, 0.0); N];
    let mut b = [vec4f(1.0, 1.0, 1.0, 1.0); N];

    // Edge cases
    a[0] = vec4f(0.0, 0.0, 0.0, 0.0); // zero vector
    b[0] = vec4f(1.0, 2.0, 3.0, 4.0);

    a[1] = vec4f(1.0, 0.0, 0.0, 0.0); // unit x
    b[1] = vec4f(0.0, 1.0, 0.0, 0.0); // unit y (orthogonal)

    a[2] = vec4f(1.0, 1.0, 1.0, 1.0);
    b[2] = vec4f(2.0, 2.0, 2.0, 2.0); // parallel

    a[3] = vec4f(-1.0, -2.0, -3.0, -4.0); // negative
    b[3] = vec4f(5.0, 6.0, 7.0, 8.0);

    a[4] = vec4f(10.0, 20.0, 30.0, 40.0); // large values
    b[4] = vec4f(2.0, 4.0, 5.0, 8.0); // safe divisors

    // Fill remaining with diverse combinations
    for i in 5..N {
        let t = i as f32;
        a[i] = vec4f(t * 0.1, t * 0.2, t * 0.3, t * 0.4);
        b[i] = vec4f((i + 1) as f32 * 0.5, 1.0, 2.0, 3.0); // avoid zero
        // divisors
    }

    (a, b)
}

/// Generates test input pairs for Vec4i binary operations.
fn vec4i_binary_inputs() -> ([wgsl_rs::std::Vec4i; N], [wgsl_rs::std::Vec4i; N]) {
    use wgsl_rs::std::vec4i;

    let mut a = [vec4i(0, 0, 0, 0); N];
    let mut b = [vec4i(1, 1, 1, 1); N];

    a[0] = vec4i(0, 0, 0, 0);
    b[0] = vec4i(1, 2, 3, 4);

    a[1] = vec4i(10, 20, 30, 40);
    b[1] = vec4i(2, 4, 5, 8);

    a[2] = vec4i(-10, -20, -30, -40);
    b[2] = vec4i(2, 4, 5, 8);

    a[3] = vec4i(100, 200, 300, 400);
    b[3] = vec4i(10, 10, 10, 10);

    for i in 4..N {
        let t = i as i32;
        a[i] = vec4i(t * 10, t * 20, t * 30, t * 40);
        b[i] = vec4i((i as i32 % 7) + 1, 2, 3, 4);
    }

    (a, b)
}

/// Generates test input pairs for Vec4u binary operations.
///
/// Note: For u32, subtraction must avoid underflow (a >= b componentwise).
/// We ensure b is always small and a is always large enough.
fn vec4u_binary_inputs() -> ([wgsl_rs::std::Vec4u; N], [wgsl_rs::std::Vec4u; N]) {
    use wgsl_rs::std::vec4u;

    let mut a = [vec4u(1000, 2000, 3000, 4000); N];
    let mut b = [vec4u(1, 2, 3, 4); N];

    // Ensure a >= b componentwise to avoid underflow
    a[0] = vec4u(100, 200, 300, 400);
    b[0] = vec4u(1, 2, 3, 4);

    a[1] = vec4u(1000, 2000, 3000, 4000);
    b[1] = vec4u(10, 20, 30, 40);

    a[2] = vec4u(10000, 20000, 30000, 40000);
    b[2] = vec4u(100, 200, 300, 400);

    a[3] = vec4u(100000, 200000, 300000, 400000);
    b[3] = vec4u(1000, 2000, 3000, 4000);

    for i in 4..N {
        let t = (i as u32 + 10) * 100;
        a[i] = vec4u(t, t * 2, t * 3, t * 4);
        // Keep b small
        b[i] = vec4u(
            (i as u32 % 10) + 1,
            (i as u32 % 10) + 2,
            (i as u32 % 10) + 3,
            (i as u32 % 10) + 4,
        );
    }

    (a, b)
}

/// Generates test inputs for vector-scalar operations (f32).
fn vec4f_scalar_inputs() -> ([wgsl_rs::std::Vec4f; N], [f32; N]) {
    use wgsl_rs::std::vec4f;

    let mut v = [vec4f(0.0, 0.0, 0.0, 0.0); N];
    let mut s = [1.0f32; N];

    v[0] = vec4f(1.0, 2.0, 3.0, 4.0);
    s[0] = 2.0;

    v[1] = vec4f(-5.0, -10.0, -15.0, -20.0);
    s[1] = 3.0;

    v[2] = vec4f(0.5, 1.5, 2.5, 3.5);
    s[2] = 0.5;

    for i in 3..N {
        let t = i as f32;
        v[i] = vec4f(t, t * 2.0, t * 3.0, t * 4.0);
        s[i] = (i % 10 + 1) as f32;
    }

    (v, s)
}

/// Generates test inputs for vector-scalar operations (i32).
fn vec4i_scalar_inputs() -> ([wgsl_rs::std::Vec4i; N], [i32; N]) {
    use wgsl_rs::std::vec4i;

    let mut v = [vec4i(0, 0, 0, 0); N];
    let mut s = [1i32; N];

    v[0] = vec4i(10, 20, 30, 40);
    s[0] = 2;

    v[1] = vec4i(-50, -100, -150, -200);
    s[1] = 3;

    v[2] = vec4i(100, 200, 300, 400);
    s[2] = 10;

    for i in 3..N {
        let t = i as i32;
        v[i] = vec4i(t * 10, t * 20, t * 30, t * 40);
        s[i] = ((i % 10) + 1) as i32;
    }

    (v, s)
}

/// Generates test inputs for vector-scalar operations (u32).
///
/// Note: For u32, to avoid underflow in both v - s and s - v operations,
/// we keep values in a narrow safe range where s is between min and max
/// components.
fn vec4u_scalar_inputs() -> ([wgsl_rs::std::Vec4u; N], [u32; N]) {
    use wgsl_rs::std::vec4u;

    let mut v = [vec4u(10, 20, 30, 40); N];
    let mut s = [30u32; N]; // Between min (10) and max (40)

    v[0] = vec4u(10, 20, 30, 40);
    s[0] = 25; // Between 10 and 40

    v[1] = vec4u(100, 200, 300, 400);
    s[1] = 250; // Between 100 and 400

    v[2] = vec4u(5, 10, 15, 20);
    s[2] = 12; // Between 5 and 20

    for i in 3..N {
        let t = ((i as u32 % 20) + 1) * 10;
        v[i] = vec4u(t, t * 2, t * 3, t * 4);
        s[i] = t * 2 + t; // t*3, which is between t (min) and t*4 (max)
    }

    (v, s)
}

/// Generates test inputs for unary negation.
fn vec_unary_inputs() -> ([wgsl_rs::std::Vec4f; N], [wgsl_rs::std::Vec4i; N]) {
    use wgsl_rs::std::{vec4f, vec4i};

    let mut vf = [vec4f(0.0, 0.0, 0.0, 0.0); N];
    let mut vi = [vec4i(0, 0, 0, 0); N];

    vf[0] = vec4f(1.0, 2.0, 3.0, 4.0);
    vi[0] = vec4i(10, 20, 30, 40);

    vf[1] = vec4f(-1.0, -2.0, -3.0, -4.0);
    vi[1] = vec4i(-10, -20, -30, -40);

    vf[2] = vec4f(0.0, 0.0, 0.0, 0.0);
    vi[2] = vec4i(0, 0, 0, 0);

    for i in 3..N {
        let t = i as f32;
        vf[i] = vec4f(t, -t, t * 2.0, -t * 2.0);
        vi[i] = vec4i(
            i as i32 * 10,
            -(i as i32 * 10),
            i as i32 * 20,
            -(i as i32 * 20),
        );
    }

    (vf, vi)
}

/// Generates test inputs for Vec2f and Vec3f.
fn vec_mixed_dims_inputs() -> ([wgsl_rs::std::Vec2f; N], [wgsl_rs::std::Vec3f; N]) {
    use wgsl_rs::std::{vec2f, vec3f};

    let mut v2 = [vec2f(0.0, 0.0); N];
    let mut v3 = [vec3f(0.0, 0.0, 0.0); N];

    v2[0] = vec2f(1.0, 2.0);
    v3[0] = vec3f(1.0, 2.0, 3.0);

    v2[1] = vec2f(-5.0, -10.0);
    v3[1] = vec3f(-5.0, -10.0, -15.0);

    for i in 2..N {
        let t = i as f32;
        v2[i] = vec2f(t, t * 2.0);
        v3[i] = vec3f(t, t * 2.0, t * 3.0);
    }

    (v2, v3)
}

/// The vector arithmetic roundtrip test.
pub struct VectorArithmeticTest;

impl RoundtripTest for VectorArithmeticTest {
    fn name(&self) -> &str {
        "vector_arithmetic"
    }

    fn description(&self) -> &str {
        "Vector arithmetic operators: +, -, *, /, %, unary -"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        let mut results = Vec::new();

        // --- vec_binary_f32 ---
        {
            let (input_a, input_b) = vec4f_binary_inputs();
            let flattened = flatten_vec4f_pairs(&input_a, &input_b);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 5 * 4 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_binary_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_binary_f32::INPUT.set(flattened);
            vec_binary_f32::OUTPUT.set([Vec4f::default(); 320]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_binary_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_binary_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("vec4_add[{}].x", i),
                        format!("vec4_add[{}].y", i),
                        format!("vec4_add[{}].z", i),
                        format!("vec4_add[{}].w", i),
                        format!("vec4_sub[{}].x", i),
                        format!("vec4_sub[{}].y", i),
                        format!("vec4_sub[{}].z", i),
                        format!("vec4_sub[{}].w", i),
                        format!("vec4_mul[{}].x", i),
                        format!("vec4_mul[{}].y", i),
                        format!("vec4_mul[{}].z", i),
                        format!("vec4_mul[{}].w", i),
                        format!("vec4_div[{}].x", i),
                        format!("vec4_div[{}].y", i),
                        format!("vec4_div[{}].z", i),
                        format!("vec4_div[{}].w", i),
                        format!("vec4_mod[{}].x", i),
                        format!("vec4_mod[{}].y", i),
                        format!("vec4_mod[{}].z", i),
                        format!("vec4_mod[{}].w", i),
                    ]
                })
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "vec_binary_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                1e-5, // Small tolerance for floating-point ops
            ));
        }

        // --- vec_binary_i32 ---
        {
            let (input_a, input_b) = vec4i_binary_inputs();
            let flattened = flatten_vec4i_pairs(&input_a, &input_b);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<i32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_binary_i32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_i32s: &[i32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_binary_i32::INPUT.set(flattened);
            vec_binary_i32::OUTPUT.set([Vec4i::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_binary_i32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_binary_i32::OUTPUT.get();
            let cpu_i32s: Vec<i32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..1024).map(|i| format!("vec4i[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            // Convert i32 to u32 for comparison
            let gpu_u32s: Vec<u32> = gpu_i32s.iter().map(|&x| x as u32).collect();
            let cpu_u32s: Vec<u32> = cpu_i32s.iter().map(|&x| x as u32).collect();

            results.push(harness::compare_u32_results(
                "vec_binary_i32",
                &gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- vec_binary_u32 ---
        {
            let (input_a, input_b) = vec4u_binary_inputs();
            let flattened = flatten_vec4u_pairs(&input_a, &input_b);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 4 * 4 * std::mem::size_of::<u32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_binary_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_binary_u32::INPUT.set(flattened);
            vec_binary_u32::OUTPUT.set([Vec4u::default(); 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_binary_u32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_binary_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..1024).map(|i| format!("vec4u[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "vec_binary_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- vec_scalar_f32 ---
        {
            let (input_v, input_s) = vec4f_scalar_inputs();
            let flattened = flatten_vec4f_scalar(&input_v, &input_s);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 8 * 4 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_scalar_f32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_scalar_f32::INPUT.set(flattened);
            vec_scalar_f32::OUTPUT.set([Vec4f::default(); 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_scalar_f32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_scalar_f32::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..2048)
                .map(|i| format!("vec_scalar_f32[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "vec_scalar_f32",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                1e-5,
            ));
        }

        // --- vec_scalar_i32 ---
        {
            let (input_v, input_s) = vec4i_scalar_inputs();
            let flattened = flatten_vec4i_scalar(&input_v, &input_s);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 8 * 4 * std::mem::size_of::<i32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_scalar_i32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_i32s: &[i32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_scalar_i32::INPUT.set(flattened);
            vec_scalar_i32::OUTPUT.set([Vec4i::default(); 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_scalar_i32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_scalar_i32::OUTPUT.get();
            let cpu_i32s: Vec<i32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..2048)
                .map(|i| format!("vec_scalar_i32[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            let gpu_u32s: Vec<u32> = gpu_i32s.iter().map(|&x| x as u32).collect();
            let cpu_u32s: Vec<u32> = cpu_i32s.iter().map(|&x| x as u32).collect();

            results.push(harness::compare_u32_results(
                "vec_scalar_i32",
                &gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- vec_scalar_u32 ---
        {
            let (input_v, input_s) = vec4u_scalar_inputs();
            let flattened = flatten_vec4u_scalar(&input_v, &input_s);
            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (N * 8 * 4 * std::mem::size_of::<u32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_scalar_u32::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_u32s: &[u32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_scalar_u32::INPUT.set(flattened);
            vec_scalar_u32::OUTPUT.set([Vec4u::default(); 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_scalar_u32::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_scalar_u32::OUTPUT.get();
            let cpu_u32s: Vec<u32> = cpu_output_guard.iter().flat_map(|v| v.to_array()).collect();

            let labels: Vec<String> = (0..2048)
                .map(|i| format!("vec_scalar_u32[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "vec_scalar_u32",
                gpu_u32s,
                &cpu_u32s,
                &label_refs,
            ));
        }

        // --- vec_unary ---
        {
            let (input_f, input_i) = vec_unary_inputs();
            // Flatten both into single input buffer
            let mut flattened = [0.0f32; 512];
            for i in 0..N {
                let base_f = i * 4;
                flattened[base_f] = input_f[i].x;
                flattened[base_f + 1] = input_f[i].y;
                flattened[base_f + 2] = input_f[i].z;
                flattened[base_f + 3] = input_f[i].w;

                let base_i = 256 + i * 4;
                flattened[base_i] = f32::from_ne_bytes(input_i[i].x.to_ne_bytes());
                flattened[base_i + 1] = f32::from_ne_bytes(input_i[i].y.to_ne_bytes());
                flattened[base_i + 2] = f32::from_ne_bytes(input_i[i].z.to_ne_bytes());
                flattened[base_i + 3] = f32::from_ne_bytes(input_i[i].w.to_ne_bytes());
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (512 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_unary::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_unary::INPUT.set(flattened);
            vec_unary::OUTPUT.set([0.0f32; 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_unary::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_unary::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N)
                .flat_map(|i| {
                    vec![
                        format!("-vec4f[{}].x", i),
                        format!("-vec4f[{}].y", i),
                        format!("-vec4f[{}].z", i),
                        format!("-vec4f[{}].w", i),
                    ]
                })
                .chain((0..N).flat_map(|i| {
                    vec![
                        format!("-vec4i[{}].x", i),
                        format!("-vec4i[{}].y", i),
                        format!("-vec4i[{}].z", i),
                        format!("-vec4i[{}].w", i),
                    ]
                }))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "vec_unary",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                0.0, // Exact for negation
            ));
        }

        // --- vec_mixed_dims ---
        {
            let (input_v2, input_v3) = vec_mixed_dims_inputs();
            // Flatten into single input buffer
            let mut flattened = [0.0f32; 320];
            for i in 0..N {
                let base2 = i * 2;
                flattened[base2] = input_v2[i].x;
                flattened[base2 + 1] = input_v2[i].y;

                let base3 = 128 + i * 3;
                flattened[base3] = input_v3[i].x;
                flattened[base3 + 1] = input_v3[i].y;
                flattened[base3 + 2] = input_v3[i].z;
            }

            let input_bytes = bytemuck::cast_slice(&flattened);
            let output_size = (384 * std::mem::size_of::<f32>()) as u64;

            let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
                device,
                queue,
                shader_source: vec_mixed_dims::linkage::shader_source(),
                entry_point: "main",
                bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
                input_data: input_bytes,
                output_size,
                workgroup_count: (1, 1, 1),
            });
            let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);

            use wgsl_rs::std::*;
            vec_mixed_dims::INPUT.set(flattened);
            vec_mixed_dims::OUTPUT.set([0.0f32; 384]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |builtins| {
                vec_mixed_dims::main(builtins.global_invocation_id);
            });
            let cpu_output_guard = vec_mixed_dims::OUTPUT.get();
            let cpu_floats: Vec<f32> = cpu_output_guard.to_vec();

            let labels: Vec<String> = (0..N * 6)
                .map(|i| format!("vec_mixed_dims[{}]", i))
                .collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_f32_results(
                "vec_mixed_dims",
                gpu_floats,
                &cpu_floats,
                &label_refs,
                1e-5,
            ));
        }

        results
    }
}
