//! Roundtrip tests for matrix builtins and arithmetic operators.
//!
//! Tests: `determinant`, `transpose`, matrix-vector multiplication,
//! and matrix-matrix multiplication.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

#[wgsl]
pub mod determinant_mat2 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 256]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    /// Computes determinant for 64 Mat2x2f inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let input = get!(INPUT);

        let m = mat2x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
        );
        get_mut!(OUTPUT)[idx] = determinant(m);
    }
}

#[wgsl]
pub mod determinant_mat3 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    /// Computes determinant for 64 Mat3x3f inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 9;
        let input = get!(INPUT);

        let m = mat3x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
            vec3f(input[base + 6], input[base + 7], input[base + 8]),
        );
        get_mut!(OUTPUT)[idx] = determinant(m);
    }
}

#[wgsl]
pub mod determinant_mat4 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 1024]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 64]);

    /// Computes determinant for 64 Mat4x4f inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 16;
        let input = get!(INPUT);

        let m = mat4x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
            vec4f(
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ),
            vec4f(
                input[base + 12],
                input[base + 13],
                input[base + 14],
                input[base + 15],
            ),
        );
        get_mut!(OUTPUT)[idx] = determinant(m);
    }
}

#[wgsl]
pub mod transpose_mat2x2 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 256]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    /// Computes transpose(Mat2x2f) for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 4;
        let input = get!(INPUT);

        let m = mat2x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
        );
        let t = mat2x2f(transpose(m)[0usize], transpose(m)[1usize]);
        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[1usize].x;
        get_mut!(OUTPUT)[base + 3] = t[1usize].y;
    }
}

#[wgsl]
pub mod transpose_mat3x3 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 576]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 576]);

    /// Computes transpose(Mat3x3f) for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 9;
        let input = get!(INPUT);

        let m = mat3x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
            vec3f(input[base + 6], input[base + 7], input[base + 8]),
        );
        let t = mat3x3f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[1usize].x;
        get_mut!(OUTPUT)[base + 4] = t[1usize].y;
        get_mut!(OUTPUT)[base + 5] = t[1usize].z;
        get_mut!(OUTPUT)[base + 6] = t[2usize].x;
        get_mut!(OUTPUT)[base + 7] = t[2usize].y;
        get_mut!(OUTPUT)[base + 8] = t[2usize].z;
    }
}

#[wgsl]
pub mod transpose_mat4x4 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 1024]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 1024]);

    /// Computes transpose(Mat4x4f) for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 16;
        let input = get!(INPUT);

        let m = mat4x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
            vec4f(
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ),
            vec4f(
                input[base + 12],
                input[base + 13],
                input[base + 14],
                input[base + 15],
            ),
        );
        let t = mat4x4f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
            transpose(m)[3usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[0usize].w;
        get_mut!(OUTPUT)[base + 4] = t[1usize].x;
        get_mut!(OUTPUT)[base + 5] = t[1usize].y;
        get_mut!(OUTPUT)[base + 6] = t[1usize].z;
        get_mut!(OUTPUT)[base + 7] = t[1usize].w;
        get_mut!(OUTPUT)[base + 8] = t[2usize].x;
        get_mut!(OUTPUT)[base + 9] = t[2usize].y;
        get_mut!(OUTPUT)[base + 10] = t[2usize].z;
        get_mut!(OUTPUT)[base + 11] = t[2usize].w;
        get_mut!(OUTPUT)[base + 12] = t[3usize].x;
        get_mut!(OUTPUT)[base + 13] = t[3usize].y;
        get_mut!(OUTPUT)[base + 14] = t[3usize].z;
        get_mut!(OUTPUT)[base + 15] = t[3usize].w;
    }
}

#[wgsl]
pub mod transpose_mat2x3 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 384]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 384]);

    /// Computes transpose(Mat2x3f) -> Mat3x2f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 6;
        let input = get!(INPUT);

        let m = mat2x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
        );
        let t = mat3x2f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[1usize].x;
        get_mut!(OUTPUT)[base + 3] = t[1usize].y;
        get_mut!(OUTPUT)[base + 4] = t[2usize].x;
        get_mut!(OUTPUT)[base + 5] = t[2usize].y;
    }
}

#[wgsl]
pub mod transpose_mat3x2 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 384]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 384]);

    /// Computes transpose(Mat3x2f) -> Mat2x3f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 6;
        let input = get!(INPUT);

        let m = mat3x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
            vec2f(input[base + 4], input[base + 5]),
        );
        let t = mat2x3f(transpose(m)[0usize], transpose(m)[1usize]);

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[1usize].x;
        get_mut!(OUTPUT)[base + 4] = t[1usize].y;
        get_mut!(OUTPUT)[base + 5] = t[1usize].z;
    }
}

#[wgsl]
pub mod transpose_mat3x4 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 768]);

    /// Computes transpose(Mat3x4f) -> Mat4x3f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 12;
        let input = get!(INPUT);

        let m = mat3x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
            vec4f(
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ),
        );
        let t = mat4x3f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
            transpose(m)[3usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[1usize].x;
        get_mut!(OUTPUT)[base + 4] = t[1usize].y;
        get_mut!(OUTPUT)[base + 5] = t[1usize].z;
        get_mut!(OUTPUT)[base + 6] = t[2usize].x;
        get_mut!(OUTPUT)[base + 7] = t[2usize].y;
        get_mut!(OUTPUT)[base + 8] = t[2usize].z;
        get_mut!(OUTPUT)[base + 9] = t[3usize].x;
        get_mut!(OUTPUT)[base + 10] = t[3usize].y;
        get_mut!(OUTPUT)[base + 11] = t[3usize].z;
    }
}

#[wgsl]
pub mod transpose_mat4x3 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 768]);

    /// Computes transpose(Mat4x3f) -> Mat3x4f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 12;
        let input = get!(INPUT);

        let m = mat4x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
            vec3f(input[base + 6], input[base + 7], input[base + 8]),
            vec3f(input[base + 9], input[base + 10], input[base + 11]),
        );
        let t = mat3x4f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[0usize].w;
        get_mut!(OUTPUT)[base + 4] = t[1usize].x;
        get_mut!(OUTPUT)[base + 5] = t[1usize].y;
        get_mut!(OUTPUT)[base + 6] = t[1usize].z;
        get_mut!(OUTPUT)[base + 7] = t[1usize].w;
        get_mut!(OUTPUT)[base + 8] = t[2usize].x;
        get_mut!(OUTPUT)[base + 9] = t[2usize].y;
        get_mut!(OUTPUT)[base + 10] = t[2usize].z;
        get_mut!(OUTPUT)[base + 11] = t[2usize].w;
    }
}

#[wgsl]
pub mod transpose_mat2x4 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 512]);

    /// Computes transpose(Mat2x4f) -> Mat4x2f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let m = mat2x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
        );
        let t = mat4x2f(
            transpose(m)[0usize],
            transpose(m)[1usize],
            transpose(m)[2usize],
            transpose(m)[3usize],
        );

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[1usize].x;
        get_mut!(OUTPUT)[base + 3] = t[1usize].y;
        get_mut!(OUTPUT)[base + 4] = t[2usize].x;
        get_mut!(OUTPUT)[base + 5] = t[2usize].y;
        get_mut!(OUTPUT)[base + 6] = t[3usize].x;
        get_mut!(OUTPUT)[base + 7] = t[3usize].y;
    }
}

#[wgsl]
pub mod transpose_mat4x2 {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 512]);

    /// Computes transpose(Mat4x2f) -> Mat2x4f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let m = mat4x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
            vec2f(input[base + 4], input[base + 5]),
            vec2f(input[base + 6], input[base + 7]),
        );
        let t = mat2x4f(transpose(m)[0usize], transpose(m)[1usize]);

        get_mut!(OUTPUT)[base] = t[0usize].x;
        get_mut!(OUTPUT)[base + 1] = t[0usize].y;
        get_mut!(OUTPUT)[base + 2] = t[0usize].z;
        get_mut!(OUTPUT)[base + 3] = t[0usize].w;
        get_mut!(OUTPUT)[base + 4] = t[1usize].x;
        get_mut!(OUTPUT)[base + 5] = t[1usize].y;
        get_mut!(OUTPUT)[base + 6] = t[1usize].z;
        get_mut!(OUTPUT)[base + 7] = t[1usize].w;
    }
}

#[wgsl]
pub mod mat2_vec2_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 384]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 128]);

    /// Computes Mat2x2f * Vec2f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 6;
        let input = get!(INPUT);

        let m = mat2x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
        );
        let v = vec2f(input[base + 4], input[base + 5]);
        let out = m * v;

        get_mut!(OUTPUT)[idx * 2] = out.x;
        get_mut!(OUTPUT)[idx * 2 + 1] = out.y;
    }
}

#[wgsl]
pub mod mat3_vec3_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 768]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 192]);

    /// Computes Mat3x3f * Vec3f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 12;
        let input = get!(INPUT);

        let m = mat3x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
            vec3f(input[base + 6], input[base + 7], input[base + 8]),
        );
        let v = vec3f(input[base + 9], input[base + 10], input[base + 11]);
        let out = m * v;

        get_mut!(OUTPUT)[idx * 3] = out.x;
        get_mut!(OUTPUT)[idx * 3 + 1] = out.y;
        get_mut!(OUTPUT)[idx * 3 + 2] = out.z;
    }
}

#[wgsl]
pub mod mat4_vec4_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 1280]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    /// Computes Mat4x4f * Vec4f for 64 inputs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 20;
        let input = get!(INPUT);

        let m = mat4x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
            vec4f(
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ),
            vec4f(
                input[base + 12],
                input[base + 13],
                input[base + 14],
                input[base + 15],
            ),
        );
        let v = vec4f(
            input[base + 16],
            input[base + 17],
            input[base + 18],
            input[base + 19],
        );
        let out = m * v;

        get_mut!(OUTPUT)[idx * 4] = out.x;
        get_mut!(OUTPUT)[idx * 4 + 1] = out.y;
        get_mut!(OUTPUT)[idx * 4 + 2] = out.z;
        get_mut!(OUTPUT)[idx * 4 + 3] = out.w;
    }
}

#[wgsl]
pub mod mat2_mat2_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    /// Computes Mat2x2f * Mat2x2f for 64 input pairs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 8;
        let input = get!(INPUT);

        let a = mat2x2f(
            vec2f(input[base], input[base + 1]),
            vec2f(input[base + 2], input[base + 3]),
        );
        let b = mat2x2f(
            vec2f(input[base + 4], input[base + 5]),
            vec2f(input[base + 6], input[base + 7]),
        );
        let out = mat2x2f((a * b)[0usize], (a * b)[1usize]);

        get_mut!(OUTPUT)[idx * 4] = out[0usize].x;
        get_mut!(OUTPUT)[idx * 4 + 1] = out[0usize].y;
        get_mut!(OUTPUT)[idx * 4 + 2] = out[1usize].x;
        get_mut!(OUTPUT)[idx * 4 + 3] = out[1usize].y;
    }
}

#[wgsl]
pub mod mat3_mat3_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 1152]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 576]);

    /// Computes Mat3x3f * Mat3x3f for 64 input pairs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 18;
        let input = get!(INPUT);

        let a = mat3x3f(
            vec3f(input[base], input[base + 1], input[base + 2]),
            vec3f(input[base + 3], input[base + 4], input[base + 5]),
            vec3f(input[base + 6], input[base + 7], input[base + 8]),
        );
        let b = mat3x3f(
            vec3f(input[base + 9], input[base + 10], input[base + 11]),
            vec3f(input[base + 12], input[base + 13], input[base + 14]),
            vec3f(input[base + 15], input[base + 16], input[base + 17]),
        );
        let out = mat3x3f((a * b)[0usize], (a * b)[1usize], (a * b)[2usize]);

        let out_base = idx * 9;
        get_mut!(OUTPUT)[out_base] = out[0usize].x;
        get_mut!(OUTPUT)[out_base + 1] = out[0usize].y;
        get_mut!(OUTPUT)[out_base + 2] = out[0usize].z;
        get_mut!(OUTPUT)[out_base + 3] = out[1usize].x;
        get_mut!(OUTPUT)[out_base + 4] = out[1usize].y;
        get_mut!(OUTPUT)[out_base + 5] = out[1usize].z;
        get_mut!(OUTPUT)[out_base + 6] = out[2usize].x;
        get_mut!(OUTPUT)[out_base + 7] = out[2usize].y;
        get_mut!(OUTPUT)[out_base + 8] = out[2usize].z;
    }
}

#[wgsl]
pub mod mat4_mat4_mul {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 2048]);
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 1024]);

    /// Computes Mat4x4f * Mat4x4f for 64 input pairs.
    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let base = idx * 32;
        let input = get!(INPUT);

        let a = mat4x4f(
            vec4f(
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ),
            vec4f(
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            ),
            vec4f(
                input[base + 8],
                input[base + 9],
                input[base + 10],
                input[base + 11],
            ),
            vec4f(
                input[base + 12],
                input[base + 13],
                input[base + 14],
                input[base + 15],
            ),
        );
        let b = mat4x4f(
            vec4f(
                input[base + 16],
                input[base + 17],
                input[base + 18],
                input[base + 19],
            ),
            vec4f(
                input[base + 20],
                input[base + 21],
                input[base + 22],
                input[base + 23],
            ),
            vec4f(
                input[base + 24],
                input[base + 25],
                input[base + 26],
                input[base + 27],
            ),
            vec4f(
                input[base + 28],
                input[base + 29],
                input[base + 30],
                input[base + 31],
            ),
        );
        let out = mat4x4f(
            (a * b)[0usize],
            (a * b)[1usize],
            (a * b)[2usize],
            (a * b)[3usize],
        );

        let out_base = idx * 16;
        get_mut!(OUTPUT)[out_base] = out[0usize].x;
        get_mut!(OUTPUT)[out_base + 1] = out[0usize].y;
        get_mut!(OUTPUT)[out_base + 2] = out[0usize].z;
        get_mut!(OUTPUT)[out_base + 3] = out[0usize].w;
        get_mut!(OUTPUT)[out_base + 4] = out[1usize].x;
        get_mut!(OUTPUT)[out_base + 5] = out[1usize].y;
        get_mut!(OUTPUT)[out_base + 6] = out[1usize].z;
        get_mut!(OUTPUT)[out_base + 7] = out[1usize].w;
        get_mut!(OUTPUT)[out_base + 8] = out[2usize].x;
        get_mut!(OUTPUT)[out_base + 9] = out[2usize].y;
        get_mut!(OUTPUT)[out_base + 10] = out[2usize].z;
        get_mut!(OUTPUT)[out_base + 11] = out[2usize].w;
        get_mut!(OUTPUT)[out_base + 12] = out[3usize].x;
        get_mut!(OUTPUT)[out_base + 13] = out[3usize].y;
        get_mut!(OUTPUT)[out_base + 14] = out[3usize].z;
        get_mut!(OUTPUT)[out_base + 15] = out[3usize].w;
    }
}

/// Writes one Mat2x2f into a flattened f32 buffer.
fn write_mat2(dst: &mut [f32], idx: usize, c0: [f32; 2], c1: [f32; 2]) {
    let base = idx * 4;
    dst[base] = c0[0];
    dst[base + 1] = c0[1];
    dst[base + 2] = c1[0];
    dst[base + 3] = c1[1];
}

/// Writes one Mat3x3f into a flattened f32 buffer.
fn write_mat3(dst: &mut [f32], idx: usize, c0: [f32; 3], c1: [f32; 3], c2: [f32; 3]) {
    let base = idx * 9;
    dst[base] = c0[0];
    dst[base + 1] = c0[1];
    dst[base + 2] = c0[2];
    dst[base + 3] = c1[0];
    dst[base + 4] = c1[1];
    dst[base + 5] = c1[2];
    dst[base + 6] = c2[0];
    dst[base + 7] = c2[1];
    dst[base + 8] = c2[2];
}

/// Writes one Mat4x4f into a flattened f32 buffer.
fn write_mat4(dst: &mut [f32], idx: usize, c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) {
    let base = idx * 16;
    dst[base] = c0[0];
    dst[base + 1] = c0[1];
    dst[base + 2] = c0[2];
    dst[base + 3] = c0[3];
    dst[base + 4] = c1[0];
    dst[base + 5] = c1[1];
    dst[base + 6] = c1[2];
    dst[base + 7] = c1[3];
    dst[base + 8] = c2[0];
    dst[base + 9] = c2[1];
    dst[base + 10] = c2[2];
    dst[base + 11] = c2[3];
    dst[base + 12] = c3[0];
    dst[base + 13] = c3[1];
    dst[base + 14] = c3[2];
    dst[base + 15] = c3[3];
}

/// Generates Mat2x2f inputs used by determinant and transpose tests.
fn mat2_inputs() -> [f32; 256] {
    let mut out = [0.0f32; 256];
    write_mat2(&mut out, 0, [1.0, 0.0], [0.0, 1.0]);
    write_mat2(&mut out, 1, [2.0, 1.0], [1.0, 3.0]);
    write_mat2(&mut out, 2, [1.0, 2.0], [2.0, 4.0]);
    write_mat2(&mut out, 3, [-2.0, 0.5], [1.25, -3.0]);

    for i in 4..N {
        let t = i as f32 * 0.125 - 4.0;
        write_mat2(
            &mut out,
            i,
            [1.0 + t * 0.1, -0.35 + t * 0.07],
            [0.2 - t * 0.04, 0.8 + t * 0.05],
        );
    }
    out
}

/// Generates Mat3x3f inputs used by determinant, transpose, and multiplication
/// tests.
fn mat3_inputs() -> [f32; 576] {
    let mut out = [0.0f32; 576];
    write_mat3(
        &mut out,
        0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    write_mat3(
        &mut out,
        1,
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    );
    write_mat3(
        &mut out,
        2,
        [2.0, -1.0, 0.5],
        [0.0, 1.5, -2.0],
        [1.25, 0.0, 3.0],
    );

    for i in 3..N {
        let t = i as f32 * 0.11 - 3.0;
        write_mat3(
            &mut out,
            i,
            [1.0 + t * 0.03, 0.2 - t * 0.02, -0.4 + t * 0.01],
            [0.1 + t * 0.04, 0.9 + t * 0.05, 0.3 - t * 0.03],
            [-0.2 + t * 0.02, 0.6 - t * 0.01, 1.2 + t * 0.03],
        );
    }
    out
}

/// Generates Mat4x4f inputs used by determinant, transpose, and multiplication
/// tests.
fn mat4_inputs() -> [f32; 1024] {
    let mut out = [0.0f32; 1024];
    write_mat4(
        &mut out,
        0,
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    );
    write_mat4(
        &mut out,
        1,
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    );

    for i in 2..N {
        let t = i as f32 * 0.09 - 2.5;
        write_mat4(
            &mut out,
            i,
            [1.0 + t * 0.03, 0.2 - t * 0.01, -0.1 + t * 0.02, 0.4],
            [0.0 + t * 0.02, 0.8 + t * 0.04, 0.3 - t * 0.02, -0.2],
            [0.1 - t * 0.02, 0.5, 1.1 + t * 0.03, 0.25 - t * 0.01],
            [-0.3, 0.15 + t * 0.01, 0.2 - t * 0.02, 0.9 + t * 0.02],
        );
    }
    out
}

/// Generates Mat2x3f flattened inputs.
fn mat2x3_inputs() -> [f32; 384] {
    let mut out = [0.0f32; 384];
    for i in 0..N {
        let t = i as f32 * 0.2 - 4.0;
        let base = i * 6;
        out[base] = 1.0 + t * 0.04;
        out[base + 1] = -0.2 + t * 0.03;
        out[base + 2] = 0.3 - t * 0.02;
        out[base + 3] = 0.5 - t * 0.01;
        out[base + 4] = 1.2 + t * 0.02;
        out[base + 5] = -0.7 + t * 0.05;
    }
    out
}

/// Generates Mat3x2f flattened inputs.
fn mat3x2_inputs() -> [f32; 384] {
    let mut out = [0.0f32; 384];
    for i in 0..N {
        let t = i as f32 * 0.18 - 3.0;
        let base = i * 6;
        out[base] = 1.0 + t * 0.03;
        out[base + 1] = -0.1 + t * 0.02;
        out[base + 2] = 0.4 - t * 0.04;
        out[base + 3] = 0.8 + t * 0.01;
        out[base + 4] = -0.5 + t * 0.05;
        out[base + 5] = 1.1 - t * 0.03;
    }
    out
}

/// Generates Mat3x4f flattened inputs.
fn mat3x4_inputs() -> [f32; 768] {
    let mut out = [0.0f32; 768];
    for i in 0..N {
        let t = i as f32 * 0.12 - 2.0;
        let base = i * 12;
        out[base] = 1.0 + t * 0.02;
        out[base + 1] = -0.2 + t * 0.01;
        out[base + 2] = 0.3 - t * 0.02;
        out[base + 3] = 0.4 + t * 0.03;
        out[base + 4] = 0.0 + t * 0.02;
        out[base + 5] = 1.1 + t * 0.04;
        out[base + 6] = -0.3 + t * 0.03;
        out[base + 7] = 0.2 - t * 0.01;
        out[base + 8] = -0.5 + t * 0.02;
        out[base + 9] = 0.7 - t * 0.03;
        out[base + 10] = 0.9 + t * 0.01;
        out[base + 11] = 1.3 + t * 0.02;
    }
    out
}

/// Generates Mat4x3f flattened inputs.
fn mat4x3_inputs() -> [f32; 768] {
    let mut out = [0.0f32; 768];
    for i in 0..N {
        let t = i as f32 * 0.14 - 2.5;
        let base = i * 12;
        out[base] = 1.0 + t * 0.02;
        out[base + 1] = 0.1 - t * 0.03;
        out[base + 2] = -0.3 + t * 0.01;
        out[base + 3] = 0.2 + t * 0.04;
        out[base + 4] = 1.0 + t * 0.02;
        out[base + 5] = 0.4 - t * 0.02;
        out[base + 6] = -0.1 + t * 0.03;
        out[base + 7] = 0.7 + t * 0.01;
        out[base + 8] = 0.8 - t * 0.02;
        out[base + 9] = -0.6 + t * 0.02;
        out[base + 10] = 0.3 + t * 0.03;
        out[base + 11] = 1.2 - t * 0.01;
    }
    out
}

/// Generates Mat2x4f flattened inputs.
fn mat2x4_inputs() -> [f32; 512] {
    let mut out = [0.0f32; 512];
    for i in 0..N {
        let t = i as f32 * 0.15 - 2.0;
        let base = i * 8;
        out[base] = 1.0 + t * 0.02;
        out[base + 1] = -0.2 + t * 0.01;
        out[base + 2] = 0.3 - t * 0.02;
        out[base + 3] = 0.4 + t * 0.01;
        out[base + 4] = 0.0 + t * 0.02;
        out[base + 5] = 1.1 - t * 0.03;
        out[base + 6] = -0.6 + t * 0.02;
        out[base + 7] = 0.9 + t * 0.01;
    }
    out
}

/// Generates Mat4x2f flattened inputs.
fn mat4x2_inputs() -> [f32; 512] {
    let mut out = [0.0f32; 512];
    for i in 0..N {
        let t = i as f32 * 0.16 - 2.8;
        let base = i * 8;
        out[base] = 1.0 + t * 0.03;
        out[base + 1] = -0.1 + t * 0.02;
        out[base + 2] = 0.2 - t * 0.03;
        out[base + 3] = 1.2 + t * 0.02;
        out[base + 4] = -0.4 + t * 0.01;
        out[base + 5] = 0.7 - t * 0.02;
        out[base + 6] = 0.6 + t * 0.03;
        out[base + 7] = -0.5 + t * 0.02;
    }
    out
}

/// Generates inputs for Mat2x2f * Vec2f.
fn mat2_vec2_inputs() -> [f32; 384] {
    let mats = mat2_inputs();
    let mut out = [0.0f32; 384];
    for i in 0..N {
        let m_base = i * 4;
        let base = i * 6;
        out[base] = mats[m_base];
        out[base + 1] = mats[m_base + 1];
        out[base + 2] = mats[m_base + 2];
        out[base + 3] = mats[m_base + 3];
        let t = i as f32 * 0.2 - 5.0;
        out[base + 4] = 0.8 + t * 0.07;
        out[base + 5] = -0.6 + t * 0.05;
    }
    out
}

/// Generates inputs for Mat3x3f * Vec3f.
fn mat3_vec3_inputs() -> [f32; 768] {
    let mats = mat3_inputs();
    let mut out = [0.0f32; 768];
    for i in 0..N {
        let m_base = i * 9;
        let base = i * 12;
        out[base..base + 9].copy_from_slice(&mats[m_base..m_base + 9]);
        let t = i as f32 * 0.14 - 4.0;
        out[base + 9] = 0.5 + t * 0.04;
        out[base + 10] = -0.7 + t * 0.03;
        out[base + 11] = 1.1 - t * 0.02;
    }
    out
}

/// Generates inputs for Mat4x4f * Vec4f.
fn mat4_vec4_inputs() -> [f32; 1280] {
    let mats = mat4_inputs();
    let mut out = [0.0f32; 1280];
    for i in 0..N {
        let m_base = i * 16;
        let base = i * 20;
        out[base..base + 16].copy_from_slice(&mats[m_base..m_base + 16]);
        let t = i as f32 * 0.11 - 3.0;
        out[base + 16] = 0.3 + t * 0.03;
        out[base + 17] = -0.4 + t * 0.02;
        out[base + 18] = 0.9 - t * 0.01;
        out[base + 19] = 1.2 + t * 0.02;
    }
    out
}

/// Generates inputs for Mat2x2f * Mat2x2f.
fn mat2_mat2_inputs() -> [f32; 512] {
    let mut out = [0.0f32; 512];
    for i in 0..N {
        let t = i as f32 * 0.13 - 3.0;
        let base = i * 8;
        out[base] = 1.0 + t * 0.02;
        out[base + 1] = 0.2 - t * 0.03;
        out[base + 2] = -0.1 + t * 0.01;
        out[base + 3] = 0.9 + t * 0.02;
        out[base + 4] = 0.7 - t * 0.02;
        out[base + 5] = 0.1 + t * 0.01;
        out[base + 6] = -0.3 + t * 0.02;
        out[base + 7] = 1.1 - t * 0.01;
    }
    out
}

/// Generates inputs for Mat3x3f * Mat3x3f.
fn mat3_mat3_inputs() -> [f32; 1152] {
    let mut out = [0.0f32; 1152];
    for i in 0..N {
        let t = i as f32 * 0.1 - 2.5;
        let base = i * 18;
        out[base] = 1.0 + t * 0.03;
        out[base + 1] = -0.1 + t * 0.02;
        out[base + 2] = 0.2 - t * 0.01;
        out[base + 3] = 0.3 + t * 0.01;
        out[base + 4] = 0.9 + t * 0.03;
        out[base + 5] = -0.4 + t * 0.02;
        out[base + 6] = -0.2 + t * 0.02;
        out[base + 7] = 0.5 - t * 0.01;
        out[base + 8] = 1.1 + t * 0.02;

        out[base + 9] = 0.8 - t * 0.02;
        out[base + 10] = 0.2 + t * 0.03;
        out[base + 11] = -0.3 + t * 0.01;
        out[base + 12] = 0.1 + t * 0.02;
        out[base + 13] = 1.0 - t * 0.01;
        out[base + 14] = 0.4 + t * 0.03;
        out[base + 15] = -0.5 + t * 0.02;
        out[base + 16] = 0.6 - t * 0.03;
        out[base + 17] = 0.9 + t * 0.01;
    }
    out
}

/// Generates inputs for Mat4x4f * Mat4x4f.
fn mat4_mat4_inputs() -> [f32; 2048] {
    let mut out = [0.0f32; 2048];
    for i in 0..N {
        let t = i as f32 * 0.08 - 2.0;
        let base = i * 32;
        for j in 0..16 {
            out[base + j] = if j % 5 == 0 {
                1.0 + t * 0.02
            } else {
                (j as f32 * 0.07) - t * 0.01
            };
            out[base + 16 + j] = if j % 5 == 0 {
                0.9 - t * 0.02
            } else {
                (j as f32 * -0.05) + t * 0.015
            };
        }
    }
    out
}

/// Runs one f32-based compute shader on the GPU and returns unpacked f32
/// output.
fn run_gpu_f32_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_source: &str,
    input: &[f32],
    output_len: usize,
) -> Vec<f32> {
    let input_bytes = bytemuck::cast_slice(input);
    let output_size = (output_len * std::mem::size_of::<f32>()) as u64;
    let gpu_bytes = harness::run_gpu_compute(&harness::GpuComputeParams {
        device,
        queue,
        shader_source,
        entry_point: "main",
        bind_group_layout_entries: &harness::STANDARD_LAYOUT_ENTRIES,
        input_data: input_bytes,
        output_size,
        workgroup_count: (1, 1, 1),
    });
    bytemuck::cast_slice(&gpu_bytes).to_vec()
}

/// Pushes one f32 comparison result with generated element labels.
fn push_f32_result(
    results: &mut Vec<ComparisonResult>,
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    epsilon: f32,
) {
    let labels: Vec<String> = (0..gpu.len()).map(|i| format!("{name}[{i}]")).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    results.push(harness::compare_f32_results(
        name,
        gpu,
        cpu,
        &label_refs,
        epsilon,
    ));
}

pub struct MatrixOperationsTest;

impl RoundtripTest for MatrixOperationsTest {
    fn name(&self) -> &str {
        "matrix_operations"
    }

    fn description(&self) -> &str {
        "determinant, transpose, matrix*vector, matrix*matrix"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        {
            let input = mat2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                determinant_mat2::linkage::shader_source(),
                &input,
                64,
            );
            determinant_mat2::INPUT.set(input);
            determinant_mat2::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                determinant_mat2::main(b.global_invocation_id)
            });
            let cpu = determinant_mat2::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "determinant_mat2", &gpu, &cpu, 1e-5);
        }

        {
            let input = mat3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                determinant_mat3::linkage::shader_source(),
                &input,
                64,
            );
            determinant_mat3::INPUT.set(input);
            determinant_mat3::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                determinant_mat3::main(b.global_invocation_id)
            });
            let cpu = determinant_mat3::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "determinant_mat3", &gpu, &cpu, 1e-4);
        }

        {
            let input = mat4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                determinant_mat4::linkage::shader_source(),
                &input,
                64,
            );
            determinant_mat4::INPUT.set(input);
            determinant_mat4::OUTPUT.set([0.0f32; 64]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                determinant_mat4::main(b.global_invocation_id)
            });
            let cpu = determinant_mat4::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "determinant_mat4", &gpu, &cpu, 1e-3);
        }

        {
            let input = mat2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat2x2::linkage::shader_source(),
                &input,
                256,
            );
            transpose_mat2x2::INPUT.set(input);
            transpose_mat2x2::OUTPUT.set([0.0f32; 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat2x2::main(b.global_invocation_id)
            });
            let cpu = transpose_mat2x2::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat2x2", &gpu, &cpu, 0.0);
        }

        {
            let input = mat3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat3x3::linkage::shader_source(),
                &input,
                576,
            );
            transpose_mat3x3::INPUT.set(input);
            transpose_mat3x3::OUTPUT.set([0.0f32; 576]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat3x3::main(b.global_invocation_id)
            });
            let cpu = transpose_mat3x3::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat3x3", &gpu, &cpu, 0.0);
        }

        {
            let input = mat4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat4x4::linkage::shader_source(),
                &input,
                1024,
            );
            transpose_mat4x4::INPUT.set(input);
            transpose_mat4x4::OUTPUT.set([0.0f32; 1024]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat4x4::main(b.global_invocation_id)
            });
            let cpu = transpose_mat4x4::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat4x4", &gpu, &cpu, 0.0);
        }

        {
            let input = mat2x3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat2x3::linkage::shader_source(),
                &input,
                384,
            );
            transpose_mat2x3::INPUT.set(input);
            transpose_mat2x3::OUTPUT.set([0.0f32; 384]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat2x3::main(b.global_invocation_id)
            });
            let cpu = transpose_mat2x3::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat2x3", &gpu, &cpu, 0.0);
        }

        {
            let input = mat3x2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat3x2::linkage::shader_source(),
                &input,
                384,
            );
            transpose_mat3x2::INPUT.set(input);
            transpose_mat3x2::OUTPUT.set([0.0f32; 384]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat3x2::main(b.global_invocation_id)
            });
            let cpu = transpose_mat3x2::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat3x2", &gpu, &cpu, 0.0);
        }

        {
            let input = mat3x4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat3x4::linkage::shader_source(),
                &input,
                768,
            );
            transpose_mat3x4::INPUT.set(input);
            transpose_mat3x4::OUTPUT.set([0.0f32; 768]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat3x4::main(b.global_invocation_id)
            });
            let cpu = transpose_mat3x4::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat3x4", &gpu, &cpu, 0.0);
        }

        {
            let input = mat4x3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat4x3::linkage::shader_source(),
                &input,
                768,
            );
            transpose_mat4x3::INPUT.set(input);
            transpose_mat4x3::OUTPUT.set([0.0f32; 768]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat4x3::main(b.global_invocation_id)
            });
            let cpu = transpose_mat4x3::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat4x3", &gpu, &cpu, 0.0);
        }

        {
            let input = mat2x4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat2x4::linkage::shader_source(),
                &input,
                512,
            );
            transpose_mat2x4::INPUT.set(input);
            transpose_mat2x4::OUTPUT.set([0.0f32; 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat2x4::main(b.global_invocation_id)
            });
            let cpu = transpose_mat2x4::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat2x4", &gpu, &cpu, 0.0);
        }

        {
            let input = mat4x2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                transpose_mat4x2::linkage::shader_source(),
                &input,
                512,
            );
            transpose_mat4x2::INPUT.set(input);
            transpose_mat4x2::OUTPUT.set([0.0f32; 512]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                transpose_mat4x2::main(b.global_invocation_id)
            });
            let cpu = transpose_mat4x2::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "transpose_mat4x2", &gpu, &cpu, 0.0);
        }

        {
            let input = mat2_vec2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat2_vec2_mul::linkage::shader_source(),
                &input,
                128,
            );
            mat2_vec2_mul::INPUT.set(input);
            mat2_vec2_mul::OUTPUT.set([0.0f32; 128]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat2_vec2_mul::main(b.global_invocation_id)
            });
            let cpu = mat2_vec2_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat2_vec2_mul", &gpu, &cpu, 1e-5);
        }

        {
            let input = mat3_vec3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat3_vec3_mul::linkage::shader_source(),
                &input,
                192,
            );
            mat3_vec3_mul::INPUT.set(input);
            mat3_vec3_mul::OUTPUT.set([0.0f32; 192]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat3_vec3_mul::main(b.global_invocation_id)
            });
            let cpu = mat3_vec3_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat3_vec3_mul", &gpu, &cpu, 1e-5);
        }

        {
            let input = mat4_vec4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat4_vec4_mul::linkage::shader_source(),
                &input,
                256,
            );
            mat4_vec4_mul::INPUT.set(input);
            mat4_vec4_mul::OUTPUT.set([0.0f32; 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat4_vec4_mul::main(b.global_invocation_id)
            });
            let cpu = mat4_vec4_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat4_vec4_mul", &gpu, &cpu, 1e-4);
        }

        {
            let input = mat2_mat2_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat2_mat2_mul::linkage::shader_source(),
                &input,
                256,
            );
            mat2_mat2_mul::INPUT.set(input);
            mat2_mat2_mul::OUTPUT.set([0.0f32; 256]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat2_mat2_mul::main(b.global_invocation_id)
            });
            let cpu = mat2_mat2_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat2_mat2_mul", &gpu, &cpu, 1e-5);
        }

        {
            let input = mat3_mat3_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat3_mat3_mul::linkage::shader_source(),
                &input,
                576,
            );
            mat3_mat3_mul::INPUT.set(input);
            mat3_mat3_mul::OUTPUT.set([0.0f32; 576]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat3_mat3_mul::main(b.global_invocation_id)
            });
            let cpu = mat3_mat3_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat3_mat3_mul", &gpu, &cpu, 1e-4);
        }

        {
            let input = mat4_mat4_inputs();
            let gpu = run_gpu_f32_shader(
                device,
                queue,
                mat4_mat4_mul::linkage::shader_source(),
                &input,
                1024,
            );
            mat4_mat4_mul::INPUT.set(input);
            mat4_mat4_mul::OUTPUT.set([0.0f32; 1024]);
            dispatch_workgroups((1, 1, 1), (N as u32, 1, 1), |b| {
                mat4_mat4_mul::main(b.global_invocation_id)
            });
            let cpu = mat4_mat4_mul::OUTPUT.get().to_vec();
            push_f32_result(&mut results, "mat4_mat4_mul", &gpu, &cpu, 1e-4);
        }

        results
    }
}
