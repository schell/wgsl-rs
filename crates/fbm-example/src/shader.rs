//! Fractal Brownian Motion shader.
//!
//! A port of the classic FBM shader by Patricio Gonzalez Vivo (2015)
//! from GLSL to wgsl-rs. The shader generates animated, organic-looking
//! noise patterns using layered (octave) noise with domain warping.

#![allow(dead_code, clippy::excessive_precision)]

use wgsl_rs::wgsl;

#[wgsl]
pub mod fbm_shader {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U_RESOLUTION: Vec2f);
    uniform!(group(0), binding(1), U_MOUSE: Vec2f);
    uniform!(group(0), binding(2), U_TIME: f32);

    /// Pseudo-random number generator based on a 2D input.
    pub fn random(st: Vec2f) -> f32 {
        fract(sin(dot(st, vec2f(12.9898, 78.233))) * 43758.5453123)
    }

    /// Value noise based on Morgan McGuire's implementation.
    ///
    /// Interpolates random values at integer grid points using a
    /// smooth Hermite curve.
    pub fn noise(st: Vec2f) -> f32 {
        let i = floor(st);
        let f = fract(st);

        // Four corners in 2D of a tile
        let a = random(i);
        let b = random(i + vec2f(1.0, 0.0));
        let c = random(i + vec2f(0.0, 1.0));
        let d = random(i + vec2f(1.0, 1.0));

        // Smooth interpolation using Hermite curve: 3f^2 - 2f^3
        let u = f * f * (vec2f(3.0, 3.0) - 2.0 * f);

        mix(a, b, u.x()) + (c - a) * u.y() * (1.0 - u.x()) + (d - b) * u.x() * u.y()
    }

    /// Fractal Brownian Motion.
    ///
    /// Sums multiple octaves of noise, each at higher frequency and
    /// lower amplitude, with a rotation to reduce axial bias.
    pub fn fbm(st_in: Vec2f) -> f32 {
        let mut st = st_in;
        let mut v = 0.0;
        let mut a = 0.5;
        let shift = vec2f(100.0, 100.0);
        // Rotation matrix to reduce axial bias
        let cos_val = cos(0.5);
        let sin_val = sin(0.5);
        let rot = mat2x2f(vec2f(cos_val, sin_val), vec2f(-sin_val, cos_val));
        for _i in 0..5 {
            v += a * noise(st);
            st = rot * st * 2.0 + shift;
            a *= 0.5;
        }
        v
    }

    /// Full-screen triangle vertex shader.
    ///
    /// Emits a single triangle that covers the entire clip space using
    /// only the vertex index (no vertex buffer needed).
    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)];
        let p = POS[vertex_index as usize];
        vec4f(p.x(), p.y(), 0.0, 1.0)
    }

    /// Fragment input providing the built-in position.
    #[input]
    pub struct FragIn {
        #[builtin(position)]
        pub frag_coord: Vec4f,
    }

    /// Fragment shader producing FBM-based domain-warped color.
    ///
    /// Uses two layers of domain warping (q and r offsets fed back
    /// into fbm) to produce organic, flowing patterns animated over
    /// time.
    #[fragment]
    pub fn frag_main(input: FragIn) -> Vec4f {
        // Load uniforms into local values.
        // The f32() / vec2f() wrappers bridge the Rust ModuleVarReadGuard
        // to plain values; in WGSL these become identity type constructors.
        let res = vec2f(get!(U_RESOLUTION).x(), get!(U_RESOLUTION).y());
        let time = f32(get!(U_TIME));
        let st = input.frag_coord.xy() / res * 3.0;

        let qx = fbm(st + 0.00 * time);
        let qy = fbm(st + vec2f(1.0, 1.0));
        let q = vec2f(qx, qy);

        let rx = fbm(st + 1.0 * q + vec2f(1.7, 9.2) + 0.15 * time);
        let ry = fbm(st + 1.0 * q + vec2f(8.3, 2.8) + 0.126 * time);
        let r = vec2f(rx, ry);

        let f = fbm(st + r);

        // Color blending — mix requires all three args to be the same type,
        // so we splat the scalar blend factor into a Vec3f.
        let blend1 = clamp(f * f * 4.0, 0.0, 1.0);
        let mut color = mix(
            vec3f(0.101961, 0.619608, 0.666667),
            vec3f(0.666667, 0.666667, 0.498039),
            vec3f(blend1, blend1, blend1),
        );

        let blend2 = clamp(length(q), 0.0, 1.0);
        color = mix(
            color,
            vec3f(0.0, 0.0, 0.164706),
            vec3f(blend2, blend2, blend2),
        );

        let blend3 = clamp(abs(r.x()), 0.0, 1.0);
        color = mix(
            color,
            vec3f(0.666667, 1.0, 1.0),
            vec3f(blend3, blend3, blend3),
        );

        let intensity = f * f * f + 0.6 * f * f + 0.5 * f;
        vec4f(
            intensity * color.x(),
            intensity * color.y(),
            intensity * color.z(),
            1.0,
        )
    }
}
