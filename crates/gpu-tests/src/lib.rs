//! GPU comparison test shader modules and helpers.
//!
//! Contains WGSL shader modules used by integration tests that compare
//! CPU-side `dispatch_fragments` results with real GPU fragment shader
//! execution.

use wgsl_rs::wgsl;

/// Shader that computes `dpdx(position.x)` and `dpdy(position.y)` and packs
/// them into the output color channels.
///
/// For a viewport where position increases by 1.0 per pixel:
/// - `dpdx(position.x)` should be 1.0 everywhere
/// - `dpdy(position.y)` should be 1.0 everywhere
/// - `fwidth(position.x)` = `|dpdx(position.x)| + |dpdy(position.x)|` = 1.0 +
///   0.0 = 1.0
/// - `fwidth(position.y)` = `|dpdx(position.y)| + |dpdy(position.y)|` = 0.0 +
///   1.0 = 1.0
///
/// Output channels:
/// - R = `dpdx(position.x)`
/// - G = `dpdy(position.y)`
/// - B = `fwidth(position.x)`
/// - A = `fwidth(position.y)`
#[wgsl]
pub mod derivative_shader {
    use wgsl_rs::std::*;

    #[input]
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        // Full-screen triangle: 3 vertices covering the entire clip space.
        // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> Vec4f {
        let position = input.position;
        let dx = dpdx(position.x);
        let dy = dpdy(position.y);
        let fwx = fwidth(position.x);
        let fwy = fwidth(position.y);
        vec4f(dx, dy, fwx, fwy)
    }
}

/// Shader that computes fine and coarse derivative variants and packs the
/// results into two render targets.
///
/// Target 0 (fine):   R=dpdx_fine(pos.x), G=dpdy_fine(pos.y),
/// B=fwidth_fine(pos.x), A=fwidth_fine(pos.y) Target 1 (coarse):
/// R=dpdx_coarse(pos.x), G=dpdy_coarse(pos.y), B=fwidth_coarse(pos.x),
/// A=fwidth_coarse(pos.y)
#[wgsl]
pub mod derivative_variants_shader {
    use wgsl_rs::std::*;

    #[input]
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[output]
    pub struct DerivativeVariantOutputs {
        #[location(0)]
        pub fine: Vec4f,
        #[location(1)]
        pub coarse: Vec4f,
    }

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> DerivativeVariantOutputs {
        let position = input.position;
        let dx_fine = dpdx_fine(position.x);
        let dy_fine = dpdy_fine(position.y);
        let fwx_fine = fwidth_fine(position.x);
        let fwy_fine = fwidth_fine(position.y);

        let dx_coarse = dpdx_coarse(position.x);
        let dy_coarse = dpdy_coarse(position.y);
        let fwx_coarse = fwidth_coarse(position.x);
        let fwy_coarse = fwidth_coarse(position.y);

        DerivativeVariantOutputs {
            fine: vec4f(dx_fine, dy_fine, fwx_fine, fwy_fine),
            coarse: vec4f(dx_coarse, dy_coarse, fwx_coarse, fwy_coarse),
        }
    }
}
