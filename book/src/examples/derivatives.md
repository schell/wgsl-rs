# Derivatives

Demonstrates all 9 WGSL derivative builtin functions used in a fragment shader: `dpdx`, `dpdy`, `fwidth`, plus their `_fine` and `_coarse` variants, applied to both scalars and vectors.

## Rust Source

```rust
#[wgsl]
pub mod derivative_example {
    //! Demonstrates all 9 WGSL derivative builtin functions used in a fragment
    //! shader.

    use wgsl_rs::std::*;

    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    pub struct DerivativeOutputs {
        #[location(0)]
        pub dx: Vec4f,
        #[location(1)]
        pub dy: Vec4f,
        #[location(2)]
        pub fw: Vec4f,
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> DerivativeOutputs {
        let position = input.position;

        // Scalar derivatives.
        let dx_scalar = dpdx(position.x());
        let dy_scalar = dpdy(position.y());
        let fw_scalar = fwidth(position.x());

        // Fine variants on a Vec2f.
        let pos_xy = vec2f(position.x(), position.y());
        let dx_fine = dpdx_fine(pos_xy);
        let dy_fine = dpdy_fine(pos_xy);
        let fw_fine = fwidth_fine(pos_xy);

        // Coarse variants on a scalar.
        let dx_coarse = dpdx_coarse(position.x());
        let dy_coarse = dpdy_coarse(position.y());
        let fw_coarse = fwidth_coarse(position.x());

        DerivativeOutputs {
            dx: vec4f(dx_scalar, dx_fine.x(), dx_fine.y(), dx_coarse),
            dy: vec4f(dy_scalar, dy_fine.x(), dy_fine.y(), dy_coarse),
            fw: vec4f(fw_scalar, fw_fine.x(), fw_fine.y(), fw_coarse),
        }
    }
}
```

## Generated WGSL

```wgsl
struct FragInput {
    @builtin(position) position: vec4f
}

struct DerivativeOutputs {
    @location(0) dx: vec4f,
    @location(1) dy: vec4f,
    @location(2) fw: vec4f
}

@fragment fn frag_main(input: FragInput) -> DerivativeOutputs {
    let position = input.position;
    let dx_scalar = dpdx(position.x);
    let dy_scalar = dpdy(position.y);
    let fw_scalar = fwidth(position.x);
    let pos_xy = vec2f(position.x, position.y);
    let dx_fine = dpdxFine(pos_xy);
    let dy_fine = dpdyFine(pos_xy);
    let fw_fine = fwidthFine(pos_xy);
    let dx_coarse = dpdxCoarse(position.x);
    let dy_coarse = dpdyCoarse(position.y);
    let fw_coarse = fwidthCoarse(position.x);
    return DerivativeOutputs(vec4f(dx_scalar, dx_fine.x, dx_fine.y, dx_coarse), vec4f(dy_scalar, dy_fine.x, dy_fine.y, dy_coarse), vec4f(fw_scalar, fw_fine.x, fw_fine.y, fw_coarse));
}
```

## Notes

- The nine functions are: `dpdx`, `dpdy`, `fwidth`, `dpdx_fine`, `dpdy_fine`, `fwidth_fine`, `dpdx_coarse`, `dpdy_coarse`, `fwidth_coarse`.
- Snake_case Rust names map to camelCase WGSL builtins (e.g. `dpdx_fine` -> `dpdxFine`).