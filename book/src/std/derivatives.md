# Derivatives

Derivatives compute per-pixel rate-of-change of a value with respect to
screen-space coordinates. They are only valid inside fragment shaders; calling
them from any other stage is a WGSL error.

## Fine / coarse variants

WGSL provides three precision tiers. wgsl-rs exposes all of them:

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `dpdx(p)` | `dpdx` | Default precision `dpCoarse`. |
| `dpdy(p)` | `dpdy` | Default precision `dpCoarse`. |
| `fwidth(p)` | `fwidth` | `abs(dpdx) + abs(dpdy)`, default precision. |
| `dpdx_fine(p)` | `dpdxFine` | Fine precision `dpFine`. |
| `dpdy_fine(p)` | `dpdyFine` | Fine precision. |
| `fwidth_fine(p)` | `fwidthFine` | Fine precision. |
| `dpdx_coarse(p)` | `dpdxCoarse` | Coarse precision `dpCoarse`. |
| `dpdy_coarse(p)` | `dpdyCoarse` | Coarse precision. |
| `fwidth_coarse(p)` | `fwidthCoarse` | Coarse precision. |

The bare `dpdx`/`dpdy`/`fwidth` map to WGSL's default-precision builtins, which
WGSL defines as coarse. Prefer the explicit `_fine` / `_coarse` variants when
the choice matters for your application.

## Common uses

- **Mip selection in non-fragment-aware sampling**: pass gradients to
  `textureSampleGrad` from `dpdx`/`dpdy` of the texture coordinates.
- **Edge detection / anti-aliasing**: `fwidth` to compute pixel-local width.
- **Screen-space dependent branching**: compare `fwidth` against a threshold.

## Example

```rust
#[wgsl]
pub mod derivatives_example {
    use wgsl_rs::std::*;

    pub struct Frame {
        pub time: f32,
    }

    pub struct DerivativesExamplePack {
        #[uniform(group(0), binding(0))]
        pub frame: Uniform<Frame>,
        #[texture(group(0), binding(1))]
        pub color_tex: Texture2D<f32>,
        #[sampler(group(0), binding(2))]
        pub color_smp: Sampler,
    }

    #[fragment]
    pub fn fs(in: VertexOutput, pack: &DerivativesExamplePack) -> Vec4f {
        let uv = in.uv;
        let dx = dpdx_fine(uv);
        let dy = dpdy_fine(uv);
        let w = fwidth_fine(uv);
        let color = texture_sample_grad(&pack.color_tex, &pack.color_smp, uv, dx, dy);
        color
    }
}
```

## CPU behavior

On the CPU, derivatives return zero for `dpdx`/`dpdy` and the input value's
magnitude for `fwidth` — enough to keep CPU tests running without panic. The
GPU is the source of truth for derivative accuracy.