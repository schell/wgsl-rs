# Packing

WGSL provides builtins to pack and unpack vectors of normalized or
floating-point values into a single `u32`. wgsl-rs exposes all of them as free
functions in `wgsl_rs::std`.

## Pack / unpack pairs

| Function | WGSL Equivalent | Description |
|----------|-----------------|-------------|
| `pack4x8snorm(v)` | `pack4x8snorm` | Pack 4× `f32` in `[-1,1]` into `u32`, 8 bits each, signed normalized. |
| `unpack4x8snorm(u)` | `unpack4x8snorm` | Inverse of `pack4x8snorm`. |
| `pack4x8unorm(v)` | `pack4x8unorm` | Pack 4× `f32` in `[0,1]` into `u32`, 8 bits each, unsigned normalized. |
| `unpack4x8unorm(u)` | `unpack4x8unorm` | Inverse of `pack4x8unorm`. |
| `pack2x16snorm(v)` | `pack2x16snorm` | Pack 2× `f32` in `[-1,1]` into `u32`, 16 bits each, signed normalized. |
| `unpack2x16snorm(u)` | `unpack2x16snorm` | Inverse of `pack2x16snorm`. |
| `pack2x16unorm(v)` | `pack2x16unorm` | Pack 2× `f32` in `[0,1]` into `u32`, 16 bits each, unsigned normalized. |
| `unpack2x16unorm(u)` | `unpack2x16unorm` | Inverse of `pack2x16unorm`. |
| `pack2x16float(v)` | `pack2x16float` | Pack 2× `f32` into `u32` as `f16` pairs. |
| `unpack2x16float(u)` | `unpack2x16float` | Inverse of `pack2x16float`. |

Inputs are `Vec4f` / `Vec2f` for the pack functions; outputs are `u32`. The
unpack functions take `u32` and return the corresponding vector.

## Rounding behavior

The pack functions round normalized floats to the nearest representable
integer using round-to-nearest-even, matching WGSL. Values outside the
normalized range are clamped.

## Example

```rust
#[wgsl]
pub mod packing_example {
    use wgsl_rs::std::*;

    pub fn encode_tangent(t: Vec4f) -> u32 {
        pack4x8snorm(t)
    }

    pub fn decode_tangent(packed: u32) -> Vec4f {
        unpack4x8snorm(packed)
    }
}
```

## CPU behavior

The CPU implementations use the same rounding and clamping as WGSL, so
`pack` followed by `unpack` returns a value within one ULP of the original.
This makes packing safe to exercise in unit tests.