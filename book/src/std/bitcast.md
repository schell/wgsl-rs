# Bitcast

`bitcast` reinterprets the bit pattern of a value as a different type without
changing any bits — unlike `f32 as u32`, which performs a numeric conversion.
WGSL's `bitcast` builtin takes a single argument and the result type is
inferred from context; wgsl-rs instead provides one named function per target
type so Rust type inference is unambiguous.

## Functions

Each function is named `bitcast_<targettype>`:

| Function | WGSL Equivalent | Input | Output |
|----------|-----------------|-------|--------|
| `bitcast_f32(e)` | `bitcast<f32>` | `i32` / `u32` | `f32` |
| `bitcast_i32(e)` | `bitcast<i32>` | `f32` / `u32` | `i32` |
| `bitcast_u32(e)` | `bitcast<u32>` | `f32` / `i32` | `u32` |
| `bitcast_vec2f(e)` | `bitcast<vec2f>` | `vec2i` / `vec2u` | `Vec2f` |
| `bitcast_vec2i(e)` | `bitcast<vec2i>` | `vec2f` / `vec2u` | `Vec2i` |
| `bitcast_vec2u(e)` | `bitcast<vec2u>` | `vec2f` / `vec2i` | `Vec2u` |
| `bitcast_vec4f(e)` | `bitcast<vec4f>` | `vec4i` / `vec4u` | `Vec4f` |
| `bitcast_vec4i(e)` | `bitcast<vec4i>` | `vec4f` / `vec4u` | `Vec4i` |
| `bitcast_vec4u(e)` | `bitcast<vec4u>` | `vec4f` / `vec4i` | `Vec4u` |

The set of accepted input types per target follows WGSL §17: the source and
target must have the same bit width, and only numeric scalar/vector types are
allowed (no `bool`).

## Why per-target functions

WGSL resolves `bitcast` overloading from the surrounding expression context,
which Rust cannot do without type annotations. Naming each target type makes
the intent explicit on the CPU side and keeps type inference deterministic.

## Example

```rust
#[wgsl]
pub mod bitcast_example {
    use wgsl_rs::std::*;

    pub fn pack_normal_as_u32(n: Vec3f) -> u32 {
        let q = vec4f(n.x() * 0.5 + 0.5,
                      n.y() * 0.5 + 0.5,
                      n.z() * 0.5 + 0.5,
                      0.0);
        bitcast_vec4u(q).x()
    }

    pub fn unpack_normal_from_u32(packed: u32) -> Vec3f {
        let q = bitcast_vec4f(vec4u(packed, 0, 0, 0));
        q.xyz() * 2.0 - 1.0
    }
}
```

## CPU behavior

On the CPU, these map to `f32::from_bits` / `f32::to_bits` (and the `Vec`
equivalents), so the bit pattern is preserved exactly. This is what makes
bitcast safe to use in roundtrip tests.