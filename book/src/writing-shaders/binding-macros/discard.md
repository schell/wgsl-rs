# `discard!`

Discards the current fragment. In WGSL this emits `discard;`; on CPU it sets a thread-local flag.

## Syntax

```rust
discard!();
```

## Behavior

- In WGSL, transpiles to `discard;` and stops further output for the fragment.
- On CPU, sets a thread-local flag that `dispatch_fragments` checks; execution continues after the call, matching WGSL's helper-invocation semantics.

## Reachable From

`discard!()` may be called from any function reachable from a `#[fragment]` entry point, including helper functions.

## Example

```rust
#[wgsl]
pub mod alpha {
    use wgsl_rs::std::*;

    texture!(group(0), binding(0), MASK: Texture2D<f32>);
    sampler!(group(0), binding(1), LIN: Sampler);

    pub fn threshold(uv: Vec2f, min: f32) {
        let a = texture_sample(MASK, LIN, uv).x();
        if a < min {
            discard!();
        }
    }

    #[fragment]
    pub fn fs_main(#[location(0)] uv: Vec2f) -> Vec4f {
        threshold(uv, 0.5);
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}
```

## Notes

- Execution after `discard!()` continues on the CPU side, so code following the call still runs. Guard any side effects accordingly.
- `discard!()` is only valid inside fragment shaders. Calling it from a compute or vertex entry point is a validation error.