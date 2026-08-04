# `discard!()`

`discard!()` aborts the current fragment's output. In WGSL it emits the
`discard;` statement; on the CPU it sets a thread-local flag.

## Syntax

```rust
discard!();
```

It is a macro (not a function) because it must be recognized by the
transpiler as a control-flow side effect.

## Semantics

- **WGSL**: execution of the rest of the fragment invocation has undefined
  behavior; outputs (color, depth) are not committed. Helper-invocation
  semantics apply — derivatives and similar may still run.
- **CPU**: sets a thread-local "discarded" flag and returns. The CPU dispatch
  runtime checks this flag after the entry point returns and skips committing
  the fragment's output. Code after `discard!()` continues to execute unless
  you explicitly return; this mirrors WGSL's helper-invocation model, where
  the shader is not abruptly terminated.

## Reachability

`discard!()` can appear in any function reachable from a fragment entry
point. The transpiler tracks this through the call graph. Using it outside
fragment-reachable code is a compile error.

## Example

```rust
#[wgsl]
pub mod discard_example {
    use wgsl_rs::std::*;

    pub struct Material {
        pub alpha_cutoff: f32,
    }
    uniform!(MATERIAL, Material);
    texture!(ALBEDO_TEX, Texture2d);
    sampler!(ALBEDO_SMP, Sampler);

    #[fragment]
    pub fn fs(in: VertexOutput) -> Vec4f {
        let albedo = textureSample(ALBEDO_TEX, ALBEDO_SMP, in.uv);
        if albedo.w() < MATERIAL.alpha_cutoff {
            discard!();
        }
        albedo
    }
}
```

## Difference from an early return

`discard!()` is **not** a return. It marks the fragment for rejection but lets
subsequent statements run. If you want to stop executing immediately, combine
it with an explicit `return`:

```rust
if albedo.w() < cutoff {
    discard!();
    return vec4f(0.0);
}
```

On the CPU this keeps the thread-local flag set and the dispatch runtime will
ignore the returned value. On the GPU the WGSL `discard;` ensures the output
is not committed regardless of what the return produces.