# Default Annotations

To keep simple shaders short, wgsl-rs applies a few default WGSL I/O annotations when an entry point returns a bare vector without an explicit struct.

## Vertex Returning `Vec4f`

A vertex entry point that returns `Vec4f` directly (rather than a struct) is automatically annotated with `@builtin(position)` on the return value:

```rust
#[vertex]
pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
    vec4f(0.0, 0.0, 0.0, 1.0)
}
```

```wgsl
@vertex
fn vtx_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
```

This matches the most common vertex shader shape: produce a clip-space position and nothing else.

## Fragment Returning `Vec4f`

A fragment entry point that returns `Vec4f` directly is automatically annotated with `@location(0)`:

```rust
#[fragment]
pub fn frag_main() -> Vec4f {
    vec4f(1.0, 0.0, 0.0, 1.0)
}
```

```wgsl
@fragment
fn frag_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

This is the single-render-target case.

## When to Use Explicit Annotations

Use a struct return type instead of the bare defaults whenever you need to emit more than one output value:

- A vertex shader that also writes inter-stage varyings (color, UV, normal, ...).
- A fragment shader writing multiple render targets (MRT) — use `#[location(N)]` per field.
- A vertex shader whose position should be `@invariant` — put `#[builtin(position)]` and `#[invariant]` on the field.
- Dual-source blending — use `#[blend_src(0)]` and `#[blend_src(1)]` on two fields.

For example, switching from the default to a struct:

```rust
pub struct VertexOutput {
    #[builtin(position)]
    pub clip_position: Vec4f,
    #[location(0)]
    pub color: Vec4f,
}

#[vertex]
pub fn vs_main(#[builtin(vertex_index)] vi: u32) -> VertexOutput {
    VertexOutput {
        clip_position: vec4f(0.0, 0.0, 0.0, 1.0),
        color: vec4f(1.0, 0.0, 0.0, 1.0),
    }
}
```

See [Inter-stage IO](./inter-stage-io.md) for the full set of field attributes.

## Compute Entry Points

Compute entry points return `()` (no value) and have no default annotations; their I/O is entirely through builtins on parameters and through binding macros. See [Vertex / Fragment / Compute](./stages.md).

## Custom IO via Struct Fields

For fragment inputs, accept a struct parameter whose fields mirror the vertex output struct. wgsl-rs allows the **same** struct to be used on both sides, which is the recommended shared-inter-stage pattern:

```rust
#[fragment]
pub fn fs_main(input: VertexOutput) -> Vec4f {
    input.color
}
```

The defaults apply only to bare `Vec4f` returns; once you return or accept a struct, every IO attribute must be explicit on its fields.