# Inter-stage IO

Vertex outputs and fragment inputs are connected by passing data through a struct whose fields carry WGSL IO attributes. wgsl-rs mirrors the WGSL pattern directly: attributes go on **struct fields**, and the same struct can serve as both a vertex return type and a fragment parameter.

## IO Attributes

| Attribute                    | Maps to                 | Applies to                       |
| ---------------------------- | ----------------------- | -------------------------------- |
| `#[builtin(NAME)]`           | `@builtin(NAME)`        | field                            |
| `#[location(N)]`             | `@location(N)`          | field                            |
| `#[interpolate(TYPE)]`       | `@interpolate(TYPE)`    | field (fragment-stage input)     |
| `#[interpolate(TYPE, SAMP)]` | `@interpolate(TYPE, SAMP)` | field                         |
| `#[blend_src(N)]`            | `@blend_src(N)`         | field (dual-source blending)     |
| `#[invariant]`               | `@invariant`            | field (position)                 |

### Interpolation

`#[interpolate(...)]` accepts a type and an optional sampling qualifier:

```rust
#[interpolate(flat)]
#[interpolate(linear)]
#[interpolate(perspective)]
#[interpolate(perspective, centroid)]
#[interpolate(perspective, sample)]
```

The default when `#[interpolate]` is omitted is `@interpolate(perspective)` with the default sampling, matching WGSL.

## Shared Inter-stage Struct

The idiomatic pattern is a single struct used as both the vertex output and the fragment input — the `shared_inter_stage` example:

```rust
#[wgsl]
pub mod shared_inter_stage {
    use wgsl_rs::std::*;

    pub struct VertexOutput {
        #[builtin(position)]
        pub clip_position: Vec4f,
        #[location(0)]
        pub color: Vec4f,
    }

    #[vertex]
    pub fn vs_main(#[builtin(vertex_index)] vertex_index: u32) -> VertexOutput {
        const POS: [Vec2f; 3] = [
            vec2f(0.0, 0.5),
            vec2f(-0.5, -0.5),
            vec2f(0.5, -0.5),
        ];
        let position = POS[vertex_index as usize];
        VertexOutput {
            clip_position: vec4f(position.x, position.y, 0.0, 1.0),
            color: vec4f(1.0, 0.0, 0.0, 1.0),
        }
    }

    #[fragment]
    pub fn fs_main(input: VertexOutput) -> Vec4f {
        input.color
    }
}
```

```wgsl
struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  /* ... */
  return VertexOutput(/* ... */);
}

@fragment
fn fs_main(input: VertexOutput) -> vec4<f32> {
  return input.color;
}
```

There is no separate attribute on the struct itself — the field-level attributes do all the work, exactly as in WGSL.

## IO Attributes are Stripped from Rust

The `#[wgsl]` macro strips `#[builtin]`, `#[location]`, `#[interpolate]`, `#[blend_src]`, and `#[invariant]` from the **emitted Rust** so the module remains valid Rust without needing wrapper attributes. You do not need to gate these annotations behind a cfg or feature; the macro removes them before the Rust compiler sees the post-expansion module.

> This means `VertexOutput` is a plain `#[derive(Wgsl)]` struct on the CPU side, and the same field list becomes a fully attributed WGSL struct on the GPU side.

## Supported Builtins

wgsl-rs recognizes the following builtin names inside `#[builtin(...)]`:

| Vertex input         | Vertex output         | Fragment input         | Fragment output      | Compute input             |
| -------------------- | --------------------- | ---------------------- | -------------------- | ------------------------- |
| `vertex_index`       | `position`            | `position`             | `frag_depth`         | `local_invocation_id`     |
| `instance_index`     |                       | `front_facing`         | `sample_mask`        | `local_invocation_index`  |
|                      |                       | `sample_index`         |                      | `global_invocation_id`    |
|                      |                       | `sample_mask`          |                      | `workgroup_id`            |
|                      |                       | `primitive_index`      |                      | `num_workgroups`          |
|                      |                       |                        |                      | `subgroup_invocation_id`  |
|                      |                       |                        |                      | `subgroup_size`           |
|                      |                       |                        |                      | `subgroup_id`             |
|                      |                       |                        |                      | `num_subgroups`           |

`position` may additionally carry `#[invariant]` on the vertex output to force invariant interpolation.

## Mixing Builtins and Locations

A struct may mix builtins and locations freely:

```rust
pub struct VertexOutput {
    #[builtin(position)]
    #[invariant]
    pub clip_position: Vec4f,
    #[location(0)]
    pub color: Vec4f,
    #[location(1)]
    #[interpolate(flat)]
    pub material_id: u32,
}
```