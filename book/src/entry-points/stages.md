# Vertex / Fragment / Compute

WGSL has three shader stages, each with its own entry-point attribute. wgsl-rs exposes them as Rust attributes that the macro translates to `@vertex`, `@fragment`, and `@compute`.

| Attribute     | WGSL        | Notes                                              |
| ------------- | ----------- | -------------------------------------------------- |
| `#[vertex]`   | `@vertex`   | One per pipeline's vertex stage.                   |
| `#[fragment]` | `@fragment` | One per pipeline's fragment stage.                 |
| `#[compute]`  | `@compute`  | Requires `#[workgroup_size(...)]`.                 |

A module may contain any combination of entry points. Functions without these attributes transpile to ordinary WGSL functions.

## Vertex

A vertex entry point takes per-vertex/per-instance inputs (builtins and location-tagged values) and returns a position. Returning a bare `Vec4f` is automatically annotated with `@builtin(position)`; see [Default Annotations](./default-annotations.md).

```rust
#[vertex]
pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
    const POS: [Vec2f; 3] = [
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5),
    ];
    let position = POS[vertex_index as usize];
    vec4f(position.x, position.y, 0.0, 1.0)
}
```

```wgsl
@vertex
fn vtx_main(@builtin(vertex_index) vertex_index: u32) -> vec4<f32> {
  /* ... */
}
```

## Fragment

A fragment entry point takes inter-stage inputs (locations and builtins such as `@builtin(front_facing)`) and returns a color. Returning a bare `Vec4f` is automatically annotated with `@location(0)`.

```rust
#[fragment]
pub fn frag_main() -> Vec4f {
    vec4f(1.0, 0.0, 0.0, 1.0)
}
```

```wgsl
@fragment
fn frag_main() -> vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

## Compute

A compute entry point must declare a workgroup size. Use a single integer for a 1D dispatch or three integers for a 3D dispatch:

```rust
#[compute]
#[workgroup_size(64)]
pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
    let idx = global_id.x() as usize;
    /* ... */
}

#[compute]
#[workgroup_size(8, 8, 1)]
pub fn tiled(#[builtin(global_invocation_id)] global_id: Vec3u) {
    let x = global_id.x();
    let y = global_id.y();
    /* ... */
}
```

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { /* ... */ }

@compute @workgroup_size(8, 8, 1)
fn tiled(@builtin(global_invocation_id) global_id: vec3<u32>) { /* ... */ }
```

Compute entry points frequently take no return value (a `()` return type) and access storage or workgroup resources via the binding macros (see [Binding Macros](../writing-shaders/binding-macros.md)).

## Inputs

Entry-point inputs are declared as ordinary function parameters. Each parameter may carry an I/O attribute:

- `#[builtin(NAME)]` — a WGSL builtin value.
- `#[location(N)]` — a per-vertex attribute (vertex stage) or inter-stage value (fragment stage).

For complex inter-stage IO, use a struct input/output; see [Inter-stage IO](./inter-stage-io.md).

## All Three Stages Together

A single module may declare all three stages. The example module below shows vertex, fragment, and compute entry points coexisting (see [Binding Macros](../writing-shaders/binding-macros.md) for `storage!` and `get_mut!`):

```rust
#[wgsl]
pub mod pipeline {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), read_write, COUNTER: u32);

    #[vertex]
    pub fn vs_main(#[builtin(vertex_index)] vi: u32) -> Vec4f {
        vec4f(0.0, 0.0, 0.0, 1.0)
    }

    #[fragment]
    pub fn fs_main() -> Vec4f {
        vec4f(1.0, 1.0, 1.0, 1.0)
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn cs_main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let i = gid.x() as usize;
        get_mut!(COUNTER);
    }
}
```