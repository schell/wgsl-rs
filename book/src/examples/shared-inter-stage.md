# Shared Inter-Stage

Demonstrates a single struct shared between vertex and fragment stages. `VertexOutput` serves as both the vertex return type and the fragment input, with `#[builtin(position)]` and `#[location]` decorations.

## Rust Source

```rust
#[wgsl]
pub mod shared_inter_stage {
    use wgsl_rs::std::*;

    /// Vertex output / fragment input — a single struct shared across stages.
    pub struct VertexOutput {
        #[builtin(position)]
        pub clip_position: Vec4f,
        #[location(0)]
        pub color: Vec4f,
    }

    #[vertex]
    pub fn vs_main(#[builtin(vertex_index)] vertex_index: u32) -> VertexOutput {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];
        let position = POS[vertex_index as usize];
        VertexOutput {
            clip_position: vec4f(position.x(), position.y(), 0.0, 1.0),
            color: vec4f(1.0, 0.0, 0.0, 1.0),
        }
    }

    #[fragment]
    pub fn fs_main(input: VertexOutput) -> Vec4f {
        input.color
    }
}
```

## Generated WGSL

```wgsl
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f
}

@vertex fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const POS: array<vec2f, 3> = array(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
    let position = POS[u32(vertex_index)];
    return VertexOutput(vec4f(position.x, position.y, 0.0, 1.0), vec4f(1.0, 0.0, 0.0, 1.0));
}

@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
```

## Notes

- One struct can serve as both the vertex shader's output and the fragment shader's input, matching the WGSL convention where vertex output structs feed fragment input structs.
- Struct literals become positional constructor calls in WGSL.