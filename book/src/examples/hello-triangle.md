# Hello Triangle

A "hello world" shader that renders a triangle with a color that changes over time. It demonstrates vertex and fragment entry points, the `uniform!` macro, glob imports, and builtins like `vertex_index`.

## Rust Source

```rust
#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing
    //! color. Original source is [here](https://google.github.io/tour-of-wgsl/).

    // Only glob-imports are supported, but hey, imports work!
    use wgsl_rs::std::*;

    // Define a uniform in both Rust and WGSL using the uniform! macro.
    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];

        let position = POS[vertex_index as usize];
        vec4f(position.x, position.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(get!(FRAME)) / 128.0), 0.0, 1.0)
    }
}
```

## Generated WGSL

```wgsl
@group(0) @binding(0) var<uniform> FRAME: u32;

@vertex fn vtx_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    const POS: array<vec2f, 3> = array(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
    let position = POS[u32(vertex_index)];
    return vec4f(position.x, position.y, 0.0, 1.0);
}

@fragment fn frag_main() -> @location(0) vec4f {
    return vec4f(1.0, sin(f32(FRAME) / 128.0), 0.0, 1.0);
}
```

## Notes

- The `uniform!` macro declares a uniform variable available in both Rust and WGSL.
- `get!(FRAME)` reads the uniform; it is a no-op in WGSL and is stripped during parsing.
- `#[vertex]` and `#[fragment]` mark entry points.