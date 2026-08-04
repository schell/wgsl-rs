# Structs

Demonstrates user-defined structs used as fragment shader inputs and outputs, mixing `#[location]` and `#[builtin]` attributes, plus `#[interpolate]`.

## Rust Source

```rust
#[wgsl]
pub mod structs {
    use wgsl_rs::std::*;

    // Mixed builtins and user-defined inputs.
    pub struct MyInputs {
        #[location(0)]
        pub x: Vec4<f32>,

        #[builtin(front_facing)]
        pub y: bool,

        #[location(1)]
        #[interpolate(flat)]
        pub z: u32,

        #[location(2)]
        pub other: f32,
    }

    pub struct MyOutputs {
        #[location(0)]
        pub x: f32,

        #[location(1)]
        pub y: Vec4<f32>,
    }

    #[fragment]
    pub fn frag_shader(in1: MyInputs) -> MyOutputs {
        MyOutputs { x: 0.0, y: in1.x }
    }
}
```

## Generated WGSL

```wgsl
struct MyInputs {
    @location(0) x: vec4f,
    @builtin(front_facing) y: bool,
    @location(1) @interpolate(flat) z: u32,
    @location(2) other: f32
}

struct MyOutputs {
    @location(0) x: f32,
    @location(1) y: vec4f
}

@fragment fn frag_shader(in1: MyInputs) -> MyOutputs {
    return MyOutputs(0.0, in1.x);
}
```

## Notes

- Struct fields may carry `#[location(...)]`, `#[builtin(...)]`, and `#[interpolate(...)]` attributes that map directly to WGSL decorations.
- Struct literals in Rust (`MyOutputs { x: 0.0, y: in1.x }`) become positional constructor calls in WGSL.