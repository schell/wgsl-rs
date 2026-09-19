# Discard

Demonstrates the `discard!()` statement for discarding fragments. The macro expands to the WGSL `discard` statement.

## Rust Source

```rust
#[wgsl]
pub mod discard_example {
    //! Demonstrates the `discard!()` statement for discarding fragments.

    use wgsl_rs::std::*;

    /// Discard fragments with shallow depth (close to the near plane).
    pub fn discard_if_shallow(pos: Vec4f) {
        if pos.z < 0.001 {
            discard!();
        }
    }

    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    #[fragment]
    pub fn frag_main(input: FragInput) -> Vec4f {
        discard_if_shallow(input.position);
        vec4f(1.0, 0.0, 0.0, 1.0)
    }
}
```

## Generated WGSL

```wgsl
fn discard_if_shallow(pos: vec4f) {
    if pos.z < 0.001 {
        discard;
    }
}

struct FragInput {
    @builtin(position) position: vec4f
}

@fragment fn frag_main(input: FragInput) -> @location(0) vec4f {
    discard_if_shallow(input.position);
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
```

## Notes

- `discard!()` is a macro that expands to the WGSL `discard` statement. It must be used inside a fragment shader.