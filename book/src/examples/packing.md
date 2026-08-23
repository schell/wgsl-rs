# Packing

Demonstrates WGSL data packing and unpacking builtin functions, which convert between vector types and packed `u32` representations. Useful for vertex attribute compression and storage optimization.

## Rust Source

```rust
#[wgsl]
pub mod packing_example {
    //! Demonstrates WGSL data packing and unpacking builtin functions.
    //!
    //! These functions convert between vector types and packed `u32`
    //! representations, useful for vertex attribute compression and storage
    //! optimization.
    use wgsl_rs::std::*;

    pub fn demo_pack4x8snorm(v: Vec4f) -> u32 {
        pack4x8snorm(v)
    }

    pub fn demo_unpack4x8snorm(e: u32) -> Vec4f {
        unpack4x8snorm(e)
    }

    pub fn demo_pack4x8unorm(v: Vec4f) -> u32 {
        pack4x8unorm(v)
    }

    pub fn demo_unpack4x8unorm(e: u32) -> Vec4f {
        unpack4x8unorm(e)
    }

    pub fn demo_pack2x16snorm(v: Vec2f) -> u32 {
        pack2x16snorm(v)
    }

    pub fn demo_unpack2x16snorm(e: u32) -> Vec2f {
        unpack2x16snorm(e)
    }

    pub fn demo_pack2x16unorm(v: Vec2f) -> u32 {
        pack2x16unorm(v)
    }

    pub fn demo_unpack2x16unorm(e: u32) -> Vec2f {
        unpack2x16unorm(e)
    }

    pub fn demo_pack2x16float(v: Vec2f) -> u32 {
        pack2x16float(v)
    }

    pub fn demo_unpack2x16float(e: u32) -> Vec2f {
        unpack2x16float(e)
    }
}
```

## Generated WGSL

```wgsl
fn demo_pack4x8snorm(v: vec4f) -> u32 {
    return pack4x8snorm(v);
}

fn demo_unpack4x8snorm(e: u32) -> vec4f {
    return unpack4x8snorm(e);
}

fn demo_pack4x8unorm(v: vec4f) -> u32 {
    return pack4x8unorm(v);
}

fn demo_unpack4x8unorm(e: u32) -> vec4f {
    return unpack4x8unorm(e);
}

fn demo_pack2x16snorm(v: vec2f) -> u32 {
    return pack2x16snorm(v);
}

fn demo_unpack2x16snorm(e: u32) -> vec2f {
    return unpack2x16snorm(e);
}

fn demo_pack2x16unorm(v: vec2f) -> u32 {
    return pack2x16unorm(v);
}

fn demo_unpack2x16unorm(e: u32) -> vec2f {
    return unpack2x16unorm(e);
}

fn demo_pack2x16float(v: vec2f) -> u32 {
    return pack2x16float(v);
}

fn demo_unpack2x16float(e: u32) -> vec2f {
    return unpack2x16float(e);
}
```

## Notes

- The Rust snake_case names (`pack4x8snorm`, etc.) map directly to the WGSL builtin names.