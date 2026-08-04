# Bitcast

Demonstrates `bitcast` builtin functions for reinterpreting the bit pattern of a value as another type. In `wgsl-rs`, each target type has a dedicated function (e.g. `bitcast_f32`, `bitcast_u32`, `bitcast_vec4i`).

## Rust Source

```rust
#[wgsl]
#[expect(dead_code, reason = "demonstration")]
pub mod bitcast_example {
    //! Demonstrates using `bitcast` to reinterpret the bits of a value as
    //! another type.
    //!
    //! WGSL `bitcast<T>(e)` reinterprets the bit pattern of `e` as type `T`
    //! without changing any bits. This is useful for packing/unpacking data,
    //! interpreting raw buffer contents, and working with IEEE 754
    //! representations.
    //!
    //! In `wgsl-rs`, each target type has a dedicated function:
    //!   - `bitcast_f32(e)` → `bitcast<f32>(e)`
    //!   - `bitcast_u32(e)` → `bitcast<u32>(e)`
    //!   - `bitcast_i32(e)` → `bitcast<i32>(e)`
    //!   - `bitcast_vec2f(e)` → `bitcast<vec2<f32>>(e)`, etc.
    use wgsl_rs::std::*;

    // Input: raw u32 data representing packed floats
    storage!(group(0), binding(0), INPUT: [u32; 256]);

    // Output: reinterpreted as floats
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    // Reinterpret a u32 bit pattern as an f32 value.
    pub fn reinterpret_as_float(bits: u32) -> f32 {
        bitcast_f32(bits)
    }

    // Reinterpret an f32 value as its u32 bit pattern.
    pub fn float_to_bits(value: f32) -> u32 {
        bitcast_u32(value)
    }

    // Reinterpret a u32 vector as an i32 vector.
    pub fn reinterpret_vec_as_signed(v: Vec4u) -> Vec4i {
        bitcast_vec4i(v)
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x() as usize;
        // Read raw u32 bits from input and reinterpret as f32
        let raw_bits = get!(INPUT)[idx];
        get_mut!(OUTPUT)[idx] = bitcast_f32(raw_bits);
    }
}
```

## Generated WGSL

```wgsl
@group(0) @binding(0) var<storage, read> INPUT: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> OUTPUT: array<f32, 256>;

fn reinterpret_as_float(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

fn float_to_bits(value: f32) -> u32 {
    return bitcast<u32>(value);
}

fn reinterpret_vec_as_signed(v: vec4u) -> vec4i {
    return bitcast<vec4<i32>>(v);
}

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = u32(global_id.x);
    let raw_bits = INPUT[idx];
    OUTPUT[idx] = bitcast<f32>(raw_bits);
}
```

## Notes

- `bitcast_<ty>(e)` maps to `bitcast<<ty>>(e)` in WGSL.
- Useful for packing/unpacking data, interpreting raw buffer contents, and working with IEEE 754 representations.