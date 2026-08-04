# Arrays & `RuntimeArray<T>`

## Fixed-Size Arrays

A Rust fixed-size array `[T; N]` transpiles to a WGSL `array<T, N>` with the size baked in.

```rust
const POS: [Vec2f; 3] = [
    vec2f(0.0, 0.5),
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
];
```

```wgsl
const POS: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>(0.5, -0.5),
);
```

## Indexing

WGSL array indexing requires the index to be a `u32`/`i32`. Because Rust idioms (and CPU-side data structures) commonly carry `usize`, wgsl-rs accepts an `i as usize` cast on the index and emits the inner expression directly:

```rust
let p = POS[vertex_index as usize];   // -> POS[vertex_index]
```

Use `arr[i as usize]` whenever the index source is a `u32` builtin (e.g. `vertex_index`, `global_invocation_id.x()`).

## Zero-Value Arrays

A Rust zero-value array `[0u32; 4]` is recognized and turned into an explicit WGSL array constructor of the same length, populated with the zero value:

```rust
let zeros: [u32; 4] = [0u32; 4];
```

```wgsl
var zeros: array<u32, 4> = array<u32, 4>();
```

The element expression must be a literal `0` of the element type. Non-zero repeated-element arrays are not given this special treatment.

## Runtime-Size Arrays

`RuntimeArray<T>` maps to the unsized WGSL `array<T>` (no count parameter). Runtime arrays are restricted by the WGSL specification:

- They may only appear in storage buffers.
- They must be the **last field** of a struct.

On the CPU side `RuntimeArray<T>` is backed by a `Vec<T>` so the same struct can be populated and read by host code.

```rust
#[derive(Wgsl)]
pub struct ParticleSystem {
    pub count: u32,
    pub particles: RuntimeArray<Particle>,
}

storage!(group(0), binding(0), read_write, PARTICLES: ParticleSystem);
```

```wgsl
struct ParticleSystem {
  count: u32,
  particles: array<Particle>,
};

@group(0) @binding(0) var<storage, read_write> PARTICLES: ParticleSystem;
```

## `array_length`

Query the length of a runtime array with `array_length(&arr)`. Pass the array field by reference:

```rust
let n = array_length(&get!(PARTICLES).particles);
```

```wgsl
let n = arrayLength(&PARTICLES.particles);
```

For fixed-size arrays the length is known statically; use `arr.len()` or the `N` from the type where convenient, but prefer `array_length` only for runtime arrays.