# Runtime Array

Demonstrates runtime-sized arrays (`RuntimeArray<T>`) in storage buffers. Runtime arrays transpile to `array<T>` in WGSL (no size) and must be the last field of a struct in a storage buffer.

## Rust Source

```rust
#[wgsl]
#[allow(dead_code)]
pub mod runtime_array_example {
    //! Demonstrates runtime-sized arrays (`RuntimeArray<T>`).
    //!
    //! Runtime-sized arrays transpile to `array<T>` in WGSL (no size
    //! parameter). They can only be used in storage buffers, typically as
    //! the last field of a struct.
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Particle {
        pub position: Vec3f,
        pub velocity: Vec3f,
    }

    #[derive(Wgsl)]
    pub struct ParticleSystem {
        pub count: u32,
        pub particles: RuntimeArray<Particle>,
    }

    storage!(group(0), binding(0), read_write, PARTICLES: ParticleSystem);

    #[compute]
    #[workgroup_size(16, 16, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let num_particles = array_length(&get!(PARTICLES).particles);
        let index = global_id.y() * 16 + global_id.x();
        if num_particles < index {
            let velocity = get!(PARTICLES).particles[index].velocity;
            let position = &mut get_mut!(PARTICLES).particles[index].position;
            *position = *position + velocity;
        }
    }
}
```

## Generated WGSL

```wgsl
struct Particle {
    position: vec3f,
    velocity: vec3f
}

struct ParticleSystem {
    count: u32,
    particles: array<Particle>
}
@group(0) @binding(0) var<storage, read_write> PARTICLES: ParticleSystem;

@compute @workgroup_size(16, 16, 1) fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let num_particles = arrayLength(&PARTICLES.particles);
    let index = global_id.y * 16 + global_id.x;
    if num_particles < index {
        let velocity = PARTICLES.particles[index].velocity;
        let position = &PARTICLES.particles[index].position;
        *position = *position + velocity;
    }
}
```

## Notes

- `RuntimeArray<T>` transpiles to `array<T>` (unsized) in WGSL.
- `array_length(&...)` maps to the WGSL `arrayLength` builtin.
- Runtime arrays may only appear in storage buffers, typically as the last field.
- The bounds check in the example (`if num_particles < index`) is intentionally verbatim from the source — note the condition is reversed from what you'd typically want (`index < num_particles`). This is a known quirk of the example, preserved for roundtrip-test compatibility.