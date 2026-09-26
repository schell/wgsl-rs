# `uniform!`

Declares a uniform buffer binding.

## Syntax

```rust
uniform!(group(N), binding(M), NAME: Type);
```

## What It Generates

WGSL:

```wgsl
@group(N) @binding(M) var<uniform> NAME: Type;
```

Rust:

```rust
pub static NAME: Uniform<Type>;
```

## Access

Read with `get!(NAME)`. The returned guard dereferences to `&Type`:

```rust
#[wgsl]
pub mod shader {
    use wgsl_rs::std::*;

    #[derive(Wgsl)]
    pub struct Camera {
        pub view: Mat4f,
        pub proj: Mat4f,
        pub pos: Vec3f,
    }

    uniform!(group(0), binding(0), CAMERA: Camera);

    pub fn world_to_clip(p: Vec3f) -> Vec4f {
        let c = get!(CAMERA);
        c.proj * c.view * vec4f(p, 1.0)
    }
}
```

`get!(NAME)` returns a guard, so field access uses `.` directly. For generic entry points, supply the type explicitly: `get!(CAMERA, Camera)`.

### `load!` — reading values

`get!` returns a guard, which cannot be used directly in arithmetic. `load!(NAME)` copies the value out and returns plain `Type`, so it works in expressions. In WGSL both are the bare variable reference:

```rust
#[wgsl]
pub mod shader {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U_TIME: f32);
    uniform!(group(0), binding(1), U_RESOLUTION: Vec2f);

    pub fn animated_st(frag_coord: Vec4f) -> Vec2f {
        // Works directly in arithmetic — no guard bridging.
        frag_coord.xy() / load!(U_RESOLUTION) * 3.0 + load!(U_TIME)
    }
}
```

- `load!(NAME)` — load a concrete variable's value.
- `load!(NAME, T)` — load with an explicit type inside generic/template entry points.
- The type must be `Copy`; all built-in value types are.

**`load!` or `get!`?** Use `load!` when you want a value (arithmetic, comparisons, passing a copy around). Use `get!` when you want borrow-style access — field access on the guard, indexing (`get!(BUFFER)[i]`), or taking a pointer for atomics (`atomic_load(&get!(COUNTER))`).

## Notes

- Uniforms are read-only on the GPU.
- `Type` should be `#[derive(Wgsl)]` so the host side can lay out and upload the buffer.
- One uniform binding per `(group, binding)` pair.