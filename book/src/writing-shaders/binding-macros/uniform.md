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

## Notes

- Uniforms are read-only on the GPU.
- `Type` should be `#[derive(Wgsl)]` so the host side can lay out and upload the buffer.
- One uniform binding per `(group, binding)` pair.