# Vectors & Swizzles

wgsl-rs exposes all 12 WGSL vector types as Rust aliases and constructor functions in `wgsl_rs::std`.

## Type Aliases

| f32       | i32       | u32       | Generic form      | WGSL          |
| --------- | --------- | --------- | ----------------- | ------------- |
| `Vec2f`   | `Vec2i`   | `Vec2u`   | `Vec2<f32>`       | `vec2<f32>`   |
| `Vec3f`   | `Vec3i`   | `Vec3u`   | `Vec3<f32>`       | `vec3<f32>`   |
| `Vec4f`   | `Vec4i`   | `Vec4u`   | `Vec4<f32>`       | `vec4<f32>`   |

The generic form (`Vec2<f32>`, `Vec3<f32>`, `Vec4<f32>`) is accepted wherever a concrete alias is, including struct fields and function signatures.

## Constructors

Constructors are lowercase functions named after the WGSL `vecN*` builtin:

```rust
let a: Vec2f = vec2f(0.0, 1.0);
let b: Vec3f = vec3f(0.0, 1.0, 2.0);
let c: Vec4f = vec4f(0.0, 1.0, 2.0, 3.0);
let u: Vec3u = vec3u(0u32, 1u32, 2u32);
let i: Vec4i = vec4i(0i32, 1i32, 2i32, 3i32);
```

Single-argument splat constructors are supported, mirroring WGSL:

```rust
let all_ones: Vec4f = vec4f(1.0);   // -> vec4<f32>(1.0)
```

These map directly to WGSL `vecN<T>(...)` constructor calls.

## Swizzles are Method Calls

WGSL lets you access vector components as fields (`v.xyz`). In wgsl-rs, **swizzles are method calls**, not field accesses:

```rust
let v: Vec4f = vec4f(1.0, 2.0, 3.0, 4.0);
let xyz: Vec3f = v.xyz();   // -> v.xyz
let x: f32    = v.x();      // -> v.x
let xy: Vec2f = v.xy();     // -> v.xy
let rgb: Vec3f = v.rgb();   // -> v.rgb
```

### Why Method Calls?

wgsl-rs maintains a single source file that must compile as **unmodified Rust** on the CPU and transpile to WGSL for the GPU (the "two worlds" constraint). Rust does not permit `.xyz` field access on a generic vector alias the way WGSL does, and the canonical Rust vector libraries (`glam`, etc.) expose swizzles via the `Vec4Swizzle`-style trait — i.e. as methods. Using method-call syntax keeps the shader source a legal Rust program that mirrors an idiomatic CPU implementation.

Field access (`v.xyz`) is therefore a parse error in your shader; always call the swizzle as a method.

## Supported Swizzle Names

Any combination of the `xyzw`, `rgba`, or `stpq` component sets, length 1-4, single-set only:

| Length | Examples                         | Returns   |
| ------ | -------------------------------- | --------- |
| 1      | `.x()`, `.r()`, `.s()`           | scalar    |
| 2      | `.xy()`, `.rg()`, `.st()`        | `Vec2*`   |
| 3      | `.xyz()`, `.rgb()`, `.stp()`     | `Vec3*`   |
| 4      | `.xyzw()`, `.rgba()`, `.stpq()`  | `Vec4*`   |

The result type matches the source vector's scalar kind: a `Vec3u` swizzle returns `u32` or `Vec2u`/`Vec3u`/`Vec4u`.