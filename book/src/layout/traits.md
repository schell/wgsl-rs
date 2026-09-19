# Layout Traits

## `WgslLayout`

`WgslLayout` is implemented for every built-in WGSL type: scalars, vectors, matrices, arrays, and atomics.

```rust
pub trait WgslLayout {
    const SIZE: usize;
    const ALIGN: usize;
}
```

| Type                  | `SIZE` | `ALIGN` |
|-----------------------|--------|---------|
| `f32`                 | 4      | 4       |
| `u32`                 | 4      | 4       |
| `vec2<f32>`           | 8      | 8       |
| `vec3<f32>`           | 12     | 16      |
| `vec4<f32>`           | 16     | 16      |
| `mat4x4<f32>`         | 64     | 16      |
| `array<f32, 4>`       | 16     | 4       |

These constants are the source of truth for all downstream layout computation.

## `Layout`

`Layout` extends `WgslLayout` with per-field metadata for composite types:

```rust
pub trait Layout: WgslLayout {
    const FIELDS: &'static [FieldLayout];
}
```

`FieldLayout` is described in [Field Layout](./field-layout.md).

## Generic Structs

Generic structs are supported. Each type parameter receives a `T: WgslLayout` bound in the generated impl, so `SIZE`, `ALIGN`, and `FIELDS` are computed in terms of the substituted type's constants:

```rust
#[derive(Layout)]
struct Cell<T> {
    value: T,
    next: u32,
}
```

The generated impl is roughly:

```rust
impl<T: WgslLayout> Layout for Cell<T> {
    const FIELDS: &'static [FieldLayout] = &[ /* computed from T::SIZE, T::ALIGN */ ];
}
```

Because the bounds propagate `WgslLayout`, generic structs compose freely with other layout-annotated types and built-in WGSL types.