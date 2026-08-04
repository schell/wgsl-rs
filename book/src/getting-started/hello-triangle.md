# Hello, Triangle

This chapter walks through the canonical `hello_triangle` example end to end. The module is ordinary Rust that the `#[wgsl]` macro transpiles to WGSL, and it is also a valid Rust module you can compile and test.

## The source

```rust
#[wgsl]
pub mod hello_triangle {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];
        let position = POS[vertex_index as usize];
        vec4f(position.x, position.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4f(1.0, sin(f32(get!(FRAME)) / 128.0), 0.0, 1.0)
    }
}
```

## `#[wgsl] pub mod hello_triangle { ... }`

The `#[wgsl]` attribute marks a module for transpilation. The macro consumes the module body, builds an owned IR, and emits a `WGSL_SOURCE` static containing the generated shader text. The module remains valid Rust: the functions are callable from the CPU side, and the types resolve against `wgsl_rs::std`.

## `use wgsl_rs::std::*`

The glob import is required. It brings the WGSL type aliases (`Vec2f`, `Vec3f`, `Vec4f`, ...), constructor functions (`vec2f`, `vec3f`, `vec4f`, ...), and the built-in WGSL functions (`sin`, `cos`, `dot`, ...) into scope so the Rust body type-checks and maps one-to-one onto WGSL declarations.

## `uniform!(...)`

```rust
uniform!(group(0), binding(0), FRAME: u32);
```

The `uniform!` macro declares a uniform binding that is visible in both worlds. It expands to a WGSL `var<uniform>` declaration in the generated source and to a Rust handle that the runtime can bind and read. Here `FRAME` is a `u32` at group 0, binding 0.

## Entry points: `#[vertex]` and `#[fragment]`

Functions annotated with `#[vertex]` and `#[fragment]` become WGSL entry points tagged with `@vertex` and `@fragment` respectively. Other functions in the module without these annotations transpile to plain WGSL functions.

## `#[builtin(vertex_index)]`

```rust
pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f
```

Argument annotations carry WGSL I/O attributes through to the generated signature. `#[builtin(vertex_index)]` becomes `@builtin(vertex_index)` in the WGSL output. The same mechanism supports `@location(n)`, `@interpolate(...)`, and other I/O attributes via the corresponding `#[...]` annotations.

## Vector types and constructors

`Vec4f` and `Vec2f` are type aliases for `vec4<f32>` and `vec2<f32>` exposed by `wgsl_rs::std`. The lowercase `vec2f` / `vec4f` functions are the matching constructors. They mirror WGSL exactly, so Rust expressions like `vec4f(1.0, 0.0, 0.0, 1.0)` transpile directly to `vec4<f32>(1.0, 0.0, 0.0, 1.0)`.

## `get!(FRAME)`

```rust
sin(f32(get!(FRAME)) / 128.0)
```

`get!(...)` is the runtime accessor for a declared uniform. On the Rust side it reads the bound value; in the generated WGSL it expands to the bare uniform reference `FRAME`. This lets the same expression serve both CPU evaluation (e.g. in dispatch-runtime tests) and the shader.

## Generated WGSL

```wgsl
@group(0) @binding(0) var<uniform> FRAME: u32;

@vertex
fn vtx_main(@builtin(vertex_index) vertex_index: u32) -> vec4<f32> {
    const POS: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.5), vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5));
    var position: vec2<f32> = POS[vertex_index];
    return vec4<f32>(position.x(), position.y(), 0.0, 1.0);
}

@fragment
fn frag_main() -> vec4<f32> {
    return vec4<f32>(1.0, sin(f32(FRAME) / 128.0), 0.0, 1.0);
}
```

Note how each Rust construct maps onto WGSL: `const` to `const`, `let` to `var`, array literals to explicit `array<T, N>(...)` constructors, and `get!(FRAME)` to the bare `FRAME` reference.

## Validation

`#[wgsl]` auto-generates a hidden test:

```rust
#[test]
fn __validate_wgsl() { /* ... */ }
```

For non-template modules this test feeds `WGSL_SOURCE` through naga and fails on any validation error. Run it with:

```sh
cargo test hello_triangle
```

A passing test means the transpiled WGSL is well-formed according to naga.