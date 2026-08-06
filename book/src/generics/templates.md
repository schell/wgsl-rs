# Template Modules & Instantiation

Generic functions and structs are monomorphized to concrete WGSL at macro time — every type parameter is resolved before the module is emitted. **Template modules** are the complementary mechanism for *deferring* type parameters until runtime: the macro emits **template WGSL** carrying `TypeParam` placeholders, and you instantiate it with concrete types at runtime to obtain a valid shader.

## When to Use a Template

Use a template module when the type of an entry point, a linkage binding, or a struct field cannot be pinned down at macro time — for example, a renderer that wants to swap `f32` precision for `f16`, or a uniform whose host-side type is chosen per pipeline.

A module becomes a template when **any** of the following appear:

- An entry-point function with type parameters.
- An entry-point function with const parameters (`const N: usize` or `const N: u32`).
- A linkage macro (`uniform!`, `storage!`, ...) whose declared type uses `impl Trait`.
- A `get!(VAR, T)` accessor that introduces a fresh type variable bound to a linkage variable.

## Defining a Template Module

The `hello_triangle_generic` example shows the pattern:

```rust
#[wgsl(validate_with_instantiation_types(f32, f32))]
pub mod hello_triangle_generic {
    use wgsl_rs::std::*;

    // The `impl Convert<f32>` syntax declares that FRAME's concrete type
    // is chosen at instantiation time.
    uniform!(group(0), binding(0), FRAME: impl Convert<f32>);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [
            vec2f(0.0, 0.5),
            vec2f(-0.5, -0.5),
            vec2f(0.5, -0.5),
        ];
        let position = POS[vertex_index as usize];
        vec4f(position.x, position.y, 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main<T: Convert<f32> + Wgsl + Clone>() -> Vec4f {
        let frame_t = get!(FRAME, T);
        vec4f(1.0, sin(f32(frame_t) / 128.0), 0.0, 1.0)
    }
}
```

Two type parameters are in play here: the `impl Convert<f32>` on `FRAME` and the `T` on `frag_main`. They are linked through `get!(FRAME, T)`.

## Generic Linkage: `impl Trait`

A linkage macro may declare its type as `impl SomeTrait`:

```rust
uniform!(group(0), binding(0), FRAME: impl Convert<f32>);
```

This says: *FRAME has some concrete type, chosen at instantiation time, that implements `Convert<f32>`*. The trait bounds are replayed onto the typestate builder's `set_frame` method, so a host caller can only bind `FRAME` to a type satisfying those bounds.

## `get!(VAR, T)` Constraints

Inside a generic entry point, `get!(VAR, T)` reads the linkage variable `VAR` and introduces a fresh type variable `T` connected to `VAR`'s declared type. This generates a constraint of the form

```
VAR: linkage::Type<Is = T>
```

on the module's `instantiate` function. The transpiler threads that constraint so that the same concrete type is used for both the binding and the body of the entry point.

## Instantiation at Runtime

A template module exposes an `instantiate` function you call with concrete turbofish type arguments. It returns a concrete `ir::Module` whose `TypeParam` placeholders have been substituted:

```rust
use example::hello_triangle_generic as tmpl;

let module: ir::Module = tmpl::instantiate::<f32, f32>();
let source: String = module.to_wgsl();
```

The number and order of type arguments match the template's declared type parameters. The resulting `ir::Module` is a fully concrete, validatable WGSL module. To build `wgpu` pipelines from an instantiated template, see [Template Linkage](../linkage/template-linkage.md).

## Validating Templates

Template modules are **not** auto-validated by `#[wgsl]`, because the raw `TypeParam` placeholders are not valid WGSL. There are two ways to validate:

1. **At test time** with the `validate_with_instantiation_types(T1, T2, ...)` attribute. The auto-generated test instantiates the template with the given types and validates the result through naga:

   ```rust
   #[wgsl(validate_with_instantiation_types(f32, f32))]
   pub mod hello_triangle_generic { /* ... */ }
   ```

2. **At runtime** by calling `module.validate()` on the instantiated `ir::Module` (requires the `validation` feature). See [Runtime Validation](../validation/runtime.md).

If you omit both, the template is never validated by `cargo test`.

## Multiple Type Parameters & Transitive Use

Templates support multiple type parameters and transitive generic calls just like monomorphized generics. Each `instantiate` call substitutes the full tuple of type arguments through the module's IR; the runtime performs deduplication of any shared monomorphized pieces inside the resulting module.

## Const Parameters on Entry Points

Entry points can also take `const N: usize` (or `const N: u32`) parameters. The module becomes a template and is instantiated with a concrete integer:

```rust
#[wgsl(skip_validation)]
pub mod entry_point {
    use wgsl_rs::std::*;

    #[compute]
    #[workgroup_size(1)]
    pub fn main<const N: usize>() -> u32 {
        let arr: [u32; N] = [0u32; N];
        arr[0]
    }
}
```

Instantiate with the concrete const value:

```rust
let module: ir::Module = entry_point::instantiate::<4>();
```

Const params use a separate positional namespace (`{fn}_c{i}`) so type and const params on the same entry point don't collide. The `instantiate::<...>()` turbofish accepts both type and const arguments in the order they're declared on the entry point.