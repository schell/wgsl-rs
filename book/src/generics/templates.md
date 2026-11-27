# Template Modules & Instantiation

Generic functions and structs are monomorphized to concrete WGSL at macro time — every type parameter is resolved before the module is emitted. **Template modules** are the complementary mechanism for *deferring* type parameters until runtime: the macro emits **template WGSL** carrying `TypeParam` placeholders, and you instantiate it with concrete types at runtime to obtain a valid shader.

## When to Use a Template

Use a template module when the type of an entry point, a linkage binding, or a struct field cannot be pinned down at macro time — for example, a renderer that wants to swap `f32` precision for `f16`, or a uniform whose host-side type is chosen per pipeline.

A module becomes a template when **any** of the following appear:

- An entry-point function with type parameters.
- An entry-point function with const parameters (`const N: usize`, `const N: u32`, `const N: i32`, or `const B: bool`).
- A [resource pack](../writing-shaders/resource-packs.md) with a type parameter, such as `FramePack<T>` with a `Uniform<T>` field.
- A sampled texture or plain sampler pack field with a bare bool filterability parameter, such as `Texture2D<f32, FILTERABLE>` or `Sampler<FILTERABLE>`.
- A pack const parameter in a field annotation, such as `#[storage(group(GROUP), binding(0))]`.

## Defining a Template Module

The `hello_triangle_generic` example shows the pattern:

```rust
#[wgsl(validate_with_instantiation_types(f32, f32))]
pub mod hello_triangle_generic {
    use wgsl_rs::std::*;

    // The pack's `T` slot declares that frame's concrete type is chosen
    // at instantiation time.
    pub struct HelloTriangleGenericPack<T> {
        #[uniform(group(0), binding(0))]
        pub frame: Uniform<T>,
    }

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
    pub fn frag_main<T: Convert<f32> + Wgsl + Clone + Copy>(
        pack: &HelloTriangleGenericPack<T>,
    ) -> Vec4f {
        let frame_t = *get!(pack.frame);
        vec4f(1.0, sin(f32(frame_t) / 128.0), 0.0, 1.0)
    }
}
```

Two type parameters are in play here: the `T` on the pack (used by the
`frame` field) and the `T` on `frag_main`. They are linked through the
`&HelloTriangleGenericPack<T>` parameter.

## Generic Linkage: Pack Type Parameters

A resource pack may declare a type parameter that its field types use:

```rust
pub struct FramePack<T> {
    #[uniform(group(0), binding(0))]
    pub frame: Uniform<T>,
}
```

This says: *the `frame` binding has some concrete type, chosen at
instantiation time*. The entry point's trait bounds are replayed onto the
`instantiate` function, so a host caller can only instantiate with types
satisfying those bounds.

## `Type<Is = ...>` Constraints

Inside a generic entry point, a pack parameter such as
`&FramePack<T>` introduces a type variable `T` connected to the pack
field's type. This generates a constraint of the form

```
FramePack_T: linkage::Type<Is = T>
```

on the module's `instantiate` function. The transpiler threads that
constraint so that the same concrete type is used for both the binding and
the body of the entry point.

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

## Const Parameters on Texture and Sampler Pack Fields

An f32-sampled texture or plain sampler can defer its host-side filterability to instantiation time. Put a bare identifier in the bool const position of the pack field's type instead of writing `true` or `false`:

```rust
#[wgsl(skip_validation)]
pub mod heightmap {
    use wgsl_rs::std::*;

    pub struct HeightmapPack<const FILTERABLE: bool> {
        #[texture(group(0), binding(0))]
        pub heights: Texture2D<f32, FILTERABLE>,
        #[sampler(group(0), binding(1))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    #[fragment]
    pub fn main(pack: &HeightmapPack<FILTERABLE>) -> Vec4f {
        texture_sample(&pack.heights, &pack.heights_smp, vec2f(0.0, 0.0))
    }
}

let module = heightmap::instantiate::<false>();
```

Repeating `FILTERABLE` on the texture and sampler fields shares the pack
const slot, so both receive one argument. Use separate pack const parameters
when independent values are needed.

These parameters belong to the pack, not to an entry point. A bind group layout is shared by every pipeline entry point that uses the binding, and module-scope Rust structs cannot reference a function's generic parameters. This is the const-valued counterpart of a pack type parameter.

Filterability changes only the generated host layout. The rendered WGSL is identical for `true` and `false`. To generate wgpu linkage, pass the instantiated IR to `analyze_ir_module` or call `generate_linkage` as described in [Template Linkage](../linkage/template-linkage.md).

Parameter order is deterministic: resource-pack type parameters come first in pack declaration order, pack const parameters follow in first-occurrence field declaration order, and entry-point parameters that do not alias a resource pack come last in their existing order. Pack consts include group, binding, and filterability values. The authoritative order is also available as `WGSL_SOURCE.module_type_params`.


## Const Parameters on Resource Packs

[Resource pack structs](../writing-shaders/resource-packs.md) can parameterize
binding addresses and filterability in their field annotations. Every distinct
pack const receives a pack-owned module slot named `PackType_PARAM`. This
namespacing keeps separate packs independent even when both declare `GROUP` or
`FILTERABLE`.

An entry-point const used as `&Pack<...>` becomes an alias of the pack slot.
It does not add a second module argument, and its body uses the same value as
the generated layout. All entry points sharing one pack therefore receive one
coherent set of layout values. Entry-point-only consts that are not tied to a
pack remain independent and retain their normal positional slots.

Matching concrete uses, such as two `&Pack<false>` parameters, specialize the
IR directly and need no generic argument. Different concrete values, or a
mixture of concrete and generic uses of one pack const, are compile-time
layout conflicts. Resource names and `(group, binding)` addresses are validated
again after generic substitution, so instantiated conflicts are rejected too.

Pack-owned binding consts appear in first-occurrence field declaration order,
before non-aliased entry-point parameters. Separate packs can be selected
independently by supplying their respective slots.

## Type Parameters on Resource Packs

[Resource pack structs](../writing-shaders/resource-packs.md) may declare type
parameters that their field types reference. Each pack type parameter
introduces one module-level type slot named `PackType_PARAM`, listed before
the pack's const slots. Unlike a pack const, a type slot is always generic:
declaring `DataPack<T>` makes the module a template, and the concrete type
is chosen at instantiation.

```rust
pub struct DataPack<T> {
    #[storage(group(0), binding(0), read_only)]
    pub data: Storage<RuntimeArray<T>, Read>,
}

#[fragment]
pub fn fragment<T: Wgsl>(position: Vec4f, pack: &DataPack<T>) -> Vec4f { /* ... */ }

// One argument for the pack slot, one for the entry point's `T`.
let module = shader::instantiate::<Vec4f, Vec4f>();
```

Type arguments at entry points tie to the pack slot through generated
`linkage::Type<Is = ...>` constraints, the same mechanism `get!(VAR, T)`
uses on an `impl Trait` linkage, so rustc rejects inconsistent instantiation.
Writing the concrete type in the entry point signature
(`&DataPack<Vec4f>`) keeps the entry-point slot out of the argument list.
Helpers may use a bare type parameter in the pack parameter only; it reads
the pack slot directly, mirroring the const aliasing, and may not appear in
any other parameter or return position.

## Const Parameters on Entry Points

Entry points can also take `const N: usize`, `const N: u32`, `const N: i32`, or `const B: bool` parameters. The module becomes a template and is instantiated with a concrete value (integer or bool):

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

A `bool` const param on an entry point instantiates the same way, with the flag value: `entry_point::instantiate::<true>()`. This is the compile-time feature-flag pattern; entry points keep their un-mangled Rust name in concrete instantiations, so linkage lookups use the plain name. See [Bool const parameters](generic-functions.md#bool-const-parameters).