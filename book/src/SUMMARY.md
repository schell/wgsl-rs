# Summary

[Introduction](./introduction.md)

---

# Getting Started

- [Installation](./getting-started/installation.md)
- [Hello, Triangle](./getting-started/hello-triangle.md)
- [Cargo Features](./getting-started/cargo-features.md)

# Writing Shaders

- [The `#[wgsl]` Macro](./writing-shaders/the-wgsl-macro.md)
- [Modules & Imports](./writing-shaders/modules-and-imports.md)
- [Functions](./writing-shaders/functions.md)
- [Structs & Impl Blocks](./writing-shaders/structs-and-impls.md)
- [Constants](./writing-shaders/constants.md)
- [Control Flow](./writing-shaders/control-flow.md)
- [Operators & Expressions](./writing-shaders/operators.md)
- [Binding Macros](./writing-shaders/binding-macros.md)
  - [`uniform!`](./writing-shaders/binding-macros/uniform.md)
  - [`storage!`](./writing-shaders/binding-macros/storage.md)
  - [`workgroup!`](./writing-shaders/binding-macros/workgroup.md)
  - [`texture!` & `sampler!`](./writing-shaders/binding-macros/texture-sampler.md)
  - [`ptr!`](./writing-shaders/binding-macros/ptr.md)
  - [`discard!`](./writing-shaders/binding-macros/discard.md)

# Types

- [Scalars & Literals](./types/scalars.md)
- [Vectors & Swizzles](./types/vectors.md)
- [Matrices](./types/matrices.md)
- [Arrays & `RuntimeArray<T>`](./types/arrays.md)
- [Atomics](./types/atomics.md)

# Generics & Templates

- [Generic Functions](./generics/generic-functions.md)
- [Generic Structs](./generics/generic-structs.md)
- [Template Modules & Instantiation](./generics/templates.md)

# Entry Points

- [Vertex / Fragment / Compute](./entry-points/stages.md)
- [Inter-stage IO](./entry-points/inter-stage-io.md)
- [Default Annotations](./entry-points/default-annotations.md)

# Validation

- [Auto-generated Tests](./validation/auto-tests.md)
- [Runtime Validation](./validation/runtime.md)
- [Disabling Validation](./validation/disabling.md)

# The Standard Library

- [Overview](./std/overview.md)
- [Numeric Builtins](./std/numeric.md)
- [Matrix & Vector Functions](./std/matrix-vector.md)
- [Texture & Sampler Functions](./std/texture-sampler.md)
- [Derivatives](./std/derivatives.md)
- [Bitcast](./std/bitcast.md)
- [Packing](./std/packing.md)
- [Synchronization](./std/synchronization.md)
- [`discard!()`](./std/discard.md)

# wgpu Linkage

- [Overview](./linkage/overview.md)
- [Bind Groups & Buffers](./linkage/bind-groups.md)
- [Pipeline Layouts](./linkage/pipeline-layouts.md)
- [Template Linkage](./linkage/template-linkage.md)
- [Per-binding Shader Stages](./linkage/shader-stages.md)

# Extensions

- [The `WgslExtension` Trait](./extensions/trait.md)
- [`modify_ir` & Run Order](./extensions/modify-ir.md)
- [IR Attributes](./extensions/ir-attributes.md)
- [Worked Examples](./extensions/examples.md)
- [Pitfalls & Constraints](./extensions/pitfalls.md)

# Memory Layout

- [Overview](./layout/overview.md)
- [`WgslLayout` & `Layout` Traits](./layout/traits.md)
- [`#[derive(Layout)]`](./layout/derive.md)
- [`FieldLayout` & `pad_after`](./layout/field-layout.md)
- [SVG Diagrams](./layout/svg-diagrams.md)

# Examples

- [Catalog](./examples/catalog.md)
- [Hello Triangle](./examples/hello-triangle.md)
- [Structs](./examples/structs.md)
- [Compute Shader](./examples/compute-shader.md)
- [Matrix Example](./examples/matrix.md)
- [Impl Example](./examples/impl.md)
- [Enum / Match-Switch](./examples/enum.md)
- [Binary Ops](./examples/binary-ops.md)
- [For Loop](./examples/for-loop.md)
- [While & Loop](./examples/while-loop.md)
- [If / Switch / Break / Return](./examples/control-flow.md)
- [Runtime Array](./examples/runtime-array.md)
- [Pointers](./examples/ptr.md)
- [Atomics](./examples/atomics.md)
- [Textures](./examples/texture.md)
- [Bitcast](./examples/bitcast.md)
- [Packing](./examples/packing.md)
- [Advanced Numeric](./examples/advanced-numeric.md)
- [Matrix Builtin](./examples/matrix-builtin.md)
- [Synchronization](./examples/synchronization.md)
- [`macro_rules!` Definitions](./examples/macro-rules.md)
- [Slab Read/Write](./examples/slab-read-write.md)
- [Derivatives](./examples/derivatives.md)
- [Discard](./examples/discard.md)
- [Generic Functions](./examples/generic-functions.md)
- [Trait Impls](./examples/trait-impls.md)
- [Renderer Specialization](./examples/renderer-specialization.md)
- [Renderer Specialization (Simple)](./examples/renderer-specialization-simple.md)
- [Generic Structs](./examples/generic-structs.md)
- [Shared Inter-stage](./examples/shared-inter-stage.md)

# Reference

- [Supported Rust Subset](./reference/supported-subset.md)
- [Macro Attributes Reference](./reference/macro-attributes.md)
- [Cargo Features Reference](./reference/cargo-features.md)
- [IR Types Reference](./reference/ir-types.md)
- [Error Reference](./reference/errors.md)

# Design Decisions

- [Highlights from the Devlog](./design-decisions/highlights.md)

# Contributing

- [AI Disclosure Policy](./contributing/ai-disclosure.md)
- [Code Style](./contributing/code-style.md)
- [xtask & CI](./contributing/xtask-ci.md)