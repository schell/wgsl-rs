# IR Types

The `wgsl_rs::ir` module exposes the intermediate representation that extensions and the linker operate on.

## Top-Level Types

| Type                | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| `ir::Module`        | Root node: `{ name, items: Vec<Item>, attrs: Vec<Attribute> }`         |
| `ir::Item`          | Enum of top-level module items (see below)                             |
| `ir::Type`          | Enum of all WGSL type expressions (see below)                          |
| `ir::Expr`          | Expression node                                                        |
| `ir::Stmt`          | Statement node                                                         |
| `ir::Block`         | Sequence of statements                                                 |
| `ir::FnArg`         | Function parameter with attributes and type                            |
| `ir::Field`         | Struct field with attributes and type                                  |
| `ir::Attribute`     | `{ path: String, args: Vec<String> }` — preserved Rust attribute       |
| `ir::FnAttrs`       | Dedicated function-level decorators (entry point, workgroup size)      |
| `ir::BuiltIn`       | WGSL built-in value identifiers                                        |
| `ir::InterStageIo`  | Inter-stage I/O descriptor (location, interpolation, blend_src)        |
| `ir::ReturnType`    | Function return type representation                                    |
| `ir::WorkgroupSize` | `{ x, y, z }` for compute entry points                                |
| `ir::ScalarType`    | Scalar type enum (f32, i32, u32, f16, bool)                            |
| `ir::AddressSpace`  | Address space enum (uniform, storage, workgroup, function, private)    |
| `ir::StorageAccess` | Storage buffer access (read, write)                                    |
| `ir::TextureKind`   | Texture type kind                                                      |
| `ir::TextureDepthKind` | Depth texture kind                                                  |

## `ir::Item` Variants

| Variant      | WGSL construct                                                       |
|--------------|----------------------------------------------------------------------|
| `Const`      | Module-scope `const` declaration                                     |
| `Uniform`    | Uniform buffer declaration                                           |
| `Storage`    | Storage buffer declaration                                           |
| `Workgroup`  | Workgroup variable declaration                                       |
| `Sampler`    | Sampler declaration                                                  |
| `Texture`    | Texture declaration                                                  |
| `Fn`         | Function (including entry points)                                    |
| `Struct`     | Struct declaration                                                   |
| `Impl`       | Impl block (transpiled to free functions)                            |
| `Enum`       | Enum declaration                                                     |

## `ir::Type` Variants

| Variant              | WGSL type                                            |
|----------------------|------------------------------------------------------|
| `Scalar`             | `f32`, `i32`, `u32`, `f16`, `bool`                   |
| `Vector`             | `vecN<T>`                                            |
| `Matrix`             | `matNxM<T>`                                          |
| `Array`              | `array<T, N>`                                        |
| `RuntimeArray`       | `array<T>` (runtime-sized)                           |
| `Atomic`             | `atomic<T>`                                          |
| `Struct`             | User-defined struct                                  |
| `Ptr`                | `ptr<AS, T, AM>`                                     |
| `Sampler`            | `sampler`                                            |
| `SamplerComparison`  | `sampler_comparison`                                 |
| `Texture`            | Sampled texture                                      |
| `TextureDepth`       | Depth texture                                        |
| `TypeParam`          | Generic type parameter (substituted at instantiation)|