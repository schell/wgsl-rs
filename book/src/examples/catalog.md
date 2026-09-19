# Examples

This section contains a catalog of example `wgsl-rs` modules. Each example demonstrates a specific feature of the transpiler, showing the Rust source and the generated WGSL output.

## Running examples

The `example` crate provides two subcommands for inspecting examples:

- `cargo run -p example -- show` — list all available example names
- `cargo run -p example -- source {name}` — print the generated WGSL for the named example

## Example catalog

| Name | Demonstrates | Page |
|------|--------------|------|
| `hello_triangle` | A "hello world" vertex+fragment shader with uniforms and builtins | [hello-triangle](./hello-triangle.md) |
| `structs` | User-defined structs as fragment inputs/outputs with locations and builtins | [structs](./structs.md) |
| `compute_shader` | A compute shader with storage buffers and the `get!`/`get_mut!` macros | [compute-shader](./compute-shader.md) |
| `matrix_example` | Matrix types and constant constructors (`mat2x2f`, `mat3x3f`, `mat4x4f`) | [matrix](./matrix.md) |
| `impl_example` | Struct `impl` blocks: associated constants and methods | [impl](./impl.md) |
| `enum_example` | Limited enum support translated to `u32` aliases and constants | [enum](./enum.md) |
| `binary_ops_example` | All supported binary operators (arithmetic, comparison, logical, bitwise) | [binary-ops](./binary-ops.md) |
| `for_loop_example` | For-loops with range expressions and `#[wgsl_allow(non_literal_loop_bounds)]` | [for-loop](./for-loop.md) |
| `while_loop_example` / `loop_example` | `while` loops and infinite `loop` statements | [while-loop](./while-loop.md) |
| `if_example` / `break_example` / `return_example` / `switch_example` | Control flow: `if`, `break`, explicit `return`, and `match`/switch | [control-flow](./control-flow.md) |
| `runtime_array_example` | Runtime-sized arrays (`RuntimeArray<T>`) in storage buffers | [runtime-array](./runtime-array.md) |
| `ptr_example` | Pointer types (`ptr!`) in function parameters | [ptr](./ptr.md) |
| `atomic_example` | Atomic types and workgroup variables in compute shaders | [atomics](./atomics.md) |
| `texture_example` | Textures, samplers, and texture builtin functions | [texture](./texture.md) |
| `bitcast_example` | `bitcast` builtin functions for type reinterpretation | [bitcast](./bitcast.md) |
| `packing_example` | Packing/unpacking builtins (`pack4x8snorm`, etc.) | [packing](./packing.md) |
| `advanced_numeric_example` | `modf`, `frexp`, and `ldexp` builtins | [advanced-numeric](./advanced-numeric.md) |
| `matrix_builtin_example` | `determinant` and `transpose` matrix builtins | [matrix-builtin](./matrix-builtin.md) |
| `synchronization_example` | `workgroupBarrier`, `storageBarrier`, `workgroupUniformLoad` | [synchronization](./synchronization.md) |
| `macro_rules_definitions` | `macro_rules!` and derive macros (stripped from WGSL output) | [macro-rules](./macro-rules.md) |
| `slab_read_write` | Reading/writing structs from u32 "slabs" via slab macros | [slab-read-write](./slab-read-write.md) |
| `derivative_example` | All 9 WGSL derivative builtin functions in a fragment shader | [derivatives](./derivatives.md) |
| `discard_example` | The `discard!()` statement for discarding fragments | [discard](./discard.md) |
| `generic_functions` | Generic functions with monomorphization | [generic-functions](./generic-functions.md) |
| `trait_impl_example` | Trait definitions and impl blocks resolved via monomorphization | [trait-impls](./trait-impls.md) |
| `renderer_specialization` | A full renderer specialized via traits and turbofish | [renderer-specialization](./renderer-specialization.md) |
| `renderer_specialization_simple` | A second specialization of the shared renderer pipeline | [renderer-specialization-simple](./renderer-specialization-simple.md) |
| `generic_structs` | Generic structs with `#[wgsl(skip_validation)]` (known bug) | [generic-structs](./generic-structs.md) |
| `shared_inter_stage` | A single struct shared between vertex and fragment stages | [shared-inter-stage](./shared-inter-stage.md) |
| `phantom_data` | `PhantomData<T>` marker fields (retained in IR, omitted from WGSL) | [phantom-data](./phantom-data.md) |