# devlog

## design decisions

1. The user must be able to write regular Rust code.
   No translation of Rust code will occur in the `#[wgsl]` macro.
   The macro is strictly additive in that it _adds_ more Rust code
   to the user's written code.
   Specifically, it will add the transpiled WGSL source code and imports.
2. The Rust type system should catch as many type errors as possible, so WGSL validation doesn't have to.
   Stated another way, `wgsl-rs` should never produce Rust code that compiles, but produces invalid WGSL.
   You can be sure that if your Rust compiles, your shader will too.
3. Macros are ok, as Rust folks are used to using macros.
   Eg. `uniform!(binding(0), group(0), BRIGHTNESS: f32);` is fine, and in fact we **must** use
   macros for **all** WGSL that can't be represented with Rust syntax.
4. WGSL builtins can be variadic and can be called with different types, and may return different types.
   This presents two problems to solve:
     1. Parameter _types_: When the type of a parameter differs between WGSL
        builtin "flavors", the support strategy in `wgsl-rs` is to use a trait
        that allows the parameter types to be dependent on a type parameter.
        Either in the type implementing the trait, an associated type or to have
        the parameter `impl AnotherTrait`. So far this has been flexible enough.
     2. Variadic functions: When a WGSL builtin can be called with varying parameter counts,
        the strategy in `wgsl-rs` is to create multiple functions - one
        for each variation. Each Rust function is then mapped to the same WGSL
        builtin using the `BUILTIN_NAME_CASE_MAP`.
        For example, (Rust => WGSL):
          * `texture_sample` => `textureSample`
          * `texture_sample_array` => `textureSample`
          * `texture_sample_array_offset` => `textureSample`

## tradeoffs

### Importing modules

With `wgsl-rs` you can import other modules. While this is a boon for development it has some significant
restrictions. Specifically you can only glob-import other WGSL modules written with `wgsl-rs`, or the
`wgsl_rs::std::*`. If you try to import arbitrary modules you'll get a compiler error about not having
`path::to::module::WGSL_MODULE` in scope.
 
### Swizzles

Swizzles are tricky because in Rust they could be accomplished with traits like `glam`'s 
`Vec4Swizzle` trait (and friends), but in WGSL they look like field accessors.
Because of design decisions the Rust must be un-altered so we use functions similar to the
`Vec4Swizzle` strategy. So to swizzle `xyz` in WGSL you would just call `.xyz()` in Rust.

### Numeric builtin functions

There's a lot to implement here. So far I've been pretty successful (3 functions done) using this AI prompt:

> Please add the `{fn}` function using the module-level documentation table as a guide, following the implementation of `abs`
> and `acos`, which used the `NumericBuiltinAbs` and `NumericBuiltinAcos` traits, respectively.

You should replace {fn} with whatever function you want to implement.

Keep in mind that the `VecN<T>` types have `glam` types as their `inner` fields, so you can use that for many of these.

Keep in mind that glam vectors are not iterators, you can't call `zip` on them. Instead, you can call `to_array` on
each vector and write into one of them, finally calling `.into()` on the array to convert it back to the vector.
See the implementations of `NumericBuiltinPow` and `NumericBuiltinStep` for an example of this.

I've gone with a "one-trait-per-function" strategy because each function has little differences, and I anticipate
having to use generic associated types for some functions.

### Vertex and fragment return values

Shaders may return user defined types (structs) that have their fields annotated with the `#[builtin(...)]`
and `#[location(...)]` macros, which corresponds to the description in
[the spec](https://gpuweb.github.io/gpuweb/wgsl/#example-ee897116):

```wgsl
// Mixed builtins and user-defined inputs.
struct MyInputs {
  @location(0) x: vec4<f32>,
  @builtin(front_facing) y: bool,
  @location(1) @interpolate(flat) z: u32
}

struct MyOutputs {
  @builtin(frag_depth) x: f32,
  @location(0) y: vec4<f32>
}

@fragment
fn fragShader(in1: MyInputs) -> MyOutputs {
  // ...
}
```

And here would be the corresponding Rust:
```rust
// Mixed builtins and user-defined inputs.
pub struct MyInputs {
    #[location(0)]
    pub x: Vec4<f32>,

    #[builtin(front_facing)]
    pub y: bool,

    #[location(1)]
    #[interpolate(flat)]
    pub z: u32
}

pub struct MyOutputs {
    #[builtin(frag_depth)]
    pub x: f32,

    #[location(0)]
    pub y: vec4<f32>
}

#[fragment]
pub fn fragShader(in1: MyInputs) -> MyOutputs {
  // ...
}
```

But in Rust we can't put annotations on return types, so we'll have to specify that in the shader
stage proc-attribute macros (`vertex`, `fragment` and `compute`).

But if the return type a vertex or fragment shader is `Vec4f`, and the macro didn't specify a
return type location or builtin, `wgsl-rs` will automatically insert an appropriate annotation.

I think most people who need to specify a return value other than the default `@builtin(position)` for
vertex shaders or `@location(0)` for fragment shaders will use a struct, so this is fine.

## features

### wgpu linkage (`linkage-wgpu` feature)

When the `linkage-wgpu` feature is enabled, the `#[wgsl]` macro generates additional wgpu-specific
code to simplify integration with wgpu applications:

**Buffer descriptors and creation functions** - For each `uniform!` and `storage!` declaration:
- A `{NAME}_BUFFER_DESCRIPTOR: wgpu::BufferDescriptor<'static>` constant
- A `create_{name}_buffer(device: &wgpu::Device) -> wgpu::Buffer` function

**Bind group modules** - For each bind group, a `linkage::bind_group_{N}` module containing:
- `LAYOUT_ENTRIES` and `LAYOUT_DESCRIPTOR` constants
- `layout(device)` - creates the bind group layout
- `create(device, layout, ...)` - type-safe bind group creation with named parameters
- `create_dynamic(device, layout, entries)` - dynamic bind group creation with a slice

**Shader entry point modules** - For vertex, fragment, and compute entry points:
- Entry point name constants
- Helper functions for creating pipeline states

This feature adds `wgpu` as a dependency to `wgsl-rs`.

### 2026-01-05: switch statement support (match → switch)

Rust `match` statements transpile to WGSL `switch` statements:
- `match x { 0 => {...}, 1 => {...}, _ => {...} }` → `switch x { case 0 {...} case 1 {...} default {...} }`
- Or-patterns `1 | 2 | 3 => {...}` → `case 1, 2, 3 {...}`
- Missing `_` arm auto-generates `default {}`
- Non-literal patterns (constants, identifiers) emit a warning suppressed with `#[wgsl_allow(non_literal_match_statement_patterns)]`
- Match expressions (in let bindings) are unsupported (WGSL switch is a statement)
- Guard clauses, range patterns, struct/tuple patterns are unsupported
- For future work regarding type checking, we may be able to get away with a trick. We _could_ alter the Rust code
  to result in the pattern matched, then use that result in an empty function that takes an integer. This would
  cause Rust to do the type checking for us, before WGSL validation. That would keep us from having to emit a warning. 

  Example input Rust:
  ```rust
  match my_expr {
    MyEnum::Variant1 => {
      do_stuff();
    }
  }
  ```

  Output Rust:
  ```rust
  let __match_result = match my_expr {
    input @ MyEnum::Variant1 => {
      do_stuff();
      input
    }
  };
  __ensure_integer(__match_result);
  ```

  Output WGSL:
  ```wgsl
  switch my_expr {
    case MyEnum_Variant1: {
      do_stuff();
    }
    default: {}
  }
  ```

  Maybe we should also look into what we can do with for-loop bounds and the `non_literal_loop_bounds` warning in this manner.

### 2026-01-03: for-loop support and warnings with #[wgsl_allow]

`for i in 0..n` transpiles to `for (var i = 0; i < n; i++)` and `for i in 0..=n` transpiles to `for (var i = 0; i <= n; i++)`.
Nested loops work correctly.
Only bounded ranges are supported (WGSL requires explicit bounds).

For-loops with non-literal bounds (where the bounds cannot be verified at compile-time to be ascending)
emit a compile error on stable, since warnings can't be emitted (there's a hack to emit them as deprecations, but it's hacky).
On nightly it emits a warning.
Use `#[wgsl_allow(non_literal_loop_bounds)]` on the for-loop to suppress these errors/warnings.

### 2026-01-29: Pointer type support (`ptr!` macro)

Added `ptr!(address_space, T)` macro for WGSL pointer types in function parameters.
- Supports `function` and `private` address spaces (the only ones passable to functions without extensions)
- Expands to `&mut T` in Rust for CPU execution
- Transpiles to `ptr<function, T>` or `ptr<private, T>` in WGSL
- Added dereference operator (`*`) support for pointer indirection
- Both `&x` and `&mut x` transpile to `&x` in WGSL (mutability is determined by access mode)

Example:
```rust
pub fn increment(p: ptr!(function, i32)) {
    *p += 1;
}

fn main() {
    let mut x: i32 = 5;
    increment(&mut x);  // x is now 6
}
```

### 2026-01-31: Atomic types and workgroup variables

Added `Atomic<T>` type for thread-safe atomic operations (WGSL `atomic<T>` where T is `i32` or `u32`).
Added `workgroup!` macro for declaring workgroup-scoped variables shared across compute shader invocations.
- `Atomic<T>` on CPU is backed by `std::sync::atomic::{AtomicI32, AtomicU32}` with full atomic operations
- `workgroup!(NAME: TYPE)` transpiles to `var<workgroup> NAME: TYPE;` in WGSL
- Workgroup variables on CPU are backed by `LazyLock<RwLock<T>>` for thread-safe shared state
- Added `AddressSpace::Workgroup` variant for future pointer support
- Atomic builtin functions (atomicLoad, atomicStore, etc.) will be added in a future update

### 2026-02-13: Synchronization builtin functions

Added `storageBarrier()`, `workgroupBarrier()`, `textureBarrier()`, and `workgroupUniformLoad()` synchronization builtins for compute shader workgroup coordination.
- Barrier functions are no-ops on the CPU side (no parallel dispatch runtime yet)
- `workgroup_uniform_load` uses a `WorkgroupUniformLoad` trait with impls for `Workgroup<T: Clone>` and `Workgroup<Atomic<{u32,i32}>>`
- Extended `ptr!` macro and parser to support `workgroup` address space
- Added name mappings in `BUILTIN_CASE_NAME_MAP` for all four sync builtins

### 2026-02-25: FBM example crate (`fbm-example`)

Added a standalone `fbm-example` crate that renders an animated fractal brownian motion
shader in a `winit` window using `wgpu`. Port of the classic FBM shader by Patricio
Gonzalez Vivo from GLSL.

Lessons learned during the port:
- **Typed literal suffixes** like `0.0_f32` are emitted verbatim into WGSL, causing parse
  errors. Use plain `0.0` instead. The transpiler does not strip Rust literal suffixes.
- **`#[fragment]` does not strip `#[builtin(...)]`** from function parameters (unlike
  `#[vertex]` and `#[compute]` which do). Workaround: use an `#[input]` struct with
  `#[builtin(position)]` on the field instead of a direct parameter attribute.
  See [#84](https://github.com/schell/wgsl-rs/issues/84).
- **Accessing uniforms in expressions**: `get!(U_TIME)` returns a `ModuleVarReadGuard<T>`
  on the Rust side. To use the value in arithmetic, wrap with the identity type constructor:
  `f32(get!(U_TIME))` for scalars, or `vec2f(get!(U_RESOLUTION).x(), get!(U_RESOLUTION).y())`
  for vectors. These become no-op constructors in WGSL. This is a bit messy and should be
  improved long-term.

### 2026-03-16: Discard statement support (`discard!()`)

Added `discard!()` macro for the WGSL `discard` statement (fragment shaders only).
- `discard!()` transpiles to `discard;` in WGSL
- On the CPU side, sets a thread-local flag checked by `dispatch_fragments` to suppress the fragment output
- Execution continues after `discard!()` (matching WGSL semantics where helper invocations
  continue running for derivative computation), but the output is discarded
- Can be called from any function reachable from a fragment entry point, not just the entry point itself
- Storage/workgroup writes are not yet suppressed on the CPU side for discarded invocations (future work)

### 2026-05-02: Bumped wgpu and naga to 29

Migrated `PipelineLayoutDescriptor::bind_group_layouts` to `&[Option<&BindGroupLayout>]`, switched `InstanceDescriptor::default()` to `new_with_display_handle` / `new_without_display_handle`, and updated `Surface::get_current_texture()` callers to match the new `CurrentSurfaceTexture` enum. Examples now reconfigure the surface on `Outdated` and skip the frame on `Timeout` / `Occluded` / `Lost` / `Validation` instead of panicking.

### 2026-03-22: Roundtrip tests — bit manipulation, bitcast, packing

Extended `roundtrip-tests` with 9 new sub-tests covering bit manipulation (clz,
popcount, ctz, reverse_bits, first_leading/trailing_bit, extract/insert_bits),
bitcast (scalar and vec4 roundtrips), and pack/unpack (4x8/2x16 snorm/unorm/float).
Discovered naga/Metal backend bug with `firstLeadingBit(0xFFFFFFFF_u)`.

### 2026-03-22: Roundtrip tests and fract/round bug fixes

Added `roundtrip-tests` crate — a standalone binary that validates GPU vs CPU coherence
for core numeric builtins (trig, exponential, rounding, clamping, geometric). Found and
fixed `fract` (was using `self - trunc(self)` instead of WGSL's `self - floor(self)`)
and `round` (was rounding half away from zero instead of WGSL's half to even).

### 2026-01-24: RuntimeArray<T> support

Added `RuntimeArray<T>` type for runtime-sized arrays (WGSL `array<T>` without size parameter).
These are used in storage buffers, typically as the last field of a struct.
On CPU, `RuntimeArray<T>` is backed by `Vec<T>` with full indexing support.
Transpiles to `array<T>` in WGSL.

### 2026-04-08: Generic function monomorphization

Added support for generic free functions via macro-time monomorphization. Turbofish syntax
required at call sites (e.g., `foo::<f32>(x)`). Trait bounds are Rust-only and stripped from
WGSL output. Supports transitive generic calls, multiple type params, and deduplication.
Generic impl methods and entry points are explicitly rejected (MVP scope).

### 2026-04-17: Generic struct monomorphization

Added same-module generic struct support. `pub struct Pair<T> { a: T, b: T }` used as
`Pair<f32>` monomorphizes to `struct Pair_f32 { a: f32, b: f32 }` in WGSL. Generic impl
blocks (`impl<T> Pair<T>`) produce concrete methods (`Pair_f32_first`, etc.). Struct
construction `Pair::<f32> { a: 1.0, b: 2.0 }` becomes `Pair_f32(1.0, 2.0)` in WGSL.
Cross-module generic structs are supported via the same template infrastructure
used for generic functions — the struct definition and all its impl methods are
combined into a single template with `__TP__` placeholders.

### 2026-04-17: Removed `#[input]`/`#[output]` attributes

Inter-stage IO structs no longer need `#[input]` or `#[output]` wrapper
attributes. The `#[wgsl]` macro now strips IO field attributes (`#[builtin]`,
`#[location]`, `#[interpolate]`, `#[blend_src]`, `#[invariant]`) directly from
the emitted Rust output via a new `StripIoAttrs` visitor, following the same
pattern as the existing `StripWgslAllowAttrs` visitor. The two attributes were
semantically identical and only existed to clean up field attributes for Rust
compilation. Removing them enables the natural pattern of using a single struct
as both vertex output and fragment input, mirroring standard WGSL.

### 2026-05-06: Generic linkages and shader entry points

Shader entry points (`#[vertex]`, `#[fragment]`, `#[compute]`) and module
linkages (`uniform!`, `storage!`, `workgroup!`) can now declare type
parameters. A module containing such generics produces a *template* WGSL
source with `__TP{name}__` placeholders; `Module::instantiate(&["f32"])`
substitutes them at runtime to produce a concrete shader. Rust-side
linkage statics use `Uniform`/`Storage`/`Workgroup` with a default
`WgslTypeVariable` type parameter, backed by a `TypeId`-keyed map inside
`ModuleVar`; access is via the new `get!(VAR, T)` / `get_mut!(VAR, T)`
two-arg form. Per-entry-point type params are unioned into a single
`module_type_params` slice on `Module`. Linkage-wgpu generation and
WGSL validation are skipped for template modules — callers must
instantiate first.

### 2026-05-07: AST-at-runtime overhaul (`wgsl-rs-ir` crate)

Replaced the string-based `__TP{name}__` placeholder system with an
owned IR. New crate `wgsl-rs-ir` defines `Module`, `Type`, `Expr`,
`Stmt`, `Item`, etc., with `String`/`Vec<T>` storage and no `syn`
dependency, plus `render_module` (IR → WGSL) and `substitute_types`
(walks the IR replacing `Type::TypeParam`). The proc-macro converts
`parse::*` to `ir::*` and emits `fn() -> ir::Module` constructors;
`Module::ir_constructor` replaces the `source: &[&str]` field, and
`Module::wgsl_source()` now returns `String`. `instantiate(&[ir::Type])`
performs IR-level substitution + `rename_items` to mangle generic
template instances (e.g. `Pair` → `Pair_f32`). `GenericTemplate.ir_constructor`
and `TemplateInstantiation.type_args_constructor` are also `fn` pointers.
Compile-time WGSL validation has been dropped; runtime validation via
`Module::validate()` and the auto-generated `__validate_wgsl` test cover
the same ground. The legacy `code_gen::formatter` is kept around for
internal `monomorphize.rs` tests but no longer drives production
output.

### 2026-05-08: Removed legacy `code_gen::formatter`

Migrated the last consumers of the legacy direct-to-WGSL formatter to
the IR pipeline and deleted `code_gen.rs` and `code_gen/formatter.rs`
(~2,165 lines). The `mono_wgsl` helper in `monomorphize.rs` tests now
runs `ir_convert::items_from_parse` + `wgsl_rs_ir::render_items`. The
~100 `to_wgsl()` test call sites in `parse.rs` are unchanged — a
test-only `ToWgsl` trait now provides the same method via the IR
pipeline. The IR crate exposes new public helpers `render_type`,
`render_expr`, `render_stmt`, `render_block`, and `render_item` for
rendering individual nodes. Test expected strings were updated where
the IR renderer's format differs from the old formatter (binary ops
gain spaces; `Vec4<f32>` renders as the WGSL shorthand `vec4f`).
Several dead helpers in `monomorphize.rs` (the
`flatten_struct_placeholder*` family and `type_to_wgsl`) were also
deleted.
