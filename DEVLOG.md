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
substitutes them at runtime to produce a concrete shader.
*(The string-based `instantiate(&[&str])` API was superseded by
`instantiate(&[ir::Type])` the next day — see the 2026-05-07 entry.)*
Rust-side linkage statics use `Uniform`/`Storage`/`Workgroup` with a
default `WgslTypeVariable` type parameter, backed by a `TypeId`-keyed
map inside `ModuleVar`; access is via the new `get!(VAR, T)` /
`get_mut!(VAR, T)` two-arg form. Per-entry-point type params are unioned
into a single `module_type_params` slice on `Module`. Linkage-wgpu
generation and WGSL validation are skipped for template modules —
callers must instantiate first.

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
output. *(The formatter was removed entirely the next day — see the
2026-05-08 entry below.)*

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

### 2026-05-09: Visitor trait refactor of `monomorphize.rs`

Replaced 7 hand-written recursive AST walkers in `monomorphize.rs`
(~1,800 lines of structural traversal boilerplate) with a single
`ParseVisitorMut` trait + family of `walk_*` functions in a new
`parse_visitor` module (~400 lines). Each walker family is now a
visitor struct that overrides only the `visit_*` methods relevant to
its job; the structural recursion is shared via `walk_*` defaults.

Walker conversions:
* `SubstituteVisitor` — replaces every `Type::TypeParam` with a
  concrete type from a `BTreeMap`, plus the equivalent rewrites in
  `FnPath::TypeMethod` and `Expr::Struct` idents.
* `RewriteNamesVisitor` — merges the previous `rewrite_calls_in_*`
  and `rewrite_struct_types_in_*` walkers into one pass that mangles
  both fn-template call sites and struct-type references.
* `CheckUnresolvedVisitor` — errors if any call to a known generic
  template lacks turbofish type args.
* `CrossModuleVisitor` — collects cross-module instantiations and
  rewrites the call sites' names.
* `ScanDepsVisitor` — collects template→template dependency edges
  with type-param index mappings.
* `MonoCtx` itself implements `ParseVisitorMut` for the
  instantiation-discovery walk.

Three correctness bugs were also fixed: `check_unresolved_in_stmt`,
`collect_cross_module_from_stmt`, and `scan_stmt_for_deps` previously
inlined a one-level `else if` peel that failed to recurse through
deeply-nested else-if chains. The visitor trait's shared `walk_if`
recurses correctly, so the bug class is now structurally impossible.
Two regression tests in `monomorphize::test` lock in the fix.

Net change: `monomorphize.rs` shrunk from 3,453 → 2,289 lines (-1,164),
plus 402 lines of new shared `parse_visitor` machinery — net **-762
lines** with cleaner structure and better correctness.

### 2026-05-10: Non-square matrices in IR + `Wgsl` impls for std types

Extended `ir::Type::Matrix` from a single `size: u8` to separate
`columns: u8, rows: u8`, enabling all 9 WGSL matrix shapes (`mat2x2`
through `mat4x4`). Parse-side recognition added for `MatCxRf` shorthand
(e.g. `Mat2x3f`, `Mat3x4f`). Added `Wgsl` impls for `bool`, all
`Vec{2,3,4}<T>` over `WgslScalar`, blanket `[T; N]` and
`RuntimeArray<T>`, all 9 matrix types, `Sampler`, `SamplerComparison`,
all sampled and depth texture types. Fixed and hardened the
`#[derive(Wgsl)]` macro: it now uses fully-qualified paths via a
`#[wgsl_path(...)]` helper attribute (defaulting to `::wgsl_rs`) and
correctly emits `Type::Scalar(...)` for enums.

### 2026-05-11: Decoupled generic linkages + typestate builder

Decoupled generic linkage variables from entry-point type parameters
and added a typestate `ModuleBuilder` for ergonomic, compile-time-safe
template instantiation.

**Linkage variable syntax** is now `impl Trait`:
`uniform!(group(0), binding(0), FRAME: impl Convert<f32>)`. The trait
bounds are preserved on the parsed `ItemUniform`/`ItemStorage`/
`ItemWorkgroup` (new `impl_bounds` field) and replayed on the builder's
setter methods. Bare-ident generic linkages (`FRAME: T`) are no longer
recognised; the previous module-level coupling between linkage and
entry-point type params has been removed.

**Module-level type parameters** in the IR now use distinct, unambiguous
names. Linkage variables contribute their own identifier (e.g.
`"FRAME"`); entry-point type parameters use a positional encoding
(`"<fn_name>_<index>"`, e.g. `"frag_main_0"`, `"frag_main_1"`). This is
implemented via a new `ParseContext::type_param_renames` map that's
populated for entry-point function bodies but left empty for non-entry
generic functions and structs, which still go through same-module
monomorphization on source names.

**Typestate builder** (`ModuleBuilder`) is generated next to
`WGSL_MODULE` for any module with `impl Trait` linkage variables or
generic entry points. One `Needs<X>` / `Has<X>` marker pair is emitted
per slot; `set_<linkage>::<T>()` and
`instantiate_<entry>::<T0, T1, ...>()` methods transition slots from
"needs" to "has", and `build()` is only available when every slot is
bound. Each method's bounds are the original user-written bounds from
the source plus a synthetic `+ Wgsl` so the builder can call
`<T as Wgsl>::to_ir()` to materialise the `ir::Type` at runtime. The
existing `Module::instantiate(&[ir::Type])` API stays as the lower
level escape hatch.

`parse::ItemFn` gained a `syn_generics: Option<syn::Generics>` field
that preserves the original signature generics on entry points so the
builder codegen can replay them verbatim.

#### Unified `instantiate` function replaces typestate builder

The typestate `ModuleBuilder` was replaced by a unified `instantiate`
function that uses `wgsl_rs::linkage::Type<Is = ...>` trait constraints
to enforce type consistency across entry points at compile time. A `get!(VAR, T)`
or `get_mut!(VAR, T)` call inside an entry point generates a constraint
`VAR: Type<Is = T>` in the `where` clause, so conflicting types are
caught by the Rust compiler.

`Module::instantiate` and `Module::instantiate_scalar` were removed in
favor of the generated `instantiate` function. The `Type` trait lives in
`wgsl_rs::linkage`: `T: Type<Is = U>` is satisfied iff `T` and `U` are
the same type.

The type argument in `get!(VAR, TYPE)` / `get_mut!(VAR, TYPE)` is stored
as a `syn::Type` (not `parse::Type`) since it represents a Rust-side type
expression used for code generation, not a WGSL type. This allows type
params inside built-in generic types like `Vec4<T>` which `Type::parse`
rejects as requiring a concrete scalar. Entry-point type params with
colliding names (e.g. `T` in two different functions) are disambiguated
with a `_fnname` suffix (e.g. `T_main_zeroable`).

#### Bug fixes

Replaced the typestate builder with unified `instantiate` function using
`Type<Is = ...>` trait constraints; added `#[allow(non_camel_case_types)]` to
suppress warnings on suffixed type params like `T_frag_main`.

#### Test validation

- Replaced typestate builder with unified `instantiate` function using
  `Type<Is = ...>` trait constraints; added `#[allow(non_camel_case_types)]` to
  suppress warnings on suffixed type params like `T_frag_main`.

- Added auto-generated WGSL validation tests for all modules (not just imports);
  added `validate_with_instantiation_types(T1, T2, ...)` attribute for template
  modules; removed dead `WgslValidate` variant and naga dep from proc-macro
  crate; added `validate_wgsl_source()` free function; surfaced pre-existing
  monomorphization bug in `generic_structs` example.

### 2026-05-12: Address PR #101 review comments

Fixed stale `FRAME: T` docs to use `impl Trait` syntax; added `Module.id`
field (atomic counter) for diamond-import deduplication and module-identity-
based instantiation dedupe; boxed large enum variants in `parse.rs` per
clippy; added `WgslTextureScalar` trait (f32/i32/u32 only) to prevent
`Texture2D<bool>`; added compile-time rejection of `get!`/`get_mut!` in
const initializers with trybuild test.

### 2026-05-13: PR #101 review round 2 — validation feature gating, README fix, doc cleanup

- Fixed README validation table: all non-template modules get auto-generated
  `__validate_wgsl` tests, not just modules with imports.
- Added `validation` feature to `wgsl-rs-macros` Cargo.toml; `wgsl-rs`
  propagates it. `gen_validation_test` and `gen_instantiated_validation_tests`
  now check `cfg!(feature = "validation")` so no tests are generated when
  the feature is off (avoids compile errors when `default-features = false`).
- Removed stray doc comment above `collect_linkage_constraints` in builder.rs.
- Replied to all Copilot review comments (validate Err, validation gating,
  README correction, doc cleanup).

### 2026-05-18: Robust name mangling (issue #112)

- Replaced ad-hoc `{a}_{b}` concatenation with a centralized
  `wgsl_rs_ir::mangle` module that adopts a simplified subset of
  wesl-rs's `EscapeMangler`: each component containing `N>0` underscores
  is rewritten as `_N{comp}` before components are joined with `_`,
  making mangling a bijection. Migrated all sites: `render.rs`
  (impl-method/const, enum variant, `FnPath::TypeMethod`,
  `Expr::TypePath`), `monomorphize.rs` (`mangle_name`, `mangle_type`,
  `type_to_key`, all 9 duplicate `{self_ty}_{method}` sites), and the
  runtime instance-name builder in `wgsl-rs/src/lib.rs`. Entry-point
  type-param slot encoding `{fn}_{i}` is intentionally left raw — its
  collision risk is already covered by the reserved-names check and
  changing it would break the public dispatch API surface. Added 9
  collision-pair regression tests covering the cases from the issue
  (`Foo_bar::baz` vs `Foo::bar_baz`, `Color_Red::Hot` vs
  `Color::Red_Hot`, underscored consts, underscored methods on
  underscored types, etc.).
