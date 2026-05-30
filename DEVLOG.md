# devlog

## design decisions

### 2025-12-08: User must write regular Rust code

No translation of Rust code will occur in the `#[wgsl]` macro.
The macro is strictly additive in that it _adds_ more Rust code
to the user's written code.
Specifically, it will add the transpiled WGSL source code and imports.

### 2025-12-08: Swizzles as function calls

Swizzles are tricky because in Rust they could be accomplished with traits like `glam`'s
`Vec4Swizzle` trait (and friends), but in WGSL they look like field accessors.
Because the Rust code must be un-altered (see 2025-12-08 design decision above), we use
functions similar to the `Vec4Swizzle` strategy. So to swizzle `xyz` in WGSL you would
just call `.xyz()` in Rust.

### 2025-12-08: One-trait-per-numeric-builtin strategy

WGSL numeric builtin functions have small differences across type overloads, and some
may eventually need generic associated types for correct typing. Rather than a single
mega-trait, each builtin gets its own trait (e.g., `NumericBuiltinAbs`, `NumericBuiltinAcos`).
This has proven flexible enough to handle all builtins encountered so far.

### 2025-12-10: Default annotations on vertex/fragment return values

Rust can't put annotations on return types, so IO locations and builtins must be specified
on the shader stage attribute macros (`#[vertex]`, `#[fragment]`, `#[compute]`).

When a vertex shader returns a plain `Vec4f` without an explicit location or builtin,
`wgsl-rs` automatically inserts `@builtin(position)`. For fragment shaders returning `Vec4f`,
it inserts `@location(0)`. For custom IO, users annotate struct fields with `#[builtin(...)]`
and `#[location(...)]`, mirroring WGSL's `@builtin` / `@location` syntax.

### 2025-12-27: Rust type system catches all WGSL errors

`wgsl-rs` should never produce Rust code that compiles but produces invalid WGSL.
If your Rust compiles, your shader will too. The Rust type system serves as the
first line of defense against shader errors.

### 2025-12-27: Macros for non-Rust WGSL constructs

Rust folks are used to macros, and we **must** use macros for all WGSL syntax that
can't be represented with valid Rust. For example, `uniform!(binding(0), group(0), BRIGHTNESS: f32);`
declares a WGSL uniform binding — there is no way to express this in plain Rust syntax.

### 2025-12-27: Module import strategy

With `wgsl-rs` you can import other modules, but with significant restrictions.
You can only glob-import other WGSL modules written with `wgsl-rs`, or `wgsl_rs::std::*`.
Importing arbitrary modules produces a compiler error about `WGSL_MODULE` not being in scope.
This keeps the import graph visible to the proc-macro for full-module transpilation.

### 2026-01-03: `#[wgsl_allow]` for suppressing for-loop bounds warnings

`for i in 0..n` transpiles to `for (var i = 0; i < n; i++)` and `for i in 0..=n` transpiles
to `for (var i = 0; i <= n; i++)`. Only bounded ranges are supported (WGSL requires explicit bounds).

For-loops with non-literal bounds (where the bounds cannot be verified at compile-time to be
ascending) emit a compile error on stable since warnings can't be emitted through the
proc-macro system. On nightly they emit a warning. Use `#[wgsl_allow(non_literal_loop_bounds)]`
on the for-loop to suppress these errors/warnings.

### 2026-01-24: `RuntimeArray<T>` representation

`RuntimeArray<T>` represents WGSL runtime-sized arrays (`array<T>` without a size parameter),
typically used as the last field of a struct in storage buffers. On CPU, `RuntimeArray<T>`
is backed by `Vec<T>` with full indexing support. Transpiles to `array<T>` in WGSL.

### 2026-01-29: Pointer type representation with `ptr!` macro

`ptr!(address_space, T)` declares WGSL pointer types in function parameters.
Supports `function` and `private` address spaces (the only ones passable to functions
without extensions). Expands to `&mut T` on CPU and transpiles to `ptr<function, T>` or
`ptr<private, T>` in WGSL. Both `&x` and `&mut x` transpile to `&x` in WGSL since
mutability is determined by access mode. Includes dereference operator (`*`) support.

### 2026-01-31: Atomic types and workgroup variables

`Atomic<T>` represents WGSL `atomic<T>` (where T is `i32` or `u32`), backed on CPU by
`std::sync::atomic::{AtomicI32, AtomicU32}` with full atomic operations.

`workgroup!(NAME: TYPE)` declares workgroup-scoped variables shared across compute shader
invocations. Transpiles to `var<workgroup> NAME: TYPE;` in WGSL. On CPU, backed by
`LazyLock<RwLock<T>>` for thread-safe shared state.

### 2026-02-11: Variadic WGSL builtins via multi-function name mapping

WGSL builtins can be variadic and called with different types, potentially returning
different types. This presents two sub-problems:

1. **Parameter types**: When a parameter type differs between WGSL builtin "flavors",
   we use a trait that makes the parameter types dependent on a type parameter — either
   through an associated type or `impl AnotherTrait` bounds.

2. **Variadic functions**: When a WGSL builtin accepts varying parameter counts,
   we create multiple Rust functions, one per variation. Each Rust function maps to the
   same WGSL builtin via `BUILTIN_NAME_CASE_MAP`. For example:
   * `texture_sample` → `textureSample`
   * `texture_sample_array` → `textureSample`
   * `texture_sample_array_offset` → `textureSample`

### 2026-02-13: Synchronization builtins as CPU no-ops

Barrier functions (`storageBarrier()`, `workgroupBarrier()`, `textureBarrier()`) are no-ops
on the CPU side since there is no parallel dispatch runtime yet. `workgroupUniformLoad()`
uses a `WorkgroupUniformLoad` trait with impls for `Workgroup<T: Clone>` and
`Workgroup<Atomic<{u32,i32}>>`. The `ptr!` macro and parser were extended to support the
`workgroup` address space.

### 2026-03-16: `discard!()` via thread-local flag

`discard!()` transpiles to `discard;` in WGSL fragment shaders. On the CPU side, it sets a
thread-local flag checked by `dispatch_fragments` to suppress the fragment output. Execution
continues after `discard!()` — matching WGSL semantics where helper invocations keep running
for derivative computation — but the output is discarded. Works from any function reachable
from a fragment entry point, not just the entry point itself.

### 2026-04-08: Generic function monomorphization at macro time

Generic free functions are monomorphized at macro time. Turbofish syntax is required at call
sites (e.g., `foo::<f32>(x)`). Trait bounds are Rust-only and stripped from WGSL output.
Supports transitive generic calls, multiple type parameters, and deduplication. Generic impl
methods and entry points are explicitly rejected (MVP scope).

### 2026-04-17: Generic struct monomorphization

Same-module generic structs monomorphize to concrete WGSL structs. `pub struct Pair<T> { a: T, b: T }`
used as `Pair<f32>` becomes `struct Pair_f32 { a: f32, b: f32 }` in WGSL. Generic impl blocks
(`impl<T> Pair<T>`) produce concrete methods (`Pair_f32_first`, etc.). Struct construction
`Pair::<f32> { a: 1.0, b: 2.0 }` becomes `Pair_f32(1.0, 2.0)` in WGSL. Cross-module generic
structs use the same template infrastructure as generic functions — the struct definition and
all its impl methods are combined into a single template with `__TP__` placeholders.

### 2026-04-17: Strip IO attrs via `StripIoAttrs` visitor

Inter-stage IO structs no longer need `#[input]` or `#[output]` wrapper attributes. The
`#[wgsl]` macro strips IO field attributes (`#[builtin]`, `#[location]`, `#[interpolate]`,
`#[blend_src]`, `#[invariant]`) from emitted Rust output via a `StripIoAttrs` visitor, following
the same pattern as the existing `StripWgslAllowAttrs` visitor. The two wrapper attributes were
semantically identical and only existed to clean up field attributes for Rust compilation.
Removing them enables the natural pattern of using a single struct as both vertex output and
fragment input, mirroring standard WGSL.

### 2026-05-06: Generic linkages via template modules

Shader entry points (`#[vertex]`, `#[fragment]`, `#[compute]`) and module linkages
(`uniform!`, `storage!`, `workgroup!`) can declare type parameters. A module with such
generics produces a *template* WGSL source with `__TP{name}__` placeholders;
`Module::instantiate` substitutes concrete types at runtime. Rust-side linkage statics
use `Uniform`/`Storage`/`Workgroup` with a default `WgslTypeVariable` type parameter,
backed by a `TypeId`-keyed map inside `ModuleVar`. Access is via the `get!(VAR, T)` /
`get_mut!(VAR, T)` two-arg form. Per-entry-point type params are unioned into a single
`module_type_params` slice on `Module`. Linkage-wgpu generation and WGSL validation are
skipped for template modules — callers must instantiate first.

### 2026-05-07: IR crate for runtime type substitution

Replaced the string-based `__TP{name}__` placeholder system with an owned IR. New crate
`wgsl-rs-ir` defines `Module`, `Type`, `Expr`, `Stmt`, `Item`, etc., with `String`/`Vec<T>`
storage and no `syn` dependency, plus `render_module` (IR → WGSL) and `substitute_types`
(walks the IR replacing `Type::TypeParam`). The proc-macro converts `parse::*` to `ir::*`
and emits `fn() -> ir::Module` constructors. `instantiate(&[ir::Type])` performs IR-level
substitution with `rename_items` to mangle generic template instances (e.g., `Pair` →
`Pair_f32`). Compile-time WGSL validation was dropped in favor of runtime validation via
`Module::validate()` and auto-generated `__validate_wgsl` tests.

### 2026-05-10: Non-square matrix IR representation

`ir::Type::Matrix` stores separate `columns: u8, rows: u8` fields, enabling all 9 WGSL
matrix shapes (`mat2x2` through `mat4x4`). Parse-side recognition added for `MatCxRf`
shorthand (e.g., `Mat2x3f`, `Mat3x4f`).

### 2026-05-11: Decoupled generic linkages

Generic linkage variables are decoupled from entry-point type parameters. Linkage variable
syntax is now `impl Trait`: `uniform!(group(0), binding(0), FRAME: impl Convert<f32>)`.
Bare-ident generics (`FRAME: T`) are no longer recognised; the previous module-level
coupling between linkage and entry-point type params has been removed.

Module-level type parameters in the IR use distinct, unambiguous names. Linkage variables
contribute their own identifier (e.g., `"FRAME"`); entry-point type parameters use a
positional encoding (`"<fn_name>_<index>"`, e.g., `"frag_main_0"`). This is implemented
via `ParseContext::type_param_renames`, populated for entry-point function bodies but left
empty for non-entry generic functions and structs, which still go through same-module
monomorphization on source names.

### 2026-05-11: `Type<Is = ...>` trait for compile-time type enforcement

A unified `instantiate` function replaces the typestate `ModuleBuilder`. `get!(VAR, T)` or
`get_mut!(VAR, T)` inside an entry point generates a constraint `VAR: Type<Is = T>` in
the `where` clause of `instantiate`, so conflicting types are caught by the Rust compiler
at compile time. The `Type` trait lives in `wgsl_rs::linkage`: `T: Type<Is = U>` is
satisfied iff `T` and `U` are the same type.

### 2026-05-15: WgslExtension trait and IR attributes

Downstream crates need to inspect and modify the IR before type instantiation
(e.g., a `crabslab` extension adding `SlabItem`-aware field offsets).
`ir::Attribute` stores Rust `#[...]` attributes as `(path: String, args:
Vec<String>)` pairs on every IR item, `Field`, `FnArg`, and `Module`. Attributes
are not rendered in WGSL output — they exist solely for extension inspection,
intentionally duplicating some information already in dedicated fields like
`fn_attrs` and `inter_stage_io`.
The `WgslExtension` trait (`fn modify_ir(&mut Module)`) is called in the
generated constructor via `#[wgsl(extensions = [path::Ext1, ...])]`, after IR
construction but before type instantiation. `FnArgs` now accept multiple
attributes, restricting only to one inter-stage IO annotation.

### 2026-05-18: Bijective name mangling

Replaced ad-hoc `{a}_{b}` concatenation with a centralized `wgsl_rs_ir::mangle` module
using a simplified subset of `wesl-rs`'s `EscapeMangler`: each component containing `N>0`
underscores is rewritten as `_N{comp}` before components are joined with `_`, making
mangling a bijection. This prevents collision bugs like `Foo_bar::baz` vs `Foo::bar_baz`.
Entry-point type-param slot encoding `{fn}_{i}` is intentionally left raw — its collision
risk is covered by the reserved-names check and changing it would break the public API.

### 2026-05-26: Doc-visible binding macros for runtime access

`texture!` and `sampler!` declare a two-level binding pattern: a `#[doc(hidden)]`
private backing `static __NAME` that owns the resource, and a `pub const NAME`
reference for callers. Previously both items were `#[doc(hidden)]`, which hid
bindings from downstream users who need to know what resources a module exposes.
The `pub const NAME` binding is now doc-visible, consistent with `storage!`,
`uniform!`, and `workgroup!` which all generate visible `pub static` bindings.
Only the backing `static __NAME` remains `#[doc(hidden)]` — it exists solely so
callers can pass the binding by value (without `&`) to texture/sampler functions.

### 2026-05-26: Depth texture fill via `frag_depth` render pass

Metal forbids uploading data to `Depth32Float` textures. Rather than gating
depth tests per-platform, depth values are rendered using a fragment shader
that outputs `#[builtin(frag_depth)]` with a deterministic position-based
formula. The CPU path writes the same data directly via `TextureDepth2D::set()`.
Both paths compute identical pixel values — only the delivery mechanism differs.
This pattern is reusable for any future cross-platform test that needs
pre-populated depth data.

### 2026-05-29: `wgsl-rs-layout` — a standalone extension crate

`wgsl-rs` needs to inform tools of how to marshal data to/from the GPU. The
WGSL spec §14.4.1 defines strict alignment, size, and offset rules for all
types, and getting these wrong silently produces bad data. A `#[derive(Layout)]`
macro computes these values at compile time.

The extension lives in two new crates:
- `wgsl-rs-layout` (regular lib): `WgslLayout` and `Layout` traits, `FieldLayout`
  struct, and `WgslLayout` impls for all WGSL scalar/vector/matrix/array types.
- `wgsl-rs-layout-macros` (proc-macro): `#[derive(Layout)]` for structs.

This is the first extension that dogfoods `wgsl-rs` as a dependency — it depends
on `wgsl-rs` for concrete type definitions (`Vec3f`, `Mat4x4f`, `Atomic<u32>`,
etc.) but is otherwise self-contained. It demonstrates the extension pattern
where downstream crates consume `wgsl-rs` types directly.

**How it works:**
1. The derive macro generates an inherent `impl` with private associated constants
   (`__OFFSET_0`, `__SIZE_0`, `__ALIGN_0`, ...) for each field, computed via
   `roundUp` per the spec's recursive definition. These are accessible from both
   the `WgslLayout` and `Layout` trait impls through `Self::`.
2. `WgslLayout` is implemented for all built-in types (scalars, vectors, matrices,
   arrays). Each type is a simple const mapping per the spec table.
3. `Layout` extends `WgslLayout` with `FIELDS: &'static [FieldLayout]` for
   per-field offset/size/align/pad_before info.
4. Generic structs are supported — type parameters receive a `T: WgslLayout` bound.

**`FieldLayout::pad_before`:**
Inter-field padding is the gap between successive fields caused by alignment
requirements. `pad_before` on field `i` is the dead bytes between the end of
field `i-1` and the start of field `i`. It is always 0 for the first field.
Tools *must* write zero bytes into these gaps when marshalling data. The
`FieldLayout` struct documents this prominently with concrete examples.

**Runtime arrays:** `RuntimeArray<T>::SIZE` is 0 (runtime-dependent).

**Empty structs:** `SIZE = 0, ALIGN = 1, FIELDS = &[]` (identity element,
consistent with common practice; WGSL spec does not define this case).

**What this is NOT:** The derive only computes WGSL memory layout — it does not
attempt to align Rust CPU-side layout to match. `#[repr(C)]` on a struct with a
`Vec3f` field gives Rust align 4, but WGSL `vec3<f32>` has align 16. The crate
stays focused on answering "where do bytes go in the GPU buffer?"

### 2026-05-29: Inherent consts over deeply nested inline expressions

The initial inline expression approach (`roundUp(roundUp(0 + s0,a1) + s1,a2)`)
broke on structs with 3+ fields because Rust's const evaluator hits complexity
limits. The fix uses inherent associated constants (`Self::__OFFSET_0`,
`Self::__SIZE_0`, etc.) to break the recursive computation into individually
evaluable steps. These consts live on the struct (not any trait impl) so they're
visible from both the `WgslLayout` and `Layout` trait impls.
