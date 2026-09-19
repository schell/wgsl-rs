# Runtime Validation

In addition to the auto-generated `cargo test` checks, wgsl-rs exposes runtime validation APIs that let you validate shader source on demand — useful for instantiated templates, dynamically composed pipelines, and CI scripts that want to fail fast on bad WGSL.

## `WGSL_SOURCE.validate()`

Every `#[wgsl]` module exposes a `pub static WGSL_SOURCE: &str` containing the transpiled WGSL. When the `validation` feature is enabled, calling `.validate()` on it runs naga and returns a `Result`:

```rust
use example::hello_triangle::WGSL_SOURCE;

WGSL_SOURCE.validate().expect("hello_triangle failed validation");
```

This is exactly what the auto-generated `__validate_wgsl` test calls. See [Auto-generated Tests](./auto-tests.md).

## `module.validate()`

For template modules you instantiate at runtime, the returned `ir::Module` also has a `.validate()` method (with the `validation` feature):

```rust
use example::hello_triangle_generic as tmpl;
use wgsl_rs::ir;

let module: ir::Module = tmpl::instantiate::<f32, f32>();
module.validate().expect("instantiated template failed validation");
let source: String = module.to_wgsl();
```

`module.validate()` runs naga over the substituted, concrete WGSL — the same path the `validate_with_instantiation_types` attribute uses at test time, but driven from your own code.

## `validate_with_instantiation_types` at Runtime

The `validate_with_instantiation_types(T1, T2, ...)` helper is also callable as a runtime function on a template module:

```rust
use example::hello_triangle_generic as tmpl;

tmpl::validate_with_instantiation_types(f32, f32)
    .expect("template validation failed");
```

This is convenient when you want to validate several instantiations of the same template from a single test or application entry point. The argument list mirrors the template's declared type parameters.

## The `validation` Feature

Both `WGSL_SOURCE.validate()` and `module.validate()` require the `validation` feature on the `wgsl-rs` crate. It is enabled by default. If you turn it off (see [Disabling Validation](./disabling.md)), those methods are removed and any call site will fail to compile — there is no stub that silently returns `Ok`.

## Error Reporting

Validation errors surface naga's diagnostics directly. A typical failure looks like:

```text
WGSL validation failed: SomeWrongSnafu { ... }
  in module `hello_triangle`
  at @vertex fn vtx_main(...)
```

The error includes the offending module name (or the instantiated template's type arguments) and the naga span where available. For template modules, validate **after** instantiation so the error points at concrete WGSL rather than `TypeParam` placeholders.