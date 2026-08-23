# Auto-generated Tests

For every non-template `#[wgsl]` module, the macro automatically generates a hidden test that validates the transpiled WGSL through [naga](https://github.com/gfx-rs/naga). Running `cargo test` therefore validates every shader in the crate.

## What Gets Generated

Given:

```rust
#[wgsl]
pub mod example {
    use wgsl_rs::std::*;
}
```

the macro emits (approximately):

```rust
#[test]
fn __validate_wgsl() {
    example::WGSL_SOURCE.validate().expect("WGSL validation failed");
}
```

The test name is fixed; there is one per `#[wgsl]` module. A failure prints the naga diagnostic and the failing module path.

## Running the Tests

```sh
cargo test                          # validate every #[wgsl] module
cargo test example                  # narrow to one module
cargo test __validate_wgsl          # run only the auto-generated validation tests
```

A passing test means the generated WGSL parses and passes naga's type-check for the module's declared bindings, entry points, and function bodies.

## Template Modules are Excluded

A template module (one with type-parameterized entry points or `impl Trait` linkages) emits WGSL containing `TypeParam` placeholders. That text is **not** valid WGSL on its own, so the macro does not generate an auto-test for templates.

To validate a template, supply concrete types via:

```rust
#[wgsl(validate_with_instantiation_types(f32, f32))]
pub mod hello_triangle_generic { /* ... */ }
```

The attribute makes the auto-test instantiate the template with the given types and validate the resulting concrete module. The number and order of types must match the template's declared type parameters.

See [Runtime Validation](./runtime.md) for validating instantiated templates from your own code, and [Templates](../generics/templates.md) for the broader template story.

## Skipping a Single Module

If a specific module must opt out of auto-validation (e.g. it intentionally exercises a known transpiler bug, or it depends on an extension naga does not yet support), annotate it:

```rust
#[wgsl(skip_validation)]
pub mod example { /* ... */ }
```

No `__validate_wgsl` test is generated for that module. Other modules are unaffected. See [Disabling Validation](./disabling.md).