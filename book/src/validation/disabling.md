# Disabling Validation

Validation is on by default and runs through naga. There are two granularities at which you can disable it: per module, or globally via Cargo features.

## Per Module: `#[wgsl(skip_validation)]`

Annotate a single module to suppress its auto-generated `__validate_wgsl` test:

```rust
#[wgsl(skip_validation)]
pub mod generic_structs { /* ... */ }
```

Effects:

- No `#[test] fn __validate_wgsl()` is emitted for that module.
- `WGSL_SOURCE` is still produced and usable at runtime.
- Other modules in the crate are still validated normally.

Use this when a module intentionally cannot pass naga (e.g. it exercises a known transpiler bug such as the generic struct constructor mangling issue, or it relies on a WGSL extension naga does not yet support). The `generic_structs` example uses it for exactly this reason:

```rust
#[wgsl(skip_validation)]
pub mod generic_structs {
    pub struct Pair<T: Copy> { pub a: T, pub b: T }
}
```

## Globally: `default-features = false`

Validation pulls in naga, which is a non-trivial dependency. To remove it entirely from your build, disable the default features of the `wgsl-rs` crate:

```toml
[dependencies]
wgsl-rs = { version = "...", default-features = false }
```

Effects:

- The `validation` feature is off.
- `WGSL_SOURCE.validate()` and `ir::Module::validate()` are **removed** (not stubbed). Any call site will fail to compile, so you must also remove your own calls to these methods.
- The auto-generated `__validate_wgsl` tests are **not** emitted for any module, so `cargo test` no longer validates shaders.
- `WGSL_SOURCE` and `ir::Module::to_wgsl()` are still available — generation is unaffected, only validation is removed.

If you later want validation back in a specific build (e.g. CI), enable the feature explicitly:

```sh
cargo test --features wgsl-rs/validation
```

## Choosing a Granularity

| Goal                                          | Use                                        |
| --------------------------------------------- | ------------------------------------------ |
| Skip one known-bad or extension-requiring module | `#[wgsl(skip_validation)]`                |
| Ship a binary without naga in the dependency tree | `default-features = false`              |
| Validate in CI but not in release builds      | feature-gate your own `validate()` calls   |

Disabling validation does **not** change the generated WGSL text — only whether it is checked. Always run validation somewhere in your pipeline (CI, dev builds, or an explicit test) before shipping shaders.