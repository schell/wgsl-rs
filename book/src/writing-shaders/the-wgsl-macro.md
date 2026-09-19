# The `#[wgsl]` Macro

The `#[wgsl]` attribute macro is the entry point to wgsl-rs. It is applied to a module and transpiles the Rust inside it to WGSL. The generated source string is stored in a `WGSL_SOURCE` constant, and the module remains valid Rust that runs on the CPU.

## Syntax and Placement

`#[wgsl]` is placed on a `pub mod`:

```rust
#[wgsl]
pub mod example {
    use wgsl_rs::std::*;

    pub fn square(x: f32) -> f32 {
        x * x
    }
}
```

The macro produces a `pub static WGSL_SOURCE: &str` containing the transpiled WGSL. You can read it at runtime:

```rust
println!("{}", example::WGSL_SOURCE);
```

For the module above, the generated WGSL is:

```wgsl
fn square(x: f32) -> f32 {
  return x * x;
}
```

The code inside the module is ordinary Rust: type-checks on the CPU, runs in `cargo test`, and transpiles to WGSL for the GPU.

## Macro Attributes

Attributes are passed inside the `#[wgsl(...)]` list.

### `crate_path`

When the macro cannot locate the `wgsl_rs` crate (for example, from within the crate itself), set the path explicitly:

```rust
#[wgsl(crate_path = "crate")]
pub mod example {
    use wgsl_rs::std::*;
}
```

### `skip_validation`

Disable validation of the generated WGSL:

```rust
#[wgsl(skip_validation)]
pub mod example { use wgsl_rs::std::*; }
```

### `validate_with_instantiation_types`

Validate the module with concrete types for generic/template entry points:

```rust
#[wgsl(validate_with_instantiation_types(f32, u32))]
pub mod example { use wgsl_rs::std::*; }
```

Multiple types may be passed as a comma-separated list.

### `extensions`

Enable WGSL extensions during validation:

```rust
#[wgsl(extensions = [wgsl_rs::WgslExtension::Fxaalp32)] // pseudonymous
pub mod example { use wgsl_rs::std::*; }
```

Extensions are listed inside the `extensions = [...]` array and must be referenced by their full path in `wgsl_rs::WgslExtension`.

## `#[wgsl_ignore]`

Items annotated with `#[wgsl_ignore]` are compiled as Rust but omitted from WGSL generation:

```rust
#[wgsl]
pub mod example {
    use wgsl_rs::std::*;

    pub fn gpu_only(x: f32) -> f32 {
        x * 2.0
    }

    #[wgsl_ignore]
    pub fn cpu_helper(x: f32) -> f32 {
        x.sin() // not transpiled
    }
}
```

## `#[wgsl_allow(...)]`

Suppress transpiler warnings on an expression. Allowed flags:

| Flag | Purpose |
| --- | --- |
| `non_literal_loop_bounds` | A `for` loop bound that is not a literal or `const`. |
| `non_literal_match_statement_patterns` | `match` patterns that are not literal/const (e.g. or-patterns). |

```rust
#[wgsl]
pub mod example {
    use wgsl_rs::std::*;

    pub fn loopy(n: u32) -> u32 {
        let mut s: u32 = 0;
        #[wgsl_allow(non_literal_loop_bounds)]
        for i in 0..n {
            s += i;
        }
        s
    }
}
```

See [Control Flow](./control-flow.md) for related usage of these flags.