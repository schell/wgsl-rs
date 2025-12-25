# wgsl-rs

**wgsl-rs** lets you write a subset of Rust code and automatically generate WGSL shaders and `wgpu` runtime linkage. With
`wgsl-rs`, you can run your code on the CPU in Rust, and use the generated WGSL on the GPU.

Procedural macros are provided by the [`wgsl-rs-macros`](./crates/wgsl-rs-macros) crate.

---

## Roadmap

- [x] `wgsl` macro for modules
  - [x] Translate a subset of Rust into WGSL
    - [x] Types
      - [x] Concrete scalars
      - [x] `vec{N}<{scalar}>`
      - [x] Vector aliases
      - [x] Arrays
      - [ ] Matrices
      - [x] Structs
      - [ ] Textures
      - [ ] Atomics
    - [x] Descriptor sets, bindings, etc.
    - [ ] Expand support for all WGSL syntax
  - [x] Glob-importing other `wgsl-rs` modules
  - [x] WGSL standard library for Rust
    - [x] Vector constructors (`vec2f`, `vec3f`, etc.)
    - [x] Binary operators (`+`, `-`, `*`, `/`, etc.)
  - [x] Validate translated WGSL and map it back to Rust source spans, which displays
        in your IDE, through rust-analyzer. Yeah!
  - [ ] Generate linkage info
  - [ ] Generate `wgpu` linkage (possibly via a separate crate like `wgsl-rs-wgpu`)

### Can it Hello World?

Yes! See the [example](crates/example/src/main.rs), which transpiles the shader from
[Tour of WGSL](https://google.github.io/tour-of-wgsl/).

---

## Validation

`wgsl-rs` validates your WGSL at compile-time using [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga).
Validation errors are mapped back to Rust source spans, so they show up in your IDE via rust-analyzer.

### Validation Strategy

| Module Type | Validation |
|-------------|------------|
| Standalone (no imports) | Compile-time via naga |
| With imports from other `#[wgsl]` modules | Test-time via auto-generated `#[test]` function |
| `#[wgsl(skip_validation)]` | No validation |

**Why test-time for modules with imports?**

Modules that import from other `#[wgsl]` modules cannot be validated at compile-time because
the imported symbols aren't available during macro expansion. Instead, `wgsl-rs` generates a
`#[test] fn __validate_wgsl()` that validates the concatenated WGSL source at test-time.

### Example

```rust
// Standalone module - validated at compile-time
#[wgsl]
pub mod constants {
    pub const PI: f32 = 3.14159;
}

// Module with imports - validated at test-time
#[wgsl]
pub mod shader {
    use super::constants::*;
    
    pub fn circle_area(r: f32) -> f32 {
        PI * r * r
    }
}

// Skip all validation
#[wgsl(skip_validation)]
pub mod experimental {
    // ...
}
```

### Runtime Validation

You can also validate modules at runtime using `Module::validate()`:

```rust
use wgsl_rs::wgsl;

#[wgsl]
pub mod my_shader {
    // ...
}

fn main() {
    // Validate manually (requires "validation" feature)
    my_shader::WGSL_MODULE.validate().expect("WGSL validation failed");
}
```

### Disabling Validation

To disable validation entirely, disable the `validation` feature:

```toml
[dependencies]
wgsl-rs = { version = "...", default-features = false }
```

---

## Should I use Rust-GPU instead?

**Maybe — it depends on your needs.**

### Pros of wgsl-rs

- **Lower barrier to entry:** No custom Rust compiler backend required.
- **Works with stable Rust:** No need for nightly or custom toolchains.
- **Editor support:** The `#[wgsl]` macro makes supported syntax explicit, so
  your editor (via rust-analyzer) can help you write valid code.
- **Immediate WGSL output:** Use, inspect, and debug the generated WGSL anywhere
  WGSL is supported, including browsers and non-Rust projects.
- **Human readable WGSL output:** The WGSL that `wgsl-rs` produces is very close
  in structure to the Rust code you write, including binding names and types.
- **Easy interop:** Generated WGSL can be used in any WebGPU environment.

### Cons of wgsl-rs

- **WGSL only:** Only works on platforms that support WGSL.
- **Limited to WebGPU features:** No support for features not present in WGSL (e.g., bindless resources).
- **Subset of Rust:** Only a strict subset of Rust is supported.
  - No traits
  - No borrowing
  - Very restricted module support

> **Note:** wgsl-rs and Rust-GPU are not mutually exclusive!
  You can start with wgsl-rs and switch to Rust-GPU when you need more advanced features.
  I'm working on making them co-habitable.

---

## Getting Involved

The project is split into a few parts:

- **`wgsl-rs-macros`**
  Provides the `wgsl` procedural macro for writing WGSL modules in Rust. Handles
  parsing and code generation for the supported Rust subset.
- **`wgsl-rs`**
  Provides the `Module` type, `wgsl::std`, and exports the `wgsl` macro.

There's also a [devlog](DEVLOG.md) that explains some of the decisions and tradeoffs made during the making
of this library.

---

Contributions, feedback, and questions are welcome!
