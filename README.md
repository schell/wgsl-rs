# wgsl-rs

With **wgsl-rs** you write a subset of Rust code and it automatically generates WGSL shaders and `wgpu` runtime linkage. 
Rust code written this way is fully operational (it can be run on the CPU) while the transpiled WGSL is isomorphic and 
should generate the same results on the GPU. 

In short, with `wgsl-rs`, you can unit test and run your code on the CPU in Rust, and use the generated WGSL on the GPU,
while sharing the same type definitions between the two.

Procedural macros are provided by the [`wgsl-rs-macros`](./crates/wgsl-rs-macros) crate.

---

This project is funded through [NGI Zero Commons](https://nlnet.nl/commonsfund/), a fund established by [NLnet](https://nlnet.nl) 
with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. 
Learn more at the [2025 NLnet project page](https://nlnet.nl/project/Renderling-Ecosystem/).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)

[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

## 🫶 Sponsor this!

This work will always be free and open source. 
If you use it (outright or for inspiration), please consider donating.

[💰 Sponsor 💝](https://github.com/sponsors/schell)

---

## Roadmap to Beta

There is a project plan for getting to beta [here](https://github.com/users/schell/projects/3/views/1)

### Can it Hello World?

Yes! See the [example](crates/example/src/main.rs), which transpiles the shader from
[Tour of WGSL](https://google.github.io/tour-of-wgsl/).

---

## Validation

`wgsl-rs` validates WGSL using [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga).
The `#[wgsl]` macro auto-generates `#[test] fn __validate_wgsl()` for every
non-template module — running `cargo test` will surface any validation failures.

### Validation Strategy

| Module Type | Auto-generated validation test? |
|-------------|----------------------------------|
| Non-template module (with or without imports) | Yes — `__validate_wgsl` |
| Template module (with type parameters) | No — instantiate first, then validate |
| `#[wgsl(skip_validation)]` | No |

All non-template modules get an auto-generated validation test because their
full WGSL source (including any imports) is assembled at runtime.
Template modules contain unresolved placeholders and must be instantiated before
they produce valid WGSL; use `validate_with_instantiation_types(T1, T2, ...)`
to specify concrete types for validation.

### Example

```rust
// Non-template module - auto-validated at test time.
#[wgsl]
pub mod constants {
    pub const PI: f32 = 3.14159;
}

// Module with imports - also auto-validated at test time.
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

When the `validation` feature is disabled, `Module::validate()` and
`validate_wgsl_source()` are unavailable, and the `#[wgsl]` macro does not
generate auto-validation tests.

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
