# Macro Rules

Demonstrates that `macro_rules!` definitions and derive macros inside a `#[wgsl]` module are stripped from WGSL code generation but remain available in the Rust source. The struct below carries `#[derive(Debug, Clone, Copy)]`, which produces no WGSL output.

## Rust Source

```rust
#[wgsl]
pub mod macro_rules_definitions {
    //! It is possible to define `macro_rules!` within a WGSL module.
    //!
    //! Macros defined this way **will not generate WGSL code**, but will pass
    //! through to Rust code.
    //!
    //! Said another way - `macro_rules!` definitions will be stripped from WGSL
    //! code generation but will remain in your Rust source.

    #[expect(unused_macros)]
    macro_rules! my_macro {
        ($id:ident) => {
            id
        };
    }

    // It's also possible to use derive macros.
    //
    // Derive macros pass through without generating any extra WGSL.
    #[derive(Debug, Clone, Copy)]
    pub struct Data {
        pub inner: f32,
    }
}
```

## Generated WGSL

```wgsl
struct Data {
    inner: f32
}
```

## Notes

- `macro_rules!` definitions are stripped from WGSL output entirely but remain usable in Rust.
- Derive macros (e.g. `#[derive(Debug, Clone, Copy)]`) pass through without generating any extra WGSL; only the struct definition itself is emitted.