# wgsl-rs

`wgsl-rs` allows you to write a subset of Rust and have it automatically
generate WGSL code and `wgpu` runtime linkage. Rust code written with `wgsl-rs`
can then be run on the CPU, and the generated WGSL can be run on the GPU.

Procedural macros are provided by `wgsl-rs-macros`.

## roadmap

- [x] module-level wgsl macro 
  - [x] translates a subset of Rust into WGSL
    - [x] types
      - [x] concrete scalars
      - [x] vec{N}<{scalar}>
      - [x] vec aliases
      - [ ] arrays
      - [ ] matrices
      - [ ] structs
      - [ ] textures
      - [ ] atomics
    - [ ] descriptor sets, bindings, etc
    - [ ] make this more extensive until all WGSL syntax is handled
  - [x] allows glob-importing other `wgsl-rs` modules
  - [x] WGSL std for Rust
    - [x] vector constructors `vec2f`, `vec3f`, etc.
    - [ ] binary ops (+, -, *, /, etc)
  - [ ] generates linkage info
  - [ ] generates linkage in `wgpu`, maybe through another crate like `wgsl-rs-wgpu` 

### can it hello-world?

No, `wgsl-rs` is not hello-world-able yet.
Specifically, the shader at <https://google.github.io/tour-of-wgsl/> cannot be transpiled.

## pro/cons vs rust-gpu

### pros  

* no rustc backend necessary, which means
  * no provisioning rustc_codegen

## getting involved

The project is divided into a few parts:

- **`wgsl-rs-macros`**
  Provides the `wgsl` proc-macro that allows writing WGSL modules within Rust source files.
  This crate contains parsing from `syn` types into a strict subset of Rust.
  This crate also performs the "code gen", if you can call it that. It's just an implementation
  of `quote::ToTokens` for the subset AST. 
- **`wgsl-rs`**
  Provides `Module`, exports the `wgsl` proc-macro
