# wgsl-rs

`wgsl-rs` allows you to write a subset of Rust and have it automatically
generate WGSL code and `wgpu` runtime linkage. Rust code written with `wgsl-rs`
can then be run on the CPU, and the generated WGSL can be run on the GPU.

Procedural macros are provided by `wgsl-rs-macros`.
