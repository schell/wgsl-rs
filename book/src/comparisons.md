# wgsl-rs vs Other Shader Tools

Rust and WGSL programmers have several options for writing shaders and GPU
kernels without hand-writing GLSL or HLSL string literals. This chapter compares
`wgsl-rs` with the four tools users most often ask about: [Rust-GPU], [Rust
CUDA], [CubeCL], and [WESL].

They differ along two axes: **what language you write** (Rust vs WGSL) and
**where the result runs** (WebGPU only, one vendor, or many backends).

> This information is current as of October 2026. All of these projects move quickly;
> check their repositories for status.

## At a Glance

| Tool | You write | How it compiles | Runs on | Status |
|------|-----------|-----------------|---------|--------|
| **wgsl-rs** | Rust subset (`#[wgsl]` modules, stable Rust) | Proc macro to IR, rendered to WGSL | Any WebGPU/WGSL consumer | 0.1 beta |
| **Rust-GPU** | Rust with `spirv-std` (nightly) | `rustc` codegen backend to SPIR-V | Vulkan | In development, not production-ready |
| **Rust CUDA** | Rust device crates (nightly) | `rustc` backend to PTX | NVIDIA CUDA | Early development (2025 reboot) |
| **CubeCL** | Rust with `#[cube]` (stable) | Macro to IR, JIT per backend | NVIDIA, AMD, Apple, Vulkan, WebGPU, CPU | Alpha (in production via Burn) |
| **WESL** | WGSL + extensions | Linker flattens to plain WGSL | Any WebGPU/WGSL consumer | WESL 0.2 spec, `wesl` crate 0.5 |

## The Contenders

### Rust-GPU

_Full disclosure, the author used to be a Rust-GPU maintainer_

Rust-GPU is the most ambitious approach: a `rustc` compiler backend (`-Z codegen-backend`,
the same mechanism as Cranelift) that emits Vulkan SPIR-V.

- Requires a specific nightly Rust and a separate shader crate built via `spirv-builder` or `cargo-gpu`.
- Because it rides the full `rustc` pipeline, it supports more of Rust than any
  macro-based approach: rustc itself does the monomorphization, so generics,
  traits, and iterators work like normal Rust (though it must be a `no_std` GPU-legal subset, and determining
  that subset is error prone).
- Output is SPIR-V binary, so you get Vulkan portability but no readable WGSL
  and no direct WebGPU story; DXIL/WGSL targets are listed only as future
  possibilities.
- Kernel code does not run on the CPU; testing happens against the GPU. Code surrounding kernels, like helper
  function _can_ run on the CPU, though. And types can be shared, just like `wgsl-rs`.
- Created at Embark, handed to the community in 2024; its own README says it is
  not production-ready and makes no backwards-compatibility guarantees.

**Choose it when** you want the largest Rust language surface on Vulkan and can
accept nightly, SPIR-V-only output.

### Rust CUDA

An ecosystem of crates and tools for writing CUDA kernels entirely in Rust,
compiled to PTX through a custom `rustc` backend, with host-side runtime
crates and OptiX support.

- NVIDIA only. Compute and ray tracing; no graphics pipeline.
- Highest performance ceiling on NVIDIA hardware (direct PTX access), total
  vendor lock-in.
- Pinned nightly toolchain; after a dormant period it was rebooted in January
  2025 with a status update in August 2025. Its README warns to expect bugs,
  safety issues, and missing features.
- No CPU execution of kernels; testing requires an NVIDIA device.

**Choose it when** you are committed to NVIDIA and want PTX-level control for
GPGPU or OptiX workloads.

### CubeCL

A Rust language extension plus JIT compiler plus runtimes from the team behind
the Burn deep-learning framework. A `#[cube]` function compiles on demand to
CUDA, HIP, Metal, SPIR-V, WGSL, or CPU SIMD.

- Compute only; there are no vertex or fragment entry points. Low-level by
  design: explicit parallelism axes (vector, plane, cube) that good kernels
  specialize per backend at comptime.
- JIT, not ahead-of-time: only the variants you launch are compiled, with
  compilation and autotune caches you can ship warm.
- CPU is a first-class target, so kernels run and test on the CPU, similar in
  spirit to wgsl-rs's "two worlds" goal.
- Mechanically, CubeCL is the closest cousin to wgsl-rs: the `#[cube]` macro
  rewrites the function into Rust code that *builds an IR at runtime*, exactly
  the trick `wgsl-rs` uses for `WGSL_MODULE`.
- Used in production by Burn, but the public API is alpha and expected to break.

**Choose it when** you need portable, high-performance compute kernels (ML,
HPC) across many backends and are willing to manage hardware specialization
yourself.

### WESL

The odd one out: not Rust at all. WESL is a community standard that extends
WGSL itself, with a Rust implementation (`wesl-rs`), a JavaScript one, and a
tooling ecosystem.

- A strict superset of WGSL: existing `.wgsl` files work unchanged. Adds
  imports (split shaders into files), conditional compilation
  (`@if`/`@elif`/`@else`), and packages (shader libraries published to
  crates.io *and* npm).
- Experimental: generics, and eval/exec, which runs WESL code on the CPU
  (tested against the WebGPU Conformance Test Suite).
- Tooling-first: bundler plugins for Vite/Webpack, `wgsl-test` (GPU shader
  testing), `wgsl-studio` (VS Code), `wgsl-analyzer` LSP (forthcoming), and a
  playground. A framework-agnostic alternative to Bevy's `naga_oil`.
- Where `wgsl-rs` answers "WGSL is too small, use Rust instead", WESL answers
  "WGSL is too small, extend WGSL".

**Choose it when** you want to stay in WGSL but need modules, variants, and
shareable shader libraries across Rust and JavaScript projects.

## Feature Comparison

| | wgsl-rs | Rust-GPU | Rust CUDA | CubeCL | WESL |
|---|---------|----------|-----------|--------|------|
| Vertex/fragment | ✅ | ✅ | ❌ | ❌ | ✅ |
| Compute | ✅ | ✅ | ✅ | ✅ | ✅ |
| Stable Rust | ✅ (1.87+) | Nightly | Nightly | ✅ | n/a |
| When compiled | Compile time (macro) | Compile time (backend) | Compile time (backend) | JIT at first launch | Build/link time |
| Output | Readable WGSL | SPIR-V binary | PTX | Per-backend binaries | Flattened WGSL |
| Backends | WebGPU/WGSL | Vulkan | NVIDIA | NVIDIA, AMD, Apple, Vulkan, WebGPU, CPU | WebGPU/WGSL |
| Runs on CPU | ✅ by design | ❌ | ❌ | ✅ (CPU runtime) | Experimental (eval/exec) |
| Validation | Automatic `naga` tests | External SPIR-V tools | rustc | Runtime compile errors | Experimental; `wgsl-test` |
| Generics | Monomorphized at macro time; runtime template instantiation | Full rustc generics | Full rustc generics | Comptime specialization | Experimental |
| Shader libraries | Import other `#[wgsl]` modules defined in Cargo crates | Cargo crates | Cargo crates | Cargo crates (`cubek`) | WESL packages (crates.io, npm) |
| wgpu linkage | Auto-generated | Manual | Manual | Manual | Manual |
| Memory layout help | `#[derive(Wgsl, Layout)]` | Manual (`repr(C)`) | Manual | Tensor abstractions | n/a |

## Which Should You Choose?

| Pick | When |
|------|------|
| **wgsl-rs** | WebGPU-first graphics or compute; you want CPU unit tests, shared type definitions, automatic wgpu linkage, and readable WGSL, all on stable Rust. |
| **Rust-GPU** | You need most of Rust's language features on Vulkan, can accept a pinned nightly compiler, plus SPIR-V-only output. |
| **Rust CUDA** | You are committed to NVIDIA and want PTX-level control for GPGPU or OptiX. |
| **CubeCL** | You need portable, peak-performance compute kernels across NVIDIA, AMD, Apple, and CPU. |
| **WESL** | You want to keep writing WGSL but need modules, build variants, and shareable shader libraries across Rust and JS. |

None of these are mutually exclusive. A single application can render with
`wgsl-rs` shaders and run heavy compute through CubeCL kernels. And because
`wgsl-rs` emits plain WGSL, any WGSL tooling, including WESL's, applies to its
output.

## When wgsl-rs Is the Wrong Tool

- You need native backends beyond WebGPU (see CubeCL, Rust-GPU, Rust CUDA) and cannot rely on `naga`/`wgpu` to translate.
- You need Vulkan features WGSL cannot express, such as bindless resources
  (see Rust-GPU, or hand-written shader sources).
- You need Rust language features outside the supported subset (see
  [Supported Rust Subset](../reference/supported-subset.md)).

## Sources

- Rust-GPU: https://github.com/Rust-GPU/rust-gpu
- Rust CUDA: https://github.com/Rust-GPU/rust-cuda
- CubeCL: https://github.com/tracel-ai/cubecl
- WESL spec: https://wesl-lang.dev / https://github.com/webgpu-tools/wesl-spec
- wesl-rs: https://github.com/webgpu-tools/wesl-rs

[Rust-GPU]: https://github.com/Rust-GPU/rust-gpu
[Rust CUDA]: https://github.com/Rust-GPU/rust-cuda
[CubeCL]: https://github.com/tracel-ai/cubecl
[WESL]: https://wesl-lang.dev
