//! wgpu linkage generation for wgsl-rs shader modules.
//!
//! Linkage output is wgpu *types* (`BindGroupLayout`, `ShaderModule`,
//! `VertexState`, ...), and wgpu majors do not interop: types from
//! wgpu 29 and wgpu 30 cannot be mixed in one build. This crate solves
//! that with **one canonical source** (`src/linkage.rs`) compiled into
//! a feature-gated module per supported wgpu major, against that
//! major's pinned `wgpu`:
//!
//! - the `v30` feature (default) compiles it against wgpu 30 and re-exports it
//!   at the crate root,
//! - the `v29` feature compiles it against wgpu 29 as the `v29` module,
//! - the wgpu-free core (IR assembly, WGSL §14.4.1 sizing, stage analysis)
//!   lives in [`analysis`] and is compiled exactly once.
//!
//! ```ignore
//! use wgsl_rs_linkage_wgpu as wgpu_linkage;
//!
//! let linkage = wgpu_linkage::analyze_wgsl_module(&hello_triangle::WGSL_SOURCE).unwrap();
//! let shader_module = linkage.shader_module(&device);
//! let bg_layout = linkage.bind_group(0).unwrap().layout(&device);
//! let bg = linkage.bind_group(0).unwrap().create(
//!     &device, &bg_layout, &[frame_uniform.as_entire_binding()],
//! );
//! ```
//!
//! Pinned to an older wgpu major, disable the defaults and enable the
//! matching feature; the same API is available under the versioned
//! module (and, in a pinned build, at the crate root):
//!
//! ```ignore
//! // Cargo.toml: wgsl-rs-linkage-wgpu = { version = "0.1", default-features = false, features = ["v29"] }
//! let linkage = wgsl_rs_linkage_wgpu::v29::analyze_wgsl_module(&source).unwrap();
//! ```
//!
//! Both majors may be enabled at once (feature unification): the
//! modules coexist with distinct, incompatible wgpu types, and the
//! crate root re-exports the latest enabled major. Code that must
//! survive feature unification in a larger dependency graph should use
//! the stable versioned module path instead.
//!
//! Why linkage is not inside `wgsl-rs`: the linkage implementation
//! depends on `wgsl-rs` (for `Source`, `ir`, rendering), so it cannot
//! live in the same crate without a dependency cycle. `wgsl-rs` itself
//! stays wgpu-free.
//!
//! Given an IR module (produced by the macro-emitted `instantiate()`
//! function — zero arguments for concrete modules, generic
//! `instantiate::<T>()` for templates), the analyzer
//! extracts the binding and entry-point information needed to build
//! wgpu pipelines at runtime — no proc-macro-time code generation — so
//! template instantiations get the same linkage analysis as concrete
//! sources.

/// The wgpu-free core of the linkage crate: errors,
/// WGSL §14.4.1 sizing, binding metadata, and the shader-stage
/// analysis. Compiled exactly once, regardless of which wgpu majors
/// are enabled; a bare `--no-default-features` build compiles this
/// module and nothing else.
pub mod analysis;

/// Linkage compiled against wgpu 29.
#[cfg(feature = "v29")]
pub mod v29 {
    pub use wgpu29 as wgpu;
    include!("linkage.rs");
}

/// Linkage compiled against wgpu 30.
#[cfg(feature = "v30")]
pub mod v30 {
    pub use wgpu30 as wgpu;
    include!("linkage.rs");
}

// The crate root implies the latest enabled major, and only that. A
// pinned vNN-only build re-exports that major so pinned code can use
// the root path too; code that must survive feature unification (a
// graph enabling multiple majors) should use the stable versioned
// module path instead, so it can never be silently re-pointed at
// incompatible types.
#[cfg(feature = "v30")]
pub use crate::v30::*;

#[cfg(all(feature = "v29", not(feature = "v30")))]
pub use crate::v29::*;
