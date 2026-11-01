# Template Linkage

A [template module](../generics/templates.md) is generic: it cannot be
analyzed for wgpu linkage until it is instantiated with concrete type
arguments. This chapter covers how to go from a generic shader to a usable
`WgpuLinkage`.

## Templates cannot be analyzed directly

Calling `analyze_wgsl_module` on the `Source` of a template fails:

```rust
let linkage = analyze_wgsl_module(&template::SOURCE);
// Err(Error::TemplateResolution)
```

The analyzer needs concrete types to compute binding sizes and entry-point
signatures. A template's `Source` still has unsubstituted generic type
parameters, so there is nothing concrete to link.

## Instantiate, then analyze

The correct flow is:

1. Call the module's `instantiate::<T1, T2, ...>()` to produce a concrete
   `ir::Module` (imports flattened).
2. Pass the concrete IR to `analyze_ir_module`.

```rust
use wgsl_rs::ir;
use wgsl_rs::std::Vec4f;
use wgsl_rs_linkage_wgpu::analyze_ir_module;

let concrete_ir: ir::Module = renderer::instantiate::<f32, Vec4f>();
let linkage = analyze_ir_module(concrete_ir);
```

`ir::Module::generate_linkage()` (the `IrModuleExt` re-export) combines
both steps in one call:

```rust
use wgsl_rs::std::Vec4f;
use wgsl_rs_linkage_wgpu::IrModuleExt;

let linkage = renderer::instantiate::<f32, Vec4f>().generate_linkage();
```

## `WgpuLinkage` owns the concrete IR

`WgpuLinkage` holds the `ir::Module` it was given. This matters: the WGSL
source it returns via `wgsl_source()` is rendered from that exact concrete
IR, so the source always matches what was analyzed. There is no risk of
re-rendering a different template instantiation.

For a non-template `Source`, `analyze_wgsl_module` builds the IR internally and
owns it the same way.

## Example: instantiate a generic shader

```rust
#[wgsl]
pub mod tinted_output {
    use wgsl_rs::std::*;

    // The uniform's concrete type is chosen at instantiation time.
    uniform!(group(0), binding(0), TINT: impl Convert<f32>);

    #[fragment]
    pub fn fs<T: Convert<f32> + Wgsl + Clone>() -> Vec4f {
        let tint = get!(TINT, T);
        vec4f(f32(tint), f32(tint) * 0.5, 0.0, 1.0)
    }
}

// In application code: one type argument per module parameter, in
// `module_type_params` order — linkage variables first, then entry
// point parameters. Here: TINT, then fs's `T`, both f32.
use wgsl_rs_linkage_wgpu::IrModuleExt;

let mut linkage = tinted_output::instantiate::<f32, f32>().generate_linkage();
let shader = linkage.shader_module(&device);
let layout = linkage.pipeline_layout(&device, Some("tint"));
// ... build pipeline as usual
```

## Summary

| Step | API |
|------|-----|
| Instantiate a concrete module | `module::instantiate() -> ir::Module` |
| Instantiate a template module | `module::instantiate::<T...>() -> ir::Module` |
| Analyze concrete IR | `analyze_ir_module(ir) -> WgpuLinkage` |
| Or, combined | `instantiate().generate_linkage() -> WgpuLinkage` |
| Analyze a concrete source | `analyze_wgsl_module(&source) -> WgpuLinkage` |