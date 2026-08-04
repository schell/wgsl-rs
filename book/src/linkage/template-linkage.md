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

1. Get the template's IR (or `Source`).
2. Call `instantiate::<T1, T2, ...>()` to produce a concrete `ir::Module`.
3. Pass the concrete IR to `analyze_ir_module`.

```rust
use wgsl_rs::linkage::wgpu::analyze_ir_module;
use wgsl_rs::*;

let template: &Source = &renderer::SOURCE;
let concrete_ir: ir::Module = template.instantiate::<f32, Vec4f>()?;
let linkage = analyze_ir_module(concrete_ir);
```

`ir::Module::generate_linkage()` (the `IrModuleExt` re-export) does the
second step in one call:

```rust
let linkage = concrete_ir.generate_linkage()?;
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
pub mod generic_blit {
    use wgsl_rs::std::*;

    pub struct Params<T> { pub scale: T, pub bias: T }

    uniform!(PARAMS, Params<Vec4f>);     // concrete after instantiation
    texture!(SRC, Texture2d);
    sampler!(SMP, Sampler);

    #[vertex]
    pub fn vs(...) -> VertexOutput { /* ... */ }

    #[fragment]
    pub fn fs(in: VertexOutput) -> Vec4f {
        let c = textureSample(SRC, SMP, in.uv);
        c * PARAMS.scale + PARAMS.bias
    }
}

// In application code:
use wgsl_rs::*;

let concrete = generic_blit::SOURCE.instantiate::<Vec4f>()?;
let mut linkage = concrete.generate_linkage()?;
let shader = linkage.shader_module(&device);
let layout = linkage.pipeline_layout(&mut linkage, &device, Some("blit"));
// ... build pipeline as usual
```

## Summary

| Step | API |
|------|-----|
| Instantiate template IR | `source.instantiate::<T...>() -> ir::Module` |
| Analyze concrete IR | `analyze_ir_module(ir) -> WgpuLinkage` |
| Or, combined | `ir.generate_linkage() -> WgpuLinkage` |
| Analyze concrete source | `analyze_wgsl_module(&source) -> WgpuLinkage` |