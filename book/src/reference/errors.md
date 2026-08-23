# Errors

## `SourceError`

Raised by `wgsl_rs::Source` methods.

| Variant        | When                                                                  |
|----------------|-----------------------------------------------------------------------|
| `TemplateWgsl` | `wgsl_source()` is called on a `Source` that is still a template (has unresolved `TypeParam` nodes). Call `instantiate(...)` first to produce concrete WGSL. |

### Recovery

```rust
match source.wgsl_source() {
    Ok(wgsl) => { /* write or compile */ }
    Err(SourceError::TemplateWgsl) => {
        let concrete = source.instantiate(&[concrete_type])?;
        let wgsl = concrete.wgsl_source()?;
    }
}
```

## `linkage::wgpu::Error`

Raised by the `linkage-wgpu` feature when reflecting bind groups and pipeline layouts.

| Variant              | When                                                                |
|----------------------|---------------------------------------------------------------------|
| `TemplateResolution` | A linkage query was made against a template that has not been instantiated with concrete types. |
| `NoSuchBindGroup`    | The queried bind group index does not exist in the module.          |
| `NoSuchBinding`      | The queried binding within a bind group does not exist.             |
| `TypeMismatch`       | A binding's type does not match the expected wgpu binding type.     |

### Recovery

```rust
match source.linkage().bind_group(0) {
    Ok(group) => { /* build BindGroupLayout */ }
    Err(linkage::wgpu::Error::NoSuchBindGroup) => {
        // bind group 0 not declared in this shader
    }
    Err(linkage::wgpu::Error::TemplateResolution) => {
        let concrete = source.instantiate(&[concrete_type])?;
        let group = concrete.linkage().bind_group(0)?;
    }
}
```

## General Pattern

Template errors are recoverable by instantiating with concrete types (see [Template Modules & Instantiation](../generics/templates.md) and [Template Linkage](../linkage/template-linkage.md)). Bind group / binding errors indicate a mismatch between the host-side layout code and the shader declaration — inspect `ir::Module` items (`Uniform`, `Storage`, `Sampler`, `Texture`) to reconcile.