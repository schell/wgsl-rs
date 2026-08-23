# The `#[derive(Layout)]` Macro

`#[derive(Layout)]` (from `wgsl-rs-layout-macros`) generates the `WgslLayout` and `Layout` impls for a struct by computing offsets, sizes, and alignments per WGSL spec 14.4.1.

## Generated Inherent Associated Constants

In addition to the trait impls, the derive emits **inherent** associated constants on the struct itself:

```rust
impl Particle {
    const __OFFSET_0: usize = /* field 0 offset */;
    const __OFFSET_1: usize = /* field 1 offset */;
    const __SIZE_0: usize   = /* field 0 size   */;
    const __SIZE_1: usize   = /* field 1 size   */;
    const __ALIGN_0: usize  = /* field 0 align  */;
    const __ALIGN_1: usize  = /* field 1 align  */;
    // ... one triple per field
}
```

These are accessible from both the `WgslLayout` and `Layout` impls, as well as from user code that wants a specific field's layout without indexing into `FIELDS`.

## Computation

Each field's offset is computed via `roundUp(current_offset, field_align)`, matching the WGSL spec. The struct alignment is the maximum of all field alignments. The struct size is `roundUp(last_field_end, struct_align)`.

## Why Inherent Constants

The derive emits inherent constants rather than inline `const` expressions inside the trait impl because the Rust const evaluator has complexity limits when evaluating deeply nested `roundUp` expressions inside associated const bodies. Moving the computed values to inherent constants keeps the trait impls thin and avoids hitting those limits on large structs.

The constant values are identical whether accessed via the inherent constants or via `FIELDS` — they are generated from the same computation in the proc-macro.