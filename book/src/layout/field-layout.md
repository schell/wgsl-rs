# Field Layout

## `FieldLayout`

```rust
pub struct FieldLayout {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub alignment: usize,
    pub pad_after: usize,
}
```

Each entry in `Layout::FIELDS` describes one field of the struct.

| Field        | Meaning                                                       |
|--------------|---------------------------------------------------------------|
| `name`       | Field identifier as written in Rust.                          |
| `offset`     | Byte offset of the field within the struct.                   |
| `size`       | WGSL size of the field's type.                                |
| `alignment`  | WGSL alignment of the field's type.                           |
| `pad_after`  | Zero bytes to write after this field's data.                  |

## `pad_after` Semantics

`pad_after` is the number of padding bytes between the end of this field's data and the start of the next field (or the end of the struct). When writing a struct to a buffer byte-by-byte:

1. Write the field's `size` bytes.
2. Write `pad_after` zero bytes.
3. Proceed to the next field.

The final field's `pad_after` accounts for struct-end padding so that the total equals `SIZE`.

## `RuntimeArray<T>`

A runtime-sized array has no statically knowable size:

```rust
assert_eq!(<RuntimeArray<f32> as WgslLayout>::SIZE, 0);
```

`SIZE` is 0 because the array length is runtime-dependent. Such arrays may only appear as the last field of a storage-buffer struct.

## Empty Structs

An empty struct is the identity element for layout:

```rust
assert_eq!(Empty::SIZE, 0);
assert_eq!(Empty::ALIGN, 1);
assert!(Empty::FIELDS.is_empty());
```

The WGSL spec does not define the empty-struct case; `wgsl-rs-layout` defines `ALIGN = 1` so that `roundUp(offset, 1)` is a no-op and empty structs compose without disturbing surrounding layout.