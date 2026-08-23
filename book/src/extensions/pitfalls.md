# Pitfalls and Constraints

## Run Order Is Declaration Order

Extensions run in the order listed in `#[wgsl(extensions = [...])]`. There is no priority field, no topological sort, no guaranteed order beyond the list. If two extensions conflict, order them explicitly in the attribute.

## Type Substitution Happens After `modify_ir`

`modify_ir` runs before type instantiation. Do not attempt to outsmart substitution by string-replacing type names or pre-resolving `TypeParam` nodes. Operate on IR nodes and let the instantiation pass substitute `TypeParam` nodes automatically into anything you inject.

## Attributes Are Not in WGSL Output

`ir::Attribute` values are preserved on IR nodes for extension inspection only. They are never rendered into WGSL. Do not rely on them appearing in the final shader source, and do not try to emit WGSL by stuffing text into attribute args.

## Extension Types Must Be Visible at the Call Site

The paths in `#[wgsl(extensions = [path::ExtA, path::ExtB])]` must resolve at the location of the `wgsl` attribute. Import the extension types or use fully-qualified paths.

## Non-`WgslExtension` Types Are Rejected at Compile Time

The macro generates code that calls `Ext::modify_ir`. If a listed type does not implement `WgslExtension`, the error surfaces as a compile-time trait bound failure, not a runtime error.

## Don't Rely on `WGSL_MODULE` Being Mutable

Extensions receive `&mut ir::Module`, not `&mut wgsl_rs::Source`. The surrounding `Source` is not mutable from within an extension. Do all your work through the `Module` reference.

## Trait Is Re-exported at the Crate Root

Import the trait from `wgsl_rs`, not `wgsl_rs::extension`:

```rust
use wgsl_rs::WgslExtension;
```

Both paths work, but the crate-root re-export is the documented public path.