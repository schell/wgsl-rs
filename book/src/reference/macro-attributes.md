# Macro Attributes

## `wgsl` Container Attributes

| Attribute                                          | Syntax                                          | Purpose                                                       |
|---------------------------------------------------|-------------------------------------------------|---------------------------------------------------------------|
| Crate path                                        | `#[wgsl(crate_path = path)]`                    | Override the path to the `wgsl_rs` crate (for re-exports)     |
| Skip validation                                   | `#[wgsl(skip_validation)]`                      | Disable compile-time WGSL validation for this module          |
| Validate with instantiation types                 | `#[wgsl(validate_with_instantiation_types(T1, T2))]` | Validate template modules by instantiating with the given types |
| Extensions                                        | `#[wgsl(extensions = [Ext1, Ext2])]`            | Run `WgslExtension` impls on the IR before instantiation      |

## Field / Item Attributes

| Attribute                                          | Applies to          | Purpose                                                       |
|---------------------------------------------------|---------------------|---------------------------------------------------------------|
| `#[wgsl_ignore]`                                  | Items, fields       | Exclude this item/field from transpilation                    |
| `#[wgsl_allow(non_literal_loop_bounds)]`          | `for` loops         | Permit non-literal loop bounds (requires runtime support)     |
| `#[wgsl_allow(non_literal_match_statement_patterns)]` | `match`         | Permit non-literal match patterns                             |

## Entry-Point Attributes

| Attribute                                          | Applies to          | Purpose                                                       |
|---------------------------------------------------|---------------------|---------------------------------------------------------------|
| `#[vertex]`                                       | Functions           | Mark a vertex entry point                                     |
| `#[fragment]`                                     | Functions           | Mark a fragment entry point                                   |
| `#[compute]`                                      | Functions           | Mark a compute entry point                                    |
| `#[workgroup_size(N)]`                            | Compute functions   | Set workgroup size to `N x 1 x 1`                             |
| `#[workgroup_size(x, y, z)]`                      | Compute functions   | Set explicit 3D workgroup size                                |

## I/O Decorator Attributes

| Attribute                                          | Applies to          | Purpose                                                       |
|---------------------------------------------------|---------------------|---------------------------------------------------------------|
| `#[builtin(name)]`                                | Fn args, struct fields | Bind to a WGSL built-in (`position`, `vertex_index`, etc.) |
| `#[location(N)]`                                  | Fn args, struct fields | Bind to inter-stage location `N`                            |
| `#[interpolate(...)]`                             | Fn args, struct fields | Set interpolation type/filter                              |
| `#[blend_src(N)]`                                 | Fn args, struct fields | Set `@blend_src` for dual-source blending                  |
| `#[invariant]`                                     | Fn args, struct fields | Mark position output as `@invariant`                        |

## Derive Attributes

| Attribute                | Crate                  | Purpose                                                       |
|--------------------------|------------------------|---------------------------------------------------------------|
| `#[derive(Wgsl)]`        | `wgsl-rs`              | Transpile the annotated type into a WGSL struct               |
| `#[derive(Layout)]`      | `wgsl-rs-layout-macros`| Generate `WgslLayout` / `Layout` impls and inherent constants |