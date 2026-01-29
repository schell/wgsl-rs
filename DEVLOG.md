# devlog

## design decisions

1. The user must be able to write regular Rust code.
   No translation of Rust code will occur in the `#[wgsl]` macro.
   The macro is strictly additive.
   Specifically, it will add the transpiled WGSL source code and imports.
2. The Rust type system should catch as many type errors as possible, so WGSL validation doesn't have to.
   Stated another way, `wgsl-rs` should never produce Rust code that compiles, but produces invalid WGSL.
   You can be sure that if your Rust compiles, your shader will too.
3. Macros are ok, as Rust folks are used to using macros.
   Eg. `uniform!(binding(0), group(0), BRIGHTNESS: f32);` is fine, and in fact we **must** use
   macros for **all** WGSL that can't be represented with Rust syntax.

## tradeoffs

### Importing modules

With `wgsl-rs` you can import other modules. While this is a boon for development it has some significant
restrictions. Specifically you can only glob-import other WGSL modules written with `wgsl-rs`, or the
`wgsl_rs::std::*`. If you try to import arbitrary modules you'll get a compiler error about not having
`path::to::module::WGSL_MODULE` in scope.
 
### Swizzles

Swizzles are tricky because in Rust they could be accomplished with traits like `glam`'s 
`Vec4Swizzle` trait (and friends), but in WGSL they look like field accessors.
Because of design decisions the Rust must be un-altered so we use functions similar to the
`Vec4Swizzle` strategy. So to swizzle `xyz` in WGSL you would just call `.xyz()` in Rust.

### Numeric builtin functions

There's a lot to implement here. So far I've been pretty successful (3 functions done) using this AI prompt:

> Please add the `{fn}` function using the module-level documentation table as a guide, following the implementation of `abs`
> and `acos`, which used the `NumericBuiltinAbs` and `NumericBuiltinAcos` traits, respectively.

You should replace {fn} with whatever function you want to implement.

Keep in mind that the `VecN<T>` types have `glam` types as their `inner` fields, so you can use that for many of these.

Keep in mind that glam vectors are not iterators, you can't call `zip` on them. Instead, you can call `to_array` on
each vector and write into one of them, finally calling `.into()` on the array to convert it back to the vector.
See the implementations of `NumericBuiltinPow` and `NumericBuiltinStep` for an example of this.

I've gone with a "one-trait-per-function" strategy because each function has little differences, and I anticipate
having to use generic associated types for some functions.

### Vertex and fragment return values

Shaders may return user defined types (structs) that have their fields annotated with the `#[builtin(...)]`
and `#[location(...)]` macros, which corresponds to the description in
[the spec](https://gpuweb.github.io/gpuweb/wgsl/#example-ee897116):

```wgsl
// Mixed builtins and user-defined inputs.
struct MyInputs {
  @location(0) x: vec4<f32>,
  @builtin(front_facing) y: bool,
  @location(1) @interpolate(flat) z: u32
}

struct MyOutputs {
  @builtin(frag_depth) x: f32,
  @location(0) y: vec4<f32>
}

@fragment
fn fragShader(in1: MyInputs) -> MyOutputs {
  // ...
}
```

And here would be the corresponding Rust:
```rust
// Mixed builtins and user-defined inputs.
pub struct MyInputs {
    #[location(0)]
    pub x: Vec4<f32>,

    #[builtin(front_facing)]
    pub y: bool,

    #[location(1)]
    #[interpolate(flat)]
    pub z: u32
}

pub struct MyOutputs {
    #[builtin(frag_depth)]
    pub x: f32,

    #[location(0)]
    pub y: vec4<f32>
}

#[fragment]
pub fn fragShader(in1: MyInputs) -> MyOutputs {
  // ...
}
```

But in Rust we can't put annotations on return types, so we'll have to specify that in the shader
stage proc-attribute macros (`vertex`, `fragment` and `compute`).

But if the return type a vertex or fragment shader is `Vec4f`, and the macro didn't specify a
return type location or builtin, `wgsl-rs` will automatically insert an appropriate annotation.

I think most people who need to specify a return value other than the default `@builtin(position)` for
vertex shaders or `@location(0)` for fragment shaders will use a struct, so this is fine.

## features

### wgpu linkage (`linkage-wgpu` feature)

When the `linkage-wgpu` feature is enabled, the `#[wgsl]` macro generates additional wgpu-specific
code to simplify integration with wgpu applications:

**Buffer descriptors and creation functions** - For each `uniform!` and `storage!` declaration:
- A `{NAME}_BUFFER_DESCRIPTOR: wgpu::BufferDescriptor<'static>` constant
- A `create_{name}_buffer(device: &wgpu::Device) -> wgpu::Buffer` function

**Bind group modules** - For each bind group, a `linkage::bind_group_{N}` module containing:
- `LAYOUT_ENTRIES` and `LAYOUT_DESCRIPTOR` constants
- `layout(device)` - creates the bind group layout
- `create(device, layout, ...)` - type-safe bind group creation with named parameters
- `create_dynamic(device, layout, entries)` - dynamic bind group creation with a slice

**Shader entry point modules** - For vertex, fragment, and compute entry points:
- Entry point name constants
- Helper functions for creating pipeline states

This feature adds `wgpu` as a dependency to `wgsl-rs`.

### 2026-01-05: switch statement support (match → switch)

Rust `match` statements transpile to WGSL `switch` statements:
- `match x { 0 => {...}, 1 => {...}, _ => {...} }` → `switch x { case 0 {...} case 1 {...} default {...} }`
- Or-patterns `1 | 2 | 3 => {...}` → `case 1, 2, 3 {...}`
- Missing `_` arm auto-generates `default {}`
- Non-literal patterns (constants, identifiers) emit a warning suppressed with `#[wgsl_allow(non_literal_match_statement_patterns)]`
- Match expressions (in let bindings) are unsupported (WGSL switch is a statement)
- Guard clauses, range patterns, struct/tuple patterns are unsupported
- For future work regarding type checking, we may be able to get away with a trick. We _could_ alter the Rust code
  to result in the pattern matched, then use that result in an empty function that takes an integer. This would
  cause Rust to do the type checking for us, before WGSL validation. That would keep us from having to emit a warning. 

  Example input Rust:
  ```rust
  match my_expr {
    MyEnum::Variant1 => {
      do_stuff();
    }
  }
  ```

  Output Rust:
  ```rust
  let __match_result = match my_expr {
    input @ MyEnum::Variant1 => {
      do_stuff();
      input
    }
  };
  __ensure_integer(__match_result);
  ```

  Output WGSL:
  ```wgsl
  switch my_expr {
    case MyEnum_Variant1: {
      do_stuff();
    }
    default: {}
  }
  ```

  Maybe we should also look into what we can do with for-loop bounds and the `non_literal_loop_bounds` warning in this manner.

### 2026-01-03: for-loop support and warnings with #[wgsl_allow]

`for i in 0..n` transpiles to `for (var i = 0; i < n; i++)` and `for i in 0..=n` transpiles to `for (var i = 0; i <= n; i++)`.
Nested loops work correctly.
Only bounded ranges are supported (WGSL requires explicit bounds).

For-loops with non-literal bounds (where the bounds cannot be verified at compile-time to be ascending)
emit a compile error on stable, since warnings can't be emitted (there's a hack to emit them as deprecations, but it's hacky).
On nightly it emits a warning.
Use `#[wgsl_allow(non_literal_loop_bounds)]` on the for-loop to suppress these errors/warnings.

### 2026-01-29: Pointer type support (`ptr!` macro)

Added `ptr!(address_space, T)` macro for WGSL pointer types in function parameters.
- Supports `function` and `private` address spaces (the only ones passable to functions without extensions)
- Expands to `&mut T` in Rust for CPU execution
- Transpiles to `ptr<function, T>` or `ptr<private, T>` in WGSL
- Added dereference operator (`*`) support for pointer indirection
- Both `&x` and `&mut x` transpile to `&x` in WGSL (mutability is determined by access mode)

Example:
```rust
pub fn increment(p: ptr!(function, i32)) {
    *p += 1;
}

fn main() {
    let mut x: i32 = 5;
    increment(&mut x);  // x is now 6
}
```

### 2026-01-24: RuntimeArray<T> support

Added `RuntimeArray<T>` type for runtime-sized arrays (WGSL `array<T>` without size parameter).
These are used in storage buffers, typically as the last field of a struct.
On CPU, `RuntimeArray<T>` is backed by `Vec<T>` with full indexing support.
Transpiles to `array<T>` in WGSL.
