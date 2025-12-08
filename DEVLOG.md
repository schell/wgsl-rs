# devlog

## design decisions

1. The user must write regular Rust code.
   No translation of Rust code will occur in the `#[wgsl]` macro.
   The macro is strictly additive.

## tradeoffs

### Swizzles

Swizzles are tricky because in Rust they could be accomplished with traits like `glam`'s 
`Vec4Swizzle` trait (and friends), but in WGSL they look like field accessors.
Because of design decisions the Rust must be un-altered so we use functions backed
by a trait (or something) in Rust and then translate the syntax to WGSL swizzle.

### Numeric builtin functions

There's a lot to implement here. So far I've been pretty successful (3 functions done) using this AI prompt:

> Please add the `{fn}` function using the module-level documentation table as a guide, following the implementation of `abs`
> and `acos`, which used the `NumericBuiltinAbs` and `NumericBuiltinAcos` traits, respectively.

You should replace {fn} with whatever function you want to implement.

I've gone with a "one-trait-per-function" strategy because each function has little differences, and I anticipate
having to use generic associated types for some functions.
