# PhantomData

Demonstrates `PhantomData<T>` marker fields on `#[wgsl]` structs. `PhantomData` is re-exported from `wgsl_rs::std`. The proc-macro recognizes `PhantomData<_>` fields specially: they are retained in the IR (so extensions can observe which type parameter each phantom slot binds) but omitted from the rendered WGSL. Construction expressions using the bare `PhantomData` value are likewise stripped so the rendered positional constructor has the correct arity.

## Rust Source

```rust
#[wgsl(skip_validation)]
pub mod phantom_data {
    use wgsl_rs::std::*;

    /// A typed identifier carrying a phantom type tag. The `phantom`
    /// field is dropped from the WGSL output, leaving only `index`.
    pub struct Id<T> {
        pub index: u32,
        pub phantom: PhantomData<T>,
    }

    /// A struct binding two type parameters to two phantom slots. An
    /// extension inspecting the IR sees `t: PhantomData<T>` and
    /// `a: PhantomData<A>` and can reconstruct which field binds which
    /// parameter.
    pub struct Tagged<T, A> {
        pub x: f32,
        pub t: PhantomData<T>,
        pub a: PhantomData<A>,
    }

    pub fn make_id() -> Id<f32> {
        Id {
            index: 0u32,
            phantom: PhantomData,
        }
    }

    pub fn make_tagged() -> Tagged<f32, u32> {
        Tagged {
            x: 1.0,
            t: PhantomData,
            a: PhantomData,
        }
    }

    pub fn read_id(i: Id<f32>) -> u32 {
        i.index
    }

    pub fn read_tagged(t: Tagged<f32, u32>) -> f32 {
        t.x
    }
}
```

## Generated WGSL

```wgsl
fn make_id() -> Id_f32 {
    return Id(0u);
}

fn make_tagged() -> Tagged_f32_u32 {
    return Tagged(1.0);
}

fn read_id(i: Id_f32) -> u32 {
    return i.index;
}

fn read_tagged(t: Tagged_f32_u32) -> f32 {
    return t.x;
}

struct Id_f32 {
    index: u32
}

struct Tagged_f32_u32 {
    x: f32
}
```

## Notes

- `PhantomData<T>` fields are dropped from the WGSL struct definition — `Id_f32` has only `index: u32`, `Tagged_f32_u32` has only `x: f32`.
- Construction expressions use the bare `PhantomData` value (no turbofish). The macro strips it from the positional constructor call so the rendered arity matches the non-phantom field count.
- The IR retains phantom fields as `Type::Phantom { elem }` so extensions can see the full type-parameter binding structure.
- This example uses `#[wgsl(skip_validation)]` because the auto-validation test doesn't cover phantom field stripping.