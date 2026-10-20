//! Turbofish on associated-const paths (wgsl-rs#240): `Id::<T>::NONE`
//! and `<Id<T>>::NONE` are synonyms, in every context.

use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod slab {
    use wgsl_rs::std::*;

    pub struct Id<T> {
        pub index: u32,
        _marker: PhantomData<T>,
    }

    impl<T> Id<T> {
        pub const NONE: Self = Self {
            index: 0u32,
            _marker: PhantomData,
        };

        pub const fn new(index: u32) -> Self {
            Id::<T> {
                index,
                _marker: PhantomData,
            }
        }
    }

    pub struct Array<T> {
        pub id: Id<T>,
        pub len: u32,
    }

    impl<T> Array<T> {
        /// The issue's `NONE_ERRS` spelling, which used to be a parse
        /// error; it must render exactly like the `<Id<T>>::NONE`
        /// spelling does.
        pub const NONE_TURBOFISH: Self = Array::<T> {
            id: Id::<T>::NONE,
            len: 0u32,
        };

        /// The `NONE_WORKS` (qualified-self) spelling.
        pub const NONE_QSELF: Self = Array::<T> {
            id: <Id<T>>::NONE,
            len: 0u32,
        };
    }

    /// A second generic struct with an associated const that this
    /// module never instantiates itself, so the cross-module const
    /// path below must record AND render the instance chunk.
    pub struct Cell<T> {
        pub v: T,
    }

    impl<T> Cell<T> {
        pub const UNIT: u32 = 1u32;
    }

    /// Concrete instantiation inside the defining module: exercises
    /// the compile-time monomorphizer's structural substitution of the
    /// retained path type, for both spellings, and publishes
    /// `Id_f32`/`Array_f32` into this chunk (which the cross-module
    /// test below relies on for its dedup case).
    pub fn array_none() -> Id<f32> {
        let a = Array::<f32> {
            id: Id::<f32>::NONE,
            len: 0u32,
        };
        a.id
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod use_arr {
    use super::slab::*;

    /// A concrete turbofish const path, cross-module, against a struct
    /// the provider never instantiates itself: the path must record
    /// and render the `Cell<f32>` instance.
    pub fn cell_unit() -> u32 {
        Cell::<f32>::UNIT
    }

    /// A concrete turbofish const path against a provider that already
    /// contains the compile-time instantiation: the runtime instance
    /// chunk must deduplicate against it rather than redefining
    /// the struct.
    pub fn none_index() -> u32 {
        Id::<f32>::NONE.index
    }

    /// The qualified-self spelling, same context.
    pub fn none_index_qself() -> u32 {
        <Id<f32>>::NONE.index
    }

    /// A qualified-self method call, which records the same
    /// instantiation (deduplicated with the const-path one).
    pub fn made() -> Id<f32> {
        <Id<f32>>::new(1u32)
    }
}

#[test]
fn const_paths_render_mangled_instantiations() {
    // CPU world: the const paths are ordinary associated consts.
    assert_eq!(use_arr::cell_unit(), 1u32);
    assert_eq!(use_arr::none_index(), 0u32);
    assert_eq!(use_arr::none_index_qself(), 0u32);
    assert_eq!(use_arr::made().index, 1u32);
    assert_eq!(slab::array_none().index, 0u32);
    assert_eq!(slab::Array::<f32>::NONE_TURBOFISH.len, 0u32);
    assert_eq!(slab::Array::<f32>::NONE_QSELF.id.index, 0u32);
    assert_eq!(slab::Cell::<f32> { v: 1.5 }.v, 1.5);
    let array = slab::Array::<f32> {
        id: slab::Id::<f32>::NONE,
        len: 5u32,
    };
    assert_eq!(array.len, 5u32);

    // Compile-time path: both spellings inside the generic Array impl
    // resolve to the concrete `Id_f32` const for the compile-time
    // Array<f32> instantiation.
    let local = slab::WGSL_SOURCE.wgsl_source().unwrap();
    for mangled in ["_1Array_f32__1NONE_TURBOFISH", "_1Array_f32__1NONE_QSELF"] {
        assert!(
            local.contains(&format!(
                "{mangled}: Array_f32 = Array_f32(_1Id_f32_NONE, 0u);"
            )),
            "expected {mangled} to resolve its const path, got:\n{local}"
        );
    }

    // Cross-module path: const paths render the mangled instance
    // names, the not-yet-instantiated `Cell<f32>` instance chunk is
    // rendered exactly once, and the already-present `Id_f32` chunk is
    // deduplicated away.
    let cross = use_arr::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        cross.contains("return _1Id_f32_NONE.index;"),
        "expected the cross-module const path to render the mangled const, got:\n{cross}"
    );
    assert!(
        cross.contains("return _1Cell_f32_UNIT;"),
        "expected the Cell const path to render, got:\n{cross}"
    );
    assert_eq!(
        cross.matches("struct Id_f32 {").count(),
        1,
        "Id_f32 must be defined exactly once, got:\n{cross}"
    );
    assert_eq!(
        cross.matches("struct Cell_f32 {").count(),
        1,
        "Cell_f32 must be defined exactly once, got:\n{cross}"
    );
    // The macro-generated `__validate_wgsl` tests cover naga validation
    // for all modules; this double-checks the assembled source. Gated
    // so the rendering and dedup assertions still run on
    // `--no-default-features` builds.
    #[cfg(feature = "validation")]
    wgsl_rs::validate_wgsl_source(&cross).expect("assembled source must validate");
}
