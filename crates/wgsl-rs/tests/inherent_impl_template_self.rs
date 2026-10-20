//! `Self` in inherent impls of generic structs, instantiated through the
//! cross-module template machinery (wgsl-rs#236, wgsl-rs#237).
//!
//! The struct template ships its impl blocks with `Self` resolved to the
//! template's base ident; the runtime rename pass then mangles every
//! reference in lockstep with the struct definition. Before the fix, the
//! instantiated items rendered `Self` literally in method signatures,
//! const types, and `Self::…` call sites, and naga rejected the source
//! with a far-removed "unknown identifier" error.

use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
mod id_slab {
    use wgsl_rs::std::*;

    /// Sentinel index, visible to WGSL (consts referenced from a wgsl
    /// module must be defined inside one).
    pub const ID_NONE: u32 = 0u32;

    /// The live crabslab pattern: a generic handle with a phantom marker.
    pub struct Id<T> {
        pub index: u32,
        _marker: PhantomData<T>,
    }

    impl<T> Id<T> {
        pub const fn new(index: u32) -> Self {
            Id::<T> {
                index,
                _marker: PhantomData,
            }
        }

        /// `Self` in a const type position. WGSL forbids calling user
        /// functions in const initializers, so the initializer uses the
        /// struct-literal form (which renders the positional
        /// constructor, a const-expression) rather than `Self::new(..)`.
        pub const NONE: Self = Self {
            index: ID_NONE,
            _marker: PhantomData,
        };

        pub const MAGIC: u32 = 7u32;

        /// `Self` in parameter and return positions.
        pub fn then(id: Self, index: u32) -> Self {
            Id::<T> {
                index: id.index + index,
                _marker: PhantomData,
            }
        }

        /// A `Self::CONST` associated-constant path in a method body.
        pub fn magic_plus(x: u32) -> u32 {
            Self::MAGIC + x
        }

        /// A `Self::method(..)` call and an identity turbofish call to
        /// the own struct, in a method body.
        pub fn zero_index(id: Self) -> u32 {
            Self::none().index + Id::<T>::none().index + id.index
        }

        pub fn none() -> Self {
            Self {
                index: ID_NONE,
                _marker: PhantomData,
            }
        }
    }
}

#[wgsl(crate_path = wgsl_rs)]
mod use_id {
    use super::id_slab::*;

    pub fn make() -> Id<f32> {
        Id::<f32>::new(3u32)
    }

    pub fn read_zero() -> u32 {
        Id::<f32>::zero_index(Id::<f32>::new(0u32))
    }

    pub fn magic() -> u32 {
        Id::<f32>::magic_plus(1u32)
    }
}

#[test]
fn inherent_impl_self_renders_mangled_instantiation() {
    // CPU world: the module is plain Rust and runs on the host.
    let id = use_id::make();
    assert_eq!(id.index, 3u32);
    assert_eq!(use_id::read_zero(), 0u32);
    assert_eq!(use_id::magic(), 8u32);
    let bumped = id_slab::Id::<f32>::then(id, 1u32);
    assert_eq!(bumped.index, 4u32);
    let none: id_slab::Id<f32> = id_slab::Id::<f32>::NONE;
    assert_eq!(none.index, 0u32);

    // WGSL world: every `Self` position resolves to the concrete
    // instantiation, with no literal `Self` anywhere.
    let src = use_id::WGSL_SOURCE.wgsl_source().unwrap();
    assert!(
        src.contains("fn _1Id_f32_new(index: u32) -> Id_f32"),
        "expected the mangled `new` signature, got:\n{src}"
    );
    assert!(
        src.contains("const _1Id_f32_NONE: Id_f32 = Id_f32(ID_NONE);"),
        "expected the const's type and initializer to resolve, got:\n{src}"
    );
    assert!(
        src.contains("const _1Id_f32_MAGIC: u32 = 7u;"),
        "expected the associated const definition, got:\n{src}"
    );
    assert!(
        src.contains("_1Id_f32_none()"),
        "expected the `Self::none()` call to resolve to the mangled method, got:\n{src}"
    );
    assert!(
        !src.contains("Self"),
        "literal `Self` must not reach WGSL, got:\n{src}"
    );
    // The macro-generated `__validate_wgsl` tests cover naga validation
    // for both modules; this double-checks the assembled source.
    wgsl_rs::validate_wgsl_source(&src).expect("assembled source must validate");
}

/// A cross-module call to a generic struct method without turbofish
/// cannot be flagged at macro time (the consumer's macro cannot know the
/// imported name is generic), so source assembly reports it instead
/// (wgsl-rs#237). Auto-validation is skipped for this module on purpose:
/// its generated `__validate_wgsl` test would fail by design.
#[wgsl(crate_path = wgsl_rs, skip_validation)]
mod bare_caller {
    use super::id_slab::*;

    pub fn make_bare() -> Id<f32> {
        Id::new(0u32)
    }
}

#[test]
fn cross_module_bare_call_errors_at_source_assembly() {
    // CPU world: the bare call is ordinary Rust and runs fine on the
    // host — which is exactly why it used to fail silently in WGSL.
    assert_eq!(bare_caller::make_bare().index, 0u32);

    let err = bare_caller::WGSL_SOURCE.wgsl_source().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("calls generic template 'Id::new' without generic arguments"),
        "expected an actionable missing-turbofish error, got: {msg}"
    );
    assert!(
        msg.contains("turbofish"),
        "the error must point at the turbofish fix, got: {msg}"
    );
}
