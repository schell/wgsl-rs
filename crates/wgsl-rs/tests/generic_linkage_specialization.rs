use wgsl_rs::wgsl;

#[wgsl]
pub mod generic_linkage {
    //! This module exercises an issue where one generic linkage is used as
    //! different concrete types.
    //!
    //! This illustrates a puzzle with generic linkages.
    //!
    //! ### Solution
    //! During parsing, each time a generic linkage is accessed we need to log a
    //! type constraint. Then during code gen, we can list these constraints
    //! on the builder methods. This should catch conflicting constraints at
    //! instantiation.
    use wgsl_rs::std::*;

    pub trait Zeroable {
        fn zero() -> Self;
    }

    impl Zeroable for u32 {
        fn zero() -> u32 {
            0
        }
    }

    impl Zeroable for f32 {
        fn zero() -> f32 {
            0.0
        }
    }

    impl Zeroable for Vec4f {
        fn zero() -> Vec4f {
            Vec4f::default()
        }
    }

    // Here BINS is defined with the constraint `impl std::any::Any`.
    storage!(group(0), binding(0), BINS: impl Zeroable);

    #[compute]
    #[workgroup_size(64)]
    pub fn main_array<T: WgslScalar + Zeroable>(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Here BINS is accessed as Vec4<T>, which means now we know BINS is
        // `Vec4<T>: std::any::Any`
        // and
        // `T: WgslScalar + Zeroable`
        let mut bins = get_mut!(BINS, Vec4<T>);
        bins[global_id.x as usize] = T::zero();
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main_zeroable<T: Wgsl + Zeroable>(#[builtin(global_invocation_id)] _global_id: Vec3u) {
        // Here BINS is accessed as T, so we know BINS is
        // `Vec4<T>: std::any::Any + Zeroable`
        // and
        // `T: Wgsl + Zeroable`
        //
        // These constraints are still solvable, as Vec4<f32> is `Zeroable`.
        let mut bins = get_mut!(BINS, T);
        *bins = T::zero();
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main_f32(#[builtin(global_invocation_id)] _global_id: Vec3u) {
        // Here BINS is access as f32, concretizing the type within this module.
        // This should set up the constraints as:
        // `f32: std::any::Any + WgslScalar + Zeroable`
        // but we also need to show that f32 != [T; 4].
        // Maybe we could use associated types here?
        let mut bins = get_mut!(BINS, f32);
        *bins = 0.0;
    }
}
