use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
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

    // BINS is declared once as a pack field; each entry point supplies its
    // view through the pack's type argument, tied by generated
    // `linkage::Type<Is = ...>` constraints exactly like the old `get!(VAR, T)`
    // forms were.
    pub struct BinsPack<T> {
        #[storage(group(0), binding(0), read_write)]
        pub bins: Storage<T>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main_array<T: WgslScalar + Zeroable>(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        pack: &BinsPack<Vec4<T>>,
    ) {
        // BINS viewed as `Vec4<T>`: the pack's type argument carries the
        // constraint that used to come from `get_mut!(BINS, Vec4<T>)`.
        // Assign through the accessor directly — a local bound from
        // `get_mut!` holds a guard on the CPU and a value in WGSL, so
        // dereferencing it would render invalid WGSL (issue #153).
        let mut bins = get_mut!(pack.bins);
        bins[global_id.x as usize] = T::zero();
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main_zeroable<T: Wgsl + Zeroable>(
        #[builtin(global_invocation_id)] _global_id: Vec3u,
        pack: &BinsPack<T>,
    ) {
        // BINS viewed as T, with `T: Wgsl + Zeroable`.
        // These constraints are still solvable, as Vec4<f32> is
        // `Wgsl + Zeroable`.
        // Assign through the accessor directly (see `main_array`).
        *get_mut!(pack.bins) = T::zero();
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main_f32(#[builtin(global_invocation_id)] _global_id: Vec3u, pack: &BinsPack<f32>) {
        // BINS concretized to f32 within this entry point's view.
        // Assign through the accessor directly (see `main_array`).
        *get_mut!(pack.bins) = 0.0;
    }
}
