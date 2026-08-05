//! Failing test: `PhantomData<T>` is only supported as a struct field
//! type. Using it in a function parameter, return type, local binding,
//! or any other non-field type position is rejected at parse time so
//! that `render::write_type`'s `Type::Phantom` arm is truly unreachable.
//!
//! See `tests/ui/pass/phantom_data.rs` for the supported struct-field
//! form.
use wgsl_rs::wgsl;

#[wgsl(crate_path = wgsl_rs)]
pub mod phantom_in_param {
    use wgsl_rs::std::*;

    // Rejected: `PhantomData<T>` as a function parameter type.
    pub fn bad(x: PhantomData<f32>) -> f32 {
        0.0
    }
}

fn main() {}