use wgsl_rs::wgsl;

// Calling a method of a generic struct without turbofish renders the
// un-mangled name (`Id_new`), which fails WGSL validation far from the
// mistake. The transpiler cannot infer the concrete type arguments, so
// this is a hard error (wgsl-rs#237).
#[wgsl(skip_validation)]
mod generic_struct_method_missing_turbofish {
    pub struct Id<T> {
        pub index: u32,
    }

    impl<T> Id<T> {
        pub fn new(index: u32) -> Self {
            Id::<T> { index }
        }
    }

    pub fn make() -> Id<f32> {
        Id::new(0u32)
    }
}

fn main() {}