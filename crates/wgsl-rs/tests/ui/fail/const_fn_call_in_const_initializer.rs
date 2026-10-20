use wgsl_rs::wgsl;

// WGSL const initializers must be const-expressions. Rust happily
// accepts a `const fn` call in a const initializer, but the rendered
// WGSL is invalid: user functions can never be const-functions (WGSL
// §11.3), so this is rejected at parse time with guidance toward the
// struct-literal form (wgsl-rs#239).
#[wgsl(skip_validation)]
mod const_fn_call_in_const {
    pub struct MyThing {
        pub times: u32,
    }

    impl MyThing {
        pub const fn thing() -> MyThing {
            MyThing { times: 0 }
        }
        pub const THING: MyThing = MyThing { times: 0 };
        pub const OTHER_THING: MyThing = MyThing::thing();
    }
}

fn main() {}