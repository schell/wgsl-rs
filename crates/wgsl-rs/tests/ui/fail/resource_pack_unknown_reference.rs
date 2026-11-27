use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
mod unknown_reference {
    pub struct NotAPack {
        pub value: f32,
    }

    pub fn helper(value: &NotAPack) {}
}

fn main() {}
