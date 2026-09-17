//! `Vec3f` has `f32` components and `f32` is neither `Eq` nor `Hash`, so the
//! derived `Eq`/`Hash` impls must not apply (issue #161). Only component
//! types implementing `Hash`/`Eq` (u32, i32, bool) get them.

use std::collections::HashMap;

use wgsl_rs::std::{vec3f, Vec3f};

fn main() {
    let mut map: HashMap<Vec3f, u32> = HashMap::new();
    map.insert(vec3f(1.0, 2.0, 3.0), 1);
}