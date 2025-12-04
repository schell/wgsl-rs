use glam::{Vec2, Vec4};

pub fn vertex(vertex_index: u32) -> Vec4 {
    const POS: &[Vec2] = &[
        Vec2::new(0.0, 0.5),
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
    ];

    let position = POS[vertex_index as usize];
    Vec4::new(position.x, position.y, 0.0, 1.0)
}

fn fragment(frame: u32) -> Vec4 {
    Vec4::new(1.0, ((frame as f32) / 128.0).sin(), 0.0, 1.0)
}

fn main() {
    println!("Hello, world!");
}
