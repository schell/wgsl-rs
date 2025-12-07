use wgsl_rs::wgsl;

#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing color.
    //! Original source is [here](https://google.github.io/tour-of-wgsl/).
    use wgsl_rs::std::*;

    pub fn vertex(vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];

        let position = POS[vertex_index as usize];
        vec4(position.x(), position.y(), 0.0, 1.0)
    }

    pub fn fragment(frame: u32) -> Vec4f {
        vec4(1.0, ((frame as f32) / 128.0).sin(), 0.0, 1.0)
    }
}

pub fn main() {
    println!("hello!");
}
