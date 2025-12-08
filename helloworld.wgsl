@ group(0) @ binding(0) var < uniform > FRAME : f32;

fn vertex(vertex_index : u32) -> vec4f
{
    const POS : [vec2f; 3] = [
        vec2f(0.0, 0.5), vec2f(- 0.5, - 0.5), vec2f(0.5, - 0.5)
    ];
    let position = POS [vertex_index as usize];
    return vec4(position.x, position.y, 0.0, 1.0);
}

fn fragment() -> vec4f {
    return vec4(1.0, (f32(FRAME) / 128.0).sin, 0.0, 1.0);
}
