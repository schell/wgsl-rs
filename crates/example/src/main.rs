use wgsl_rs::wgsl;

#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing color.
    //! Original source is [here](https://google.github.io/tour-of-wgsl/).

    // Only glob-imports are supported, but hey, imports work!
    use wgsl_rs::std::*;

    // Define a uniform in both Rust and WGSL using the uniform! macro.
    uniform!(group(0), binding(0), FRAME: u32);

    #[vertex]
    pub fn vtx_main(#[builtin(vertex_index)] vertex_index: u32) -> Vec4f {
        const POS: [Vec2f; 3] = [vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)];

        let position = POS[vertex_index as usize];
        vec4(position.x(), position.y(), 0.0, 1.0)
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        vec4(1.0, sin(f32(FRAME) / 128.0), 0.0, 1.0)
    }
}

pub fn main() {
    let source = hello_triangle::WGSL_MODULE.wgsl_source().join("\n");
    println!("raw source:\n\n{source}\n\n");

    // Parse the source into a Module.
    let module: naga::Module = naga::front::wgsl::parse_str(&source).unwrap();

    // Validate the module.
    // Validation can be made less restrictive by changing the ValidationFlags.
    let result = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module);

    let info = match result {
        Err(e) => {
            panic!("{}", e.emit_to_string(&source));
        }
        Ok(i) => i,
    };

    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    println!("naga source:\n\n{wgsl}");
}
