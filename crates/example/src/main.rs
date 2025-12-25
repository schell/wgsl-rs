use wgsl_rs::wgsl;

#[wgsl]
pub mod hello_triangle {
    //! This is a "hello world" shader that shows a triangle with changing
    //! color. Original source is [here](https://google.github.io/tour-of-wgsl/).

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

#[wgsl]
pub mod structs {
    use wgsl_rs::std::*;

    // pub struct A {
    //     pub inner: f32,
    // }

    // pub struct B {
    //     pub a: A,
    // }

    // pub struct C {
    //     pub b: B,
    // }

    // pub fn f32_from_c(c: C) -> f32 {
    //     c.b.a.inner
    // }

    // Mixed builtins and user-defined inputs.
    #[input]
    pub struct MyInputs {
        #[location(0)]
        pub x: Vec4<f32>,

        #[builtin(front_facing)]
        pub y: bool,

        #[location(1)]
        #[interpolate(flat)]
        pub z: u32,

        #[location(2)]
        pub other: f32,
    }

    #[output]
    pub struct MyOutputs {
        #[location(0)]
        pub x: f32,

        #[location(1)]
        pub y: Vec4<f32>,
    }

    #[fragment]
    pub fn frag_shader(in1: MyInputs) -> MyOutputs {
        MyOutputs { x: 0.0, y: in1.x }
    }
}

#[wgsl]
pub mod compute_shader {
    //! A simple compute shader that demonstrates storage buffers.
    use wgsl_rs::std::*;

    // Read-only input buffer
    storage!(group(0), binding(0), INPUT: [f32; 256]);

    // Read-write output buffer
    storage!(group(0), binding(1), read_write, OUTPUT: [f32; 256]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Compute the index from global invocation ID
        // Note: Storage buffer access requires additional implementation
        // This demonstrates the compute shader structure with storage buffers
        let _idx = global_id.x() as usize;
    }
}

fn validate_and_print_source(source: &str) {
    println!("raw source:\n\n{source}\n\n");

    // Parse the source into a Module.
    let module: naga::Module = naga::front::wgsl::parse_str(source).unwrap();

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
            panic!("{}", e.emit_to_string(source));
        }
        Ok(i) => i,
    };

    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    println!("naga source:\n\n{wgsl}");
}

pub fn main() {
    {
        // hello_triangle
        let source = hello_triangle::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
    {
        // structs
        let source = structs::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
    {
        // compute_shader
        let source = compute_shader::WGSL_MODULE.wgsl_source().join("\n");
        validate_and_print_source(&source);
    }
}
