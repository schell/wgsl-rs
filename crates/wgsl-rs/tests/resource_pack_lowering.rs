//! Resource-pack parameters lower to globals while remaining ordinary Rust.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod shader {
    use wgsl_rs::std::*;

    pub struct TextureBindGroup<const FILTERABLE: bool> {
        #[texture(group(0), binding(0))]
        pub heights: Texture2D<f32, FILTERABLE>,

        #[sampler(group(0), binding(1))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    pub fn shade<const FILTERABLE: bool>(
        texture_bg: &TextureBindGroup<FILTERABLE>,
        coords: Vec2f,
    ) -> Vec4f {
        texture_sample(&texture_bg.heights, &texture_bg.heights_smp, coords)
    }

    #[fragment]
    pub fn fragment<const FILTERABLE: bool>(
        #[location(0)] clip_pos: Vec4f,
        texture: &TextureBindGroup<FILTERABLE>,
    ) -> Vec4f {
        shade(texture, clip_pos.xy())
    }
}

#[test]
fn helper_and_entry_point_run_on_cpu_with_a_constructed_pack() {
    let pack = shader::TextureBindGroup::<false>::new();
    pack.heights.init(1, 1);
    pack.heights.set_pixel(0, 0, [0.125, 0.25, 0.5, 1.0]);
    pack.heights_smp.init();

    let color = shader::fragment::<false>(vec4f(0.0, 0.0, 0.0, 1.0), &pack);
    assert_eq!(color, vec4f(0.125, 0.25, 0.5, 1.0));
}

#[test]
fn pack_parameters_and_calls_disappear_from_wgsl() {
    assert_eq!(
        shader::WGSL_SOURCE.module_type_params,
        &["TextureBindGroup_FILTERABLE"]
    );
    for filterable in [false, true] {
        let module = if filterable {
            shader::instantiate::<true>()
        } else {
            shader::instantiate::<false>()
        };
        let source = module.wgsl_source();
        wgsl_rs::validate_wgsl_source(&source).unwrap();

        assert!(
            source
                .contains("@group(0) @binding(0) var texturebindgroup__heights: texture_2d<f32>;")
        );
        assert!(
            source.contains("@group(0) @binding(1) var texturebindgroup__heights_smp: sampler;")
        );
        assert!(source.contains("fn shade(coords: vec2f) -> vec4f"));
        assert!(source.contains(
            "textureSample(texturebindgroup__heights, texturebindgroup__heights_smp, coords)"
        ));
        assert!(source.contains("return shade(clip_pos.xy);"), "{source}");
        assert!(!source.contains("TextureBindGroup"));
        assert!(!source.contains("texture_bg"));
        assert!(!source.contains("texture: &"));
        assert!(!source.contains("shade(texture"));
    }
}

#[wgsl]
mod associated {
    use wgsl_rs::std::*;

    pub struct Pack {
        #[texture(group(0), binding(0))]
        pub texture: Texture2D<f32>,
        #[sampler(group(0), binding(1))]
        pub smp: Sampler,
    }

    #[allow(dead_code)]
    pub struct Ops {
        pub unused: f32,
    }

    impl Ops {
        pub fn sample(pack: &Pack, coords: Vec2f) -> Vec4f {
            texture_sample(&pack.texture, &pack.smp, coords)
        }
    }

    #[fragment]
    pub fn fragment(#[location(0)] position: Vec4f, pack: &Pack) -> Vec4f {
        Ops::sample(pack, position.xy())
    }
}

#[test]
fn associated_functions_accept_packs_without_receiver_lowering() {
    let pack = associated::Pack::new();
    pack.texture.init(1, 1);
    pack.texture.set_pixel(0, 0, [1.0, 0.5, 0.25, 1.0]);
    pack.smp.init();
    assert_eq!(
        associated::fragment(vec4f(0.0, 0.0, 0.0, 1.0), &pack),
        vec4f(1.0, 0.5, 0.25, 1.0)
    );

    let source = associated::instantiate().wgsl_source();
    assert!(
        source.contains("fn Ops_sample(coords: vec2f) -> vec4f"),
        "{source}"
    );
    assert!(
        source.contains("return Ops_sample(position.xy);"),
        "{source}"
    );
    assert!(!source.contains("pack: &Pack"), "{source}");
}
