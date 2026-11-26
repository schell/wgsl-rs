//! Pack-field access through the `get!`/`get_mut!`/`load!` macros.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod shader {
    use wgsl_rs::std::*;

    pub struct FramePack {
        #[uniform(group(0), binding(0))]
        pub time: Uniform<f32>,
    }

    pub struct BinPack {
        #[storage(group(0), binding(1), read_write)]
        pub bins: Storage<[f32; 4]>,
    }

    #[fragment]
    pub fn scaled(#[location(0)] _position: Vec4f, frame: &FramePack) -> Vec4f {
        vec4f(load!(frame.time) * 2.0, 0.0, 0.0, 1.0)
    }

    #[compute]
    #[workgroup_size(4)]
    pub fn seed(bins: &BinPack) {
        get_mut!(bins.bins)[1] = 7.0;
        get_mut!(&bins.bins)[2] = 3.0;
    }
}

#[test]
fn macro_access_runs_on_cpu_with_a_constructed_pack() {
    let frame = shader::FramePack {
        time: Uniform::new(0, 0),
    };
    frame.time.set(2.0);
    assert_eq!(
        shader::scaled(vec4f(0.0, 0.0, 0.0, 1.0), &frame),
        vec4f(4.0, 0.0, 0.0, 1.0)
    );

    let bins = shader::BinPack {
        bins: Storage::new(0, 1),
    };
    bins.bins.set([0.0, 0.0, 0.0, 0.0]);
    shader::seed(&bins);
    assert_eq!(*bins.bins.get(), [0.0, 7.0, 3.0, 0.0]);
}

#[test]
fn macro_access_lowers_to_mangled_globals() {
    let source = shader::instantiate().wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(
        source.contains("return vec4f((framepack__time * 2.0), 0.0, 0.0, 1.0);"),
        "{source}"
    );
    assert!(source.contains("binpack__bins[1] = 7.0;"), "{source}");
    assert!(source.contains("binpack__bins[2] = 3.0;"), "{source}");
    assert!(!source.contains("FramePack"), "{source}");
    assert!(!source.contains("BinPack"), "{source}");
}
