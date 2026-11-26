//! Type-generic resource packs: a pack's type parameters become module-level
//! generics substituted into field types at instantiation.

use wgsl_rs::{std::*, wgsl};

#[wgsl(skip_validation)]
mod positions {
    use wgsl_rs::std::*;

    pub struct PositionPack<const GROUP: u32, T> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub data: RuntimeArray<T>,
    }

    /// A helper whose const and type parameters are both tied to the pack's
    /// module-level slots.
    pub fn count<const GROUP: u32, T: Wgsl>(positions: &PositionPack<GROUP, T>) -> u32 {
        array_length(&positions.data)
    }

    /// An entry point using a concrete type argument.
    #[vertex]
    pub fn vertex<const GROUP: u32>(
        #[builtin(vertex_index)] vertex_index: u32,
        positions: &PositionPack<GROUP, Vec4f>,
    ) -> Vec4f {
        let count = count(positions);
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x + count as f32 * 0.0, y, 0.0, 1.0)
    }

    /// An entry point whose type parameter ties to the pack slot through a
    /// generated `Type<Is = ...>` constraint.
    #[vertex]
    pub fn measure<const GROUP: u32, T: Wgsl>(
        #[builtin(vertex_index)] vertex_index: u32,
        positions: &PositionPack<GROUP, T>,
    ) -> Vec4f {
        let count = count(positions);
        vec4f(count as f32, vertex_index as f32, 0.0, 1.0)
    }
}

#[test]
fn generic_packs_run_on_cpu_with_constructed_packs() {
    let pack = positions::PositionPack::<1, Vec4f>::new();
    pack.data.set(RuntimeArray {
        data: vec![
            vec4f(1.0, 0.0, 0.0, 1.0),
            vec4f(0.0, 1.0, 0.0, 1.0),
            vec4f(0.0, 0.0, 1.0, 1.0),
        ],
    });

    let position = positions::vertex::<1>(0, &pack);
    assert_eq!(position, vec4f(-1.0, -1.0, 0.0, 1.0));

    let measured = positions::measure::<1, Vec4f>(0, &pack);
    assert_eq!(measured, vec4f(3.0, 0.0, 0.0, 1.0));
}

#[test]
fn pack_type_slots_substitute_into_wgsl() {
    // Pack type slots come before pack const slots, then entry-point
    // parameters: `measure`'s `T` survives as `measure_1`, while both
    // `GROUP` consts are shared aliases of the pack slot.
    assert_eq!(
        positions::WGSL_SOURCE.module_type_params,
        &["PositionPack_T", "PositionPack_GROUP", "measure_0"]
    );
    let module = positions::instantiate::<Vec4f, 1, Vec4f>();
    let source = module.wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(
        source
            .contains("@group(1) @binding(0) var<storage, read> positionpack__data: array<vec4f>;"),
        "{source}"
    );
    assert!(source.contains("fn count() -> u32"), "{source}");
    assert!(
        source.contains("return arrayLength(&positionpack__data);"),
        "{source}"
    );
    assert!(!source.contains("PositionPack"), "{source}");
    assert!(!source.contains("Vec4f"), "{source}");
}

#[wgsl(skip_validation)]
mod config {
    use wgsl_rs::std::*;

    pub struct ConfigPack<T> {
        #[uniform(group(2), binding(0))]
        pub value: T,
    }

    #[fragment]
    pub fn fragment(#[location(0)] position: Vec4f, _config: &ConfigPack<Vec4f>) -> Vec4f {
        position
    }
}

#[test]
fn generic_uniform_fields_substitute_and_construct_on_cpu() {
    assert_eq!(config::WGSL_SOURCE.module_type_params, &["ConfigPack_T"]);
    let pack = config::ConfigPack::<Vec4f>::new();
    pack.value.set(vec4f(4.0, 3.0, 2.0, 1.0));
    assert_eq!(*pack.value.get(), vec4f(4.0, 3.0, 2.0, 1.0));
    assert_eq!(
        config::fragment(vec4f(0.0, 0.0, 0.0, 1.0), &pack),
        vec4f(0.0, 0.0, 0.0, 1.0)
    );

    let module = config::instantiate::<Vec4f>();
    let source = module.wgsl_source();
    wgsl_rs::validate_wgsl_source(&source).unwrap();
    assert!(
        source.contains("@group(2) @binding(0) var<uniform> configpack__value: vec4f;"),
        "{source}"
    );
}
