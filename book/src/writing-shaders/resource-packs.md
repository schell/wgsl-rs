# Resource Pack Structs

Resource packs group related GPU bindings in an ordinary Rust struct. The
fields carry binding annotations, paralleling the annotations on interstage IO
structs. A pack owns its CPU-side resources, while shader entry points and
helpers borrow it.

## Declaring a complete vertex + fragment shader

```rust
use wgsl_rs::wgsl;

#[wgsl(skip_validation)]
pub mod render {
    use wgsl_rs::std::*;

    pub struct SlabBindGroup<const GROUP: u32> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub slab: RuntimeArray<Vec4f>,
    }

    pub struct TextureBindGroup<const GROUP: u32, const FILTERABLE: bool> {
        #[texture(group(GROUP), binding(0))]
        pub heights: Texture2D<f32, FILTERABLE>,

        #[sampler(group(GROUP), binding(1))]
        pub heights_smp: Sampler<FILTERABLE>,
    }

    // Builtins and interstage data stay in ordinary IO structs.
    pub struct FragInput {
        #[builtin(position)]
        pub position: Vec4f,
    }

    pub fn shade<const GROUP: u32, const FILTERABLE: bool>(
        texture_bg: &TextureBindGroup<GROUP, FILTERABLE>,
        coords: Vec2f,
    ) -> Vec4f {
        texture_sample(
            &texture_bg.heights,
            &texture_bg.heights_smp,
            coords,
        )
    }

    #[vertex]
    pub fn vertex<const GROUP: u32>(
        #[builtin(vertex_index)] vertex_index: u32,
        storage: &SlabBindGroup<GROUP>,
    ) -> Vec4f {
        let count = array_length(&storage.slab);
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x + count as f32 * 0.0, y, 0.0, 1.0)
    }

    #[fragment]
    pub fn fragment<const GROUP: u32, const FILTERABLE: bool>(
        input: FragInput,
        texture: &TextureBindGroup<GROUP, FILTERABLE>,
    ) -> Vec4f {
        let p = input.position;
        let dims = texture_dimensions(&texture.heights);
        let coords = vec2f(p.x / dims.x() as f32, p.y / dims.y() as f32);
        shade(texture, coords)
    }
}
```

The macro does not emit `SlabBindGroup` or `TextureBindGroup` as WGSL structs.
Each field becomes one global resource named after the field. Pack parameters
are removed from WGSL function signatures, and their corresponding call
arguments are removed. A direct borrow such as `&texture_bg.heights` resolves
to the global `heights` resource. Storage access is explicit in the annotation;
`read_only` is the default, and `read_write` selects read-write storage.

## Constructing packs and making direct CPU calls

`new()` constructs every field from its annotation. No addresses are repeated
at the call site:

```rust
use wgsl_rs::std::*;

let storage = render::SlabBindGroup::<1>::new();
storage.slab.set(RuntimeArray {
    data: vec![vec4f(0.0, 0.0, 0.0, 1.0)],
});

let textures = render::TextureBindGroup::<0, false>::new();
textures.heights.init(1, 1);
textures.heights.set_pixel(0, 0, [0.25, 0.5, 0.75, 1.0]);
textures.heights_smp.init();

let position = render::vertex::<1>(0, &storage);
let color = render::fragment::<0, false>(
    render::FragInput {
        position: vec4f(0.0, 0.0, 0.0, 1.0),
    },
    &textures,
);
assert_eq!(color, vec4f(0.25, 0.5, 0.75, 1.0));
```

Texture and sampler fields retain their declared runtime types. Uniform and
storage declarations name their inner value type, while the emitted Rust field
uses the existing `Uniform<T>` or `Storage<T, A>` runtime resource. The
constructor takes care of that distinction.

## Instantiating layouts

Group and binding addresses accept integer literals or the pack's `u32`/`usize`
const parameters. Texture and sampler filterability uses a bool const. Select
all three values through the module's generated `instantiate` function:

```rust
use wgsl_rs_linkage_wgpu::IrModuleExt;

let module = render::instantiate::<1, 0, false>();
let linkage = module.generate_linkage();
```

The argument order above is `SlabBindGroup_GROUP`, `TextureBindGroup_GROUP`,
and `TextureBindGroup_FILTERABLE`, matching `WGSL_SOURCE.module_type_params`.
Use the same values for constructed CPU packs and IR instantiation. For a
non-filtering layout, use an all-Nearest sampler on the GPU. Filterability is
host-only and never changes the WGSL texture or sampler type.

Each pack const has one deterministic, pack-owned slot. Entry-point consts
that feed a pack are aliases, not extra arguments. Multiple entry points using
one pack therefore share its values. Separate packs remain independent even
when their const parameters have the same source name. Matching concrete uses
need no generic slot; conflicting concrete or concrete/generic uses are
compile-time errors. Duplicate names and addresses are checked again after
instantiation. See [Template Modules & Instantiation](../generics/templates.md)
for the full ordering rules.

## V1 boundaries

- Every pack field must have exactly one `#[texture]`, `#[sampler]`,
  `#[storage]`, or `#[uniform]` annotation. Non-resource and workgroup fields
  are rejected.
- Builtins and interstage IO stay in separate IO structs. Mixing binding and IO
  annotations is an error.
- Resource access uses direct field borrows such as `&pack.field`.
  Destructuring, local resource aliases, by-value pack parameters, and pack
  return values are not supported.
- Existing receiver rules are unchanged. Associated functions may explicitly
  accept `&Pack`, but `self`, `&self`, and `&mut self` are not added.
- Resource names and concrete `(group, binding)` pairs must be unique.
