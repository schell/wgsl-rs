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
        pub slab: Storage<RuntimeArray<Vec4f>, Read>,
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
Each field becomes one global resource named `packlower__field` (the lowercased
pack name, a double underscore, and the field name), so packs never collide on
field names and the generated WGSL is attributable at a glance. Pack parameters
are removed from WGSL function signatures, and their corresponding call
arguments are removed. A direct borrow such as `&texture_bg.heights` resolves
to the global `heights` resource. Storage access comes from the annotation or
the type marker; `read` is the default when neither names one.

## Constructing packs and making direct CPU calls

No constructors are generated. Hosts construct packs by hand over `pub`
fields, seeding values with `set` and textures with `init`:

```rust
use wgsl_rs::std::*;

let storage = render::SlabBindGroup::<1> {
    slab: Storage::new(1, 0),
};
storage.slab.set(RuntimeArray {
    data: vec![vec4f(0.0, 0.0, 0.0, 1.0)],
});

let textures = render::TextureBindGroup::<0, false> {
    heights: Texture2D::new(0, 0),
    heights_smp: Sampler::new(0, 1),
};
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

Uniform and storage fields declare their runtime wrapper types
(`Uniform<T>`, `Storage<T, A>`); the transpiler extracts the WGSL type from
the wrapper's type argument. Storage access may come from the annotation
(`read_only`/`read_write`), the type marker (`Read`/`ReadWrite`), or both;
when both are present they must agree, and an annotation-named access whose
marker is missing is inserted into the emitted Rust automatically. Texture
and sampler fields are their own runtime types.

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
The annotations, not the CPU-side constructor arguments, drive layout: pass
the same values to `instantiate` that the annotations describe. For a
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

## Binding-key consts

Every pack gains generated consts that key its bindings without hand-mangled
strings. A binding is addressed by two axes, `(group, "Pack::field")`:

```rust
impl<const FILTERABLE: bool> AtlasBindGroup<FILTERABLE> {
    pub const SLAB: &'static str = "AtlasBindGroup::slab";
    pub const TEXTURE: &'static str = "AtlasBindGroup::texture";
    pub const SMP: &'static str = "AtlasBindGroup::smp";
}
```

Consts that change only layout (`FILTERABLE`) vanish from the key: every
instantiation shares one spelling, so the key never needs to mention them.
Consts that select the group axis cannot be baked into a string (Rust has no
const string formatting), so the group rides a value const instead:

```rust
impl<const GROUP: u32> MyBg<GROUP> {
    pub const GROUP_N: u32 = GROUP;
    pub const SLAB: &'static str = "MyBg::slab";
}

// The group axis selects the bind group; the key stays the same.
let bg = linkage.create_bind_group_named(
    MyBg::<1>::GROUP_N,
    &device,
    &[(MyBg::<1>::SLAB, slab_buffer.as_entire_binding())],
)?;
```

The linkage accepts both spellings everywhere bindings are named:
`create_bind_group_named`, `BindGroupInfo::create_named`, and the
`buffer`/`take_buffer` lookups all canonicalize `"Pack::field"` to the
mangled WGSL global name automatically. See
[Bind Groups & Buffers](../linkage/bind-groups.md).

For the string-free path, derive
[`wgsl_rs_linkage_wgpu::BindGroup`](../linkage/bind-groups.md#typed-bind-groups-derivebindgroup)
on the pack: the derive generates a typed `Resources` struct and a
`create_bind_group` that needs no keys at all.

## Type parameters on packs

Packs may declare type parameters alongside their consts. A type parameter
must be used by at least one annotated field, either as the bare field type
or as an element type such as `RuntimeArray<T>`:

```rust
#[wgsl(skip_validation)]
pub mod generic {
    use wgsl_rs::std::*;

    pub struct DataPack<const GROUP: u32, T> {
        #[storage(group(GROUP), binding(0), read_only)]
        pub data: Storage<RuntimeArray<T>, Read>,
    }

    pub fn count<const GROUP: u32, T: Wgsl>(pack: &DataPack<GROUP, T>) -> u32 {
        array_length(&pack.data)
    }

    #[vertex]
    pub fn vertex<const GROUP: u32>(
        #[builtin(vertex_index)] vertex_index: u32,
        pack: &DataPack<GROUP, Vec4f>,
    ) -> Vec4f {
        let count = count(pack);
        let x = f32((vertex_index & 1u32) * 2u32) * 2.0 - 1.0;
        let y = f32((vertex_index >> 1u32) * 2u32) * 2.0 - 1.0;
        vec4f(x + count as f32 * 0.0, y, 0.0, 1.0)
    }
}
```

Each pack type parameter owns one module-level slot named `DataPack_T`,
listed before the pack's const slots:

```rust
let module = generic::instantiate::<Vec4f, 1>();
```

Entry points supply type arguments either concretely
(`&DataPack<GROUP, Vec4f>`) or through their own generic parameters, which
tie to the pack slot through generated constraints so rustc rejects
inconsistent instantiation. Helpers may use a bare type parameter in the
pack parameter only; it reads the pack slot directly, mirroring const
aliasing, and may not appear in any other parameter or return position.

## Workgroup fields

Workgroup variables join packs with the `#[workgroup]` annotation and the
`Workgroup<T>` wrapper. They carry no group or binding — `#[workgroup]` takes
no arguments — and render as `var<workgroup> packlower__field: T;`:

```rust
pub struct WorkgroupData<T> {
    #[workgroup]
    pub accumulator: Workgroup<T>,
}

#[compute]
#[workgroup_size(4)]
pub fn accumulate(wg: &WorkgroupData<i32>) {
    let current = load!(wg.accumulator);
    *get_mut!(wg.accumulator) = current + 1;
}
```

Hosts seed the value at construction:
`WorkgroupData::<i32> { accumulator: Workgroup::new(0) }`. Workgroup fields may
mix with binding fields in one pack.

## V1 boundaries

- Every pack field must have exactly one `#[texture]`, `#[sampler]`,
  `#[storage]`, `#[uniform]`, or `#[workgroup]` annotation. Non-resource
  fields are rejected.
- Builtins and interstage IO stay in separate IO structs. Mixing binding and IO
  annotations is an error.
- Resource access uses direct field borrows such as `&pack.field`.
  Destructuring, local resource aliases, by-value pack parameters, and pack
  return values are not supported.
- Existing receiver rules are unchanged. Associated functions may explicitly
  accept `&Pack`, but `self`, `&self`, and `&mut self` are not added.
- Resource names and concrete `(group, binding)` pairs must be unique.
