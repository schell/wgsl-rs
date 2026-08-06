# Renderer Specialization

A full renderer example specialized via traits and turbofish. Three trait axes (`Material`, `LightModel`, `NormalSource`) are composed in a single generic `shade_fragment` function. Each concrete configuration (selected via turbofish like `shade_fragment::<Gradient, BlinnPhong, PerturbNormal>`) monomorphizes into a fully-inlined, specialized WGSL function with zero overhead.

## Rust Source

```rust
#[wgsl]
pub mod renderer_specialization {
    use wgsl_rs::std::*;

    // ===== Traits: each axis of rendering variation =====

    /// How a material produces a surface color at a given UV coordinate.
    pub trait Material {
        fn surface_color(uv: Vec2f) -> Vec4f;
    }

    /// How lighting is computed for a given surface color and geometry.
    pub trait LightModel {
        fn apply_lighting(
            surface: Vec4f,
            normal: Vec3f,
            light_dir: Vec3f,
            view_dir: Vec3f,
        ) -> Vec4f;
    }

    /// How the surface normal is determined.
    pub trait NormalSource {
        fn get_normal(uv: Vec2f, geom_normal: Vec3f) -> Vec3f;
    }

    // ===== Strategy structs =====
    //
    // Each struct represents a concrete rendering strategy. In a real
    // renderer these might hold configuration data; here they serve as
    // type-level tags that select which code path to monomorphize.

    /// Simple checkerboard material — procedural, no textures needed.
    pub struct Checker {
        pub _tag: u32,
    }

    /// Vertical gradient material — warm-to-cool procedural color.
    pub struct Gradient {
        pub _tag: u32,
    }

    /// Lambert diffuse lighting model.
    pub struct Lambert {
        pub _tag: u32,
    }

    /// Blinn-Phong lighting with specular highlights.
    pub struct BlinnPhong {
        pub _tag: u32,
    }

    /// Use the raw geometric normal as-is.
    pub struct GeomNormal {
        pub _tag: u32,
    }

    /// Perturb the geometric normal (simulates a normal map).
    pub struct PerturbNormal {
        pub _tag: u32,
    }

    // ===== Trait implementations =====

    impl Material for Checker {
        fn surface_color(uv: Vec2f) -> Vec4f {
            let checker: f32 = floor(uv.x() * 4.0) + floor(uv.y() * 4.0);
            let c: f32 = (checker % 2.0) * 0.5 + 0.25;
            vec4f(c, c, c, 1.0)
        }
    }

    impl Material for Gradient {
        fn surface_color(uv: Vec2f) -> Vec4f {
            vec4f(uv.y() * 0.8, 0.3, (1.0 - uv.y()) * 0.9, 1.0)
        }
    }

    impl LightModel for Lambert {
        fn apply_lighting(
            surface: Vec4f,
            normal: Vec3f,
            light_dir: Vec3f,
            _view_dir: Vec3f,
        ) -> Vec4f {
            let ndotl: f32 = max(dot(normal, light_dir), 0.0);
            surface * ndotl
        }
    }

    impl LightModel for BlinnPhong {
        fn apply_lighting(
            surface: Vec4f,
            normal: Vec3f,
            light_dir: Vec3f,
            view_dir: Vec3f,
        ) -> Vec4f {
            let ndotl: f32 = max(dot(normal, light_dir), 0.0);
            let diffuse: Vec4f = surface * ndotl;
            let half_vec: Vec3f = normalize(light_dir + view_dir);
            let spec: f32 = pow(max(dot(normal, half_vec), 0.0), 32.0);
            diffuse + vec4f(spec, spec, spec, 0.0)
        }
    }

    impl NormalSource for GeomNormal {
        fn get_normal(_uv: Vec2f, geom_normal: Vec3f) -> Vec3f {
            normalize(geom_normal)
        }
    }

    impl NormalSource for PerturbNormal {
        fn get_normal(uv: Vec2f, geom_normal: Vec3f) -> Vec3f {
            let perturb: Vec3f = vec3f(uv.x() * 0.1 - 0.05, uv.y() * 0.1 - 0.05, 1.0);
            normalize(geom_normal + perturb)
        }
    }

    // ===== Generic shader pipeline =====

    /// The single generic fragment shading function. It composes material,
    /// lighting, and normal sourcing through trait bounds. After
    /// monomorphization, each configuration produces a fully-inlined,
    /// specialized WGSL function with zero overhead.
    pub fn shade_fragment<M: Material, L: LightModel, N: NormalSource>(
        uv: Vec2f,
        geom_normal: Vec3f,
        light_dir: Vec3f,
        view_dir: Vec3f,
    ) -> Vec4f {
        let normal: Vec3f = N::get_normal(uv, geom_normal);
        let surface: Vec4f = M::surface_color(uv);
        L::apply_lighting(surface, normal, light_dir, view_dir)
    }

    // ===== Concrete shader variants (each is one turbofish line) =====

    /// Fancy renderer: gradient + Blinn-Phong + perturbed normals.
    pub fn shade_fancy(uv: Vec2f, normal: Vec3f, light_dir: Vec3f, view_dir: Vec3f) -> Vec4f {
        shade_fragment::<Gradient, BlinnPhong, PerturbNormal>(uv, normal, light_dir, view_dir)
    }

    /// Mix-and-match: checkerboard + Blinn-Phong + perturbed normals.
    /// Demonstrates that each axis of variation is independent.
    pub fn shade_hybrid(uv: Vec2f, normal: Vec3f, light_dir: Vec3f, view_dir: Vec3f) -> Vec4f {
        shade_fragment::<Checker, BlinnPhong, PerturbNormal>(uv, normal, light_dir, view_dir)
    }
}
```

## Generated WGSL

```wgsl
struct Checker {
    _tag: u32
}

struct Gradient {
    _tag: u32
}

struct Lambert {
    _tag: u32
}

struct BlinnPhong {
    _tag: u32
}

struct GeomNormal {
    _tag: u32
}

struct PerturbNormal {
    _tag: u32
}

fn Checker__1surface_color(uv: vec2f) -> vec4f {
    let checker: f32 = floor(uv.x * 4.0) + floor(uv.y * 4.0);
    let c: f32 = (checker % 2.0) * 0.5 + 0.25;
    return vec4f(c, c, c, 1.0);
}

fn Gradient__1surface_color(uv: vec2f) -> vec4f {
    return vec4f(uv.y * 0.8, 0.3, (1.0 - uv.y) * 0.9, 1.0);
}

fn Lambert__1apply_lighting(surface: vec4f, normal: vec3f, light_dir: vec3f, _view_dir: vec3f) -> vec4f {
    let ndotl: f32 = max(dot(normal, light_dir), 0.0);
    return surface * ndotl;
}

fn BlinnPhong__1apply_lighting(surface: vec4f, normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    let ndotl: f32 = max(dot(normal, light_dir), 0.0);
    let diffuse: vec4f = surface * ndotl;
    let half_vec: vec3f = normalize(light_dir + view_dir);
    let spec: f32 = pow(max(dot(normal, half_vec), 0.0), 32.0);
    return diffuse + vec4f(spec, spec, spec, 0.0);
}

fn GeomNormal__1get_normal(_uv: vec2f, geom_normal: vec3f) -> vec3f {
    return normalize(geom_normal);
}

fn PerturbNormal__1get_normal(uv: vec2f, geom_normal: vec3f) -> vec3f {
    let perturb: vec3f = vec3f(uv.x * 0.1 - 0.05, uv.y * 0.1 - 0.05, 1.0);
    return normalize(geom_normal + perturb);
}

fn shade_fancy(uv: vec2f, normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    return _1shade_fragment_Gradient_BlinnPhong_PerturbNormal(uv, normal, light_dir, view_dir);
}

fn shade_hybrid(uv: vec2f, normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    return _1shade_fragment_Checker_BlinnPhong_PerturbNormal(uv, normal, light_dir, view_dir);
}

fn _1shade_fragment_Gradient_BlinnPhong_PerturbNormal(uv: vec2f, geom_normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    let normal: vec3f = PerturbNormal__1get_normal(uv, geom_normal);
    let surface: vec4f = Gradient__1surface_color(uv);
    return BlinnPhong__1apply_lighting(surface, normal, light_dir, view_dir);
}

fn _1shade_fragment_Checker_BlinnPhong_PerturbNormal(uv: vec2f, geom_normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    let normal: vec3f = PerturbNormal__1get_normal(uv, geom_normal);
    let surface: vec4f = Checker__1surface_color(uv);
    return BlinnPhong__1apply_lighting(surface, normal, light_dir, view_dir);
}
```

## Notes

- Each trait method becomes a free function mangled as `Type__1method` (e.g. `Checker__1surface_color`).
- The generic `shade_fragment<M, L, N>` is monomorphized into one function per turbofish configuration, named `_1shade_fragment_<M>_<L>_<N>`.
- Strategy structs become empty-tagged WGSL structs; they exist only to drive monomorphization.