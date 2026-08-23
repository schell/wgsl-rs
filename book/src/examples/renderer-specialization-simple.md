# Renderer Specialization (Simple)

A second specialization of the shared renderer pipeline from [renderer-specialization](./renderer-specialization.md). This module selects `Checker + Lambert + GeomNormal` via turbofish, producing a distinct specialized WGSL function. It re-exports the trait implementations from the parent module via `use super::renderer_specialization::*`.

## Rust Source

```rust
#[wgsl]
pub mod renderer_specialization_simple {
    use super::renderer_specialization::*;
    use wgsl_rs::std::*;

    /// Simple renderer: checkerboard + Lambert + geometric normals.
    pub fn shade_simple(uv: Vec2f, normal: Vec3f, light_dir: Vec3f, view_dir: Vec3f) -> Vec4f {
        shade_fragment::<Checker, Lambert, GeomNormal>(uv, normal, light_dir, view_dir)
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
fn shade_simple(uv: vec2f, normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    return _1shade_fragment_Checker_Lambert_GeomNormal(uv, normal, light_dir, view_dir);
}
fn _1shade_fragment_Checker_Lambert_GeomNormal(uv: vec2f, geom_normal: vec3f, light_dir: vec3f, view_dir: vec3f) -> vec4f {
    let normal: vec3f = GeomNormal__1get_normal(uv, geom_normal);
    let surface: vec4f = Checker__1surface_color(uv);
    return Lambert__1apply_lighting(surface, normal, light_dir, view_dir);
}
```

## Notes

- `use super::renderer_specialization::*` pulls in all the strategy structs, trait impls, and the generic `shade_fragment` from the sibling module, causing them to be re-emitted in this module's WGSL.
- The new `shade_simple` turbofish line (`shade_fragment::<Checker, Lambert, GeomNormal>`) produces a new specialized function `_1shade_fragment_Checker_Lambert_GeomNormal`.
- This demonstrates that each axis of variation is independently combinable.