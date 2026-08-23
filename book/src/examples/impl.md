# Impl

Demonstrates struct `impl` blocks with associated constants and methods. Methods are called with explicit `Type::method(receiver, args)` syntax and constants with `Type::CONSTANT`; both translate to `Type_member` in WGSL.

## Rust Source

```rust
#[wgsl]
pub mod impl_example {
    //! Demonstrates struct impl blocks with explicit receiver syntax.
    //!
    //! Methods and constants are defined in impl blocks.
    //! - Methods are called using `Type::method(receiver, args)` syntax
    //! - Constants are accessed using `Type::CONSTANT` syntax
    //!
    //! Both translate to `Type_member` in WGSL output.
    use wgsl_rs::std::*;

    pub struct Light {
        pub position: Vec3f,
        pub intensity: f32,
    }

    impl Light {
        // Associated constants
        pub const DEFAULT_INTENSITY: f32 = 1.0;
        pub const DEFAULT_RANGE: f32 = 10.0;

        // Create a new light at the given position with the given intensity.
        pub fn new(position: Vec3f, intensity: f32) -> Light {
            Light {
                position,
                intensity,
            }
        }

        // Calculate light attenuation based on distance.
        // Uses inverse-square falloff.
        pub fn attenuate(light: Light, distance: f32) -> f32 {
            light.intensity / (distance * distance)
        }

        // Get the light's position.
        pub fn get_position(light: Light) -> Vec3f {
            light.position
        }
    }

    #[fragment]
    pub fn frag_main() -> Vec4f {
        // Create a light using the explicit receiver syntax
        let light = Light::new(vec3f(0.0, 5.0, 0.0), Light::DEFAULT_INTENSITY);

        // Call a method using explicit path syntax: Type::method(receiver, args)
        let attenuation = Light::attenuate(light, Light::DEFAULT_RANGE / 5.0);

        // Return a color based on attenuation
        vec4f(attenuation, attenuation, attenuation, 1.0)
    }
}
```

## Generated WGSL

```wgsl
struct Light {
    position: vec3f,
    intensity: f32
}
const Light__1DEFAULT_INTENSITY: f32 = 1.0;
const Light__1DEFAULT_RANGE: f32 = 10.0;

fn Light_new(position: vec3f, intensity: f32) -> Light {
    return Light(position, intensity);
}

fn Light_attenuate(light: Light, distance: f32) -> f32 {
    return light.intensity / (distance * distance);
}

fn Light__1get_position(light: Light) -> vec3f {
    return light.position;
}

@fragment fn frag_main() -> @location(0) vec4f {
    let light = Light_new(vec3f(0.0, 5.0, 0.0), Light__1DEFAULT_INTENSITY);
    let attenuation = Light_attenuate(light, Light__1DEFAULT_RANGE / 5.0);
    return vec4f(attenuation, attenuation, attenuation, 1.0);
}
```

## Notes

- Associated constants and methods are mangled into `Type_member` (with `_1` separating the type name from the member name when needed for uniqueness).
- `Type::method(receiver, args)` calls become free functions `Type_method(receiver, args)` in WGSL.