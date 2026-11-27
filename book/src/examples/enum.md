# Enum

Demonstrates limited enum support. `#[repr(u32)]` enums are translated to a `u32` alias and a set of `u32` constants, one per variant. `match` on enum variants becomes a WGSL `switch`.

## Rust Source

```rust
#[wgsl]
pub mod enum_example {
    //! Limited support for enums.
    use wgsl_rs::std::*;

    /// Analytical lighting types.
    #[repr(u32)]
    pub enum LightType {
        Directional = 1337,
        Spot = 420,
        Point = 666,
    }

    #[repr(u32)]
    #[derive(Wgsl)]
    pub enum Holidays {
        // Syntax error!
        // Halloween = -23,
        AprilFoolsDay,
        WaitangiDay,
    }

    pub struct EnumExamplePack {
        #[storage(group(0), binding(0), read_write)]
        pub input: Storage<[Holidays; 256], ReadWrite>,
    }

    #[compute]
    #[workgroup_size(16)]
    pub fn compute_holidays(
        #[builtin(global_invocation_id)] global_id: Vec3u,
        pack: &EnumExamplePack,
    ) {
        let index = global_id.x();

        let holiday = &mut get_mut!(pack.input)[index as usize];

        #[wgsl_allow(non_literal_match_statement_patterns)]
        match *holiday {
            Holidays::AprilFoolsDay => {
                *holiday = Holidays::WaitangiDay;
            }
            Holidays::WaitangiDay => {
                *holiday = Holidays::AprilFoolsDay;
            }
        }
    }
}
```

## Generated WGSL

```wgsl
alias LightType = u32;
const LightType_Directional: u32 = 1337u;
const LightType_Spot: u32 = 420u;
const LightType_Point: u32 = 666u;
alias Holidays = u32;
const Holidays_AprilFoolsDay: u32 = 0u;
const Holidays_WaitangiDay: u32 = 1u;
@group(0) @binding(0) var<storage, read_write> enumexamplepack__input: array<Holidays, 256>;

@compute @workgroup_size(16) fn compute_holidays(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    let holiday = &enumexamplepack__input[u32(index)];
    switch *holiday {
        case Holidays_AprilFoolsDay: {
            *holiday = Holidays_WaitangiDay;
        }
        case Holidays_WaitangiDay: {
            *holiday = Holidays_AprilFoolsDay;
        }
        default: { }
    }
}
```

## Notes

- Enums must be `#[repr(u32)]`; variants become `u32` constants (auto-numbered from 0 if not explicitly assigned).
- `#[wgsl_allow(non_literal_match_statement_patterns)]` is required when `match` arms use enum variant paths rather than literal patterns, because WGSL switch cases must be literal — the transpiler substitutes the variant constants.
- `#[derive(Wgsl)]` enables enum use in storage buffers.
- **Defaults**: a variant marked `#[default]` is honored on both sides. The transpiler synthesizes an inherent method rendering `fn Kind_default() -> Kind { return Kind_Alpha; }`, so `Kind::default()` works inside `#[wgsl]` modules: on the GPU it renders the mangled call, and on the CPU the expression resolves through the `Default` trait impl that `#[derive(Default)]` generates — `#[default]` requires that derive. Use the `<Type>::default()` spelling — trait-qualified calls like `Default::default()` cannot resolve at GPU call sites. A module global with the same mangled name (a free `fn Kind_default()`, an impl method named `default` on the same type, a variant named `default`) is a compile error naming the collision — rename one of them.
- Explicit `impl <Enum> { .. }` blocks work like struct impls: methods render mangled against the enum's `u32` alias, and `Self` resolves to the enum name.