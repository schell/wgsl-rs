//! Textures and samplers.
use std::marker::PhantomData;

/// A shader sampler for texture sampling operations.
///
/// In WGSL, this transpiles to the `sampler` type. Samplers control how
/// textures are sampled, including filtering modes and address wrapping
/// behavior.
///
/// On the CPU side, this is a marker type that stores the group and binding
/// indices for resource binding purposes. Actual texture sampling operations
/// are handled by the GPU.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);
///
/// // In WGSL, you would use the sampler with texture sampling functions:
/// // let color = textureSample(my_texture, TEX_SAMPLER, uv);
/// ```
///
/// # WGSL Output
///
/// ```wgsl
/// @group(0) @binding(1) var TEX_SAMPLER: sampler;
/// ```
pub struct Sampler {
    group: u32,
    binding: u32,
}

impl Sampler {
    /// Creates a new sampler with the given group and binding indices.
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    /// Returns the group index of this sampler.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Returns the binding index within its group of this sampler.
    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A shader comparison sampler for depth texture sampling operations.
///
/// In WGSL, this transpiles to the `sampler_comparison` type. Comparison
/// samplers are used for depth texture comparisons, typically in shadow
/// mapping where a depth value is compared against a reference value.
///
/// On the CPU side, this is a marker type that stores the group and binding
/// indices for resource binding purposes. Actual comparison operations
/// are handled by the GPU.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// sampler!(group(0), binding(2), SHADOW_SAMPLER: SamplerComparison);
///
/// // In WGSL, you would use the sampler with comparison sampling functions:
/// // let shadow = textureSampleCompare(shadow_map, SHADOW_SAMPLER, uv, depth);
/// ```
///
/// # WGSL Output
///
/// ```wgsl
/// @group(0) @binding(2) var SHADOW_SAMPLER: sampler_comparison;
/// ```
pub struct SamplerComparison {
    group: u32,
    binding: u32,
}

impl SamplerComparison {
    /// Creates a new comparison sampler with the given group and binding
    /// indices.
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    /// Returns the group index of this sampler.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Returns the binding index within its group of this sampler.
    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 1D sampled texture.
///
/// In WGSL, this transpiles to `texture_1d<T>` where T is f32, i32, or u32.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), LUT_TEX: Texture1D<f32>);
/// ```
pub struct Texture1D<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> Texture1D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 2D sampled texture.
///
/// In WGSL, this transpiles to `texture_2d<T>` where T is f32, i32, or u32.
/// This is the most common texture type, used for diffuse maps, normal maps,
/// etc.
///
/// # Example
///
/// ```rust
/// use wgsl_rs::wgsl;
///
/// #[wgsl]
/// mod tex {
///     use wgsl_rs::std::*;
///
///     texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
///     sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);
///
///     // In shader code, you would sample like:
///     // let color = textureSample(DIFFUSE_TEX, TEX_SAMPLER, uv);
/// }
/// ```
pub struct Texture2D<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> Texture2D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 2D sampled texture array.
///
/// In WGSL, this transpiles to `texture_2d_array<T>` where T is f32, i32, or
/// u32. Useful for texture atlases or animated textures.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), TEXTURE_ARRAY: Texture2DArray<f32>);
/// ```
pub struct Texture2DArray<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> Texture2DArray<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 3D sampled texture (volume texture).
///
/// In WGSL, this transpiles to `texture_3d<T>` where T is f32, i32, or u32.
/// Useful for volumetric effects, 3D lookup tables, etc.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), VOLUME_TEX: Texture3D<f32>);
/// ```
pub struct Texture3D<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> Texture3D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A cube sampled texture.
///
/// In WGSL, this transpiles to `texture_cube<T>` where T is f32, i32, or u32.
/// Used for environment mapping, skyboxes, etc.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), SKYBOX: TextureCube<f32>);
/// ```
pub struct TextureCube<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> TextureCube<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A cube sampled texture array.
///
/// In WGSL, this transpiles to `texture_cube_array<T>` where T is f32, i32, or
/// u32.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), CUBE_ARRAY: TextureCubeArray<f32>);
/// ```
pub struct TextureCubeArray<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> TextureCubeArray<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A multisampled 2D texture.
///
/// In WGSL, this transpiles to `texture_multisampled_2d<T>` where T is f32,
/// i32, or u32. Used for MSAA (multisample anti-aliasing) render targets.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), MSAA_TEX: TextureMultisampled2D<f32>);
/// ```
pub struct TextureMultisampled2D<T> {
    group: u32,
    binding: u32,
    _marker: PhantomData<T>,
}

impl<T> TextureMultisampled2D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            _marker: PhantomData,
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 2D depth texture.
///
/// In WGSL, this transpiles to `texture_depth_2d`.
/// Used for shadow mapping and depth buffer access.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), SHADOW_MAP: TextureDepth2D);
/// sampler!(group(0), binding(1), SHADOW_SAMPLER: SamplerComparison);
///
/// // In shader code, you would sample like:
/// // let shadow = textureSampleCompare(SHADOW_MAP, SHADOW_SAMPLER, uv, depth);
/// ```
pub struct TextureDepth2D {
    group: u32,
    binding: u32,
}

impl TextureDepth2D {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A 2D depth texture array.
///
/// In WGSL, this transpiles to `texture_depth_2d_array`.
/// Used for cascaded shadow maps or multiple shadow maps.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), CASCADE_SHADOW_MAP: TextureDepth2DArray);
/// ```
pub struct TextureDepth2DArray {
    group: u32,
    binding: u32,
}

impl TextureDepth2DArray {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A cube depth texture.
///
/// In WGSL, this transpiles to `texture_depth_cube`.
/// Used for omnidirectional shadow mapping (point light shadows).
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), POINT_SHADOW: TextureDepthCube);
/// ```
pub struct TextureDepthCube {
    group: u32,
    binding: u32,
}

impl TextureDepthCube {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A cube depth texture array.
///
/// In WGSL, this transpiles to `texture_depth_cube_array`.
/// Used for multiple omnidirectional shadow maps.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), POINT_SHADOWS: TextureDepthCubeArray);
/// ```
pub struct TextureDepthCubeArray {
    group: u32,
    binding: u32,
}

impl TextureDepthCubeArray {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

/// A multisampled 2D depth texture.
///
/// In WGSL, this transpiles to `texture_depth_multisampled_2d`.
/// Used for MSAA depth buffers.
///
/// # Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), MSAA_DEPTH: TextureDepthMultisampled2D);
/// ```
pub struct TextureDepthMultisampled2D {
    group: u32,
    binding: u32,
}

impl TextureDepthMultisampled2D {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self { group, binding }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}
