//! Textures and samplers.
//!
//! This module provides WGSL-compatible texture and sampler types that work
//! in both the "Rust world" (CPU) and "WGSL world" (GPU).
//!
//! On the CPU side, textures can optionally hold actual pixel data via the
//! `image` crate, enabling texture operations to produce real results during
//! CPU execution.
//!
//! On the GPU side, these types transpile to their WGSL equivalents.

use crate::std::{
    ModuleVar, ModuleVarReadGuard, Vec2f, Vec2i, Vec2u, Vec3u, Vec4f, Vec4i, Vec4u, vec2u, vec3u,
    vec4f, vec4i, vec4u,
};

mod builtins;
pub use builtins::*;

/// A shader sampler for texture sampling operations.
///
/// In WGSL, this transpiles to the `sampler` type. Samplers control how
/// textures are sampled, including filtering modes and address wrapping
/// behavior.
///
/// On the CPU side, this stores sampler state that can be used to control
/// texture sampling behavior.
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
    /// CPU-side sampler state
    data: ModuleVar<SamplerState>,
}

impl Sampler {
    /// Creates a new sampler with the given group and binding indices.
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    /// Returns the group index of this sampler.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Returns the binding index within its group of this sampler.
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the sampler state for CPU-side sampling.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get(&self) -> ModuleVarReadGuard<'_, SamplerState> {
        self.data.read()
    }

    /// Set the sampler state for CPU-side sampling.
    ///
    /// Not available in WGSL.
    pub fn set(&self, state: SamplerState) {
        self.data.set(state);
    }

    /// Initialize the sampler with a default value.
    ///
    /// Not available in WGSL.
    pub fn init(&self) {
        self.set(SamplerState::default());
    }
}

/// A shader comparison sampler for depth texture sampling operations.
///
/// In WGSL, this transpiles to the `sampler_comparison` type. Comparison
/// samplers are used for depth texture comparisons, typically in shadow
/// mapping where a depth value is compared against a reference value.
///
/// On the CPU side, this stores sampler state including the comparison
/// function.
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
    /// CPU-side comparison sampler state
    data: ModuleVar<SamplerComparisonState>,
}

impl SamplerComparison {
    /// Creates a new comparison sampler with the given group and binding
    /// indices.
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    /// Returns the group index of this sampler.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Returns the binding index within its group of this sampler.
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the comparison sampler state for CPU-side
    /// sampling.
    ///
    /// Not available in WGSL.
    ///
    /// ## Panics
    /// Dereferencing the returned guard will panic if it has not previously
    /// been set.
    pub fn get(&self) -> ModuleVarReadGuard<'_, SamplerComparisonState> {
        self.data.read()
    }

    /// Set the comparison sampler state for CPU-side sampling.
    ///
    /// Not available in WGSL.
    pub fn set(&self, state: SamplerComparisonState) {
        self.data.set(state);
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
    data: ModuleVar<TextureData1D<T>>,
}

impl<T> Texture1D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureData1D<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureData1D<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> Texture1D<T> {
    /// Initialize the texture with the given width.
    ///
    /// Not available in WGSL.
    pub fn init(&self, width: u32) {
        self.set(TextureData1D::new(width));
    }
}

// Implement query traits for Texture1D
impl<T> TextureDimensionsQuery for Texture1D<T> {
    type Output = u32;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        self.get().width(level)
    }
}

impl<T> TextureNumLevelsQuery for Texture1D<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
    }
}

// Implement load for Texture1D - only for f32 since that's the most common case
// WGSL requires texture_load to return vec4<T> where T matches the texture
// type, but our Vec4<T> requires ScalarCompOfVec<4>. We implement for concrete
// types.
impl<L: IntoLevel> TextureLoad<u32, L> for Texture1D<f32> {
    type Output = Vec4f;

    fn load(&self, coords: u32, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords, level)
            .map(|p| vec4f(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4f(0.0, 0.0, 0.0, 0.0))
    }
}

impl<L: IntoLevel> TextureLoad<i32, L> for Texture1D<f32> {
    type Output = Vec4f;

    fn load(&self, coords: i32, level: L) -> Self::Output {
        let x = if coords < 0 { 0u32 } else { coords as u32 };
        let level = level.into_level();

        self.get()
            .get_pixel(x, level)
            .map(|p| vec4f(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4f(0.0, 0.0, 0.0, 0.0))
    }
}

// TextureLoad implementations for Texture1D<i32>
impl<L: IntoLevel> TextureLoad<u32, L> for Texture1D<i32> {
    type Output = Vec4i;

    fn load(&self, coords: u32, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords, level)
            .map(|p| vec4i(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4i(0, 0, 0, 0))
    }
}

impl<L: IntoLevel> TextureLoad<i32, L> for Texture1D<i32> {
    type Output = Vec4i;

    fn load(&self, coords: i32, level: L) -> Self::Output {
        let x = if coords < 0 { 0u32 } else { coords as u32 };
        let level = level.into_level();

        self.get()
            .get_pixel(x, level)
            .map(|p| vec4i(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4i(0, 0, 0, 0))
    }
}

// TextureLoad implementations for Texture1D<u32>
impl<L: IntoLevel> TextureLoad<u32, L> for Texture1D<u32> {
    type Output = Vec4u;

    fn load(&self, coords: u32, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords, level)
            .map(|p| vec4u(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4u(0, 0, 0, 0))
    }
}

impl<L: IntoLevel> TextureLoad<i32, L> for Texture1D<u32> {
    type Output = Vec4u;

    fn load(&self, coords: i32, level: L) -> Self::Output {
        let x = if coords < 0 { 0u32 } else { coords as u32 };
        let level = level.into_level();

        self.get()
            .get_pixel(x, level)
            .map(|p| vec4u(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4u(0, 0, 0, 0))
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
    data: ModuleVar<TextureData2D<T>>,
}

impl<T> Texture2D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureData2D<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureData2D<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> Texture2D<T> {
    /// Initialize the texture with the given dimensions.
    ///
    /// Not available in WGSL.
    pub fn init(&self, width: u32, height: u32) {
        self.set(TextureData2D::new(width, height));
    }

    /// Set a pixel at the given coordinates.
    ///
    /// Not available in WGSL.
    pub fn set_pixel(&self, x: u32, y: u32, value: [T; 4]) {
        self.data.write().set_pixel(x, y, 0, value);
    }
}

// Implement query traits for Texture2D
impl<T> TextureDimensionsQuery for Texture2D<T> {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl<T> TextureNumLevelsQuery for Texture2D<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
    }
}

// Implement load for Texture2D - concrete implementations for f32
impl<L: IntoLevel> TextureLoad<Vec2u, L> for Texture2D<f32> {
    type Output = Vec4f;

    fn load(&self, coords: Vec2u, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords.x(), coords.y(), level)
            .map(|p| vec4f(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4f(0.0, 0.0, 0.0, 0.0))
    }
}

impl<L: IntoLevel> TextureLoad<Vec2i, L> for Texture2D<f32> {
    type Output = Vec4f;

    fn load(&self, coords: Vec2i, level: L) -> Self::Output {
        let x = if coords.x() < 0 {
            0u32
        } else {
            coords.x() as u32
        };
        let y = if coords.y() < 0 {
            0u32
        } else {
            coords.y() as u32
        };
        self.load(vec2u(x, y), level)
    }
}

// TextureLoad implementations for Texture2D<i32>
impl<L: IntoLevel> TextureLoad<Vec2u, L> for Texture2D<i32> {
    type Output = Vec4i;

    fn load(&self, coords: Vec2u, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords.x(), coords.y(), level)
            .map(|p| vec4i(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4i(0, 0, 0, 0))
    }
}

impl<L: IntoLevel> TextureLoad<Vec2i, L> for Texture2D<i32> {
    type Output = Vec4i;

    fn load(&self, coords: Vec2i, level: L) -> Self::Output {
        let x = if coords.x() < 0 {
            0u32
        } else {
            coords.x() as u32
        };
        let y = if coords.y() < 0 {
            0u32
        } else {
            coords.y() as u32
        };
        self.load(vec2u(x, y), level)
    }
}

// TextureLoad implementations for Texture2D<u32>
impl<L: IntoLevel> TextureLoad<Vec2u, L> for Texture2D<u32> {
    type Output = Vec4u;

    fn load(&self, coords: Vec2u, level: L) -> Self::Output {
        let level = level.into_level();

        self.get()
            .get_pixel(coords.x(), coords.y(), level)
            .map(|p| vec4u(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4u(0, 0, 0, 0))
    }
}

impl<L: IntoLevel> TextureLoad<Vec2i, L> for Texture2D<u32> {
    type Output = Vec4u;

    fn load(&self, coords: Vec2i, level: L) -> Self::Output {
        let x = if coords.x() < 0 {
            0u32
        } else {
            coords.x() as u32
        };
        let y = if coords.y() < 0 {
            0u32
        } else {
            coords.y() as u32
        };
        let level = level.into_level();

        self.get()
            .get_pixel(x, y, level)
            .map(|p| vec4u(p[0], p[1], p[2], p[3]))
            .unwrap_or(vec4u(0, 0, 0, 0))
    }
}

// TextureSample implementation for Texture2D<f32>
impl TextureSample<Vec2f> for Texture2D<f32> {
    type Output = Vec4f;

    fn sample(&self, sampler: &Sampler, coords: Vec2f) -> Self::Output {
        let sampler_state = sampler.get();
        let data = self.get();
        let result = sample_texture_2d(&data, &sampler_state, coords.x(), coords.y(), 0);
        vec4f(result[0], result[1], result[2], result[3])
    }
}

// TextureSampleLevel implementation for Texture2D<f32>
impl TextureSampleLevel<Vec2f, f32> for Texture2D<f32> {
    type Output = Vec4f;

    fn sample_level(&self, sampler: &Sampler, coords: Vec2f, level: f32) -> Self::Output {
        let sampler_state = sampler.get();
        let level = level.floor() as u32; // Use floor for mip level selection
        let data = self.get();
        let result = sample_texture_2d(&data, &sampler_state, coords.x(), coords.y(), level);
        vec4f(result[0], result[1], result[2], result[3])
    }
}

impl TextureSampleLevel<Vec2f, i32> for Texture2D<f32> {
    type Output = Vec4f;

    fn sample_level(&self, sampler: &Sampler, coords: Vec2f, level: i32) -> Self::Output {
        let sampler_state = sampler.get();
        let level = if level < 0 { 0u32 } else { level as u32 };
        let data = self.get();
        let result = sample_texture_2d(&data, &sampler_state, coords.x(), coords.y(), level);
        vec4f(result[0], result[1], result[2], result[3])
    }
}

impl TextureSampleLevel<Vec2f, u32> for Texture2D<f32> {
    type Output = Vec4f;

    fn sample_level(&self, sampler: &Sampler, coords: Vec2f, level: u32) -> Self::Output {
        let sampler_state = sampler.get();
        let data = self.get();
        let result = sample_texture_2d(&data, &sampler_state, coords.x(), coords.y(), level);
        vec4f(result[0], result[1], result[2], result[3])
    }
}

// TextureSampleBaseClampToEdge implementation for Texture2D<f32>
impl TextureSampleBaseClampToEdge<Vec2f> for Texture2D<f32> {
    type Output = Vec4f;

    fn sample_base_clamp_to_edge(&self, sampler: &Sampler, coords: Vec2f) -> Self::Output {
        let sampler_state = sampler.get();
        let data = self.get();
        let (w, h) = data.dimensions(0);
        // Clamp coords to [0.5/dim, 1.0 - 0.5/dim] per the WGSL spec.
        // This prevents edge wrapping regardless of sampler address mode.
        let half_texel_u = if w > 0 { 0.5 / w as f32 } else { 0.0 };
        let half_texel_v = if h > 0 { 0.5 / h as f32 } else { 0.0 };
        let u = coords.x().clamp(half_texel_u, 1.0 - half_texel_u);
        let v = coords.y().clamp(half_texel_v, 1.0 - half_texel_v);
        let result = sample_texture_2d(&data, &sampler_state, u, v, 0);
        vec4f(result[0], result[1], result[2], result[3])
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
    data: ModuleVar<TextureData2DArray<T>>,
}

impl<T> Texture2DArray<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureData2DArray<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureData2DArray<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> Texture2DArray<T> {
    /// Initialize the texture with the given dimensions.
    pub fn init(&self, width: u32, height: u32, layers: u32) {
        self.set(TextureData2DArray::new(width, height, layers))
    }
}

// Implement query traits for Texture2DArray
impl<T> TextureDimensionsQuery for Texture2DArray<T> {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl<T> TextureNumLayersQuery for Texture2DArray<T> {
    fn query_num_layers(&self) -> u32 {
        self.get().num_layers()
    }
}

impl<T: Default + Clone> TextureNumLevelsQuery for Texture2DArray<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureData3D<T>>,
}

impl<T> Texture3D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

impl<T: Default + Copy> Texture3D<T> {
    /// Initialize the texture with the given dimensions.
    pub fn init(&self, width: u32, height: u32, depth: u32) {
        self.set(TextureData3D::new(width, height, depth))
    }
}

impl<T> Texture3D<T> {
    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureData3D<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureData3D<T>) {
        self.data.set(data);
    }
}

// Implement query traits for Texture3D
impl<T: Default + Clone> TextureDimensionsQuery for Texture3D<T> {
    type Output = Vec3u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h, d) = self.get().dimensions(level);
        vec3u(w, h, d)
    }
}

impl<T: Default + Clone> TextureNumLevelsQuery for Texture3D<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataCube<T>>,
}

impl<T> TextureCube<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataCube<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataCube<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> TextureCube<T> {
    /// Initialize the texture with the given face size.
    pub fn init(&self, size: u32) {
        self.set(TextureDataCube::new(size))
    }
}

// Implement query traits for TextureCube
impl<T: Default + Copy> TextureDimensionsQuery for TextureCube<T> {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl<T: Default + Clone> TextureNumLevelsQuery for TextureCube<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataCubeArray<T>>,
}

impl<T> TextureCubeArray<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataCubeArray<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataCubeArray<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> TextureCubeArray<T> {
    /// Initialize the texture with the given face size and layer count.
    pub fn init(&self, size: u32, layers: u32) {
        self.set(TextureDataCubeArray::new(size, layers))
    }
}

// Implement query traits for TextureCubeArray
impl<T> TextureDimensionsQuery for TextureCubeArray<T> {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl<T> TextureNumLayersQuery for TextureCubeArray<T> {
    fn query_num_layers(&self) -> u32 {
        self.get().num_layers()
    }
}

impl<T> TextureNumLevelsQuery for TextureCubeArray<T> {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataMultisampled2D<T>>,
}

impl<T> TextureMultisampled2D<T> {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataMultisampled2D<T>> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataMultisampled2D<T>) {
        self.data.set(data);
    }
}

impl<T: Default + Copy> TextureMultisampled2D<T> {
    /// Initialize the texture with the given dimensions and sample count.
    pub fn init(&self, width: u32, height: u32, sample_count: u32) {
        self.set(TextureDataMultisampled2D::new(width, height, sample_count));
    }
}

// Implement query traits for TextureMultisampled2D
impl<T> TextureDimensionsQuery for TextureMultisampled2D<T> {
    type Output = Vec2u;

    fn query_dimensions(&self, _level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions();
        vec2u(w, h)
    }
}

impl<T: Default + Clone> TextureNumSamplesQuery for TextureMultisampled2D<T> {
    fn query_num_samples(&self) -> u32 {
        self.get().num_samples()
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
    data: ModuleVar<TextureDataDepth2D>,
}

impl TextureDepth2D {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataDepth2D> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataDepth2D) {
        self.data.set(data);
    }

    /// Initialize the texture with the given dimensions.
    pub fn init(&self, width: u32, height: u32) {
        self.set(TextureDataDepth2D::new(width, height))
    }
}

// Implement query traits for TextureDepth2D
impl TextureDimensionsQuery for TextureDepth2D {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl TextureNumLevelsQuery for TextureDepth2D {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
    }
}

// Implement load for TextureDepth2D
impl<L: IntoLevel> TextureLoad<Vec2u, L> for TextureDepth2D {
    type Output = f32;

    fn load(&self, coords: Vec2u, level: L) -> Self::Output {
        let x = coords.x();
        let y = coords.y();
        let level = level.into_level();

        self.get().get_depth(x, y, level).unwrap_or(0.0)
    }
}

impl<L: IntoLevel> TextureLoad<Vec2i, L> for TextureDepth2D {
    type Output = f32;

    fn load(&self, coords: Vec2i, level: L) -> Self::Output {
        let x = if coords.x() < 0 {
            0u32
        } else {
            coords.x() as u32
        };
        let y = if coords.y() < 0 {
            0u32
        } else {
            coords.y() as u32
        };

        self.load(vec2u(x, y), level)
    }
}

// TextureSampleCompare implementation for TextureDepth2D
impl TextureSampleCompare<Vec2f, f32> for TextureDepth2D {
    fn sample_compare(&self, sampler: &SamplerComparison, coords: Vec2f, depth_ref: f32) -> f32 {
        let sampler_state = sampler.get();
        let sampled_depth = sample_depth_texture_2d(
            &self.get(),
            &sampler_state.sampler,
            coords.x(),
            coords.y(),
            0,
        );
        sampler_state.compare.compare(sampled_depth, depth_ref)
    }
}

// TextureSampleCompareLevel implementation for TextureDepth2D
impl TextureSampleCompareLevel<Vec2f, f32, f32> for TextureDepth2D {
    fn sample_compare_level(
        &self,
        sampler: &SamplerComparison,
        coords: Vec2f,
        depth_ref: f32,
        level: f32,
    ) -> f32 {
        let sampler_state = sampler.get();
        let level = level.floor() as u32;
        let sampled_depth = sample_depth_texture_2d(
            &self.get(),
            &sampler_state.sampler,
            coords.x(),
            coords.y(),
            level,
        );
        sampler_state.compare.compare(sampled_depth, depth_ref)
    }
}

impl TextureSampleCompareLevel<Vec2f, f32, u32> for TextureDepth2D {
    fn sample_compare_level(
        &self,
        sampler: &SamplerComparison,
        coords: Vec2f,
        depth_ref: f32,
        level: u32,
    ) -> f32 {
        let sampler_state = sampler.get();
        let sampled_depth = sample_depth_texture_2d(
            &self.get(),
            &sampler_state.sampler,
            coords.x(),
            coords.y(),
            level,
        );
        sampler_state.compare.compare(sampled_depth, depth_ref)
    }
}

impl TextureSampleCompareLevel<Vec2f, f32, i32> for TextureDepth2D {
    fn sample_compare_level(
        &self,
        sampler: &SamplerComparison,
        coords: Vec2f,
        depth_ref: f32,
        level: i32,
    ) -> f32 {
        let sampler_state = sampler.get();
        let level = if level < 0 { 0u32 } else { level as u32 };
        let sampled_depth = sample_depth_texture_2d(
            &self.get(),
            &sampler_state.sampler,
            coords.x(),
            coords.y(),
            level,
        );
        sampler_state.compare.compare(sampled_depth, depth_ref)
    }
}

// TextureGatherCompare implementation for TextureDepth2D
impl TextureGatherCompare<Vec2f, f32> for TextureDepth2D {
    fn gather_compare(&self, sampler: &SamplerComparison, coords: Vec2f, depth_ref: f32) -> Vec4f {
        let sampler_state = sampler.get();
        let data = self.get();
        let texels = gather_4_depth_texels(&data, &sampler_state.sampler, coords.x(), coords.y());
        // Perform depth comparison on each of the 4 gathered texels.
        vec4f(
            sampler_state.compare.compare(texels[0], depth_ref),
            sampler_state.compare.compare(texels[1], depth_ref),
            sampler_state.compare.compare(texels[2], depth_ref),
            sampler_state.compare.compare(texels[3], depth_ref),
        )
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
    data: ModuleVar<TextureDataDepth2DArray>,
}

impl TextureDepth2DArray {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataDepth2DArray> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataDepth2DArray) {
        self.data.set(data);
    }

    /// Initialize the texture with the given dimensions.
    pub fn init(&self, width: u32, height: u32, layers: u32) {
        self.set(TextureDataDepth2DArray::new(width, height, layers))
    }
}

// Implement query traits for TextureDepth2DArray
impl TextureDimensionsQuery for TextureDepth2DArray {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl TextureNumLayersQuery for TextureDepth2DArray {
    fn query_num_layers(&self) -> u32 {
        self.get().num_layers()
    }
}

impl TextureNumLevelsQuery for TextureDepth2DArray {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataDepthCube>,
}

impl TextureDepthCube {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataDepthCube> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataDepthCube) {
        self.data.set(data);
    }

    /// Initialize the texture with the given face size.
    pub fn init(&self, size: u32) {
        self.set(TextureDataDepthCube::new(size))
    }
}

// Implement query traits for TextureDepthCube
impl TextureDimensionsQuery for TextureDepthCube {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl TextureNumLevelsQuery for TextureDepthCube {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataDepthCubeArray>,
}

impl TextureDepthCubeArray {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataDepthCubeArray> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataDepthCubeArray) {
        self.data.set(data);
    }

    /// Initialize the texture with the given face size and layer count.
    pub fn init(&self, size: u32, layers: u32) {
        self.set(TextureDataDepthCubeArray::new(size, layers));
    }
}

// Implement query traits for TextureDepthCubeArray
impl TextureDimensionsQuery for TextureDepthCubeArray {
    type Output = Vec2u;

    fn query_dimensions(&self, level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions(level);
        vec2u(w, h)
    }
}

impl TextureNumLayersQuery for TextureDepthCubeArray {
    fn query_num_layers(&self) -> u32 {
        self.get().num_layers()
    }
}

impl TextureNumLevelsQuery for TextureDepthCubeArray {
    fn query_num_levels(&self) -> u32 {
        self.get().num_levels()
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
    data: ModuleVar<TextureDataDepthMultisampled2D>,
}

impl TextureDepthMultisampled2D {
    pub const fn new(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            data: ModuleVar::new(),
        }
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns a reference to the inner texture data.
    ///
    /// Not available in WGSL.
    pub fn get(&self) -> ModuleVarReadGuard<'_, TextureDataDepthMultisampled2D> {
        self.data.read()
    }

    /// Set the texture data.
    ///
    /// Not available in WGSL.
    pub fn set(&self, data: TextureDataDepthMultisampled2D) {
        self.data.set(data);
    }

    /// Initialize the texture with the given dimensions and sample count.
    pub fn init(&self, width: u32, height: u32, sample_count: u32) {
        self.set(TextureDataDepthMultisampled2D::new(
            width,
            height,
            sample_count,
        ))
    }
}

// Implement query traits for TextureDepthMultisampled2D
impl TextureDimensionsQuery for TextureDepthMultisampled2D {
    type Output = Vec2u;

    fn query_dimensions(&self, _level: u32) -> Self::Output {
        let (w, h) = self.get().dimensions();
        vec2u(w, h)
    }
}

impl TextureNumSamplesQuery for TextureDepthMultisampled2D {
    fn query_num_samples(&self) -> u32 {
        self.get().num_samples()
    }
}

#[cfg(test)]
mod tests {
    use crate::std::{vec2f, vec2i};

    use super::*;

    #[test]
    fn test_texture_2d_dimensions() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(64, 32);

        let dims = texture_dimensions(&tex);
        assert_eq!(dims.x(), 64);
        assert_eq!(dims.y(), 32);
    }

    #[test]
    fn test_texture_2d_num_levels() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(64, 32);

        let levels = texture_num_levels(&tex);
        assert_eq!(levels, 1);
    }

    #[test]
    fn test_texture_2d_load_f32() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(4, 4);

        // Set a pixel
        tex.set_pixel(1, 2, [0.5, 0.25, 0.125, 1.0]);

        // Load the pixel using Vec2u coords
        let pixel = texture_load(&tex, vec2u(1, 2), 0u32);
        assert!((pixel.x() - 0.5).abs() < 0.001);
        assert!((pixel.y() - 0.25).abs() < 0.001);
        assert!((pixel.z() - 0.125).abs() < 0.001);
        assert!((pixel.w() - 1.0).abs() < 0.001);

        // Load using Vec2i coords
        let pixel = texture_load(&tex, vec2i(1, 2), 0i32);
        assert!((pixel.x() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_texture_2d_load_i32() {
        let tex: Texture2D<i32> = Texture2D::new(0, 0);
        tex.init(4, 4);

        tex.set_pixel(1, 2, [10, 20, 30, 40]);

        let pixel = texture_load(&tex, vec2u(1, 2), 0u32);
        assert_eq!(pixel.x(), 10);
        assert_eq!(pixel.y(), 20);
        assert_eq!(pixel.z(), 30);
        assert_eq!(pixel.w(), 40);
    }

    #[test]
    fn test_texture_2d_load_u32() {
        let tex: Texture2D<u32> = Texture2D::new(0, 0);
        tex.init(4, 4);

        tex.set_pixel(1, 2, [100, 200, 150, 255]);

        let pixel = texture_load(&tex, vec2u(1, 2), 0u32);
        assert_eq!(pixel.x(), 100);
        assert_eq!(pixel.y(), 200);
        assert_eq!(pixel.z(), 150);
        assert_eq!(pixel.w(), 255);
    }

    #[test]
    fn test_texture_1d_load_f32() {
        let tex: Texture1D<f32> = Texture1D::new(0, 0);
        tex.init(8);

        tex.data.write().set_pixel(3, 0, [0.1, 0.2, 0.3, 0.4]);

        let pixel = texture_load(&tex, 3u32, 0u32);
        assert!((pixel.x() - 0.1).abs() < 0.001);
        assert!((pixel.y() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_texture_2d_sample() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(2, 2);

        // Set all 4 pixels to different values
        tex.set_pixel(0, 0, [1.0, 0.0, 0.0, 1.0]); // Red
        tex.set_pixel(1, 0, [0.0, 1.0, 0.0, 1.0]); // Green
        tex.set_pixel(0, 1, [0.0, 0.0, 1.0, 1.0]); // Blue
        tex.set_pixel(1, 1, [1.0, 1.0, 0.0, 1.0]); // Yellow

        let sampler: Sampler = Sampler::new(0, 0);
        sampler.init(); // Nearest filtering by default

        // Sample at (0.25, 0.25) - should be near top-left (red)
        let color = texture_sample(&tex, &sampler, vec2f(0.25, 0.25));
        assert!((color.x() - 1.0).abs() < 0.001); // Red
        assert!((color.y() - 0.0).abs() < 0.001);
        assert!((color.z() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_texture_2d_sample_level() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(2, 2);

        tex.set_pixel(0, 0, [0.5, 0.5, 0.5, 1.0]);

        let sampler: Sampler = Sampler::new(0, 0);
        sampler.init();

        // Sample at level 0
        let color = texture_sample_level(&tex, &sampler, vec2f(0.25, 0.25), 0.0f32);
        assert!((color.x() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_texture_depth_2d_load() {
        let tex: TextureDepth2D = TextureDepth2D::new(0, 0);
        tex.init(4, 4);

        {
            let mut guard = tex.data.write();
            guard.mips[0][2][1] = 0.75; // y=2, x=1
        }

        let depth = texture_load(&tex, vec2u(1, 2), 0u32);
        assert!((depth - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_texture_depth_2d_sample_compare() {
        let tex: TextureDepth2D = TextureDepth2D::new(0, 0);
        tex.init(4, 4);

        {
            // Set all depth values to 0.5
            let mut guard = tex.data.write();
            for row in guard.mips[0].iter_mut() {
                for depth in row.iter_mut() {
                    *depth = 0.5;
                }
            }
        }


        let sampler: SamplerComparison = SamplerComparison::new(0, 0);
        sampler.set(SamplerComparisonState {
            compare: CompareFunction::Less,
            ..Default::default()
        });

        // Per WebGPU spec, Less passes when depth_ref < sampled_depth.
        // Sampled depth is 0.5, depth_ref is 0.3: 0.3 < 0.5 => true => 1.0
        let result = texture_sample_compare(&tex, &sampler, vec2f(0.5, 0.5), 0.3);
        assert!((result - 1.0).abs() < 0.001);

        // Sampled depth is 0.5, depth_ref is 0.6: 0.6 < 0.5 => false => 0.0
        let result = texture_sample_compare(&tex, &sampler, vec2f(0.5, 0.5), 0.6);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_texture_2d_array_queries() {
        let tex: Texture2DArray<f32> = Texture2DArray::new(0, 0);
        tex.init(16, 8, 4);

        let dims = texture_dimensions(&tex);
        assert_eq!(dims.x(), 16);
        assert_eq!(dims.y(), 8);

        let layers = texture_num_layers(&tex);
        assert_eq!(layers, 4);

        let levels = texture_num_levels(&tex);
        assert_eq!(levels, 1);
    }

    #[test]
    fn test_texture_3d_queries() {
        let tex: Texture3D<f32> = Texture3D::new(0, 0);
        tex.init(8, 4, 2);

        let dims = texture_dimensions(&tex);
        assert_eq!(dims.x(), 8);
        assert_eq!(dims.y(), 4);
        assert_eq!(dims.z(), 2);
    }

    #[test]
    fn test_texture_cube_queries() {
        let tex: TextureCube<f32> = TextureCube::new(0, 0);
        tex.init(32);

        let dims = texture_dimensions(&tex);
        assert_eq!(dims.x(), 32);
        assert_eq!(dims.y(), 32);
    }

    #[test]
    fn test_texture_multisampled_queries() {
        let tex: TextureMultisampled2D<f32> = TextureMultisampled2D::new(0, 0);
        tex.init(64, 64, 4);

        let dims = texture_dimensions(&tex);
        assert_eq!(dims.x(), 64);
        assert_eq!(dims.y(), 64);

        let samples = texture_num_samples(&tex);
        assert_eq!(samples, 4);
    }

    #[test]
    fn test_compare_function() {
        // Per WebGPU spec, compare(sample, reference) passes when reference <op>
        // sample. Less: reference < sample
        assert!((CompareFunction::Less.compare(0.5, 0.3) - 1.0).abs() < 0.001); // 0.3 < 0.5
        assert!((CompareFunction::Less.compare(0.3, 0.5) - 0.0).abs() < 0.001); // 0.5 not < 0.3
        // Greater: reference > sample
        assert!((CompareFunction::Greater.compare(0.3, 0.5) - 1.0).abs() < 0.001); // 0.5 > 0.3
        assert!((CompareFunction::Greater.compare(0.5, 0.3) - 0.0).abs() < 0.001); // 0.3 not > 0.5
        // Equal, LessEqual, GreaterEqual are symmetric or pass at equality
        assert!((CompareFunction::Equal.compare(0.5, 0.5) - 1.0).abs() < 0.001);
        assert!((CompareFunction::LessEqual.compare(0.5, 0.5) - 1.0).abs() < 0.001);
        assert!((CompareFunction::GreaterEqual.compare(0.5, 0.5) - 1.0).abs() < 0.001);
        assert!((CompareFunction::Always.compare(0.0, 1.0) - 1.0).abs() < 0.001);
        assert!((CompareFunction::Never.compare(0.5, 0.5) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_address_modes() {
        // Clamp to edge
        assert!(
            (SamplerState::apply_address_mode(AddressMode::ClampToEdge, -0.5) - 0.0).abs() < 0.001
        );
        assert!(
            (SamplerState::apply_address_mode(AddressMode::ClampToEdge, 1.5) - 1.0).abs() < 0.001
        );
        assert!(
            (SamplerState::apply_address_mode(AddressMode::ClampToEdge, 0.5) - 0.5).abs() < 0.001
        );

        // Repeat
        assert!((SamplerState::apply_address_mode(AddressMode::Repeat, 1.5) - 0.5).abs() < 0.001);
        assert!((SamplerState::apply_address_mode(AddressMode::Repeat, 2.25) - 0.25).abs() < 0.001);

        // Mirror repeat
        let mirror = SamplerState::apply_address_mode(AddressMode::MirrorRepeat, 1.5);
        assert!((mirror - 0.5).abs() < 0.001); // 1.5 -> 2.0 - 1.5 = 0.5
    }

    #[test]
    fn test_texture_sample_base_clamp_to_edge() {
        let tex: Texture2D<f32> = Texture2D::new(0, 0);
        tex.init(4, 4);

        // Set corner pixels to distinct colors
        tex.set_pixel(0, 0, [1.0, 0.0, 0.0, 1.0]); // top-left: red
        tex.set_pixel(3, 0, [0.0, 1.0, 0.0, 1.0]); // top-right: green
        tex.set_pixel(0, 3, [0.0, 0.0, 1.0, 1.0]); // bottom-left: blue
        tex.set_pixel(3, 3, [1.0, 1.0, 0.0, 1.0]); // bottom-right: yellow

        let sampler: Sampler = Sampler::new(0, 0);
        sampler.init(); // Nearest filtering

        // Sampling at (-0.5, -0.5) — far out of bounds — should clamp to near
        // top-left rather than wrapping. With a 4-wide texture,
        // half_texel = 0.5/4 = 0.125, so coords clamp to 0.125 which maps to
        // texel 0.
        let color = texture_sample_base_clamp_to_edge(&tex, &sampler, vec2f(-0.5, -0.5));
        assert!((color.x() - 1.0).abs() < 0.001); // red channel
        assert!((color.y() - 0.0).abs() < 0.001);

        // Sampling at (1.5, 1.5) — far out of bounds — should clamp to near
        // bottom-right. Coords clamp to 1.0 - 0.125 = 0.875 which maps to
        // texel 3.
        let color = texture_sample_base_clamp_to_edge(&tex, &sampler, vec2f(1.5, 1.5));
        assert!((color.x() - 1.0).abs() < 0.001); // yellow: r=1
        assert!((color.y() - 1.0).abs() < 0.001); // yellow: g=1
        assert!((color.z() - 0.0).abs() < 0.001); // yellow: b=0

        // Sampling at center (0.5, 0.5) should behave identically to
        // texture_sample since coords are already within the clamped range.
        let clamped = texture_sample_base_clamp_to_edge(&tex, &sampler, vec2f(0.5, 0.5));
        let normal = texture_sample(&tex, &sampler, vec2f(0.5, 0.5));
        assert!((clamped.x() - normal.x()).abs() < 0.001);
        assert!((clamped.y() - normal.y()).abs() < 0.001);
        assert!((clamped.z() - normal.z()).abs() < 0.001);
        assert!((clamped.w() - normal.w()).abs() < 0.001);
    }

    #[test]
    fn test_texture_gather_compare() {
        let tex: TextureDepth2D = TextureDepth2D::new(0, 0);
        tex.init(4, 4);

        {
            let mut guard = tex.data.write();
            // Set a 2x2 block with varying depths:
            //   (0,0)=0.2  (1,0)=0.4
            //   (0,1)=0.6  (1,1)=0.8
            guard.mips[0][0][0] = 0.2;
            guard.mips[0][0][1] = 0.4;
            guard.mips[0][1][0] = 0.6;
            guard.mips[0][1][1] = 0.8;
        }

        let sampler: SamplerComparison = SamplerComparison::new(0, 0);
        sampler.set(SamplerComparisonState {
            compare: CompareFunction::Less,
            ..Default::default()
        });

        // Sample near (0.25, 0.25) on a 4x4 texture.
        // The 4 gathered texels should be the 2x2 block at (0,0),(1,0),(0,1),(1,1).
        // Per WebGPU spec, Less comparison passes when depth_ref < sampled_depth:
        //   (0,0)=0.2: 0.5 < 0.2 => false => 0.0   (u_min, v_min) => w component
        //   (1,0)=0.4: 0.5 < 0.4 => false => 0.0   (u_max, v_min) => z component
        //   (0,1)=0.6: 0.5 < 0.6 => true  => 1.0   (u_min, v_max) => x component
        //   (1,1)=0.8: 0.5 < 0.8 => true  => 1.0   (u_max, v_max) => y component
        let result = texture_gather_compare(&tex, &sampler, vec2f(0.25, 0.25), 0.5);
        assert!((result.x() - 1.0).abs() < 0.001); // (u_min, v_max): 0.5 < 0.6
        assert!((result.y() - 1.0).abs() < 0.001); // (u_max, v_max): 0.5 < 0.8
        assert!((result.z() - 0.0).abs() < 0.001); // (u_max, v_min): 0.5 not < 0.4
        assert!((result.w() - 0.0).abs() < 0.001); // (u_min, v_min): 0.5 not < 0.2

        // With depth_ref = 0.9, no texels should pass (0.9 is not < any of
        // 0.2,0.4,0.6,0.8)
        let result = texture_gather_compare(&tex, &sampler, vec2f(0.25, 0.25), 0.9);
        assert!((result.x() - 0.0).abs() < 0.001);
        assert!((result.y() - 0.0).abs() < 0.001);
        assert!((result.z() - 0.0).abs() < 0.001);
        assert!((result.w() - 0.0).abs() < 0.001);

        // With depth_ref = 0.1, all texels should pass (0.1 < all of 0.2,0.4,0.6,0.8)
        let result = texture_gather_compare(&tex, &sampler, vec2f(0.25, 0.25), 0.1);
        assert!((result.x() - 1.0).abs() < 0.001);
        assert!((result.y() - 1.0).abs() < 0.001);
        assert!((result.z() - 1.0).abs() < 0.001);
        assert!((result.w() - 1.0).abs() < 0.001);
    }
}
