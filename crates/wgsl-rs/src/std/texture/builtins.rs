//! Texture builtin functions for WGSL.
//!
//! This module provides CPU-side implementations of WGSL texture builtin
//! functions. These functions allow textures to be sampled, loaded, and queried
//! on the CPU, enabling shader code written with `wgsl-rs` to produce
//! equivalent results in both "Rust world" (CPU) and "WGSL world" (GPU).
//!
//! See <https://gpuweb.github.io/gpuweb/wgsl/#texture-builtin-functions>.
//!
//! ## Supported Functions
//!
//! ### Query Functions
//! - `texture_dimensions` - Get texture dimensions
//! - `texture_num_layers` - Get number of array layers
//! - `texture_num_levels` - Get number of mip levels
//! - `texture_num_samples` - Get number of samples (multisampled textures)
//!
//! ### Load/Store Functions
//! - `texture_load` - Load texel without sampling
//! - `texture_store` - Store value to storage texture (requires storage
//!   textures)
//!
//! ### Sampling Functions
//! - `texture_sample` - Sample with implicit LOD
//! - `texture_sample_bias` - Sample with LOD bias
//! - `texture_sample_level` - Sample at explicit LOD
//! - `texture_sample_grad` - Sample with explicit gradients
//! - `texture_sample_compare` - Depth comparison sample
//! - `texture_sample_compare_level` - Depth comparison at explicit LOD
//! - `texture_sample_base_clamp_to_edge` - Sample at base level with clamping
//!
//! ### Gather Functions
//! - `texture_gather` - Gather single component from 4 texels
//! - `texture_gather_compare` - Gather with depth comparison

// Note: We use `std::vec::Vec` explicitly throughout this module to avoid
// name collision with our `Vec<N, T>` vector type from vectors.rs.

use crate::std::{
    texture::{Sampler, SamplerComparison},
    vectors::Vec4f,
};

/// CPU-side texture data for 1D textures.
///
/// Stores mip levels as a vector of 1D image strips.
#[derive(Clone)]
pub struct TextureData1D<T> {
    /// Mip levels, each containing width pixels.
    pub mips: std::vec::Vec<std::vec::Vec<[T; 4]>>,
}

impl<T: Default + Copy> TextureData1D<T> {
    /// Create a new 1D texture with the given width.
    pub fn new(width: u32) -> Self {
        let pixels: std::vec::Vec<[T; 4]> = (0..width)
            .map(|_| std::array::from_fn(|_| T::default()))
            .collect();
        Self { mips: vec![pixels] }
    }
}

impl<T> TextureData1D<T> {
    /// Get the width at the given mip level.
    pub fn width(&self, level: u32) -> u32 {
        self.mips
            .get(level as usize)
            .map(|m| m.len() as u32)
            .unwrap_or(0)
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Get a pixel at the given coordinate and mip level.
    pub fn get_pixel(&self, x: u32, level: u32) -> Option<&[T; 4]> {
        self.mips.get(level as usize)?.get(x as usize)
    }

    /// Set a pixel at the given coordinate and mip level.
    pub fn set_pixel(&mut self, x: u32, level: u32, value: [T; 4]) {
        if let Some(mip) = self.mips.get_mut(level as usize)
            && let Some(pixel) = mip.get_mut(x as usize)
        {
            *pixel = value;
        }
    }
}

/// CPU-side texture data for 2D textures.
#[derive(Clone)]
pub struct TextureData2D<T> {
    /// Mip levels, each containing a 2D image
    pub mips: std::vec::Vec<std::vec::Vec<std::vec::Vec<[T; 4]>>>,
}

impl<T: Default + Clone> TextureData2D<T> {
    /// Create a new 2D texture with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let pixels: std::vec::Vec<std::vec::Vec<[T; 4]>> = (0..height)
            .map(|_| {
                (0..width)
                    .map(|_| std::array::from_fn(|_| T::default()))
                    .collect()
            })
            .collect();
        Self { mips: vec![pixels] }
    }
}

impl<T> TextureData2D<T> {
    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.mips
            .get(level as usize)
            .map(|m| {
                let height = m.len() as u32;
                let width = m.first().map(|r| r.len() as u32).unwrap_or(0);
                (width, height)
            })
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Get a pixel at the given coordinate and mip level.
    pub fn get_pixel(&self, x: u32, y: u32, level: u32) -> Option<&[T; 4]> {
        self.mips
            .get(level as usize)?
            .get(y as usize)?
            .get(x as usize)
    }

    /// Set a pixel at the given coordinate and mip level.
    pub fn set_pixel(&mut self, x: u32, y: u32, level: u32, value: [T; 4]) {
        if let Some(mip) = self.mips.get_mut(level as usize)
            && let Some(row) = mip.get_mut(y as usize)
            && let Some(pixel) = row.get_mut(x as usize)
        {
            *pixel = value;
        }
    }
}

impl<T: Default + Clone> TextureData2D<T> {
    /// Generate mip levels for the texture.
    pub fn generate_mips(&mut self)
    where
        T: Copy + Into<f32> + From<f32>,
    {
        // Start with base level dimensions
        let (mut w, mut h) = self.dimensions(0);

        while w > 1 || h > 1 {
            let new_w = Ord::max(w / 2, 1);
            let new_h = Ord::max(h / 2, 1);

            let prev_level = self.mips.len() - 1;
            let mut new_mip: std::vec::Vec<std::vec::Vec<[T; 4]>> = (0..new_h)
                .map(|_| {
                    (0..new_w)
                        .map(|_| std::array::from_fn(|_| T::default()))
                        .collect()
                })
                .collect();

            for y in 0..new_h {
                for x in 0..new_w {
                    // Sample 2x2 block from previous level
                    let mut sum = [0.0f32; 4];
                    let mut count = 0;

                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = Ord::min(x * 2 + dx, w - 1);
                            let sy = Ord::min(y * 2 + dy, h - 1);
                            if let Some(pixel) = self.mips[prev_level]
                                .get(sy as usize)
                                .and_then(|row| row.get(sx as usize))
                            {
                                for c in 0..4 {
                                    sum[c] += pixel[c].into();
                                }
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        let avg = [
                            T::from(sum[0] / count as f32),
                            T::from(sum[1] / count as f32),
                            T::from(sum[2] / count as f32),
                            T::from(sum[3] / count as f32),
                        ];
                        new_mip[y as usize][x as usize] = avg;
                    }
                }
            }

            self.mips.push(new_mip);
            w = new_w;
            h = new_h;
        }
    }
}

/// CPU-side texture data for 2D array textures.
#[derive(Clone)]
pub struct TextureData2DArray<T> {
    /// Array of 2D textures
    pub layers: std::vec::Vec<TextureData2D<T>>,
}

impl<T: Default + Copy> TextureData2DArray<T> {
    /// Create a new 2D array texture.
    pub fn new(width: u32, height: u32, layers: u32) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| TextureData2D::new(width, height))
                .collect(),
        }
    }
}

impl<T> TextureData2DArray<T> {
    /// Get the number of array layers.
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.layers
            .first()
            .map(|l| l.dimensions(level))
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.layers.first().map(|l| l.num_levels()).unwrap_or(0)
    }
}

/// CPU-side texture data for 3D textures.
#[derive(Clone)]
pub struct TextureData3D<T> {
    /// Mip levels, each containing a 3D volume
    pub mips: std::vec::Vec<std::vec::Vec<std::vec::Vec<std::vec::Vec<[T; 4]>>>>,
}

impl<T: Default + Copy> TextureData3D<T> {
    /// Create a new 3D texture with the given dimensions.
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        let volume: std::vec::Vec<std::vec::Vec<std::vec::Vec<[T; 4]>>> = (0..depth)
            .map(|_| {
                (0..height)
                    .map(|_| {
                        (0..width)
                            .map(|_| std::array::from_fn(|_| T::default()))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self { mips: vec![volume] }
    }
}

impl<T> TextureData3D<T> {
    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32, u32) {
        self.mips
            .get(level as usize)
            .map(|m| {
                let depth = m.len() as u32;
                let height = m.first().map(|s| s.len() as u32).unwrap_or(0);
                let width = m
                    .first()
                    .and_then(|s| s.first())
                    .map(|r| r.len() as u32)
                    .unwrap_or(0);
                (width, height, depth)
            })
            .unwrap_or((0, 0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Get a pixel at the given coordinate and mip level.
    pub fn get_pixel(&self, x: u32, y: u32, z: u32, level: u32) -> Option<&[T; 4]> {
        self.mips
            .get(level as usize)?
            .get(z as usize)?
            .get(y as usize)?
            .get(x as usize)
    }
}

/// CPU-side texture data for cube textures.
///
/// A cube texture has 6 faces: +X, -X, +Y, -Y, +Z, -Z
#[derive(Clone)]
pub struct TextureDataCube<T> {
    /// The 6 faces of the cube, each is a 2D texture
    pub faces: [TextureData2D<T>; 6],
}

impl<T: Default + Copy> TextureDataCube<T> {
    /// Create a new cube texture with the given face size.
    pub fn new(size: u32) -> Self {
        Self {
            faces: std::array::from_fn(|_| TextureData2D::new(size, size)),
        }
    }
}

impl<T> TextureDataCube<T> {
    /// Get the dimensions (face width/height) at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.faces[0].dimensions(level)
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.faces[0].num_levels()
    }
}

/// CPU-side texture data for cube array textures.
#[derive(Clone)]
pub struct TextureDataCubeArray<T> {
    /// Array of cube textures
    pub cubes: std::vec::Vec<TextureDataCube<T>>,
}

impl<T: Default + Copy> TextureDataCubeArray<T> {
    /// Create a new cube array texture.
    pub fn new(size: u32, layers: u32) -> Self {
        Self {
            cubes: (0..layers).map(|_| TextureDataCube::new(size)).collect(),
        }
    }
}

impl<T> TextureDataCubeArray<T> {
    /// Get the number of array layers.
    pub fn num_layers(&self) -> u32 {
        self.cubes.len() as u32
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.cubes
            .first()
            .map(|c| c.dimensions(level))
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.cubes.first().map(|c| c.num_levels()).unwrap_or(0)
    }
}

/// CPU-side texture data for multisampled 2D textures.
#[derive(Clone)]
pub struct TextureDataMultisampled2D<T> {
    /// Width of the texture
    pub width: u32,
    /// Height of the texture
    pub height: u32,
    /// Number of samples per pixel
    pub sample_count: u32,
    /// Pixel data: [y][x][sample]
    pub samples: std::vec::Vec<std::vec::Vec<std::vec::Vec<[T; 4]>>>,
}

impl<T: Default + Clone> TextureDataMultisampled2D<T> {
    /// Create a new multisampled 2D texture.
    pub fn new(width: u32, height: u32, sample_count: u32) -> Self {
        let samples: std::vec::Vec<std::vec::Vec<std::vec::Vec<[T; 4]>>> = (0..height)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        (0..sample_count)
                            .map(|_| std::array::from_fn(|_| T::default()))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            width,
            height,
            sample_count,
            samples,
        }
    }
}

impl<T> TextureDataMultisampled2D<T> {
    /// Get the dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> u32 {
        self.sample_count
    }

    /// Get a sample at the given coordinate.
    pub fn get_sample(&self, x: u32, y: u32, sample: u32) -> Option<&[T; 4]> {
        self.samples
            .get(y as usize)?
            .get(x as usize)?
            .get(sample as usize)
    }
}

/// CPU-side depth texture data for 2D depth textures.
#[derive(Clone)]
pub struct TextureDataDepth2D {
    /// Mip levels
    pub mips: std::vec::Vec<std::vec::Vec<std::vec::Vec<f32>>>,
}

impl TextureDataDepth2D {
    /// Create a new 2D depth texture.
    pub fn new(width: u32, height: u32) -> Self {
        let row = vec![0.0f32; width as usize];
        let pixels = vec![row; height as usize];
        Self { mips: vec![pixels] }
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.mips
            .get(level as usize)
            .map(|m| {
                let height = m.len() as u32;
                let width = m.first().map(|r| r.len() as u32).unwrap_or(0);
                (width, height)
            })
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Get a depth value at the given coordinate and mip level.
    pub fn get_depth(&self, x: u32, y: u32, level: u32) -> Option<f32> {
        self.mips
            .get(level as usize)?
            .get(y as usize)?
            .get(x as usize)
            .copied()
    }
}

/// CPU-side depth texture data for 2D depth array textures.
#[derive(Clone)]
pub struct TextureDataDepth2DArray {
    /// Array of depth textures
    pub layers: std::vec::Vec<TextureDataDepth2D>,
}

impl TextureDataDepth2DArray {
    /// Create a new 2D depth array texture.
    pub fn new(width: u32, height: u32, layers: u32) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| TextureDataDepth2D::new(width, height))
                .collect(),
        }
    }

    /// Get the number of array layers.
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.layers
            .first()
            .map(|l| l.dimensions(level))
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.layers.first().map(|l| l.num_levels()).unwrap_or(0)
    }
}

/// CPU-side depth texture data for cube depth textures.
#[derive(Clone)]
pub struct TextureDataDepthCube {
    /// The 6 faces of the cube
    pub faces: [TextureDataDepth2D; 6],
}

impl TextureDataDepthCube {
    /// Create a new cube depth texture.
    pub fn new(size: u32) -> Self {
        Self {
            faces: std::array::from_fn(|_| TextureDataDepth2D::new(size, size)),
        }
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.faces[0].dimensions(level)
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.faces[0].num_levels()
    }
}

/// CPU-side depth texture data for cube array depth textures.
#[derive(Clone)]
pub struct TextureDataDepthCubeArray {
    /// Array of cube depth textures
    pub cubes: std::vec::Vec<TextureDataDepthCube>,
}

impl TextureDataDepthCubeArray {
    /// Create a new cube array depth texture.
    pub fn new(size: u32, layers: u32) -> Self {
        Self {
            cubes: (0..layers)
                .map(|_| TextureDataDepthCube::new(size))
                .collect(),
        }
    }

    /// Get the number of array layers.
    pub fn num_layers(&self) -> u32 {
        self.cubes.len() as u32
    }

    /// Get the dimensions at the given mip level.
    pub fn dimensions(&self, level: u32) -> (u32, u32) {
        self.cubes
            .first()
            .map(|c| c.dimensions(level))
            .unwrap_or((0, 0))
    }

    /// Get the number of mip levels.
    pub fn num_levels(&self) -> u32 {
        self.cubes.first().map(|c| c.num_levels()).unwrap_or(0)
    }
}

/// CPU-side multisampled depth texture data.
#[derive(Clone)]
pub struct TextureDataDepthMultisampled2D {
    /// Width of the texture
    pub width: u32,
    /// Height of the texture
    pub height: u32,
    /// Number of samples per pixel
    pub sample_count: u32,
    /// Depth samples: [y][x][sample]
    pub samples: std::vec::Vec<std::vec::Vec<std::vec::Vec<f32>>>,
}

impl TextureDataDepthMultisampled2D {
    /// Create a new multisampled 2D depth texture.
    pub fn new(width: u32, height: u32, sample_count: u32) -> Self {
        let samples_per_pixel = vec![0.0f32; sample_count as usize];
        let row = vec![samples_per_pixel; width as usize];
        let pixels = vec![row; height as usize];
        Self {
            width,
            height,
            sample_count,
            samples: pixels,
        }
    }

    /// Get the dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the number of samples.
    pub fn num_samples(&self) -> u32 {
        self.sample_count
    }
}

/// Filter mode for texture sampling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterMode {
    /// Nearest-neighbor filtering (point sampling)
    #[default]
    Nearest,
    /// Linear filtering (bilinear/trilinear interpolation)
    Linear,
}

/// Address mode for texture coordinate wrapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddressMode {
    /// Clamp to edge - coordinates outside [0, 1] are clamped
    #[default]
    ClampToEdge,
    /// Repeat - coordinates wrap around
    Repeat,
    /// Mirror repeat - coordinates mirror at boundaries
    MirrorRepeat,
}

/// CPU-side sampler state.
#[derive(Clone, Debug)]
pub struct SamplerState {
    /// Magnification filter
    pub mag_filter: FilterMode,
    /// Minification filter
    pub min_filter: FilterMode,
    /// Mipmap filter
    pub mipmap_filter: FilterMode,
    /// Address mode for U coordinate
    pub address_mode_u: AddressMode,
    /// Address mode for V coordinate
    pub address_mode_v: AddressMode,
    /// Address mode for W coordinate
    pub address_mode_w: AddressMode,
    /// LOD min clamp
    pub lod_min_clamp: f32,
    /// LOD max clamp
    pub lod_max_clamp: f32,
}

impl Default for SamplerState {
    fn default() -> Self {
        Self {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
        }
    }
}

impl SamplerState {
    /// Create a new sampler state with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a sampler state with linear filtering.
    pub fn linear() -> Self {
        Self {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        }
    }

    /// Apply address mode to a coordinate.
    pub fn apply_address_mode(mode: AddressMode, coord: f32) -> f32 {
        match mode {
            AddressMode::ClampToEdge => coord.clamp(0.0, 1.0),
            AddressMode::Repeat => coord.rem_euclid(1.0),
            AddressMode::MirrorRepeat => {
                let t = coord.rem_euclid(2.0);
                if t > 1.0 { 2.0 - t } else { t }
            }
        }
    }
}

/// Comparison sampler state.
#[derive(Clone, Debug)]
pub struct SamplerComparisonState {
    /// Base sampler state
    pub sampler: SamplerState,
    /// Comparison function
    pub compare: CompareFunction,
}

/// Comparison function for depth testing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompareFunction {
    /// Never pass
    Never,
    /// Pass if less than
    Less,
    /// Pass if equal
    Equal,
    /// Pass if less than or equal
    #[default]
    LessEqual,
    /// Pass if greater than
    Greater,
    /// Pass if not equal
    NotEqual,
    /// Pass if greater than or equal
    GreaterEqual,
    /// Always pass
    Always,
}

impl CompareFunction {
    /// Evaluate the comparison function per WebGPU spec semantics.
    ///
    /// The `reference` (depth_ref, the "provided value") is compared against
    /// the `sample` (fetched texel, the "sampled value"). Per the WebGPU spec,
    /// `Less` passes when `reference < sample`, etc.
    ///
    /// See <https://www.w3.org/TR/webgpu/#enumdef-gpucomparefunction>.
    pub fn compare(self, sample: f32, reference: f32) -> f32 {
        let result = match self {
            CompareFunction::Never => false,
            CompareFunction::Less => reference < sample,
            CompareFunction::Equal => (sample - reference).abs() < f32::EPSILON,
            CompareFunction::LessEqual => reference <= sample,
            CompareFunction::Greater => reference > sample,
            CompareFunction::NotEqual => (sample - reference).abs() >= f32::EPSILON,
            CompareFunction::GreaterEqual => reference >= sample,
            CompareFunction::Always => true,
        };
        if result { 1.0 } else { 0.0 }
    }
}

impl Default for SamplerComparisonState {
    fn default() -> Self {
        Self {
            sampler: SamplerState::default(),
            compare: CompareFunction::LessEqual,
        }
    }
}

/// Trait for textures that support `textureDimensions`.
pub trait TextureDimensionsQuery {
    /// The output type for dimensions (u32, Vec2u, or Vec3u)
    type Output;

    /// Get the dimensions at the given mip level.
    fn query_dimensions(&self, level: u32) -> Self::Output;
}

/// Trait for textures that support `textureNumLayers`.
pub trait TextureNumLayersQuery {
    /// Get the number of array layers.
    fn query_num_layers(&self) -> u32;
}

/// Trait for textures that support `textureNumLevels`.
pub trait TextureNumLevelsQuery {
    /// Get the number of mip levels.
    fn query_num_levels(&self) -> u32;
}

/// Trait for textures that support `textureNumSamples`.
pub trait TextureNumSamplesQuery {
    /// Get the number of samples.
    fn query_num_samples(&self) -> u32;
}

/// Trait for textures that support `textureLoad`.
pub trait TextureLoad<Coords, Level> {
    /// The output type (Vec4<T> for sampled textures, f32 for depth)
    type Output;

    /// Load a texel at the given coordinates and mip level.
    fn load(&self, coords: Coords, level: Level) -> Self::Output;
}

/// Trait for multisampled textures that support `textureLoad` with sample
/// index.
pub trait TextureLoadMultisampled<Coords, SampleIndex> {
    /// The output type
    type Output;

    /// Load a sample at the given coordinates and sample index.
    fn load_multisampled(&self, coords: Coords, sample_index: SampleIndex) -> Self::Output;
}

/// Trait for array textures that support `textureLoad` with array index.
pub trait TextureLoadArray<Coords, ArrayIndex, Level> {
    /// The output type
    type Output;

    /// Load a texel at the given coordinates, array index, and mip level.
    fn load_array(&self, coords: Coords, array_index: ArrayIndex, level: Level) -> Self::Output;
}

/// Trait for textures that support `textureSample`.
pub trait TextureSample<Coords> {
    /// The output type
    type Output;

    /// Sample the texture at the given coordinates.
    fn sample(&self, sampler: &Sampler, coords: Coords) -> Self::Output;
}

/// Trait for textures that support `textureSample` with offset.
pub trait TextureSampleOffset<Coords, Offset> {
    /// The output type
    type Output;

    /// Sample the texture at the given coordinates with offset.
    fn sample_offset(&self, sampler: &Sampler, coords: Coords, offset: Offset) -> Self::Output;
}

/// Trait for array textures that support `textureSample` with array index.
pub trait TextureSampleArray<Coords, ArrayIndex> {
    /// The output type
    type Output;

    /// Sample the texture at the given coordinates and array index.
    fn sample_array(
        &self,
        sampler: &Sampler,
        coords: Coords,
        array_index: ArrayIndex,
    ) -> Self::Output;
}

/// Trait for textures that support `textureSampleLevel`.
pub trait TextureSampleLevel<Coords, Level> {
    /// The output type
    type Output;

    /// Sample the texture at the given coordinates and explicit LOD.
    fn sample_level(&self, sampler: &Sampler, coords: Coords, level: Level) -> Self::Output;
}

/// Trait for textures that support `textureSampleBias`.
pub trait TextureSampleBias<Coords, Bias> {
    /// The output type
    type Output;

    /// Sample the texture at the given coordinates with LOD bias.
    fn sample_bias(&self, sampler: &Sampler, coords: Coords, bias: Bias) -> Self::Output;
}

/// Trait for textures that support `textureSampleGrad`.
pub trait TextureSampleGrad<Coords, Ddx, Ddy> {
    /// The output type
    type Output;

    /// Sample the texture with explicit gradients.
    fn sample_grad(&self, sampler: &Sampler, coords: Coords, ddx: Ddx, ddy: Ddy) -> Self::Output;
}

/// Trait for textures that support `textureSampleCompare`.
pub trait TextureSampleCompare<Coords, DepthRef> {
    /// Sample the depth texture with comparison.
    fn sample_compare(
        &self,
        sampler: &SamplerComparison,
        coords: Coords,
        depth_ref: DepthRef,
    ) -> f32;
}

/// Trait for textures that support `textureSampleCompareLevel`.
pub trait TextureSampleCompareLevel<Coords, DepthRef, Level> {
    /// Sample the depth texture with comparison at explicit LOD.
    fn sample_compare_level(
        &self,
        sampler: &SamplerComparison,
        coords: Coords,
        depth_ref: DepthRef,
        level: Level,
    ) -> f32;
}

/// Trait for textures that support `textureSampleBaseClampToEdge`.
///
/// Samples a texture at mip level 0 with coordinates clamped to
/// `[0.5/dim, 1.0 - 0.5/dim]` to prevent edge wrapping artifacts.
pub trait TextureSampleBaseClampToEdge<Coords> {
    /// The output type
    type Output;

    /// Sample the texture at base level with coordinates clamped to the edge.
    fn sample_base_clamp_to_edge(&self, sampler: &Sampler, coords: Coords) -> Self::Output;
}

/// Trait for textures that support `textureGather`.
pub trait TextureGather<Coords> {
    /// The output type
    type Output;

    /// Gather one component from 4 texels.
    fn gather(&self, component: u32, sampler: &Sampler, coords: Coords) -> Self::Output;
}

/// Trait for depth textures that support `textureGather` (no component
/// parameter).
pub trait TextureGatherDepth<Coords> {
    /// Gather depth values from 4 texels.
    fn gather_depth(&self, sampler: &Sampler, coords: Coords) -> Vec4f;
}

/// Trait for textures that support `textureGatherCompare`.
pub trait TextureGatherCompare<Coords, DepthRef> {
    /// Gather depth values with comparison from 4 texels.
    fn gather_compare(
        &self,
        sampler: &SamplerComparison,
        coords: Coords,
        depth_ref: DepthRef,
    ) -> Vec4f;
}

/// Trait for types that can be used as texture coordinates.
pub trait IntoCoord {
    fn into_i32(self) -> i32;
    fn into_u32(self) -> u32;
}

impl IntoCoord for i32 {
    fn into_i32(self) -> i32 {
        self
    }
    fn into_u32(self) -> u32 {
        self.max(0) as u32
    }
}

impl IntoCoord for u32 {
    fn into_i32(self) -> i32 {
        self as i32
    }
    fn into_u32(self) -> u32 {
        self
    }
}

/// Trait for types that can be used as mip levels.
pub trait IntoLevel {
    fn into_level(self) -> u32;
}

impl IntoLevel for i32 {
    fn into_level(self) -> u32 {
        self.max(0) as u32
    }
}

impl IntoLevel for u32 {
    fn into_level(self) -> u32 {
        self
    }
}

/// Returns the dimensions of a texture.
///
/// For 1D textures, returns `u32`.
/// For 2D textures, returns `Vec2u`.
/// For 3D textures, returns `Vec3u`.
///
/// To specify a mipmap level use [`texture_dimensions_level`].
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let dims = textureDimensions(my_texture);
/// ```
pub fn texture_dimensions<T: TextureDimensionsQuery>(t: &T) -> T::Output {
    t.query_dimensions(0)
}

/// Returns the dimensions of a texture at a given mipmap level.
///
/// For 1D textures, returns `u32`.
/// For 2D textures, returns `Vec2u`.
/// For 3D textures, returns `Vec3u`.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let dims_at_level = textureDimensions(my_texture, 2);
/// ```
pub fn texture_dimensions_level<T: TextureDimensionsQuery>(
    t: &T,
    level: impl IntoLevel,
) -> T::Output {
    t.query_dimensions(level.into_level())
}

/// Returns the number of array layers in an array texture.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let layers = textureNumLayers(my_texture_array);
/// ```
pub fn texture_num_layers<T: TextureNumLayersQuery>(t: &T) -> u32 {
    t.query_num_layers()
}

/// Returns the number of mip levels in a texture.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let levels = textureNumLevels(my_texture);
/// ```
pub fn texture_num_levels<T: TextureNumLevelsQuery>(t: &T) -> u32 {
    t.query_num_levels()
}

/// Returns the number of samples in a multisampled texture.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let samples = textureNumSamples(my_multisampled_texture);
/// ```
pub fn texture_num_samples<T: TextureNumSamplesQuery>(t: &T) -> u32 {
    t.query_num_samples()
}

/// Reads a single texel from a texture without sampling or filtering.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let texel = textureLoad(my_texture, coords, level);
/// ```
pub fn texture_load<T, C, L>(t: &T, coords: C, level: L) -> T::Output
where
    T: TextureLoad<C, L>,
{
    t.load(coords, level)
}

/// Reads a single texel from a multisampled texture.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let texel = textureLoad(my_multisampled_texture, coords, sample_index);
/// ```
pub fn texture_load_multisampled<T, C, S>(t: &T, coords: C, sample_index: S) -> T::Output
where
    T: TextureLoadMultisampled<C, S>,
{
    t.load_multisampled(coords, sample_index)
}

/// Reads a single texel from an array texture.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let texel = textureLoad(my_array_texture, coords, array_index, level);
/// ```
pub fn texture_load_array<T, C, A, L>(t: &T, coords: C, array_index: A, level: L) -> T::Output
where
    T: TextureLoadArray<C, A, L>,
{
    t.load_array(coords, array_index, level)
}

/// Samples a texture.
///
/// Must only be used in a fragment shader stage (enforced by WGSL, not by
/// Rust). On CPU, this samples at mip level 0 since screen-space derivatives
/// are not available.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSample(my_texture, my_sampler, uv);
/// ```
pub fn texture_sample<T, C>(t: &T, s: &Sampler, coords: C) -> T::Output
where
    T: TextureSample<C>,
{
    t.sample(s, coords)
}

/// Samples a texture with an offset applied to the texture coordinates.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSample(my_texture, my_sampler, uv, vec2<i32>(1, 0));
/// ```
pub fn texture_sample_offset<T, C, O>(t: &T, s: &Sampler, coords: C, offset: O) -> T::Output
where
    T: TextureSampleOffset<C, O>,
{
    t.sample_offset(s, coords, offset)
}

/// Samples a texture array.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSample(my_texture_array, my_sampler, uv, array_index);
/// ```
pub fn texture_sample_array<T, C, A>(t: &T, s: &Sampler, coords: C, array_index: A) -> T::Output
where
    T: TextureSampleArray<C, A>,
{
    t.sample_array(s, coords, array_index)
}

/// Samples a texture at an explicit mip level.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSampleLevel(my_texture, my_sampler, uv, 2.0);
/// ```
pub fn texture_sample_level<T, C, L>(t: &T, s: &Sampler, coords: C, level: L) -> T::Output
where
    T: TextureSampleLevel<C, L>,
{
    t.sample_level(s, coords, level)
}

/// Samples a texture with a bias added to the mip level.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSampleBias(my_texture, my_sampler, uv, 1.0);
/// ```
pub fn texture_sample_bias<T, C, B>(t: &T, s: &Sampler, coords: C, bias: B) -> T::Output
where
    T: TextureSampleBias<C, B>,
{
    t.sample_bias(s, coords, bias)
}

/// Samples a texture with explicit gradients for LOD calculation.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSampleGrad(my_texture, my_sampler, uv, ddx, ddy);
/// ```
pub fn texture_sample_grad<T, C, D>(t: &T, s: &Sampler, coords: C, ddx: D, ddy: D) -> T::Output
where
    T: TextureSampleGrad<C, D, D>,
{
    t.sample_grad(s, coords, ddx, ddy)
}

/// Samples a depth texture and compares the result against a reference value.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let shadow = textureSampleCompare(shadow_map, shadow_sampler, uv, depth_ref);
/// ```
pub fn texture_sample_compare<T, C, R>(t: &T, s: &SamplerComparison, coords: C, depth_ref: R) -> f32
where
    T: TextureSampleCompare<C, R>,
{
    t.sample_compare(s, coords, depth_ref)
}

/// Samples a depth texture at an explicit mip level and compares the result.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let shadow = textureSampleCompareLevel(shadow_map, shadow_sampler, uv, depth_ref, 0);
/// ```
pub fn texture_sample_compare_level<T, C, R, L>(
    t: &T,
    s: &SamplerComparison,
    coords: C,
    depth_ref: R,
    level: L,
) -> f32
where
    T: TextureSampleCompareLevel<C, R, L>,
{
    t.sample_compare_level(s, coords, depth_ref, level)
}

/// Samples a texture at its base mip level with coordinates clamped to the
/// edge.
///
/// The coordinates are clamped to `[0.5/dim, 1.0 - 0.5/dim]` before sampling,
/// which prevents wrapping artifacts regardless of the sampler's address mode.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let color = textureSampleBaseClampToEdge(my_texture, my_sampler, uv);
/// ```
pub fn texture_sample_base_clamp_to_edge<T, C>(t: &T, s: &Sampler, coords: C) -> T::Output
where
    T: TextureSampleBaseClampToEdge<C>,
{
    t.sample_base_clamp_to_edge(s, coords)
}

/// Gathers one component from 4 texels that would be used in bilinear
/// filtering.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let gathered = textureGather(0, my_texture, my_sampler, uv);
/// ```
pub fn texture_gather<T, C>(component: impl IntoCoord, t: &T, s: &Sampler, coords: C) -> T::Output
where
    T: TextureGather<C>,
{
    t.gather(component.into_u32(), s, coords)
}

/// Gathers depth values from 4 texels (for depth textures).
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let gathered = textureGather(depth_texture, my_sampler, uv);
/// ```
pub fn texture_gather_depth<T, C>(t: &T, s: &Sampler, coords: C) -> Vec4f
where
    T: TextureGatherDepth<C>,
{
    t.gather_depth(s, coords)
}

/// Gathers depth values with comparison from 4 texels.
///
/// # WGSL Equivalent
///
/// ```wgsl
/// let gathered = textureGatherCompare(depth_texture, shadow_sampler, uv, depth_ref);
/// ```
pub fn texture_gather_compare<T, C, R>(
    t: &T,
    s: &SamplerComparison,
    coords: C,
    depth_ref: R,
) -> Vec4f
where
    T: TextureGatherCompare<C, R>,
{
    t.gather_compare(s, coords, depth_ref)
}

/// Helper function for bilinear interpolation.
fn bilinear_interpolate<T: Copy + Default + Into<f32> + From<f32>>(
    tl: [T; 4],
    tr: [T; 4],
    bl: [T; 4],
    br: [T; 4],
    fx: f32,
    fy: f32,
) -> [T; 4] {
    std::array::from_fn(|i| {
        let top = tl[i].into() * (1.0 - fx) + tr[i].into() * fx;
        let bottom = bl[i].into() * (1.0 - fx) + br[i].into() * fx;
        T::from(top * (1.0 - fy) + bottom * fy)
    })
}

/// Helper function to convert normalized coordinates to texel coordinates.
fn normalized_to_texel(coord: f32, size: u32) -> (u32, f32) {
    let scaled = coord * size as f32;
    let texel = (scaled.floor() as i32).clamp(0, (size as i32) - 1) as u32;
    let frac = scaled.fract();
    (texel, frac)
}

/// Sample a 2D texture at the given normalized coordinates.
pub fn sample_texture_2d<T: Copy + Default + Into<f32> + From<f32>>(
    data: &TextureData2D<T>,
    sampler_state: &SamplerState,
    u: f32,
    v: f32,
    level: u32,
) -> [T; 4] {
    let u = SamplerState::apply_address_mode(sampler_state.address_mode_u, u);
    let v = SamplerState::apply_address_mode(sampler_state.address_mode_v, v);

    let (width, height) = data.dimensions(level);
    if width == 0 || height == 0 {
        return std::array::from_fn(|_| T::default());
    }

    match sampler_state.min_filter {
        FilterMode::Nearest => {
            let x = Ord::min((u * width as f32).floor() as u32, width - 1);
            let y = Ord::min((v * height as f32).floor() as u32, height - 1);
            data.get_pixel(x, y, level)
                .copied()
                .unwrap_or_else(|| std::array::from_fn(|_| T::default()))
        }
        FilterMode::Linear => {
            let (x0, fx) = normalized_to_texel(u - 0.5 / width as f32, width);
            let (y0, fy) = normalized_to_texel(v - 0.5 / height as f32, height);
            let x1 = Ord::min(x0 + 1, width - 1);
            let y1 = Ord::min(y0 + 1, height - 1);

            let default_pixel = || std::array::from_fn(|_| T::default());
            let tl = data
                .get_pixel(x0, y0, level)
                .copied()
                .unwrap_or_else(default_pixel);
            let tr = data
                .get_pixel(x1, y0, level)
                .copied()
                .unwrap_or_else(default_pixel);
            let bl = data
                .get_pixel(x0, y1, level)
                .copied()
                .unwrap_or_else(default_pixel);
            let br = data
                .get_pixel(x1, y1, level)
                .copied()
                .unwrap_or_else(default_pixel);

            bilinear_interpolate(tl, tr, bl, br, fx, fy)
        }
    }
}

/// Sample a depth 2D texture at the given normalized coordinates.
pub fn sample_depth_texture_2d(
    data: &TextureDataDepth2D,
    sampler_state: &SamplerState,
    u: f32,
    v: f32,
    level: u32,
) -> f32 {
    let u = SamplerState::apply_address_mode(sampler_state.address_mode_u, u);
    let v = SamplerState::apply_address_mode(sampler_state.address_mode_v, v);

    let (width, height) = data.dimensions(level);
    if width == 0 || height == 0 {
        return 0.0;
    }

    match sampler_state.min_filter {
        FilterMode::Nearest => {
            let x = Ord::min((u * width as f32).floor() as u32, width - 1);
            let y = Ord::min((v * height as f32).floor() as u32, height - 1);
            data.get_depth(x, y, level).unwrap_or(0.0)
        }
        FilterMode::Linear => {
            let (x0, fx) = normalized_to_texel(u - 0.5 / width as f32, width);
            let (y0, fy) = normalized_to_texel(v - 0.5 / height as f32, height);
            let x1 = Ord::min(x0 + 1, width - 1);
            let y1 = Ord::min(y0 + 1, height - 1);

            let tl = data.get_depth(x0, y0, level).unwrap_or(0.0);
            let tr = data.get_depth(x1, y0, level).unwrap_or(0.0);
            let bl = data.get_depth(x0, y1, level).unwrap_or(0.0);
            let br = data.get_depth(x1, y1, level).unwrap_or(0.0);

            let top = tl * (1.0 - fx) + tr * fx;
            let bottom = bl * (1.0 - fx) + br * fx;
            top * (1.0 - fy) + bottom * fy
        }
    }
}

/// Gather the 4 depth texels that would be used in bilinear filtering at mip
/// level 0, returning them in the WGSL-specified gather component order:
///   x = (u_min, v_max), y = (u_max, v_max),
///   z = (u_max, v_min), w = (u_min, v_min).
pub fn gather_4_depth_texels(
    data: &TextureDataDepth2D,
    sampler_state: &SamplerState,
    u: f32,
    v: f32,
) -> [f32; 4] {
    let u = SamplerState::apply_address_mode(sampler_state.address_mode_u, u);
    let v = SamplerState::apply_address_mode(sampler_state.address_mode_v, v);

    let (width, height) = data.dimensions(0);
    if width == 0 || height == 0 {
        return [0.0; 4];
    }

    // Find the 4 texels that bilinear filtering would use.
    // The texel grid center is offset by half a texel from the normalized
    // coordinate origin, so we subtract 0.5/dim before computing the floor.
    let (x0, _fx) = normalized_to_texel(u - 0.5 / width as f32, width);
    let (y0, _fy) = normalized_to_texel(v - 0.5 / height as f32, height);
    let x1 = Ord::min(x0 + 1, width - 1);
    let y1 = Ord::min(y0 + 1, height - 1);

    let tl = data.get_depth(x0, y0, 0).unwrap_or(0.0); // (u_min, v_min)
    let tr = data.get_depth(x1, y0, 0).unwrap_or(0.0); // (u_max, v_min)
    let bl = data.get_depth(x0, y1, 0).unwrap_or(0.0); // (u_min, v_max)
    let br = data.get_depth(x1, y1, 0).unwrap_or(0.0); // (u_max, v_max)

    // WGSL gather order: x=(u_min,v_max), y=(u_max,v_max), z=(u_max,v_min),
    // w=(u_min,v_min)
    [bl, br, tr, tl]
}
