//! WGSL builtin function name translations.
//!
//! Maps Rust snake_case names to WGSL camelCase / templated names for
//! builtin functions and type aliases that require translation during
//! rendering. Names that match between Rust and WGSL (e.g. `sin`, `cos`,
//! `abs`) are NOT included here.
//!
//! This table is duplicated from `wgsl-rs-macros/src/builtins.rs` so that
//! `wgsl-rs-ir` can stay completely standalone (no dependency on the
//! proc-macro crate). The two should be kept in sync; tests should pin
//! both copies.

const TABLE: &[(&str, &str)] = &[
    // Boolean vector aliases
    ("vec2b", "vec2<bool>"),
    ("vec3b", "vec3<bool>"),
    ("vec4b", "vec4<bool>"),
    // Arrays
    ("array_length", "arrayLength"),
    // Bitcast
    ("bitcast_f32", "bitcast<f32>"),
    ("bitcast_i32", "bitcast<i32>"),
    ("bitcast_u32", "bitcast<u32>"),
    ("bitcast_vec2f", "bitcast<vec2<f32>>"),
    ("bitcast_vec2i", "bitcast<vec2<i32>>"),
    ("bitcast_vec2u", "bitcast<vec2<u32>>"),
    ("bitcast_vec3f", "bitcast<vec3<f32>>"),
    ("bitcast_vec3i", "bitcast<vec3<i32>>"),
    ("bitcast_vec3u", "bitcast<vec3<u32>>"),
    ("bitcast_vec4f", "bitcast<vec4<f32>>"),
    ("bitcast_vec4i", "bitcast<vec4<i32>>"),
    ("bitcast_vec4u", "bitcast<vec4<u32>>"),
    // Atomic operations
    ("atomic_add", "atomicAdd"),
    ("atomic_add_i32", "atomicAdd"),
    ("atomic_and", "atomicAnd"),
    ("atomic_and_i32", "atomicAnd"),
    ("atomic_compare_exchange_weak", "atomicCompareExchangeWeak"),
    (
        "atomic_compare_exchange_weak_i32",
        "atomicCompareExchangeWeak",
    ),
    ("atomic_exchange", "atomicExchange"),
    ("atomic_exchange_i32", "atomicExchange"),
    ("atomic_load", "atomicLoad"),
    ("atomic_load_i32", "atomicLoad"),
    ("atomic_max", "atomicMax"),
    ("atomic_max_i32", "atomicMax"),
    ("atomic_min", "atomicMin"),
    ("atomic_min_i32", "atomicMin"),
    ("atomic_or", "atomicOr"),
    ("atomic_or_i32", "atomicOr"),
    ("atomic_store", "atomicStore"),
    ("atomic_store_i32", "atomicStore"),
    ("atomic_sub", "atomicSub"),
    ("atomic_sub_i32", "atomicSub"),
    ("atomic_xor", "atomicXor"),
    ("atomic_xor_i32", "atomicXor"),
    // Synchronization
    ("storage_barrier", "storageBarrier"),
    ("texture_barrier", "textureBarrier"),
    ("workgroup_barrier", "workgroupBarrier"),
    ("workgroup_uniform_load", "workgroupUniformLoad"),
    // Bit manipulation
    ("count_leading_zeros", "countLeadingZeros"),
    ("count_one_bits", "countOneBits"),
    ("count_trailing_zeros", "countTrailingZeros"),
    ("extract_bits", "extractBits"),
    ("first_leading_bit", "firstLeadingBit"),
    ("first_trailing_bit", "firstTrailingBit"),
    ("insert_bits", "insertBits"),
    ("reverse_bits", "reverseBits"),
    // Derivative builtins
    ("dpdx_coarse", "dpdxCoarse"),
    ("dpdx_fine", "dpdxFine"),
    ("dpdy_coarse", "dpdyCoarse"),
    ("dpdy_fine", "dpdyFine"),
    ("fwidth_coarse", "fwidthCoarse"),
    ("fwidth_fine", "fwidthFine"),
    // Numeric builtins with camelCase
    ("face_forward", "faceForward"),
    ("inverse_sqrt", "inverseSqrt"),
    // Texture functions
    ("texture_dimensions", "textureDimensions"),
    ("texture_dimensions_level", "textureDimensions"),
    // textureGather variants
    ("texture_gather", "textureGather"),
    ("texture_gather_array", "textureGather"),
    ("texture_gather_array_offset", "textureGather"),
    ("texture_gather_depth", "textureGather"),
    ("texture_gather_depth_array", "textureGather"),
    ("texture_gather_depth_array_offset", "textureGather"),
    ("texture_gather_depth_offset", "textureGather"),
    ("texture_gather_offset", "textureGather"),
    // textureGatherCompare variants
    ("texture_gather_compare", "textureGatherCompare"),
    ("texture_gather_compare_array", "textureGatherCompare"),
    (
        "texture_gather_compare_array_offset",
        "textureGatherCompare",
    ),
    ("texture_gather_compare_offset", "textureGatherCompare"),
    // textureLoad variants
    ("texture_load", "textureLoad"),
    ("texture_load_array", "textureLoad"),
    ("texture_load_multisampled", "textureLoad"),
    // Query functions
    ("texture_num_layers", "textureNumLayers"),
    ("texture_num_levels", "textureNumLevels"),
    ("texture_num_samples", "textureNumSamples"),
    // textureSample variants
    ("texture_sample", "textureSample"),
    ("texture_sample_array", "textureSample"),
    ("texture_sample_array_offset", "textureSample"),
    ("texture_sample_offset", "textureSample"),
    (
        "texture_sample_base_clamp_to_edge",
        "textureSampleBaseClampToEdge",
    ),
    // textureSampleBias variants
    ("texture_sample_bias", "textureSampleBias"),
    ("texture_sample_bias_array", "textureSampleBias"),
    ("texture_sample_bias_array_offset", "textureSampleBias"),
    ("texture_sample_bias_offset", "textureSampleBias"),
    // textureSampleCompare variants
    ("texture_sample_compare", "textureSampleCompare"),
    ("texture_sample_compare_array", "textureSampleCompare"),
    (
        "texture_sample_compare_array_offset",
        "textureSampleCompare",
    ),
    ("texture_sample_compare_offset", "textureSampleCompare"),
    // textureSampleCompareLevel variants
    ("texture_sample_compare_level", "textureSampleCompareLevel"),
    (
        "texture_sample_compare_level_array",
        "textureSampleCompareLevel",
    ),
    (
        "texture_sample_compare_level_array_offset",
        "textureSampleCompareLevel",
    ),
    (
        "texture_sample_compare_level_offset",
        "textureSampleCompareLevel",
    ),
    // textureSampleGrad variants
    ("texture_sample_grad", "textureSampleGrad"),
    ("texture_sample_grad_array", "textureSampleGrad"),
    ("texture_sample_grad_array_offset", "textureSampleGrad"),
    ("texture_sample_grad_offset", "textureSampleGrad"),
    // textureSampleLevel variants
    ("texture_sample_level", "textureSampleLevel"),
    ("texture_sample_level_array", "textureSampleLevel"),
    ("texture_sample_level_array_offset", "textureSampleLevel"),
    ("texture_sample_level_offset", "textureSampleLevel"),
    // textureStore variants
    ("texture_store", "textureStore"),
    ("texture_store_array", "textureStore"),
];

/// Translate a Rust-style builtin function name to the WGSL form. Returns
/// `None` if no translation is needed (the name should be used as-is).
pub fn lookup(name: &str) -> Option<&'static str> {
    TABLE
        .iter()
        .find(|(rust, _)| *rust == name)
        .map(|(_, wgsl)| *wgsl)
}
