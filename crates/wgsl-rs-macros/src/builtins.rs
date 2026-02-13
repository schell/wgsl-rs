//! WGSL builtin function name mappings.
//!
//! Maps Rust snake_case names to WGSL camelCase names for builtin functions
//! that require name translation during code generation.

/// Lookup table for builtin functions that need name translation.
///
/// Format: (rust_snake_case, wgslCamelCase)
///
/// Note: Functions where Rust and WGSL names match (e.g., `sin`, `cos`, `abs`)
/// are NOT included here since they don't need translation.
pub const BUILTIN_CASE_NAME_MAP: &[(&str, &str)] = &[
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
    ("atomic_and", "atomicAnd"),
    ("atomic_compare_exchange_weak", "atomicCompareExchangeWeak"),
    ("atomic_exchange", "atomicExchange"),
    ("atomic_load", "atomicLoad"),
    ("atomic_max", "atomicMax"),
    ("atomic_min", "atomicMin"),
    ("atomic_or", "atomicOr"),
    ("atomic_store", "atomicStore"),
    ("atomic_sub", "atomicSub"),
    ("atomic_xor", "atomicXor"),
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

/// Looks up the WGSL name for a Rust function name.
///
/// Returns `Some(wgsl_name)` if translation is needed, `None` if the name
/// should be used as-is.
pub fn lookup_wgsl_name(rust_name: &str) -> Option<&'static str> {
    BUILTIN_CASE_NAME_MAP
        .iter()
        .find(|(rust, _)| *rust == rust_name)
        .map(|(_, wgsl)| *wgsl)
}

/// Checks if a name (either Rust or WGSL form) is reserved for a builtin.
///
/// Returns `Some((rust_name, wgsl_name))` if reserved, `None` otherwise.
pub fn is_reserved_builtin(name: &str) -> Option<(&'static str, &'static str)> {
    BUILTIN_CASE_NAME_MAP
        .iter()
        .find(|(rust, wgsl)| *rust == name || *wgsl == name)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_existing_builtin() {
        assert_eq!(
            lookup_wgsl_name("count_leading_zeros"),
            Some("countLeadingZeros")
        );
        assert_eq!(lookup_wgsl_name("inverse_sqrt"), Some("inverseSqrt"));
        assert_eq!(
            lookup_wgsl_name("texture_dimensions"),
            Some("textureDimensions")
        );
    }

    #[test]
    fn lookup_non_builtin_returns_none() {
        assert_eq!(lookup_wgsl_name("sin"), None);
        assert_eq!(lookup_wgsl_name("my_custom_function"), None);
    }

    #[test]
    fn is_reserved_matches_rust_name() {
        let result = is_reserved_builtin("count_leading_zeros");
        assert_eq!(result, Some(("count_leading_zeros", "countLeadingZeros")));
    }

    #[test]
    fn is_reserved_matches_wgsl_name() {
        let result = is_reserved_builtin("countLeadingZeros");
        assert_eq!(result, Some(("count_leading_zeros", "countLeadingZeros")));
    }

    #[test]
    fn is_reserved_returns_none_for_non_builtin() {
        assert_eq!(is_reserved_builtin("my_function"), None);
        assert_eq!(is_reserved_builtin("sin"), None);
    }

    #[test]
    fn lookup_bitcast_builtins() {
        assert_eq!(lookup_wgsl_name("bitcast_f32"), Some("bitcast<f32>"));
        assert_eq!(lookup_wgsl_name("bitcast_u32"), Some("bitcast<u32>"));
        assert_eq!(lookup_wgsl_name("bitcast_i32"), Some("bitcast<i32>"));
        assert_eq!(
            lookup_wgsl_name("bitcast_vec2f"),
            Some("bitcast<vec2<f32>>")
        );
        assert_eq!(
            lookup_wgsl_name("bitcast_vec4i"),
            Some("bitcast<vec4<i32>>")
        );
    }

    #[test]
    fn bitcast_names_are_reserved() {
        assert!(is_reserved_builtin("bitcast_f32").is_some());
        assert!(is_reserved_builtin("bitcast_u32").is_some());
        assert!(is_reserved_builtin("bitcast_i32").is_some());
        assert!(is_reserved_builtin("bitcast_vec3u").is_some());
    }
}
