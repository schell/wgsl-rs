//! WGSL builtin function name mappings.
//!
//! Maps Rust snake_case names to WGSL camelCase names for builtin functions
//! that require name translation during code generation.

/// Lookup table for builtin functions that need name translation.
///
/// Format: (rust_snake_case, wgsl_camelCase)
///
/// Note: Functions where Rust and WGSL names match (e.g., `sin`, `cos`, `abs`)
/// are NOT included here since they don't need translation.
pub static BUILTIN_NAME_MAP: &[(&str, &str)] = &[
    // Arrays
    ("array_length", "arrayLength"),
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
    ("texture_gather", "textureGather"),
    ("texture_gather_compare", "textureGatherCompare"),
    ("texture_load", "textureLoad"),
    ("texture_num_layers", "textureNumLayers"),
    ("texture_num_levels", "textureNumLevels"),
    ("texture_num_samples", "textureNumSamples"),
    ("texture_sample", "textureSample"),
    (
        "texture_sample_base_clamp_to_edge",
        "textureSampleBaseClampToEdge",
    ),
    ("texture_sample_bias", "textureSampleBias"),
    ("texture_sample_compare", "textureSampleCompare"),
    ("texture_sample_compare_level", "textureSampleCompareLevel"),
    ("texture_sample_grad", "textureSampleGrad"),
    ("texture_sample_level", "textureSampleLevel"),
    ("texture_store", "textureStore"),
];

/// Looks up the WGSL name for a Rust function name.
///
/// Returns `Some(wgsl_name)` if translation is needed, `None` if the name
/// should be used as-is.
pub fn lookup_wgsl_name(rust_name: &str) -> Option<&'static str> {
    BUILTIN_NAME_MAP
        .iter()
        .find(|(rust, _)| *rust == rust_name)
        .map(|(_, wgsl)| *wgsl)
}

/// Checks if a name (either Rust or WGSL form) is reserved for a builtin.
///
/// Returns `Some((rust_name, wgsl_name))` if reserved, `None` otherwise.
pub fn is_reserved_builtin(name: &str) -> Option<(&'static str, &'static str)> {
    BUILTIN_NAME_MAP
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
}
