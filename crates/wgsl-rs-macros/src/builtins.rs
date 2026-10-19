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
    // Boolean vector aliases that _should_ exist in WGSL
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
    ("texture_load_storage", "textureLoad"),
    ("texture_load_storage_array", "textureLoad"),
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

/// Rust builtin function names that lower to binary operators during
/// rendering. A two-argument call to one of these renders as the infix
/// operator applied to its arguments: `cmp_eq(a, b)` -> `(a == b)`. This
/// is the componentwise comparison escape hatch for vectors (#164).
///
/// The render-side lowering lives in `wgsl-rs-ir`'s `builtin_lookup`
/// (`BINARY_OPS`); this copy exists so the macro can (a) reserve the
/// names against user definitions and (b) reject bad arity at parse time.
/// The two tables are pinned in sync by tests in this module.
pub const BUILTIN_BINARY_OPS: &[(&str, &str)] = &[
    // (rust_snake_case, wgsl operator)
    ("cmp_eq", "=="),
    ("cmp_ne", "!="),
];

/// Checks if a name refers to a binary-operator builtin (`cmp_eq`,
/// `cmp_ne`). Returns the WGSL operator it renders as.
pub fn is_operator_builtin(name: &str) -> Option<&'static str> {
    BUILTIN_BINARY_OPS
        .iter()
        .find(|(rust, _)| *rust == name)
        .map(|(_, op)| *op)
}

/// Looks up the WGSL name for a Rust function name.
///
/// Returns `Some(wgsl_name)` if translation is needed, `None` if the name
/// should be used as-is.
///
/// Note: production WGSL rendering goes through `wgsl_rs_ir`, which keeps
/// its own copy of the table. This function is retained for parity and
/// for the in-crate tests that pin the table contents.
#[allow(dead_code)]
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
    if let Some(entry) = BUILTIN_CASE_NAME_MAP
        .iter()
        .find(|(rust, wgsl)| *rust == name || *wgsl == name)
    {
        return Some(*entry);
    }
    BUILTIN_BINARY_OPS
        .iter()
        .find(|(rust, _)| *rust == name)
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgsl_rs_ir::{BinOp, render::builtin_lookup as ir_lookup};

    /// The name-translation table is duplicated between this crate and
    /// `wgsl-rs-ir` (which must stay standalone and cannot depend on a
    /// proc-macro crate). This pins the two copies together: adding,
    /// removing or editing an entry on one side without the other fails
    /// here.
    #[test]
    fn builtin_name_map_matches_ir_table() {
        assert_eq!(
            BUILTIN_CASE_NAME_MAP.len(),
            ir_lookup::TABLE.len(),
            "BUILTIN_CASE_NAME_MAP and wgsl-rs-ir's TABLE drifted: lengths differ"
        );
        for (rust, wgsl) in BUILTIN_CASE_NAME_MAP {
            assert!(
                ir_lookup::TABLE.iter().any(|(r, w)| r == rust && w == wgsl),
                "entry ({rust}, {wgsl}) is missing from wgsl-rs-ir's TABLE"
            );
        }
        for (rust, wgsl) in ir_lookup::TABLE {
            assert!(
                BUILTIN_CASE_NAME_MAP
                    .iter()
                    .any(|(r, w)| r == rust && w == wgsl),
                "entry ({rust}, {wgsl}) is missing from BUILTIN_CASE_NAME_MAP"
            );
        }
    }

    /// The binary-operator table is likewise duplicated. The macros copy
    /// stores the WGSL operator as a string (for error messages), the ir
    /// copy as a `BinOp` (for rendering); the mapping must agree.
    #[test]
    fn builtin_binary_ops_match_ir_table() {
        assert_eq!(
            BUILTIN_BINARY_OPS.len(),
            ir_lookup::BINARY_OPS.len(),
            "BUILTIN_BINARY_OPS and wgsl-rs-ir's BINARY_OPS drifted: lengths differ"
        );
        for (rust, op_str) in BUILTIN_BINARY_OPS {
            let expected = match *op_str {
                "==" => BinOp::Eq,
                "!=" => BinOp::Ne,
                other => panic!("unknown operator string {other} in BUILTIN_BINARY_OPS"),
            };
            assert_eq!(
                ir_lookup::lookup_binary_op(rust),
                Some(expected),
                "BUILTIN_BINARY_OPS says {rust} renders as {op_str}, but wgsl-rs-ir's BINARY_OPS \
                 disagrees"
            );
        }
        for (rust, _) in ir_lookup::BINARY_OPS {
            assert!(
                BUILTIN_BINARY_OPS.iter().any(|(r, _)| r == rust),
                "operator builtin {rust} is missing from BUILTIN_BINARY_OPS"
            );
        }
    }

    /// The builtin *type*-alias table in `wgsl-rs-ir`
    /// (`render::builtin_lookup::TYPE_TABLE`) duplicates this crate's
    /// `parse::builtin_wgsl_type_name` resolution (duplicated so
    /// `wgsl-rs-ir` stays standalone). This pins the copy against the
    /// macro's resolution for the full input space, both directions
    /// (wgsl-rs#230).
    #[test]
    fn ir_type_table_matches_builtin_wgsl_type_name() {
        use crate::parse::builtin_wgsl_type_name;

        // The full input space `builtin_wgsl_type_name` recognizes.
        let mut inputs: Vec<String> = Vec::new();
        for n in ["2", "3", "4"] {
            for suffix in ["f", "i", "u", "b"] {
                inputs.push(format!("Vec{n}{suffix}"));
            }
        }
        for c in ["2", "3", "4"] {
            inputs.push(format!("Mat{c}f"));
            for r in ["2", "3", "4"] {
                inputs.push(format!("Mat{c}x{r}f"));
            }
        }

        for input in &inputs {
            let expected = builtin_wgsl_type_name(input)
                .unwrap_or_else(|| panic!("`{input}` should resolve via builtin_wgsl_type_name"));
            assert_eq!(
                ir_lookup::lookup_type_name(input),
                Some(expected.as_str()),
                "wgsl-rs-ir's TYPE_TABLE drifted from builtin_wgsl_type_name for `{input}`"
            );
        }

        // No extra table entries outside the recognized space.
        for (rust, wgsl) in ir_lookup::TYPE_TABLE {
            assert_eq!(
                builtin_wgsl_type_name(rust),
                Some((*wgsl).to_string()),
                "wgsl-rs-ir's TYPE_TABLE entry ({rust}, {wgsl}) has no matching \
                 builtin_wgsl_type_name resolution"
            );
        }
    }

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
    fn operator_builtins_are_reserved() {
        assert!(is_reserved_builtin("cmp_eq").is_some());
        assert!(is_reserved_builtin("cmp_ne").is_some());
        assert_eq!(is_operator_builtin("cmp_eq"), Some("=="));
        assert_eq!(is_operator_builtin("cmp_ne"), Some("!="));
        assert_eq!(is_operator_builtin("cmp_lt"), None);
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
