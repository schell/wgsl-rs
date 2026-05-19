//! Robust name mangling for WGSL identifiers.
//!
//! `wgsl-rs` composes WGSL identifiers from multiple Rust identifier
//! components in several places (impl methods, impl constants, enum
//! variants, monomorphized generics). A naive `{a}_{b}` join is ambiguous
//! when components themselves contain underscores: `("Foo_bar", "baz")`
//! and `("Foo", "bar_baz")` both produce `Foo_bar_baz`.
//!
//! This module implements a simplified subset of the
//! [wesl-rs `EscapeMangler`](https://github.com/webgpu-tools/wesl-rs/blob/main/crates/wesl/src/mangle.rs)
//! scheme: each component that contains `N>0` underscores is rewritten as
//! `_N{component}` before the components are joined with `_`. This makes
//! the mangling a bijection.
//!
//! # Examples
//!
//! ```
//! use wgsl_rs_ir::mangle::{mangle, unmangle};
//!
//! assert_eq!(mangle(&["Foo", "bar"]), "Foo_bar");
//! assert_eq!(mangle(&["Foo_bar", "baz"]), "_1Foo_bar_baz");
//! assert_eq!(mangle(&["Foo", "bar_baz"]), "Foo__1bar_baz");
//!
//! assert_eq!(
//!     unmangle("Foo_bar").as_deref(),
//!     Some(&["Foo".to_string(), "bar".to_string()][..])
//! );
//! ```
//!
//! Inputs are required to be valid Rust identifiers (matching
//! `[A-Za-z_][A-Za-z0-9_]*`). No other characters are escaped.
//!
//! For nested mangling (e.g. building a mangled name from a type tree),
//! treat each inner mangled string as one opaque component and call
//! [`mangle`] again at the outer level. Inner underscores will be
//! re-counted and escaped at the outer level so the composition remains
//! unambiguous.

/// Escape a single identifier component. Components containing `_` are
/// prefixed with `_N` where `N` is the underscore count.
fn escape_component(comp: &str) -> String {
    let n = comp.chars().filter(|c| *c == '_').count();
    if n == 0 {
        comp.to_string()
    } else {
        format!("_{n}{comp}")
    }
}

/// Reverse `escape_component`, consuming as many `_`-separated parts from
/// `parts` as needed to reconstruct a single original component.
///
/// `part` is the current part; if empty, the next part begins with the
/// count digits and we consume `count` additional parts after the first
/// in order to glue them back with `_`.
fn unescape_component<'a>(part: &str, parts: &mut impl Iterator<Item = &'a str>) -> Option<String> {
    if !part.is_empty() {
        // Component with no underscores: pass through unchanged.
        return Some(part.to_string());
    }
    // An empty `part` signals that the next part starts with a digit
    // count followed by the first chunk of the original component.
    let first = parts.next()?;
    let digits: usize = first.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let count: usize = first[..digits].parse().ok()?;
    let rem = &first[digits..];
    let mut chunks: Vec<String> = Vec::with_capacity(count + 1);
    chunks.push(rem.to_string());
    for _ in 0..count {
        chunks.push(parts.next()?.to_string());
    }
    Some(chunks.join("_"))
}

/// Mangle a list of identifier components into a single WGSL-safe
/// identifier.
///
/// Each component containing underscores is rewritten as `_N{component}`
/// (where `N` is the underscore count) before the components are joined
/// with `_`. The result is a valid WGSL identifier and the mangling is
/// reversible via [`unmangle`].
pub fn mangle(components: &[&str]) -> String {
    let mut out = String::new();
    for (i, c) in components.iter().enumerate() {
        if i > 0 {
            out.push('_');
        }
        out.push_str(&escape_component(c));
    }
    out
}

/// Reverse of [`mangle`]. Returns `None` if `s` is not a well-formed
/// mangled string.
pub fn unmangle(s: &str) -> Option<Vec<String>> {
    let mut components = Vec::new();
    let mut parts = s.split('_');
    while let Some(part) = parts.next() {
        components.push(unescape_component(part, &mut parts)?);
    }
    Some(components)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_underscore_components_unchanged() {
        assert_eq!(mangle(&["Foo", "bar"]), "Foo_bar");
        assert_eq!(mangle(&["Color", "Red"]), "Color_Red");
    }

    #[test]
    fn underscore_in_first_component_escapes() {
        assert_eq!(mangle(&["Foo_bar", "baz"]), "_1Foo_bar_baz");
        assert_eq!(mangle(&["My_Struct", "do_thing"]), "_1My_Struct__1do_thing");
    }

    #[test]
    fn underscore_in_second_component_escapes() {
        assert_eq!(mangle(&["Foo", "bar_baz"]), "Foo__1bar_baz");
    }

    #[test]
    fn many_underscores_use_correct_count() {
        assert_eq!(mangle(&["a_b_c"]), "_2a_b_c");
        assert_eq!(mangle(&["___"]), "_3___");
    }

    /// The canonical collision case from issue #112: an impl on a
    /// underscored type vs. an impl method with an underscored name must
    /// produce distinct mangled names.
    #[test]
    fn collision_pair_impl_method_underscored_type_vs_method() {
        let a = mangle(&["Foo_bar", "baz"]);
        let b = mangle(&["Foo", "bar_baz"]);
        assert_ne!(a, b, "{a} == {b}");
    }

    #[test]
    fn collision_pair_enum_variant_underscored_name() {
        let a = mangle(&["Color_Red", "Hot"]);
        let b = mangle(&["Color", "Red_Hot"]);
        assert_ne!(a, b);
    }

    #[test]
    fn unmangle_inverts_mangle() {
        let cases: &[&[&str]] = &[
            &["Foo", "bar"],
            &["Foo_bar", "baz"],
            &["Foo", "bar_baz"],
            &["My_Struct", "do_thing"],
            &["only_one"],
            &["plain"],
            &["a_b_c", "d", "e_f"],
            &["___", "x"],
        ];
        for case in cases {
            let m = mangle(case);
            let parts: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            assert_eq!(
                unmangle(&m),
                Some(parts.clone()),
                "round-trip failed for {case:?} -> {m}"
            );
        }
    }

    #[test]
    fn unmangle_rejects_garbage() {
        // A leading `_` without a following digit count is invalid.
        assert_eq!(unmangle("_xFoo"), None);
    }

    /// Composition: an outer mangle of an inner mangled string must
    /// remain a bijection — and the resulting outer mangling is still
    /// roundtrippable as a flat component list at the outer level.
    #[test]
    fn nested_composition_is_unambiguous() {
        let inner_a = mangle(&["array", "f32"]); // "array_f32"
        let inner_b = mangle(&["array_f32"]); // "_1array_f32" (forced distinct)
        let outer_a = mangle(&["wrap", &inner_a]);
        let outer_b = mangle(&["wrap", &inner_b]);
        assert_ne!(outer_a, outer_b);
    }
}
