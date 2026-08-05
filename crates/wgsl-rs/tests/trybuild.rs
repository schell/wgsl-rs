#[test]
fn mixed_local_and_cross_module_generics_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/mixed_local_and_cross_module_generics.rs");
}

#[test]
fn extensions_basic_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/extensions_basic.rs");
}

#[test]
fn extensions_not_impl_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/extensions_not_impl.rs");
}

#[test]
fn match_mixed_default_selectors_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/match_mixed_default_selectors.rs");
}

#[test]
fn linkage_access_in_const_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/linkage_access_in_const.rs");
}

#[test]
fn multi_segment_type_path_in_struct_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/path_in_struct.rs");
}

#[test]
fn typed_literal_f64_suffix_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/typed_literal_f64_suffix.rs");
}

#[test]
fn generic_impl_on_array_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/generic_impl_on_array.rs");
}

#[test]
fn generic_impl_const_generic_array_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/generic_impl_const_generic_array.rs");
}

#[test]
fn const_generics_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/const_generics.rs");
}

#[test]
fn phantom_data_compiles() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/phantom_data.rs");
}

#[test]
fn phantom_data_qualified_path_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/phantom_data_qualified_path.rs");
}

#[test]
fn phantom_data_in_non_field_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/phantom_data_in_non_field.rs");
}
