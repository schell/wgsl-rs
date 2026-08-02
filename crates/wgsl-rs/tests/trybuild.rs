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
