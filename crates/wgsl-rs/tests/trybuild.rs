#[test]
fn mixed_local_and_cross_module_generics_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/mixed_local_and_cross_module_generics.rs");
}

#[test]
fn linkage_access_in_const_is_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/linkage_access_in_const.rs");
}
