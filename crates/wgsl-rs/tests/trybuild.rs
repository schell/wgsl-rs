#[test]
fn mixed_local_and_cross_module_generics_compile() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/mixed_local_and_cross_module_generics.rs");
}
