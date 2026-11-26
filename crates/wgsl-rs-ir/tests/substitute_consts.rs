//! Unit tests for const-generic substitution (`substitute_consts`) — the
//! const parallel of `substitute_types` (see smoke.rs for the type side).
//!
//! Const values are `ConstValue`. `Int` covers signed const generics:
//! non-negative values must render byte-identically to the pre-signed
//! output, and negative values must render as parenthesized literals so
//! they are unambiguous in every expression context. `Bool` covers
//! compile-time feature flags, rendering as `true` / `false`. Neither a
//! negative value nor a bool may reach an array-length position (rustc
//! guards that path for macro-authored code by requiring array sizes to
//! evaluate to `usize`; the defensive check exists for hand-rolled IR).

use std::collections::HashMap;

use wgsl_rs_ir::*;

fn ident(name: &str) -> Expr {
    Expr::Ident(name.to_string())
}

fn bare_int(digits: &str) -> Expr {
    Expr::Lit(Lit::Int {
        digits: digits.to_string(),
        suffix: String::new(),
    })
}

/// A fn `f(x: i32) -> i32` with `const_params` and the given body.
fn fn_with_body(body_stmts: Vec<Stmt>, const_params: &[&str]) -> Module {
    Module {
        name: "t",
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            const_params: const_params.iter().map(|s| s.to_string()).collect(),
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::Scalar(ScalarType::I32),
                attrs: vec![],
            }],
            return_type: ReturnType::Type {
                annotation: ReturnTypeAnnotation::None,
                ty: Type::Scalar(ScalarType::I32),
            },
            block: Block { stmts: body_stmts },
            attrs: vec![],
        })],
        attrs: vec![],
    }
}

/// Integer const map helper: each pair becomes `ConstValue::Int`.
fn consts_of(pairs: &[(&str, i64)]) -> HashMap<String, ConstValue> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), ConstValue::Int(*v)))
        .collect()
}

/// Bool const map helper: each pair becomes `ConstValue::Bool`.
fn bool_consts_of(pairs: &[(&str, bool)]) -> HashMap<String, ConstValue> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), ConstValue::Bool(*v)))
        .collect()
}

#[test]
fn positive_const_substitutes_bare_literal() {
    // Byte-identical to the pre-signed output: bare digits, no parens.
    let body = vec![Stmt::Return(Some(Expr::Binary {
        lhs: Box::new(ident("x")),
        op: BinOp::Sub,
        rhs: Box::new(ident("N")),
    }))];
    let mut m = fn_with_body(body, &["N"]);
    substitute_consts(&mut m, &consts_of(&[("N", 4)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("return (x - 4);"), "got: {wgsl}");
}

#[test]
fn negative_const_substitutes_parenthesized_literal() {
    // `x - (-4)` — the parenthesized form is unambiguous in every
    // context (`x - -4` parses but is needlessly fragile).
    let body = vec![Stmt::Return(Some(Expr::Binary {
        lhs: Box::new(ident("x")),
        op: BinOp::Sub,
        rhs: Box::new(ident("N")),
    }))];
    let mut m = fn_with_body(body, &["N"]);
    substitute_consts(&mut m, &consts_of(&[("N", -4)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("x - (-4)"), "got: {wgsl}");
    assert!(!wgsl.contains("- -4"), "got: {wgsl}");
}

#[test]
fn negative_const_in_min_max_clamp() {
    // The motivating use case: `max(min(x, MAX), MIN)` with MIN = -8.
    let call = |name: &str, params: Vec<Expr>| Expr::FnCall {
        path: FnPath::Ident(name.to_string()),
        type_args: vec![],
        params,
    };
    let body = vec![Stmt::Return(Some(call(
        "max",
        vec![call("min", vec![ident("x"), ident("MAX")]), ident("MIN")],
    )))];
    let mut m = fn_with_body(body, &["MIN", "MAX"]);
    substitute_consts(&mut m, &consts_of(&[("MIN", -8), ("MAX", 8)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("max(min(x, 8), (-8))"), "got: {wgsl}");
}

#[test]
fn negative_const_in_comparison() {
    // Loop-bound shape: `0 < N` with N = -4 renders unambiguously.
    let body = vec![Stmt::Return(Some(Expr::Binary {
        lhs: Box::new(bare_int("0")),
        op: BinOp::Lt,
        rhs: Box::new(ident("N")),
    }))];
    let mut m = fn_with_body(body, &["N"]);
    substitute_consts(&mut m, &consts_of(&[("N", -4)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("0 < (-4)"), "got: {wgsl}");
}

#[test]
#[should_panic(expected = "negative const generic value -4 substituted into an array length")]
fn negative_const_in_array_length_panics() {
    // Unreachable through wgsl-rs macros — rustc rejects non-`usize`
    // array lengths ("the constant `N` is not of type `usize`") — but
    // hand-rolled IR could produce it; the defensive check must fire
    // before invalid WGSL renders.
    let mut m = Module {
        name: "t",
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            const_params: vec!["N".to_string()],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "a".to_string(),
                ty: Type::Array {
                    elem: Box::new(Type::Scalar(ScalarType::F32)),
                    len: ident("N"),
                },
                attrs: vec![],
            }],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    substitute_consts(&mut m, &consts_of(&[("N", -4)]));
    let _ = render_module(&m);
}

#[test]
fn positive_const_in_array_length_substitutes() {
    // The legitimate array-length path (usize consts on the Rust side):
    // a non-negative const substitutes into the length untouched.
    let mut m = Module {
        name: "t",
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            const_params: vec!["N".to_string()],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "a".to_string(),
                ty: Type::Array {
                    elem: Box::new(Type::Scalar(ScalarType::F32)),
                    len: ident("N"),
                },
                attrs: vec![],
            }],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    substitute_consts(&mut m, &consts_of(&[("N", 4)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("array<f32, 4>"), "got: {wgsl}");
}

#[test]
fn bool_const_substitutes_true_and_false_literals() {
    // The motivating case: a compile-time feature flag. `ConstValue::Bool`
    // substitutes as `Lit::Bool`, the same node a written `true` /
    // `false` parses to, so the flag branch renders valid WGSL and the
    // two worlds agree (CPU Rust evaluates the same branch).
    let mk = |flag: bool| {
        let body = vec![Stmt::If(StmtIf {
            condition: ident("APPLY_FOG"),
            then_block: Block {
                stmts: vec![Stmt::Return(Some(ident("x")))],
            },
            else_branch: None,
        })];
        let mut m = fn_with_body(body, &["APPLY_FOG"]);
        substitute_consts(&mut m, &bool_consts_of(&[("APPLY_FOG", flag)]));
        render_module(&m)
    };
    let true_wgsl = mk(true);
    assert!(true_wgsl.contains("if true {"), "got: {true_wgsl}");
    assert!(!true_wgsl.contains("APPLY_FOG"), "got: {true_wgsl}");
    let false_wgsl = mk(false);
    assert!(false_wgsl.contains("if false {"), "got: {false_wgsl}");
}

#[test]
fn bool_const_in_boolean_expression_renders_parenthesized() {
    // `B && x` with B = true renders `(true && x)`: the same
    // parenthesization discipline as the integer substitution paths.
    let body = vec![Stmt::Return(Some(Expr::Binary {
        lhs: Box::new(ident("B")),
        op: BinOp::And,
        rhs: Box::new(ident("x")),
    }))];
    let mut m = fn_with_body(body, &["B"]);
    substitute_consts(&mut m, &bool_consts_of(&[("B", true)]));
    let wgsl = render_module(&m);
    assert!(wgsl.contains("return (true && x);"), "got: {wgsl}");
}

#[test]
#[should_panic(expected = "bool const generic value substituted into an array length")]
fn bool_const_in_array_length_panics() {
    // Unreachable through wgsl-rs macros for the same reason as the
    // negative-length case: rustc requires array sizes to evaluate to
    // `usize`. The defensive check fires for hand-rolled IR before
    // invalid WGSL (`array<f32, true>`) renders.
    let mut m = Module {
        name: "t",
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            const_params: vec!["N".to_string()],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "a".to_string(),
                ty: Type::Array {
                    elem: Box::new(Type::Scalar(ScalarType::F32)),
                    len: ident("N"),
                },
                attrs: vec![],
            }],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    substitute_consts(&mut m, &bool_consts_of(&[("N", true)]));
    let _ = render_module(&m);
}

fn module_with_filterability_params() -> Module {
    Module {
        name: "filterability",
        items: vec![
            Item::Texture(ItemTexture {
                group: 0.into(),
                binding: 0.into(),
                name: "TEX".into(),
                ty: Type::Texture {
                    kind: TextureKind::Texture2D,
                    sampled_type: ScalarType::F32,
                },
                filterable: Filterability::Param("TEX_FILTERABLE".into()),
                attrs: vec![],
            }),
            Item::Sampler(ItemSampler {
                group: 0.into(),
                binding: 1.into(),
                name: "SMP".into(),
                ty: Type::Sampler,
                filterable: Filterability::Param("SMP_FILTERABLE".into()),
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    }
}

#[test]
fn bool_consts_resolve_binding_filterability_without_changing_wgsl() {
    for value in [true, false] {
        let mut module = module_with_filterability_params();
        let before = render_module(&module);
        substitute_consts(
            &mut module,
            &bool_consts_of(&[("TEX_FILTERABLE", value), ("SMP_FILTERABLE", !value)]),
        );

        assert_eq!(render_module(&module), before);
        let Item::Texture(texture) = &module.items[0] else {
            unreachable!()
        };
        assert_eq!(texture.filterable, Filterability::Concrete(value));
        let Item::Sampler(sampler) = &module.items[1] else {
            unreachable!()
        };
        assert_eq!(sampler.filterable, Filterability::Concrete(!value));
    }
}

#[test]
fn filterability_params_remain_unresolved_without_a_substitution() {
    let mut module = module_with_filterability_params();
    substitute_consts(&mut module, &HashMap::new());

    let Item::Texture(texture) = &module.items[0] else {
        unreachable!()
    };
    assert_eq!(texture.filterable.concrete(), None);
    let Item::Sampler(sampler) = &module.items[1] else {
        unreachable!()
    };
    assert_eq!(sampler.filterable.concrete(), None);
}

#[test]
#[should_panic(expected = "filterability requires a bool")]
fn integer_const_does_not_coerce_to_filterability() {
    let mut module = module_with_filterability_params();
    substitute_consts(&mut module, &consts_of(&[("TEX_FILTERABLE", 1)]));
}

fn module_with_address_params() -> Module {
    Module {
        name: "addresses",
        items: vec![Item::Texture(ItemTexture {
            group: BindingAddress::Param("GROUP".into()),
            binding: BindingAddress::Param("BINDING".into()),
            name: "TEX".into(),
            ty: Type::Texture {
                kind: TextureKind::Texture2D,
                sampled_type: ScalarType::F32,
            },
            filterable: true.into(),
            attrs: vec![],
        })],
        attrs: vec![],
    }
}

#[test]
fn integer_consts_resolve_binding_addresses() {
    let mut module = module_with_address_params();
    substitute_consts(&mut module, &consts_of(&[("GROUP", 3), ("BINDING", 7)]));
    let Item::Texture(texture) = &module.items[0] else {
        unreachable!()
    };
    assert_eq!(texture.group, BindingAddress::Concrete(3));
    assert_eq!(texture.binding, BindingAddress::Concrete(7));
    assert!(render_module(&module).contains("@group(3) @binding(7) var TEX"));
}

#[test]
#[should_panic(expected = "outside the u32 address range")]
fn binding_address_substitution_rejects_out_of_range_values() {
    let mut module = module_with_address_params();
    substitute_consts(&mut module, &consts_of(&[("GROUP", u32::MAX as i64 + 1)]));
}

#[test]
#[should_panic(expected = "addresses require integers")]
fn binding_address_substitution_rejects_bool_values() {
    let mut module = module_with_address_params();
    substitute_consts(&mut module, &bool_consts_of(&[("GROUP", true)]));
}

#[test]
fn duplicate_addresses_are_rejected_after_substitution() {
    let mut module = module_with_address_params();
    module.items.push(Item::Sampler(ItemSampler {
        group: BindingAddress::Param("GROUP".into()),
        binding: BindingAddress::Param("BINDING".into()),
        name: "SMP".into(),
        ty: Type::Sampler,
        filterable: true.into(),
        attrs: vec![],
    }));
    substitute_consts(&mut module, &consts_of(&[("GROUP", 1), ("BINDING", 2)]));
    let error = validate_binding_addresses(&module).unwrap_err();
    assert!(error.contains("TEX") && error.contains("SMP"), "{error}");
}
