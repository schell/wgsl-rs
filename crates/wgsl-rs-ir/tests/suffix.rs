//! Unit tests for the type-directed integer literal suffix pass
//! (`suffix_module` / `suffix_items`, wgsl-rs#145).

use wgsl_rs_ir::*;

fn ident(name: &str) -> Expr {
    Expr::Ident(name.to_string())
}

/// An unsuffixed integer literal — what Rust emits for `0`, `1`, …
fn bare_int(digits: &str) -> Expr {
    Expr::Lit(Lit::Int {
        digits: digits.to_string(),
        suffix: String::new(),
    })
}

/// An integer literal that already carries a Rust-style suffix.
fn suffixed_int(digits: &str, suffix: &str) -> Expr {
    Expr::Lit(Lit::Int {
        digits: digits.to_string(),
        suffix: suffix.to_string(),
    })
}

fn u32_ty() -> Type {
    Type::Scalar(ScalarType::U32)
}

fn i32_ty() -> Type {
    Type::Scalar(ScalarType::I32)
}

fn returns(ty: Type) -> ReturnType {
    ReturnType::Type {
        annotation: ReturnTypeAnnotation::None,
        ty,
    }
}

fn arg(name: &str, ty: Type) -> FnArg {
    FnArg {
        inter_stage_io: vec![],
        name: name.to_string(),
        ty,
        attrs: vec![],
    }
}

fn fn_item(name: &str, inputs: Vec<FnArg>, return_type: ReturnType, stmts: Vec<Stmt>) -> Item {
    Item::Fn(ItemFn {
        type_params: vec![],
        const_params: vec![],
        fn_attrs: FnAttrs::None,
        name: name.to_string().into(),
        inputs,
        return_type,
        block: Block { stmts },
        attrs: vec![],
    })
}

fn local(name: &str, ty: Option<Type>, init: Option<Expr>) -> Stmt {
    Stmt::Local(Local {
        mutable: false,
        name: name.to_string(),
        ty,
        init,
    })
}

fn module(items: Vec<Item>) -> Module {
    Module {
        name: "t",
        items,
        attrs: vec![],
    }
}

/// Suffix, then render — the order the runtime pipeline will use.
fn render(m: &mut Module) -> String {
    suffix_module(m);
    render_module(m)
}

#[test]
fn return_statement_suffixes_from_return_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(bare_int("0")))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("return 0u;"), "got: {wgsl}");
}

#[test]
fn trailing_expression_suffixes_from_return_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Expr {
            expr: bare_int("7"),
            has_semi: false,
        }],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("7u"), "got: {wgsl}");
}

#[test]
fn i32_expectation_writes_i32_suffix() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(i32_ty()),
        vec![Stmt::Return(Some(bare_int("0")))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0i"), "got: {wgsl}");
}

#[test]
fn typed_local_initializer_is_suffixed() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local("x", Some(u32_ty()), Some(bare_int("0"))),
            Stmt::Return(Some(ident("x"))),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("= 0u;"), "got: {wgsl}");
}

#[test]
fn assignment_rhs_uses_declared_local_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local("x", Some(u32_ty()), Some(suffixed_int("3", "u32"))),
            Stmt::Assignment {
                lhs: ident("x"),
                rhs: bare_int("0"),
            },
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("x = 0u;"), "got: {wgsl}");
}

#[test]
fn assignment_rhs_uses_param_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("p", u32_ty())],
        ReturnType::Default,
        vec![Stmt::Assignment {
            lhs: ident("p"),
            rhs: bare_int("1"),
        }],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("p = 1u;"), "got: {wgsl}");
}

#[test]
fn compound_assignment_rhs_is_suffixed() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local("x", Some(u32_ty()), Some(suffixed_int("3", "u32"))),
            Stmt::CompoundAssignment {
                lhs: ident("x"),
                op: CompoundOp::AddAssign,
                rhs: bare_int("1"),
            },
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn parenthesized_expressions_propagate_expectations() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(Expr::Paren(Box::new(bare_int("9")))))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("9u"), "got: {wgsl}");
}

#[test]
fn const_item_initializer_is_suffixed() {
    let mut m = module(vec![Item::Const(ItemConst {
        name: "MAX".to_string(),
        ty: u32_ty(),
        expr: bare_int("42"),
        attrs: vec![],
    })]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("42u"), "got: {wgsl}");
}

#[test]
fn impl_method_bodies_are_walked() {
    let mut m = module(vec![Item::Impl(ItemImpl {
        type_params: vec![],
        const_params: vec![],
        self_ty: "Counter".to_string(),
        items: vec![ImplItem::Fn(ItemFn {
            type_params: vec![],
            const_params: vec![],
            fn_attrs: FnAttrs::None,
            name: "zero".to_string().into(),
            inputs: vec![],
            return_type: returns(u32_ty()),
            block: Block {
                stmts: vec![Stmt::Expr {
                    expr: bare_int("0"),
                    has_semi: false,
                }],
            },
            attrs: vec![],
        })],
        attrs: vec![],
    })]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn nested_block_scopes_resolve_assignment_targets() {
    // `if c { let y: u32 = …; y = 0; }` — the assignment inside the
    // block resolves `y` through the scope frames.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![Stmt::If(StmtIf {
            condition: ident("c"),
            then_block: Block {
                stmts: vec![
                    local("y", Some(u32_ty()), Some(suffixed_int("1", "u32"))),
                    Stmt::Assignment {
                        lhs: ident("y"),
                        rhs: bare_int("0"),
                    },
                ],
            },
            else_branch: None,
        })],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("y = 0u;"), "got: {wgsl}");
}

#[test]
fn suffixed_literals_are_never_re_suffixed() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(suffixed_int("5", "u32")))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("5u"), "got: {wgsl}");
    assert!(!wgsl.contains("5uu"), "got: {wgsl}");
}

#[test]
fn pass_is_idempotent() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(bare_int("0")))],
    )]);
    suffix_module(&mut m);
    suffix_module(&mut m);
    // IR-level check: the suffix is exactly "u32", not doubled.
    match &m.items[0] {
        Item::Fn(f) => match &f.block.stmts[0] {
            Stmt::Return(Some(Expr::Lit(Lit::Int { digits, suffix }))) => {
                assert_eq!(digits, "0");
                assert_eq!(suffix, "u32");
            }
            other => panic!("expected return of literal, got {other:?}"),
        },
        other => panic!("expected fn item, got {other:?}"),
    }
}

#[test]
fn non_scalar_expectations_leave_literals_bare() {
    // A vector return type is not a concrete scalar expectation; the
    // propagation stops and the literal stays bare.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(Type::Vector {
            elements: 2,
            scalar_ty: Some(ScalarType::F32),
        }),
        vec![Stmt::Return(Some(bare_int("0")))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("return 0;"), "got: {wgsl}");
}

// ===== Composite contexts =====

fn array_ty(elem: Type, len: u32) -> Type {
    Type::Array {
        elem: Box::new(elem),
        len: suffixed_int(&len.to_string(), "u32"),
    }
}

fn struct_item(name: &str, type_params: &[&str], fields: &[(&str, Type)]) -> Item {
    Item::Struct(ItemStruct {
        type_params: type_params.iter().map(|s| s.to_string()).collect(),
        const_params: vec![],
        name: name.to_string(),
        fields: fields
            .iter()
            .map(|(field_name, ty)| Field {
                inter_stage_io: vec![],
                name: field_name.to_string(),
                ty: ty.clone(),
                attrs: vec![],
            })
            .collect(),
        attrs: vec![],
    })
}

fn struct_expr(name: &str, type_args: Vec<Type>, fields: Vec<(&str, Expr)>) -> Expr {
    Expr::Struct {
        name: name.to_string(),
        type_args,
        fields: fields
            .into_iter()
            .map(|(member, expr)| FieldValue {
                member: member.to_string(),
                expr,
            })
            .collect(),
    }
}

#[test]
fn array_literal_elements_get_element_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(array_ty(u32_ty(), 2)),
        vec![Stmt::Return(Some(Expr::Array {
            elems: vec![bare_int("0"), bare_int("1")],
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn nested_array_literals_propagate_element_types() {
    // Return type `[[u32; 2]; 1]`, literal `[[0, 1]]` — the element
    // context must survive nesting (the shape of issue #145's repro).
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(array_ty(array_ty(u32_ty(), 2), 1)),
        vec![Stmt::Return(Some(Expr::Array {
            elems: vec![Expr::Array {
                elems: vec![bare_int("0"), bare_int("1")],
            }],
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn struct_constructor_fields_get_declared_types() {
    let mut m = module(vec![
        struct_item("Counter", &[], &[("n", u32_ty()), ("tag", i32_ty())]),
        fn_item(
            "f",
            vec![],
            returns(Type::Struct {
                name: "Counter".to_string(),
                type_args: vec![],
            }),
            vec![Stmt::Return(Some(struct_expr(
                "Counter",
                vec![],
                vec![("n", bare_int("0")), ("tag", bare_int("1"))],
            )))],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
    assert!(wgsl.contains("1i"), "got: {wgsl}");
}

#[test]
fn struct_constructor_self_anchors_without_annotation() {
    // `let c = Counter { n: 0 };` — no outer annotation; the
    // constructor's own definition provides the field types.
    let mut m = module(vec![
        struct_item("Counter", &[], &[("n", u32_ty())]),
        fn_item(
            "f",
            vec![],
            ReturnType::Default,
            vec![
                local(
                    "c",
                    None,
                    Some(struct_expr("Counter", vec![], vec![("n", bare_int("0"))])),
                ),
                Stmt::Return(None),
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn generic_struct_fields_substitute_type_args() {
    // `struct Pair<T> { first: T, count: u32 }`;
    // `fn f() -> Pair<u32> { return Pair { first: 0, count: 1 }; }`
    let mut m = module(vec![
        struct_item(
            "Pair",
            &["T"],
            &[
                (
                    "first",
                    Type::TypeParam {
                        name: "T".to_string(),
                    },
                ),
                ("count", u32_ty()),
            ],
        ),
        fn_item(
            "f",
            vec![],
            returns(Type::Struct {
                name: "Pair".to_string(),
                type_args: vec![u32_ty()],
            }),
            vec![Stmt::Return(Some(struct_expr(
                "Pair",
                vec![],
                vec![("first", bare_int("0")), ("count", bare_int("1"))],
            )))],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn cast_propagates_target_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(Expr::Cast {
            lhs: Box::new(bare_int("5")),
            ty: Box::new(u32_ty()),
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("5u"), "got: {wgsl}");
}

#[test]
fn binary_operand_anchors_from_typed_param() {
    // `let q = p + 1;` (untyped local) — `p: u32` anchors the literal.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("p", u32_ty())],
        ReturnType::Default,
        vec![
            local(
                "q",
                None,
                Some(Expr::Binary {
                    lhs: Box::new(ident("p")),
                    op: BinOp::Add,
                    rhs: Box::new(bare_int("1")),
                }),
            ),
            Stmt::Return(Some(ident("q"))),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn comparison_operand_anchors_from_typed_param() {
    // `if p < 10 { }` — the comparison anchors the bare literal.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("p", u32_ty())],
        ReturnType::Default,
        vec![Stmt::If(StmtIf {
            condition: Expr::Binary {
                lhs: Box::new(ident("p")),
                op: BinOp::Lt,
                rhs: Box::new(bare_int("10")),
            },
            then_block: Block { stmts: vec![] },
            else_branch: None,
        })],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("10u"), "got: {wgsl}");
}

#[test]
fn assignment_to_field_uses_field_type() {
    let mut m = module(vec![
        struct_item("Counter", &[], &[("n", u32_ty())]),
        fn_item(
            "f",
            vec![],
            ReturnType::Default,
            vec![
                local(
                    "c",
                    Some(Type::Struct {
                        name: "Counter".to_string(),
                        type_args: vec![],
                    }),
                    Some(struct_expr(
                        "Counter",
                        vec![],
                        vec![("n", suffixed_int("0", "u32"))],
                    )),
                ),
                Stmt::Assignment {
                    lhs: Expr::FieldAccess {
                        base: Box::new(ident("c")),
                        field: "n".to_string(),
                    },
                    rhs: bare_int("1"),
                },
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("= 1u"), "got: {wgsl}");
}

#[test]
fn assignment_to_array_element_uses_elem_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local(
                "a",
                Some(array_ty(u32_ty(), 4)),
                Some(Expr::Array {
                    elems: vec![
                        suffixed_int("0", "u32"),
                        suffixed_int("0", "u32"),
                        suffixed_int("0", "u32"),
                        suffixed_int("0", "u32"),
                    ],
                }),
            ),
            Stmt::Assignment {
                lhs: Expr::ArrayIndexing {
                    lhs: Box::new(ident("a")),
                    index: Box::new(bare_int("0")),
                },
                rhs: bare_int("1"),
            },
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("= 1u"), "got: {wgsl}");
    // The index stays bare — WGSL coerces the abstract integer.
    assert!(wgsl.contains("a[0]"), "got: {wgsl}");
}

#[test]
fn switch_selectors_get_selector_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("x", u32_ty())],
        ReturnType::Default,
        vec![Stmt::Switch(StmtSwitch {
            selector: ident("x"),
            arms: vec![SwitchArm {
                selectors: vec![
                    CaseSelector::Literal(Lit::Int {
                        digits: "0".to_string(),
                        suffix: String::new(),
                    }),
                    CaseSelector::Default,
                ],
                body: Block { stmts: vec![] },
            }],
            has_explicit_default: true,
        })],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn for_bounds_get_loop_var_type() {
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![Stmt::For(ForLoop {
            var: "i".to_string(),
            var_ty: Some(u32_ty()),
            from: bare_int("0"),
            to: bare_int("8"),
            inclusive: false,
            body: Block { stmts: vec![] },
        })],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
    assert!(wgsl.contains("8u"), "got: {wgsl}");
}

// ===== Call-site contexts =====

fn call(name: &str, params: Vec<Expr>) -> Expr {
    Expr::FnCall {
        path: FnPath::Ident(name.to_string()),
        type_args: vec![],
        params,
    }
}

fn method_call(ty: &str, method: &str, params: Vec<Expr>) -> Expr {
    Expr::FnCall {
        path: FnPath::TypeMethod {
            ty: ty.to_string(),
            method: method.to_string(),
        },
        type_args: vec![],
        params,
    }
}

#[test]
fn issue_145_repro_selects_in_u32_array_context() {
    // `fn to_array(data: bool) -> [u32; 1] { [select(0, 1, data)] }` —
    // Rust infers `0`/`1` as u32; without suffixes naga reads them as
    // i32 and rejects the module. This is the issue #145 repro.
    let mut m = module(vec![fn_item(
        "to_array",
        vec![arg("data", Type::Scalar(ScalarType::Bool))],
        returns(array_ty(u32_ty(), 1)),
        vec![Stmt::Return(Some(Expr::Array {
            elems: vec![call(
                "select",
                vec![bare_int("0"), bare_int("1"), ident("data")],
            )],
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("select(0u, 1u, data)"), "got: {wgsl}");
}

#[test]
fn select_anchors_from_typed_value_arg() {
    // `let y = select(x, 0, c);` — no outer annotation; `x: u32`
    // anchors the bare literal.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("x", u32_ty()), arg("c", Type::Scalar(ScalarType::Bool))],
        ReturnType::Default,
        vec![
            local(
                "y",
                None,
                Some(call("select", vec![ident("x"), bare_int("0"), ident("c")])),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn min_anchors_from_suffixed_value_arg() {
    // `let y = min(3u32, 1);` — the suffixed operand anchors the bare
    // literal through the shared value type.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local(
                "y",
                None,
                Some(call("min", vec![suffixed_int("3", "u32"), bare_int("1")])),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("1u"), "got: {wgsl}");
}

#[test]
fn user_fn_params_propagate_into_args() {
    let mut m = module(vec![
        fn_item(
            "helper",
            vec![arg("x", u32_ty())],
            returns(u32_ty()),
            vec![Stmt::Return(Some(ident("x")))],
        ),
        fn_item(
            "caller",
            vec![],
            ReturnType::Default,
            vec![
                local("y", None, Some(call("helper", vec![bare_int("0")]))),
                Stmt::Return(None),
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("helper(0u)"), "got: {wgsl}");
}

#[test]
fn user_fn_shadows_builtin_rule() {
    // A user-defined `select` shadows the builtin rule, mirroring Rust
    // and WGSL name resolution — here its u32 param types drive the
    // propagation.
    let mut m = module(vec![
        fn_item(
            "select",
            vec![
                arg("f", u32_ty()),
                arg("t", u32_ty()),
                arg("cond", Type::Scalar(ScalarType::Bool)),
            ],
            returns(u32_ty()),
            vec![Stmt::Return(Some(ident("f")))],
        ),
        fn_item(
            "caller",
            vec![],
            ReturnType::Default,
            vec![
                local(
                    "y",
                    None,
                    Some(call(
                        "select",
                        vec![bare_int("0"), bare_int("1"), ident("cond")],
                    )),
                ),
                Stmt::Return(None),
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("select(0u, 1u, cond)"), "got: {wgsl}");
}

#[test]
fn impl_method_params_propagate_into_args() {
    let mut m = module(vec![
        struct_item("Counter", &[], &[("n", u32_ty())]),
        Item::Impl(ItemImpl {
            type_params: vec![],
            const_params: vec![],
            self_ty: "Counter".to_string(),
            items: vec![ImplItem::Fn(ItemFn {
                type_params: vec![],
                const_params: vec![],
                fn_attrs: FnAttrs::None,
                name: "bump".to_string().into(),
                inputs: vec![
                    arg(
                        "c",
                        Type::Struct {
                            name: "Counter".to_string(),
                            type_args: vec![],
                        },
                    ),
                    arg("amount", u32_ty()),
                ],
                return_type: returns(Type::Struct {
                    name: "Counter".to_string(),
                    type_args: vec![],
                }),
                block: Block {
                    stmts: vec![Stmt::Return(Some(ident("c")))],
                },
                attrs: vec![],
            })],
            attrs: vec![],
        }),
        fn_item(
            "caller",
            vec![],
            ReturnType::Default,
            vec![
                local(
                    "y",
                    None,
                    Some(method_call(
                        "Counter",
                        "bump",
                        vec![
                            struct_expr("Counter", vec![], vec![("n", suffixed_int("0", "u32"))]),
                            bare_int("5"),
                        ],
                    )),
                ),
                Stmt::Return(None),
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("5u"), "got: {wgsl}");
}

#[test]
fn vec_constructor_u_propagates_element_type() {
    // `let v = vec4u(0, 1, 2, 3);` — the `vec4u(..., 0, 0)` pattern
    // already appears in the roundtrip bit_manipulation shader.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local(
                "v",
                None,
                Some(call(
                    "vec4u",
                    vec![bare_int("0"), bare_int("1"), bare_int("2"), bare_int("3")],
                )),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("vec4u(0u, 1u, 2u, 3u)"), "got: {wgsl}");
}

#[test]
fn abstract_vec_constructor_leaves_literals_bare() {
    // `let v = vec3(0, 1, 2);` — abstract constructors have no element
    // type to propagate; WGSL's abstract-int defaults match Rust's i32.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local(
                "v",
                None,
                Some(call(
                    "vec3",
                    vec![bare_int("0"), bare_int("1"), bare_int("2")],
                )),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("vec3(0, 1, 2)"), "got: {wgsl}");
}

// ===== Review-fix coverage =====

#[test]
fn impl_associated_const_is_anchored() {
    // `impl Counter { const MAX: u32 = 42; }` — associated consts anchor
    // from their declared type, like module-level consts.
    let mut m = module(vec![Item::Impl(ItemImpl {
        type_params: vec![],
        const_params: vec![],
        self_ty: "Counter".to_string(),
        items: vec![ImplItem::Const(ItemConst {
            name: "MAX".to_string(),
            ty: u32_ty(),
            expr: bare_int("42"),
            attrs: vec![],
        })],
        attrs: vec![],
    })]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("42u"), "got: {wgsl}");
}

#[test]
fn imported_fn_signatures_anchor_call_args() {
    // Module A defines `helper(x: u32)`; module B calls `helper(0)`.
    // Seeding B's pass with A's signatures (as `Source::collect` threads
    // them through its depth-first import walk) suffixes the argument.
    let helper = module(vec![fn_item(
        "helper",
        vec![arg("x", u32_ty())],
        returns(u32_ty()),
        vec![Stmt::Return(Some(ident("x")))],
    )]);
    let sigs = fn_signatures(&helper);
    assert_eq!(
        sigs.get("helper"),
        Some(&vec![u32_ty()]),
        "fn_signatures should harvest free-fn params"
    );

    let mut caller = module(vec![fn_item(
        "caller",
        vec![],
        ReturnType::Default,
        vec![
            local("y", None, Some(call("helper", vec![bare_int("0")]))),
            Stmt::Return(None),
        ],
    )]);
    suffix_module_with_imports(&mut caller, &sigs);
    let wgsl = render_module(&caller);
    assert!(wgsl.contains("helper(0u)"), "got: {wgsl}");
}

#[test]
fn own_fn_signatures_shadow_imported_ones() {
    // An imported `helper(x: u32)` is shadowed by the caller's own
    // `helper(x: i32)` — Rust name resolution wins.
    let imported = module(vec![fn_item(
        "helper",
        vec![arg("x", u32_ty())],
        returns(u32_ty()),
        vec![Stmt::Return(Some(ident("x")))],
    )]);
    let sigs = fn_signatures(&imported);

    let mut caller = module(vec![
        fn_item(
            "helper",
            vec![arg("x", i32_ty())],
            returns(i32_ty()),
            vec![Stmt::Return(Some(ident("x")))],
        ),
        fn_item(
            "caller",
            vec![],
            ReturnType::Default,
            vec![
                local("y", None, Some(call("helper", vec![bare_int("0")]))),
                Stmt::Return(None),
            ],
        ),
    ]);
    suffix_module_with_imports(&mut caller, &sigs);
    let wgsl = render_module(&caller);
    assert!(wgsl.contains("helper(0i)"), "got: {wgsl}");
}

#[test]
fn module_wgsl_source_applies_suffixing() {
    // `Module::wgsl_source` normalizes literals; `render_module` on its
    // own stays a pure emitter.
    let m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(bare_int("0")))],
    )]);
    assert!(render_module(&m).contains("return 0;"), "got: {m:?}");
    let wgsl = m.wgsl_source();
    assert!(wgsl.contains("return 0u;"), "got: {wgsl}");
}

#[test]
fn suffix_items_with_imports_anchors_template_calls() {
    // A cross-source template's items reference functions defined in
    // other chunks of the assembled translation unit; seeding from those
    // chunks' signatures anchors the call arguments.
    let helper = module(vec![fn_item(
        "helper",
        vec![arg("x", u32_ty())],
        returns(u32_ty()),
        vec![Stmt::Return(Some(ident("x")))],
    )]);
    let sigs = fn_signatures(&helper);

    let mut template_items = vec![fn_item(
        "caller",
        vec![],
        ReturnType::Default,
        vec![
            local("y", None, Some(call("helper", vec![bare_int("0")]))),
            Stmt::Return(None),
        ],
    )];
    suffix_items_with_imports(&mut template_items, &sigs);
    let wgsl = render_items(&template_items);
    assert!(wgsl.contains("helper(0u)"), "got: {wgsl}");
}

#[test]
fn local_const_anchors_later_expressions() {
    // `const C: u32 = 1; let y = select(C, 0, cond);` — the const's
    // registered type anchors the bare literal through the select
    // value group.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("cond", Type::Scalar(ScalarType::Bool))],
        ReturnType::Default,
        vec![
            Stmt::Const(ItemConst {
                name: "C".to_string(),
                ty: u32_ty(),
                expr: bare_int("1"),
                attrs: vec![],
            }),
            local(
                "y",
                None,
                Some(call(
                    "select",
                    vec![ident("C"), bare_int("0"), ident("cond")],
                )),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn module_const_anchors_expressions() {
    // `const MAX: u32 = 4095;` at module scope anchors expressions in
    // every function body; locals shadow it (the inner `let MAX: i32`
    // wins inside `g`).
    let mut m = module(vec![
        Item::Const(ItemConst {
            name: "MAX".to_string(),
            ty: u32_ty(),
            expr: bare_int("4095"),
            attrs: vec![],
        }),
        fn_item(
            "f",
            vec![arg("x", u32_ty())],
            ReturnType::Default,
            vec![
                local(
                    "y",
                    None,
                    Some(call("min", vec![ident("MAX"), bare_int("1")])),
                ),
                Stmt::Return(None),
            ],
        ),
        fn_item(
            "g",
            vec![arg("cond", Type::Scalar(ScalarType::Bool))],
            ReturnType::Default,
            vec![
                local("MAX", Some(i32_ty()), Some(suffixed_int("7", "i32"))),
                local(
                    "y",
                    None,
                    Some(call(
                        "select",
                        vec![ident("MAX"), bare_int("0"), ident("cond")],
                    )),
                ),
                Stmt::Return(None),
            ],
        ),
    ]);
    let wgsl = render(&mut m);
    // In `f`, the module const (u32) anchors the literal.
    assert!(wgsl.contains("min(MAX, 1u)"), "got: {wgsl}");
    // In `g`, the shadowing local (i32) wins over the module const.
    assert!(wgsl.contains("select(MAX, 0i, cond)"), "got: {wgsl}");
}

#[test]
fn slab_copy_offsets_and_size_are_u32() {
    // The renderer emits a u32 loop counter and adds the offsets to it,
    // so offsets and size flow through u32 arithmetic.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("c", Type::Scalar(ScalarType::Bool))],
        ReturnType::Default,
        vec![Stmt::SlabCopy {
            src: ident("slab"),
            src_offset: call("select", vec![bare_int("0"), bare_int("1"), ident("c")]),
            dest: ident("d"),
            dest_offset: bare_int("0"),
            size: call("select", vec![bare_int("0"), bare_int("1"), ident("c")]),
        }],
    )]);
    let wgsl = render(&mut m);
    assert!(
        wgsl.contains("select(0u, 1u, c)"),
        "slab offsets/size should be u32, got: {wgsl}"
    );
    assert!(wgsl.contains("0u + _i"), "got: {wgsl}");
}

#[test]
fn for_bounds_infer_from_typed_range_bound() {
    // `let n: u32 = 4; for i in 0..n` — parsed loops never carry a
    // loop-variable annotation, so `i`'s type is inferred from `n`.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        ReturnType::Default,
        vec![
            local("n", Some(u32_ty()), Some(suffixed_int("4", "u32"))),
            Stmt::For(ForLoop {
                var: "i".to_string(),
                var_ty: None,
                from: bare_int("0"),
                to: ident("n"),
                inclusive: false,
                body: Block { stmts: vec![] },
            }),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("var i = 0u;"), "got: {wgsl}");
}

#[test]
fn switch_selector_subtree_is_walked() {
    // `match x + select(0, 1, c)` with `x: u32` — the selector subtree
    // is walked (the select anchors through the binary operand), and
    // the case literals inherit the selector's derived type.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("x", u32_ty()), arg("c", Type::Scalar(ScalarType::Bool))],
        ReturnType::Default,
        vec![Stmt::Switch(StmtSwitch {
            selector: Expr::Binary {
                lhs: Box::new(ident("x")),
                op: BinOp::Add,
                rhs: Box::new(call(
                    "select",
                    vec![bare_int("0"), bare_int("1"), ident("c")],
                )),
            },
            arms: vec![SwitchArm {
                selectors: vec![CaseSelector::Literal(Lit::Int {
                    digits: "9".to_string(),
                    suffix: String::new(),
                })],
                body: Block { stmts: vec![] },
            }],
            has_explicit_default: false,
        })],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("select(0u, 1u, c)"), "got: {wgsl}");
    assert!(
        wgsl.contains("9u"),
        "case literal should match selector type, got: {wgsl}"
    );
}

#[test]
fn binary_walks_operands_without_expectation() {
    // `let y = select(x, 0, c) + 1;` — no outer annotation; the select
    // anchors through its own typed argument, and the `1` anchors from
    // the select's type.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("x", u32_ty()), arg("c", Type::Scalar(ScalarType::Bool))],
        ReturnType::Default,
        vec![
            local(
                "y",
                None,
                Some(Expr::Binary {
                    lhs: Box::new(call("select", vec![ident("x"), bare_int("0"), ident("c")])),
                    op: BinOp::Add,
                    rhs: Box::new(bare_int("1")),
                }),
            ),
            Stmt::Return(None),
        ],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("select(x, 0u, c)"), "got: {wgsl}");
    assert!(wgsl.contains("+ 1u"), "got: {wgsl}");
}

#[test]
fn unary_complement_propagates_expectation() {
    // `fn f() -> u32 { !0 }` — the complement preserves the expected
    // u32, so the operand is suffixed.
    let mut m = module(vec![fn_item(
        "f",
        vec![],
        returns(u32_ty()),
        vec![Stmt::Return(Some(Expr::Unary {
            op: UnOp::Complement,
            expr: Box::new(bare_int("0")),
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("0u"), "got: {wgsl}");
}

#[test]
fn array_indexing_base_gets_element_expectation() {
    // `fn f(c: bool) -> u32 { [select(0, 1, c)][0] }` — the return
    // expectation flows into the indexed array literal's elements.
    let mut m = module(vec![fn_item(
        "f",
        vec![arg("c", Type::Scalar(ScalarType::Bool))],
        returns(u32_ty()),
        vec![Stmt::Return(Some(Expr::ArrayIndexing {
            lhs: Box::new(Expr::Array {
                elems: vec![call(
                    "select",
                    vec![bare_int("0"), bare_int("1"), ident("c")],
                )],
            }),
            index: Box::new(bare_int("0")),
        }))],
    )]);
    let wgsl = render(&mut m);
    assert!(wgsl.contains("select(0u, 1u, c)"), "got: {wgsl}");
}
