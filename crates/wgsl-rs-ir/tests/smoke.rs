//! Smoke tests for the IR types, renderer, and type substitution.

use std::collections::HashMap;

use wgsl_rs_ir::*;

fn ident(name: &str) -> Expr {
    Expr::Ident(name.to_string())
}

fn lit_u(n: u32) -> Expr {
    Expr::Lit(Lit::Int {
        digits: n.to_string(),
        suffix: "u32".to_string(),
    })
}

#[test]
fn renders_simple_const() {
    let m = Module {
        name: "test".to_string(),
        items: vec![Item::Const(ItemConst {
            name: "MAX".to_string(),
            ty: Type::Scalar(ScalarType::U32),
            expr: lit_u(42),
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert_eq!(wgsl, "const MAX: u32 = 42u;\n");
}

#[test]
fn renders_simple_function() {
    let m = Module {
        name: "test".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            fn_attrs: FnAttrs::None,
            name: "double".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::Scalar(ScalarType::F32),
                attrs: vec![],
            }],
            return_type: ReturnType::Type {
                annotation: ReturnTypeAnnotation::None,
                ty: Type::Scalar(ScalarType::F32),
            },
            block: Block {
                stmts: vec![Stmt::Expr {
                    expr: Expr::Binary {
                        lhs: Box::new(ident("x")),
                        op: BinOp::Mul,
                        rhs: Box::new(Expr::Lit(Lit::Float {
                            text: "2.0".to_string(),
                        })),
                    },
                    has_semi: false,
                }],
            },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(wgsl.contains("fn double(x: f32) -> f32 {"), "got: {wgsl}");
    assert!(wgsl.contains("return x * 2.0;"), "got: {wgsl}");
}

#[test]
fn renders_compute_entry_point() {
    let m = Module {
        name: "test".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            fn_attrs: FnAttrs::Compute {
                workgroup_size: WorkgroupSize {
                    x: 64,
                    y: None,
                    z: None,
                },
            },
            name: "main".to_string().into(),
            inputs: vec![],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(
        wgsl.contains("@compute @workgroup_size(64) fn main()"),
        "got: {wgsl}"
    );
}

#[test]
fn renders_struct_and_impl() {
    let m = Module {
        name: "test".to_string(),
        items: vec![
            Item::Struct(ItemStruct {
                type_params: vec![],
                name: "Point".to_string(),
                fields: vec![
                    Field {
                        inter_stage_io: vec![],
                        name: "x".to_string(),
                        ty: Type::Scalar(ScalarType::F32),
                        attrs: vec![],
                    },
                    Field {
                        inter_stage_io: vec![],
                        name: "y".to_string(),
                        ty: Type::Scalar(ScalarType::F32),
                        attrs: vec![],
                    },
                ],
                attrs: vec![],
            }),
            Item::Impl(ItemImpl {
                type_params: vec![],
                self_ty: "Point".to_string(),
                items: vec![ImplItem::Fn(ItemFn {
                    type_params: vec![],
                    fn_attrs: FnAttrs::None,
                    name: "x_only".to_string().into(),
                    inputs: vec![FnArg {
                        inter_stage_io: vec![],
                        name: "p".to_string(),
                        ty: Type::Struct {
                            name: "Point".to_string(),
                            type_args: vec![],
                        },
                        attrs: vec![],
                    }],
                    return_type: ReturnType::Type {
                        annotation: ReturnTypeAnnotation::None,
                        ty: Type::Scalar(ScalarType::F32),
                    },
                    block: Block {
                        stmts: vec![Stmt::Expr {
                            expr: Expr::FieldAccess {
                                base: Box::new(ident("p")),
                                field: "x".to_string(),
                            },
                            has_semi: false,
                        }],
                    },
                    attrs: vec![],
                })],
                attrs: vec![],
            }),
        ],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(wgsl.contains("struct Point {"), "got: {wgsl}");
    // `x_only` contains an underscore, so under the robust mangling
    // scheme (issue #112) the joined impl-method name is escaped.
    assert!(wgsl.contains("fn Point__1x_only(p: Point)"), "got: {wgsl}");
}

#[test]
fn renders_struct_expr_positionally() {
    let e = Expr::Struct {
        name: "Point".to_string(),
        type_args: vec![],
        fields: vec![
            FieldValue {
                member: "x".to_string(),
                expr: Expr::Lit(Lit::Float {
                    text: "1.0".to_string(),
                }),
            },
            FieldValue {
                member: "y".to_string(),
                expr: Expr::Lit(Lit::Float {
                    text: "2.0".to_string(),
                }),
            },
        ],
    };
    let m = Module {
        name: "t".to_string(),
        items: vec![Item::Const(ItemConst {
            name: "P".to_string(),
            ty: Type::Struct {
                name: "Point".to_string(),
                type_args: vec![],
            },
            expr: e,
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let out = render_module(&m);
    assert!(out.contains("Point(1.0, 2.0)"), "got: {out}");
}

#[test]
fn renders_enum_with_auto_discriminants() {
    let m = Module {
        name: "t".to_string(),
        items: vec![Item::Enum(ItemEnum {
            name: "Color".to_string(),
            variants: vec![
                EnumVariant {
                    name: "Red".to_string(),
                    discriminant: None,
                },
                EnumVariant {
                    name: "Green".to_string(),
                    discriminant: Some(5),
                },
                EnumVariant {
                    name: "Blue".to_string(),
                    discriminant: None,
                },
            ],
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(wgsl.contains("alias Color = u32;"), "got: {wgsl}");
    assert!(wgsl.contains("const Color_Red: u32 = 0u;"), "got: {wgsl}");
    assert!(wgsl.contains("const Color_Green: u32 = 5u;"), "got: {wgsl}");
    assert!(wgsl.contains("const Color_Blue: u32 = 6u;"), "got: {wgsl}");
}

#[test]
fn substitute_replaces_type_params() {
    let mut m = Module {
        name: "t".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec!["T".to_string()],
            fn_attrs: FnAttrs::None,
            name: "id".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::TypeParam {
                    name: "T".to_string(),
                },
                attrs: vec![],
            }],
            return_type: ReturnType::Type {
                annotation: ReturnTypeAnnotation::None,
                ty: Type::TypeParam {
                    name: "T".to_string(),
                },
            },
            block: Block {
                stmts: vec![Stmt::Expr {
                    expr: ident("x"),
                    has_semi: false,
                }],
            },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), Type::Scalar(ScalarType::F32));
    substitute_types(&mut m, &subst);
    let wgsl = render_module(&m);
    assert!(wgsl.contains("fn id(x: f32) -> f32"), "got: {wgsl}");
    assert!(!wgsl.contains("__TP"), "got: {wgsl}");
}

#[test]
fn substitute_propagates_into_arrays_and_pointers() {
    let mut m = Module {
        name: "t".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec!["T".to_string()],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "p".to_string(),
                ty: Type::Ptr {
                    address_space: AddressSpace::Function,
                    elem: Box::new(Type::Array {
                        elem: Box::new(Type::TypeParam {
                            name: "T".to_string(),
                        }),
                        len: lit_u(4),
                    }),
                },
                attrs: vec![],
            }],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), Type::Scalar(ScalarType::I32));
    substitute_types(&mut m, &subst);
    let wgsl = render_module(&m);
    assert!(
        wgsl.contains("ptr<function, array<i32, 4u>>"),
        "got: {wgsl}"
    );
}

#[test]
fn fn_call_translates_builtin_names() {
    let m = Module {
        name: "t".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::Scalar(ScalarType::F32),
                attrs: vec![],
            }],
            return_type: ReturnType::Type {
                annotation: ReturnTypeAnnotation::None,
                ty: Type::Scalar(ScalarType::F32),
            },
            block: Block {
                stmts: vec![Stmt::Expr {
                    expr: Expr::FnCall {
                        path: FnPath::Ident("inverse_sqrt".to_string()),
                        type_args: vec![],
                        params: vec![ident("x")],
                    },
                    has_semi: false,
                }],
            },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(wgsl.contains("inverseSqrt(x)"), "got: {wgsl}");
}

#[test]
fn slab_read_lowers_to_for_loop() {
    let m = Module {
        name: "t".to_string(),
        items: vec![Item::Fn(ItemFn {
            type_params: vec![],
            fn_attrs: FnAttrs::None,
            name: "f".to_string().into(),
            inputs: vec![],
            return_type: ReturnType::Default,
            block: Block {
                stmts: vec![Stmt::SlabRead {
                    slab: ident("slab"),
                    offset: ident("o"),
                    dest: ident("d"),
                    size: lit_u(4),
                }],
            },
            attrs: vec![],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(
        wgsl.contains("for (var _i: u32 = 0u; _i < 4u; _i++)"),
        "got: {wgsl}"
    );
    assert!(wgsl.contains("d[_i] = slab[o + _i];"), "got: {wgsl}");
}

#[test]
fn attrs_are_not_rendered_in_wgsl() {
    let m = Module {
        name: "test".to_string(),
        items: vec![Item::Struct(ItemStruct {
            type_params: vec![],
            name: "Foo".to_string(),
            fields: vec![Field {
                inter_stage_io: vec![],
                name: "bar".to_string(),
                ty: Type::Scalar(ScalarType::F32),
                attrs: vec![Attribute {
                    path: "derive".to_string(),
                    args: vec!["Clone".to_string()],
                }],
            }],
            attrs: vec![Attribute {
                path: "derive".to_string(),
                args: vec!["SlabItem".to_string()],
            }],
        })],
        attrs: vec![],
    };
    let wgsl = render_module(&m);
    assert!(
        !wgsl.contains("derive"),
        "derive should not appear in WGSL, got: {wgsl}"
    );
    assert!(
        !wgsl.contains("SlabItem"),
        "SlabItem should not appear in WGSL, got: {wgsl}"
    );
    assert!(
        wgsl.contains("struct Foo"),
        "struct Foo should be in WGSL, got: {wgsl}"
    );
}

#[test]
fn substitute_preserves_attrs() {
    let mut m = Module {
        name: "t".to_string(),
        items: vec![Item::Struct(ItemStruct {
            type_params: vec![],
            name: "Pair".to_string(),
            fields: vec![Field {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::TypeParam {
                    name: "T".to_string(),
                },
                attrs: vec![Attribute {
                    path: "my_attr".to_string(),
                    args: vec!["arg1".to_string()],
                }],
            }],
            attrs: vec![Attribute {
                path: "another_attr".to_string(),
                args: vec![],
            }],
        })],
        attrs: vec![],
    };
    let mut subst = HashMap::new();
    subst.insert("T".to_string(), Type::Scalar(ScalarType::F32));
    substitute_types(&mut m, &subst);
    // After substitution, attrs should still be present
    match &m.items[0] {
        Item::Struct(s) => {
            assert_eq!(s.attrs.len(), 1);
            assert_eq!(s.attrs[0].path, "another_attr");
            assert_eq!(s.fields[0].attrs.len(), 1);
            assert_eq!(s.fields[0].attrs[0].path, "my_attr");
            // Type should be substituted
            assert_eq!(s.fields[0].ty, Type::Scalar(ScalarType::F32));
        }
        _ => panic!("expected struct"),
    }
}
