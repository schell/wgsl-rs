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
        })],
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
            name: "double".to_string(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::Scalar(ScalarType::F32),
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
        })],
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
            name: "main".to_string(),
            inputs: vec![],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
        })],
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
                    },
                    Field {
                        inter_stage_io: vec![],
                        name: "y".to_string(),
                        ty: Type::Scalar(ScalarType::F32),
                    },
                ],
            }),
            Item::Impl(ItemImpl {
                type_params: vec![],
                self_ty: "Point".to_string(),
                items: vec![ImplItem::Fn(ItemFn {
                    type_params: vec![],
                    fn_attrs: FnAttrs::None,
                    name: "x_only".to_string(),
                    inputs: vec![FnArg {
                        inter_stage_io: vec![],
                        name: "p".to_string(),
                        ty: Type::Struct {
                            name: "Point".to_string(),
                            type_args: vec![],
                        },
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
                })],
            }),
        ],
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
        })],
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
        })],
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
            name: "id".to_string(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::TypeParam {
                    name: "T".to_string(),
                },
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
        })],
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
            name: "f".to_string(),
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
            }],
            return_type: ReturnType::Default,
            block: Block { stmts: vec![] },
        })],
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
            name: "f".to_string(),
            inputs: vec![FnArg {
                inter_stage_io: vec![],
                name: "x".to_string(),
                ty: Type::Scalar(ScalarType::F32),
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
        })],
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
            name: "f".to_string(),
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
        })],
    };
    let wgsl = render_module(&m);
    assert!(
        wgsl.contains("for (var _i: u32 = 0u; _i < 4u; _i++)"),
        "got: {wgsl}"
    );
    assert!(wgsl.contains("d[_i] = slab[o + _i];"), "got: {wgsl}");
}
