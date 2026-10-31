//! Unit tests for const-generic substitution (`substitute_consts`) — the
//! const parallel of `substitute_types` (see smoke.rs for the type side).
//!
//! Const values are `i64` (signed const generics): non-negative values
//! must render byte-identically to the pre-signed output, and negative
//! values must render as parenthesized literals so they are unambiguous
//! in every expression context. A negative value must never reach an
//! array-length position (rustc guards that path for macro-authored
//! code by requiring array sizes to evaluate to `usize`; the defensive
//! check exists for hand-rolled IR).

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

fn consts_of(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
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
