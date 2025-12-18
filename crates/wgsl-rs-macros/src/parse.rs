//! WGSL abstract syntax tree-ish.
//!
//! There's a lot of hand-waving going on here, but that's ok
//! because in practice this stuff is already type checked by Rust at the
//! time it's constructed.
//!
//! The syntax here is the subset of Rust that can be interpreted as WGSL.
//!
use darling::FromMeta;
// HEY!
//
// This module is incomplete at best.
//
// See the WGSL spec
// [subsection](https://gpuweb.github.io/gpuweb/wgsl/#grammar-recursive-descent)
// on grammar for help implementing this module.
use quote::{ToTokens, quote};
use snafu::prelude::*;
use syn::{Ident, Token, parenthesized, parse::Parse, spanned::Spanned};

#[allow(unused_imports)]
use crate::parse::util::in_progress;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(
        display("Encountered unsupported Rust syntax{}",
        if note.is_empty() { ".".into() } else { format!(":\n{note}.")})
    )]
    Unsupported {
        span: proc_macro2::Span,
        note: String,
    },

    #[snafu(display(""))]
    Darling { source: darling::Error },

    #[snafu(display("Encountered currently unsupported Rust syntax.\n  {note}"))]
    CurrentlyUnsupported {
        span: proc_macro2::Span,
        note: &'static str,
    },

    #[snafu(display(
        "Unsupported use of if-then-else, WGSL if statements are a control structure, not an expression."
    ))]
    UnsupportedIfThen { span: proc_macro2::Span },

    #[snafu(display("Non-public item.\n{item} must be public"))]
    Visibility {
        span: proc_macro2::Span,
        item: &'static str,
    },

    #[snafu(display("In progress:\n{message}"))]
    InProgress {
        span: proc_macro2::Span,
        message: String,
    },
}

impl From<darling::Error> for Error {
    fn from(source: darling::Error) -> Self {
        Self::Darling { source }
    }
}

impl From<syn::Error> for Error {
    fn from(value: syn::Error) -> Self {
        UnsupportedSnafu {
            span: value.span(),
            note: value.to_string(),
        }
        .build()
    }
}

impl Error {
    pub fn span(&self) -> proc_macro2::Span {
        match self {
            Error::Unsupported { span, .. } => *span,
            Error::Darling { source } => source.span(),
            Error::CurrentlyUnsupported { span, .. } => *span,
            Error::UnsupportedIfThen { span } => *span,
            Error::InProgress { span, message: _ } => *span,
            Error::Visibility { span, .. } => *span,
        }
    }
}

impl From<Error> for syn::Error {
    fn from(e: Error) -> Self {
        syn::Error::new(e.span(), format!("Parsing error: '{e}'"))
    }
}

#[allow(dead_code)]
mod util {
    use super::*;

    pub fn some_is_unsupported<T: syn::spanned::Spanned>(
        maybe: Option<&T>,
        note: &'static str,
    ) -> Result<(), Error> {
        if let Some(inner) = maybe {
            UnsupportedSnafu {
                span: inner.span(),
                note,
            }
            .fail()
        } else {
            Ok(())
        }
    }

    pub fn in_progress<X, T: Spanned + std::fmt::Debug>(t: &T) -> Result<X, Error> {
        InProgressSnafu {
            span: t.span(),
            message: format!("{t:#?}"),
        }
        .fail()
    }
}

pub enum ScalarType {
    I32,
    U32,
    F32,
    Bool,
}

impl TryFrom<&syn::Ident> for ScalarType {
    type Error = Error;

    fn try_from(value: &syn::Ident) -> Result<Self, Self::Error> {
        Ok(match value.to_string().as_str() {
            "i32" => Self::I32,
            "u32" => Self::U32,
            "f32" => Self::F32,
            "bool" => Self::Bool,
            "usize" => Self::U32,
            other => UnsupportedSnafu {
                span: value.span(),
                note: format!(
                    "Expected i32, u32, f32, bool or usize (u32 in WGSL).\nSaw '{other}'"
                ),
            }
            .fail()?,
        })
    }
}

/// Types.
pub enum Type {
    /// Concrete scalar types:
    /// * i32
    /// * u32
    /// * f32
    /// * bool
    ///
    /// We also support `usize` for compatibility with native Rust.
    Scalar { ty: ScalarType, ident: Ident },

    /// Vector types:
    /// vec{N}<{T}>
    ///   where T is a scalar type
    #[allow(dead_code)]
    Vector {
        elements: u8,
        scalar_ty: ScalarType,
        ident: Ident,
        scalar: Option<(Token![<], Ident, Token![>])>,
    },

    /// Array type: [T; N]
    Array {
        bracket_token: syn::token::Bracket,
        elem: Box<Type>,
        semi_token: Token![;],
        len: Expr,
    },

    /// Struct type: eg. MyStruct
    Struct { ident: Ident },
}

fn split_as_vec(s: &str) -> Option<(&str, &str)> {
    let (_vec, n_prefix) = s.split_once("Vec")?;
    (n_prefix.len() == 2).then_some(())?;
    let split = n_prefix.split_at(1);
    Some(split)
}

impl TryFrom<&syn::Type> for Type {
    type Error = Error;

    fn try_from(ty: &syn::Type) -> Result<Self, Self::Error> {
        let span = ty.span();
        if let syn::Type::Array(syn::TypeArray {
            bracket_token,
            elem,
            semi_token,
            len,
        }) = ty
        {
            // Parse [T; N]
            let elem = Type::try_from(elem.as_ref())?;
            Ok(Type::Array {
                bracket_token: *bracket_token,
                elem: Box::new(elem),
                semi_token: *semi_token,
                len: Expr::try_from(len)?,
            })
        } else if let syn::Type::Path(type_path) = ty {
            util::some_is_unsupported(
                type_path.qself.as_ref(),
                "QSelf not allowed in scalar type",
            )?;
            let segment = type_path.path.segments.first().context(UnsupportedSnafu {
                span: type_path.path.segments.span(),
                note: "Unexpected type path",
            })?;
            let ident = &segment.ident;
            match &segment.arguments {
                syn::PathArguments::None => {
                    // Expect this to be a vector alias, a scalar type, or a struct
                    Ok(match ident.to_string().as_str() {
                        "i32" | "u32" | "f32" | "bool" => Type::Scalar {
                            ty: ScalarType::try_from(ident)?,
                            ident: ident.clone(),
                        },
                        "usize" => Type::Scalar {
                            ty: ScalarType::U32,
                            ident: Ident::new("u32", ident.span()),
                        },
                        other => {
                            // Check for vec
                            if let Some((n, prefix)) = split_as_vec(other) {
                                let elements = match n {
                                    "2" => 2,
                                    "3" => 3,
                                    "4" => 4,
                                    other_n => UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!("Unsupported vector type '{other}'. `{other_n}` must be one of 2, 3, or 4"),
                                    }
                                    .fail()?,
                                };
                                let scalar_ty = match prefix {
                                    "i" => ScalarType::I32,
                                    "u" => ScalarType::U32,
                                    "f" => ScalarType::F32,
                                    "b" => ScalarType::Bool,
                                    other_prefix => UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!("Unsupported vectory type '{other}'. `{other_prefix}` must be one of i, u, f or b")
                                    }.fail()?
                                };
                                Type::Vector {
                                    elements,
                                    scalar_ty,
                                    ident: ident.clone(),
                                    scalar: None,
                                }
                            } else {
                                // We assume this is a struct
                                Type::Struct {
                                    ident: ident.clone(),
                                }
                            }
                        }
                    })
                }
                syn::PathArguments::AngleBracketed(syn::AngleBracketedGenericArguments {
                    colon2_token,
                    lt_token,
                    args,
                    gt_token,
                }) => {
                    // Expect this to be a vector of the form `Vec{N}<{scalar}>`
                    util::some_is_unsupported(
                        colon2_token.as_ref(),
                        "Prefix path syntax unsupported in WGSL",
                    )?;

                    snafu::ensure!(
                        args.len() == 1,
                        UnsupportedSnafu {
                            span: args.span(),
                            note: "Unsupported generics"
                        }
                    );

                    let elements = match ident.to_string().as_str() {
                        "Vec2" => 2,
                        "Vec3" => 3,
                        "Vec4" => 4,
                        _other => UnsupportedSnafu {
                            span: ident.span(),
                            note: "Unsupported vector, must be one of Vec2, Vec3 or Vec4",
                        }
                        .fail()?,
                    };

                    let arg = args.first().expect("checked that len was 1");
                    match arg {
                        syn::GenericArgument::Type(ty) => {
                            if let Type::Scalar {
                                ty: scalar_ty,
                                ident: scalar_ident,
                            } = Type::try_from(ty)?
                            {
                                Ok(Type::Vector {
                                    elements,
                                    scalar_ty,
                                    ident: ident.clone(),
                                    scalar: Some((*lt_token, scalar_ident, *gt_token)),
                                })
                            } else {
                                UnsupportedSnafu {
                                    span: ty.span(),
                                    note: format!(
                                        "Expected concrete scalar type. Saw '{}'",
                                        ty.into_token_stream()
                                    ),
                                }
                                .fail()
                            }
                        }
                        other => UnsupportedSnafu {
                            span: other.span(),
                            note: format!("'{}' is unsupported", other.into_token_stream()),
                        }
                        .fail(),
                    }
                }
                other => UnsupportedSnafu {
                    span: other.span(),
                    note: "Unsupported type",
                }
                .fail(),
            }
        } else {
            UnsupportedSnafu {
                span,
                note: format!("Type is not a path: '{}'", ty.into_token_stream()),
            }
            .fail()
        }
    }
}

/// A literal value.
#[derive(Debug, PartialEq)]
pub enum Lit {
    Bool(syn::LitBool),
    Float(syn::LitFloat),
    Int(syn::LitInt),
}

impl TryFrom<&syn::Lit> for Lit {
    type Error = Error;

    fn try_from(value: &syn::Lit) -> Result<Self, Self::Error> {
        match value {
            syn::Lit::Int(lit_int) => Ok(Lit::Int(lit_int.clone())),
            syn::Lit::Float(lit_float) => Ok(Lit::Float(lit_float.clone())),
            syn::Lit::Bool(lit_bool) => Ok(Lit::Bool(lit_bool.clone())),
            other => UnsupportedSnafu {
                span: other.span(),
                note: format!("{} is not a literal", other.into_token_stream()),
            }
            .fail(),
        }
    }
}

impl std::fmt::Display for Lit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tokens = match self {
            Lit::Bool(lit_bool) => lit_bool.to_token_stream(),
            Lit::Float(lit_float) => lit_float.to_token_stream(),
            Lit::Int(lit_int) => lit_int.to_token_stream(),
        };
        tokens.fmt(f)
    }
}

/// A binary operator: `+` `-` `*`.
pub enum BinOp {
    Add(Token![+]),
    Sub(Token![-]),
    Mul(Token![*]),
    Div(Token![/]),
}

impl TryFrom<&syn::BinOp> for BinOp {
    type Error = Error;

    fn try_from(value: &syn::BinOp) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::BinOp::Add(t) => Self::Add(*t),
            syn::BinOp::Sub(t) => Self::Sub(*t),
            syn::BinOp::Mul(t) => Self::Mul(*t),
            syn::BinOp::Div(t) => Self::Div(*t),
            other => UnsupportedSnafu {
                span: other.span(),
                note: format!(
                    "'{}' is not a supported binary operation.",
                    other.into_token_stream()
                ),
            }
            .fail()?,
        })
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOp::Add(_) => "+",
            BinOp::Sub(_) => "-",
            BinOp::Mul(_) => "*",
            BinOp::Div(_) => "/",
        };
        f.write_str(s)
    }
}

/// A unary operator: "!" or "-"
pub enum UnOp {
    Not(Token![!]),
    Neg(Token![-]),
}

impl TryFrom<&syn::UnOp> for UnOp {
    type Error = Error;

    fn try_from(value: &syn::UnOp) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::UnOp::Not(t) => UnOp::Not(*t),
            syn::UnOp::Neg(t) => UnOp::Neg(*t),
            other => UnsupportedSnafu {
                span: other.span(),
                note: format!("Unsupported unary operator '{}'", other.into_token_stream()),
            }
            .fail()?,
        })
    }
}

pub struct FieldValue {
    pub member: Ident,
    pub colon_token: Option<Token![:]>,
    pub expr: Expr,
}

impl TryFrom<&syn::FieldValue> for FieldValue {
    type Error = Error;

    fn try_from(value: &syn::FieldValue) -> Result<Self, Self::Error> {
        Ok(FieldValue {
            member: match &value.member {
                syn::Member::Named(ident) => ident.clone(),
                unnamed => UnsupportedSnafu {
                    span: unnamed.span(),
                    note: "Unnamed field",
                }
                .fail()?,
            },
            colon_token: value.colon_token,
            expr: Expr::try_from(&value.expr)?,
        })
    }
}

/// WGSL expressions.
pub enum Expr {
    /// A literal value.
    Lit(Lit),
    /// A name for something like a variable or a function.
    ///
    /// Eg. `a` or `foo`
    Ident(syn::Ident),
    /// An array literal: `[expr1, expr2, ...]`
    Array {
        bracket_token: syn::token::Bracket,
        elems: syn::punctuated::Punctuated<Expr, syn::Token![,]>,
    },
    /// An expression enclosed in parentheses.
    ///
    /// `(a + b)`
    Paren {
        paren_token: syn::token::Paren,
        inner: Box<Expr>,
    },
    /// An infix operator like "+", "-" or "*"
    Binary {
        lhs: Box<Expr>,
        op: BinOp,
        rhs: Box<Expr>,
    },
    /// A unary operator like "!" or "-"
    Unary { op: UnOp, expr: Box<Expr> },
    /// An array indexing operation like `lhs[0]`,
    ArrayIndexing {
        lhs: Box<Expr>,
        bracket_token: syn::token::Bracket,
        index: Box<Expr>,
    },
    /// Swizzling.
    Swizzle {
        lhs: Box<Expr>,
        dot_token: Token![.],
        swizzle: Ident,
    },
    /// Type conversion.
    ///
    /// This needs special help because we want to support indexing with u32 and i32
    /// sinc WGSL supports this.
    Cast { lhs: Box<Expr>, ty: Box<Type> },
    /// A function call
    FnCall {
        lhs: Ident,
        paren_token: syn::token::Paren,
        params: syn::punctuated::Punctuated<Expr, syn::Token![,]>,
    },
    /// Struct constructor
    Struct {
        ident: Ident,
        brace_token: syn::token::Brace,
        fields: syn::punctuated::Punctuated<FieldValue, syn::Token![,]>,
    },
    /// Struct field access, e.g. `foo.bar`
    FieldAccess {
        base: Box<Expr>,
        dot_token: Token![.],
        field: Ident,
    },
}

impl TryFrom<&syn::Expr> for Expr {
    type Error = Error;

    fn try_from(value: &syn::Expr) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::Expr::Lit(syn::ExprLit { attrs: _, lit }) => Self::Lit(Lit::try_from(lit)?),
            syn::Expr::Unary(syn::ExprUnary { attrs: _, op, expr }) => {
                let op = UnOp::try_from(op)?;
                let expr = Box::new(Expr::try_from(expr.as_ref())?);
                Self::Unary { op, expr }
            }
            syn::Expr::Path(syn::PatPath {
                attrs: _,
                qself,
                path,
            }) => {
                util::some_is_unsupported(qself.as_ref(), "QSelf is unsupported")?;
                let ident = path.get_ident().context(UnsupportedSnafu {
                    span: path.span(),
                    note: format!("Expected an identifier, saw '{}'", path.into_token_stream()),
                })?;
                Self::Ident(ident.clone())
            }
            syn::Expr::Paren(syn::ExprParen {
                attrs: _,
                paren_token,
                expr,
            }) => {
                let inner = Box::new(Expr::try_from(expr.as_ref())?);
                Self::Paren {
                    paren_token: *paren_token,
                    inner,
                }
            }
            syn::Expr::Array(syn::ExprArray {
                attrs: _,
                bracket_token,
                elems,
            }) => {
                let mut expr_elems = syn::punctuated::Punctuated::new();
                for pair in elems.pairs() {
                    let expr = pair.value();
                    let parsed = Expr::try_from(*expr)?;
                    expr_elems.push_value(parsed);
                    if let Some(comma) = pair.punct() {
                        expr_elems.push_punct(**comma);
                    }
                }
                Self::Array {
                    bracket_token: *bracket_token,
                    elems: expr_elems,
                }
            }
            syn::Expr::Index(syn::ExprIndex {
                attrs: _,
                expr: lhs,
                bracket_token,
                index,
            }) => {
                let lhs = Box::new(Expr::try_from(lhs.as_ref())?);
                let index = Box::new(Expr::try_from(index.as_ref())?);
                Self::ArrayIndexing {
                    lhs,
                    bracket_token: *bracket_token,
                    index,
                }
            }
            syn::Expr::Field(syn::ExprField {
                attrs: _,
                base,
                dot_token,
                member,
            }) => {
                let base = Box::new(Expr::try_from(base.as_ref())?);
                let field = match member {
                    syn::Member::Named(ident) => ident.clone(),
                    unnamed => UnsupportedSnafu {
                        span: unnamed.span(),
                        note: "Unnamed field access is not supported in WGSL struct field access",
                    }
                    .fail()?,
                };
                Self::FieldAccess {
                    base,
                    dot_token: *dot_token,
                    field,
                }
            }
            syn::Expr::MethodCall(syn::ExprMethodCall {
                attrs: _,
                receiver,
                dot_token,
                method,
                turbofish,
                paren_token: _,
                args,
            }) => {
                util::some_is_unsupported(
                    turbofish.as_ref(),
                    "Turbofish is not supported in WGSL",
                )?;
                util::some_is_unsupported(args.first(), "Swizzling cannot accept parameters")?;
                let lhs = Box::new(Expr::try_from(receiver.as_ref())?);
                // Treat as swizzle: receiver.method
                Self::Swizzle {
                    lhs,
                    dot_token: *dot_token,
                    swizzle: method.clone(),
                }
            }
            syn::Expr::Binary(syn::ExprBinary {
                attrs: _,
                left,
                op,
                right,
            }) => Self::Binary {
                lhs: Box::new(Expr::try_from(left.as_ref())?),
                op: BinOp::try_from(op)?,
                rhs: Box::new(Expr::try_from(right.as_ref())?),
            },
            syn::Expr::Cast(syn::ExprCast {
                attrs: _,
                expr: lhs,
                as_token: _,
                ty,
            }) => {
                let lhs = Box::new(Expr::try_from(lhs.as_ref())?);
                let ty = Box::new(Type::try_from(ty.as_ref())?);
                Self::Cast { lhs, ty }
            }
            syn::Expr::Call(syn::ExprCall {
                attrs: _,
                func,
                paren_token,
                args,
            }) => match func.as_ref() {
                syn::Expr::Path(expr_path) => {
                    util::some_is_unsupported(expr_path.qself.as_ref(), "QSelf unsupported")?;
                    let lhs = expr_path
                        .path
                        .get_ident()
                        .context(UnsupportedSnafu {
                            span: expr_path.path.span(),
                            note: "Expected an identifier",
                        })?
                        .clone();
                    let paren_token = *paren_token;
                    let mut params = syn::punctuated::Punctuated::new();
                    for pair in args.pairs() {
                        let expr = pair.value();
                        let param = Expr::try_from(*expr)?;
                        params.push_value(param);
                        if let Some(comma) = pair.punct() {
                            params.push_punct(**comma);
                        }
                    }
                    Self::FnCall {
                        lhs,
                        paren_token,
                        params,
                    }
                }
                other => UnsupportedSnafu {
                    span: other.span(),
                    note: format!(
                        "Unsupported function call syntax: '{}'",
                        other.into_token_stream()
                    ),
                }
                .fail()?,
            },
            syn::Expr::Struct(syn::ExprStruct {
                attrs: _,
                qself,
                path,
                brace_token,
                fields,
                dot2_token,
                rest,
            }) => {
                util::some_is_unsupported(qself.as_ref(), "")?;
                let ident = path.get_ident().context(UnsupportedSnafu {
                    span: path.span(),
                    note: "Struct name cannot be a path",
                })?;
                util::some_is_unsupported(dot2_token.as_ref(), "Default struct construction")?;
                util::some_is_unsupported(rest.as_ref(), "Default struct construction")?;

                let mut parsed_fields = syn::punctuated::Punctuated::new();
                for pair in fields.pairs() {
                    let field = pair.value();
                    let parsed = FieldValue::try_from(*field)?;
                    parsed_fields.push_value(parsed);
                    if let Some(comma) = pair.punct() {
                        parsed_fields.push_punct(**comma);
                    }
                }

                Expr::Struct {
                    ident: ident.clone(),
                    brace_token: *brace_token,
                    fields: parsed_fields,
                }
            }
            other => UnsupportedSnafu {
                span: other.span(),
                note: format!("Unexpected expression '{}'", other.into_token_stream()),
            }
            .fail()?,
        })
    }
}

pub enum ReturnTypeAnnotation {
    None,
    BuiltIn(Ident),
    Location(Lit),
}

pub enum ReturnType {
    Default,
    Type {
        arrow: Token![->],
        annotation: ReturnTypeAnnotation,
        ty: Box<Type>,
    },
}

impl TryFrom<&syn::ReturnType> for ReturnType {
    type Error = Error;

    fn try_from(ret: &syn::ReturnType) -> Result<Self, Self::Error> {
        match ret {
            syn::ReturnType::Default => Ok(ReturnType::Default),
            syn::ReturnType::Type(arrow, ty) => {
                let scalar = Type::try_from(ty.as_ref())?;
                Ok(ReturnType::Type {
                    arrow: *arrow,
                    ty: Box::new(scalar),
                    annotation: ReturnTypeAnnotation::None,
                })
            }
        }
    }
}

pub struct LocalInit {
    pub eq_token: Token![=],
    pub expr: Expr,
}

impl TryFrom<&syn::LocalInit> for LocalInit {
    type Error = Error;

    fn try_from(value: &syn::LocalInit) -> Result<Self, Self::Error> {
        if let Some((else_token, _)) = value.diverge.as_ref() {
            UnsupportedIfThenSnafu {
                span: else_token.span(),
            }
            .fail()?;
        }
        Ok(LocalInit {
            eq_token: value.eq_token,
            expr: Expr::try_from(value.expr.as_ref())?,
        })
    }
}

pub struct Local {
    pub let_token: Token![let],
    /// If `mutability` is `Some`, this is a `var` binding, otherwise this is a `let` binding.
    pub mutability: Option<Token![mut]>,
    pub ident: Ident,
    pub ty: Option<(Token![:], Type)>,
    pub init: Option<LocalInit>,
    pub semi_token: Token![;],
}

impl TryFrom<&syn::Local> for Local {
    type Error = Error;

    fn try_from(value: &syn::Local) -> Result<Self, Self::Error> {
        let let_token = value.let_token;
        let semi_token = value.semi_token;

        struct IdentMutTy(Ident, Option<Token![mut]>, Option<(Token![:], Type)>);

        fn ident_mut_ty(pat: &syn::Pat) -> Result<IdentMutTy, Error> {
            match pat {
                syn::Pat::Ident(syn::PatIdent {
                    attrs: _,
                    by_ref,
                    mutability,
                    ident,
                    subpat,
                }) => {
                    if let Some(by_ref) = by_ref.as_ref() {
                        // WGSL doesn't support `let ref thing = ...;`
                        UnsupportedSnafu {
                            span: by_ref.span(),
                            note: "WGSL does not support 'let ref ...' bindings.",
                        }
                        .fail()?;
                    }

                    if let Some((at, subpat)) = subpat.as_ref() {
                        // WGSL doesn' support `let thing@(...) = ...`
                        let span = at.span().join(subpat.span()).unwrap();
                        UnsupportedSnafu {
                            span,
                            note: "WGSL does not support 'let ... @ ...' bindings.",
                        }
                        .fail()?;
                    }

                    Ok(IdentMutTy(ident.clone(), *mutability, None))
                }
                syn::Pat::Type(syn::PatType {
                    attrs: _,
                    pat,
                    colon_token,
                    ty,
                }) => {
                    let mut output = ident_mut_ty(pat.as_ref())?;
                    output.2 = Some((*colon_token, Type::try_from(ty.as_ref())?));
                    Ok(output)
                }
                _ => UnsupportedSnafu {
                    span: pat.span(),
                    note: format!(
                        "Unsupported pattern in let binding: '{}'",
                        pat.into_token_stream()
                    ),
                }
                .fail(),
            }
        }

        let IdentMutTy(ident, mutability, ty) = ident_mut_ty(&value.pat)?;
        let init = if let Some(init) = &value.init {
            Some(LocalInit::try_from(init)?)
        } else {
            None
        };
        Ok(Local {
            let_token,
            mutability,
            ident,
            ty,
            init,
            semi_token,
        })
    }
}

pub enum Stmt {
    Local(Box<Local>),
    Const(Box<ItemConst>),
    Expr {
        expr: Expr,
        /// If `None`, this expression is a return statement
        semi_token: Option<Token![;]>,
    },
}

impl TryFrom<&syn::Stmt> for Stmt {
    type Error = Error;

    fn try_from(value: &syn::Stmt) -> Result<Self, Self::Error> {
        match value {
            syn::Stmt::Local(local) => Ok(Stmt::Local(Box::new(Local::try_from(local)?))),
            syn::Stmt::Item(item) => match item {
                syn::Item::Const(item_const) => {
                    Ok(Stmt::Const(Box::new(ItemConst::try_from(item_const)?)))
                }
                other => UnsupportedSnafu {
                    span: other.span(),
                    note: format!("Unsupported statement item '{}'", other.into_token_stream()),
                }
                .fail(),
            },
            syn::Stmt::Expr(expr, semi_token) => Ok(Stmt::Expr {
                expr: Expr::try_from(expr)?,
                semi_token: *semi_token,
            }),
            _ => UnsupportedSnafu {
                span: value.span(),
                note: format!("Unsupported statement: '{}'", value.into_token_stream()),
            }
            .fail(),
        }
    }
}

pub struct Block {
    pub brace_token: syn::token::Brace,
    pub stmt: Vec<Stmt>,
}

impl TryFrom<&syn::Block> for Block {
    type Error = Error;

    fn try_from(value: &syn::Block) -> Result<Self, Self::Error> {
        let brace_token = value.brace_token;
        let mut stmts = Vec::new();
        for stmt in &value.stmts {
            stmts.push(Stmt::try_from(stmt)?);
        }
        Ok(Block {
            brace_token,
            stmt: stmts,
        })
    }
}

// TODO: These enums should hold a reference to their Rust span for better error reporting
#[derive(FromMeta)]
#[darling(derive_syn_parse)]
pub enum BuiltIn {
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SubgroupInvocationId,
    SubgroupSize,
    PrimitiveIndex,
    SubgroupId,
    NumSubgroups,
}

pub enum InterpolationType {
    Perspective(syn::Ident),
    Linear(syn::Ident),
    Flat(syn::Ident),
}

impl Parse for InterpolationType {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        Ok(match ident.to_string().as_str() {
            "perspective" => Self::Perspective(ident),
            "linear" => Self::Linear(ident),
            "flat" => Self::Flat(ident),
            other => Err(syn::Error::new(
                ident.span(),
                format!("Unexpected interpolation type '{other}'"),
            ))?,
        })
    }
}

pub enum InterpolationSampling {
    Center(syn::Ident),
    Centroid(syn::Ident),
    Sample(syn::Ident),
    First(syn::Ident),
    Either(syn::Ident),
}

impl Parse for InterpolationSampling {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        Ok(match ident.to_string().as_str() {
            "center" => Self::Center(ident),
            "centroid" => Self::Centroid(ident),
            "sample" => Self::Sample(ident),
            "first" => Self::First(ident),
            "either" => Self::Either(ident),
            other => Err(syn::Error::new(
                ident.span(),
                format!("Unexpected interpolation sampling '{other}'"),
            ))?,
        })
    }
}

/// <https://gpuweb.github.io/gpuweb/wgsl/#interpolation>
pub struct Interpolate {
    pub ty: InterpolationType,
    pub comma_token: Option<Token![,]>,
    pub sampling: Option<InterpolationSampling>,
}

/// Parse the _arguments_ of #[interpolate(type, sampling)].
impl Parse for Interpolate {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty = input.parse()?;
        let comma_token: Option<Token![,]> = input.parse()?;
        let sampling = if comma_token.is_some() {
            Some(input.parse()?)
        } else {
            None
        };
        Ok(Self {
            ty,
            comma_token,
            sampling,
        })
    }
}

/// A shader stage input is a datum provided to the shader stage from upstream in the pipeline.
///
/// Each datum is either a built-in input value, or a user-defined input.
///
/// A shader stage output is a datum the shader provides for further processing
/// downstream in the pipeline. Each datum is either a built-in output value, or
/// a user-defined output.
/// IO attributes are used to establish an object as a shader stage input or a
/// shader stage output, or to further describe the properties of an input or
/// output. The IO attributes are:
///
/// * builtin
///
/// * location
///
/// * blend_src
///
/// * interpolate
///
/// * invariant
///
/// See <https://gpuweb.github.io/gpuweb/wgsl/#stage-inputs-outputs>.
pub enum InterStageIo {
    BuiltIn(BuiltIn),
    /// <https://gpuweb.github.io/gpuweb/wgsl/#location-attr>
    Location(syn::LitInt),
    /// <https://gpuweb.github.io/gpuweb/wgsl/#blend-src-attr>
    /// Strictly output
    ///
    /// Contains a literal value of 0 or 1.
    BlendSrc(syn::LitInt),
    Interpolate(Interpolate),
    /// Strictly output, placed in addition to a `@builtin(position)` output.
    Invariant,
}

impl TryFrom<&syn::Attribute> for InterStageIo {
    type Error = Error;

    fn try_from(value: &syn::Attribute) -> Result<Self, Self::Error> {
        // Only handle outer attributes
        if matches!(value.style, syn::AttrStyle::Inner(_)) {
            return UnsupportedSnafu {
                span: value.span(),
                note: "Inner attributes are not supported for WGSL IO",
            }
            .fail();
        }

        let ident = value.path().get_ident().ok_or_else(|| Error::Unsupported {
            span: value.span(),
            note: "Expected a simple identifier for attribute".to_string(),
        })?;

        match ident.to_string().as_str() {
            "builtin" => {
                let built_in = BuiltIn::from_meta(&value.meta)?;
                Ok(InterStageIo::BuiltIn(built_in))
            }
            "location" => {
                let list = value.meta.require_list()?;
                let lit: syn::LitInt = syn::parse2(list.tokens.clone())?;
                Ok(InterStageIo::Location(lit))
            }
            "blend_src" => {
                // #[blend_source()]
                let list = value.meta.require_list()?;
                let lit: syn::LitInt = syn::parse2(list.tokens.clone())?;
                Ok(InterStageIo::BlendSrc(lit))
            }
            "interpolate" => {
                // #[interpolate(ty, sampling?)]
                let list = value.meta.require_list()?;
                let tokens = list.tokens.clone();
                let interpolate = syn::parse2(tokens)?;
                Ok(InterStageIo::Interpolate(interpolate))
            }
            "invariant" => {
                // #[invariant]
                Ok(InterStageIo::Invariant)
            }
            other => UnsupportedSnafu {
                span: value.span(),
                note: format!("Unknown IO attribute '{other}'"),
            }
            .fail(),
        }
    }
}

pub struct FnArg {
    pub inter_stage_io: Vec<InterStageIo>,
    pub ident: Ident,
    pub colon_token: Token![:],
    pub ty: Type,
}

impl TryFrom<&syn::FnArg> for FnArg {
    type Error = Error;

    fn try_from(value: &syn::FnArg) -> Result<Self, Self::Error> {
        match value {
            syn::FnArg::Receiver(receiver) => CurrentlyUnsupportedSnafu {
                span: receiver.span(),
                note: "wgsl-rs does not yet support &self in fn args.",
            }
            .fail()?,
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => {
                    snafu::ensure!(
                        pat_ident.mutability.is_none(),
                        CurrentlyUnsupportedSnafu {
                            span: pat_ident
                                .mutability
                                .expect("already checked that it's Some")
                                .span(),
                            note: "wgsl-rs does not yet support mutable fn args."
                        }
                    );

                    let ident = pat_ident.ident.clone();

                    let syn::PatType {
                        attrs,
                        pat: _,
                        colon_token: _,
                        ty,
                    } = pat_type;

                    snafu::ensure!(
                        attrs.len() <= 1,
                        UnsupportedSnafu {
                            span: attrs[1].span(),
                            note: "WGSL only supports a single annotation on function parameters."
                        }
                    );
                    let mut inter_stage_io = vec![];
                    for attr in attrs.iter() {
                        inter_stage_io.push(InterStageIo::try_from(attr)?);
                    }

                    let ty = Type::try_from(ty.as_ref())?;

                    Ok(FnArg {
                        inter_stage_io,
                        ident,
                        colon_token: pat_type.colon_token,
                        ty,
                    })
                }
                other => UnsupportedSnafu {
                    span: other.span(),
                    note: format!(
                        "Unsupported pattern in function argument: '{}'",
                        other.into_token_stream()
                    ),
                }
                .fail(),
            },
        }
    }
}

#[derive(Default)]
pub enum FnAttrs {
    #[default]
    None,
    Vertex,
    Fragment,
}

impl TryFrom<&Vec<syn::Attribute>> for FnAttrs {
    type Error = Error;

    fn try_from(value: &Vec<syn::Attribute>) -> Result<Self, Self::Error> {
        for syn::Attribute {
            pound_token: _,
            style,
            bracket_token: _,
            meta,
        } in value.iter()
        {
            if matches!(style, syn::AttrStyle::Inner(_)) {
                continue;
            }

            if let Some(ident) = meta.path().get_ident() {
                match ident.to_string().as_str() {
                    "vertex" => return Ok(FnAttrs::Vertex),
                    "fragment" => return Ok(FnAttrs::Fragment),
                    other => UnsupportedSnafu {
                        span: ident.span(),
                        note: format!("'{other}' is not a supported annotation"),
                    }
                    .fail()?,
                }
            }
        }
        Ok(FnAttrs::None)
    }
}

pub struct ItemFn {
    pub fn_attrs: FnAttrs,
    pub fn_token: Token![fn],
    pub ident: Ident,
    pub paren_token: syn::token::Paren,
    pub inputs: syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
    pub return_type: ReturnType,
    pub block: Block,
}

impl TryFrom<&syn::ItemFn> for ItemFn {
    type Error = Error;

    fn try_from(value: &syn::ItemFn) -> Result<Self, Self::Error> {
        let syn::ItemFn {
            attrs,
            vis,
            sig,
            block,
        } = value;
        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            VisibilitySnafu {
                span: sig.span(),
                item: "Functions"
            }
        );
        let fn_attrs = FnAttrs::try_from(attrs)?;
        let mut inputs = syn::punctuated::Punctuated::new();
        for pair in sig.inputs.pairs() {
            let input = pair.value();
            let arg = FnArg::try_from(*input)?;
            inputs.push_value(arg);
            if let Some(comma) = pair.punct() {
                inputs.push_punct(**comma);
            }
        }

        let mut return_type = ReturnType::try_from(&sig.output)?;
        match &mut return_type {
            ReturnType::Default => {}
            ReturnType::Type {
                arrow: _,
                annotation,
                ty,
            } => {
                if let Type::Vector {
                    elements: 4,
                    scalar_ty: ScalarType::F32,
                    ident,
                    ..
                } = ty.as_ref()
                {
                    *annotation = match fn_attrs {
                        FnAttrs::Vertex => {
                            ReturnTypeAnnotation::BuiltIn(Ident::new("position", ident.span()))
                        }
                        FnAttrs::Fragment => ReturnTypeAnnotation::Location(Lit::Int(
                            syn::LitInt::new("0", ident.span()),
                        )),
                        _ => ReturnTypeAnnotation::None,
                    }
                }
            }
        }

        Ok(ItemFn {
            fn_attrs,
            fn_token: sig.fn_token,
            ident: sig.ident.clone(),
            paren_token: sig.paren_token,
            inputs,
            return_type,
            block: Block::try_from(block.as_ref())?,
        })
    }
}

pub struct ItemConst {
    pub const_token: Token![const],
    pub ident: Ident,
    pub colon_token: Token![:],
    pub ty: Type,
    pub eq_token: Token![=],
    pub expr: Expr,
    pub semi_token: Token![;],
}

impl TryFrom<&syn::ItemConst> for ItemConst {
    type Error = Error;

    fn try_from(value: &syn::ItemConst) -> Result<Self, Self::Error> {
        let syn::ItemConst {
            attrs: _,
            vis: _,
            const_token,
            ident,
            generics: _,
            colon_token,
            ty,
            eq_token,
            expr,
            semi_token,
        } = value;
        Ok(ItemConst {
            const_token: *const_token,
            ident: ident.clone(),
            colon_token: *colon_token,
            ty: Type::try_from(ty.as_ref())?,
            eq_token: *eq_token,
            expr: Expr::try_from(expr.as_ref())?,
            semi_token: *semi_token,
        })
    }
}

/// A WGSL "module".
pub struct ItemMod {
    #[allow(dead_code)]
    pub ident: Ident,
    pub content: Vec<Item>,
}

impl TryFrom<&syn::ItemMod> for ItemMod {
    type Error = Error;

    fn try_from(item_mod: &syn::ItemMod) -> Result<Self, Self::Error> {
        let ident = item_mod.ident.clone();
        let mut content = Vec::new();

        // Only handle inline modules (with content)
        if let Some((_, items)) = &item_mod.content {
            for item in items {
                content.push(Item::try_from(item)?);
            }
            Ok(ItemMod { ident, content })
        } else {
            // For now, error on modules without inline content
            UnsupportedSnafu {
                span: item_mod.span(),
                note: "Modules without inline content are not supported.",
            }
            .fail()
        }
    }
}

impl ItemMod {
    pub fn imports(&self, wgsl_rs_crate_path: &syn::Path) -> Vec<proc_macro2::TokenStream> {
        fn is_wgsl_std(wgsl_rs_crate_path: &syn::Path, path: &syn::Path) -> bool {
            let wgsl_std = {
                let mut std = wgsl_rs_crate_path.clone();
                if !std.segments.empty_or_trailing() {
                    std.segments.push_punct(syn::token::PathSep::default());
                }
                std.segments.push_value(syn::PathSegment {
                    ident: quote::format_ident!("std"),
                    arguments: syn::PathArguments::None,
                });
                std
            };
            let wgsl_std = wgsl_std.into_token_stream().to_string();
            let path = path.into_token_stream().to_string();
            wgsl_std == path
        }

        let mut imports = vec![];
        for item in self.content.iter() {
            if let Item::Use(use_item) = item {
                for path in use_item.modules.iter() {
                    // If this import is `use wgsl_rs::std::*;`, skip any importing
                    // on the WGSL side.
                    if is_wgsl_std(wgsl_rs_crate_path, path) {
                        continue;
                    }

                    imports.push(quote! {
                        #path::WGSL_MODULE
                    });
                }
            }
        }
        imports
    }
}

/// A WGSL use/import statement.
/// Only supports glob imports of an entire module, e.g. `use foo::bar::*;`
pub struct ItemUse {
    pub modules: Vec<syn::Path>,
}

impl TryFrom<&syn::UseTree> for ItemUse {
    type Error = Error;

    fn try_from(value: &syn::UseTree) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::UseTree::Path(syn::UsePath {
                ident,
                colon2_token: _,
                tree,
            }) => {
                let segment = syn::PathSegment {
                    ident: ident.clone(),
                    arguments: syn::PathArguments::None,
                };
                let mut item_use = Self::try_from(tree.as_ref())?;
                // Prefix this module on the remaining paths
                for module in item_use.modules.iter_mut() {
                    module.segments.insert(0, segment.clone());
                }

                if item_use.modules.is_empty() {
                    item_use.modules.push(syn::Path {
                        leading_colon: None,
                        segments: syn::punctuated::Punctuated::from_iter(Some(syn::PathSegment {
                            ident: ident.clone(),
                            arguments: syn::PathArguments::None,
                        })),
                    });
                }

                item_use
            }
            syn::UseTree::Name(use_name) => UnsupportedSnafu {
                span: use_name.span(),
                note: "Only glob imports of modules are supported (e.g. use foo::*;).",
            }
            .fail()?,
            syn::UseTree::Rename(use_rename) => UnsupportedSnafu {
                span: use_rename.span(),
                note: "Renaming in use statements is not supported.",
            }
            .fail()?,
            syn::UseTree::Glob(_use_glob) => Self { modules: vec![] },
            syn::UseTree::Group(use_group) => UnsupportedSnafu {
                span: use_group.span(),
                note: "Grouped use statements are not supported.",
            }
            .fail()?,
        })
    }
}

// pub struct ItemStatic {}

// impl TryFrom<&syn::ItemStatic> for ItemStatic {
//     type Error = Error;

//     fn try_from(value: &syn::ItemStatic) -> Result<Self, Self::Error> {
//         let syn::ItemStatic {
//             attrs,
//             vis,
//             static_token,
//             mutability,
//             ident,
//             colon_token,
//             ty,
//             eq_token,
//             expr,
//             semi_token,
//         } = value;

//         Ok(ItemStatic {})
//     }
// }

// impl ToTokens for ItemStatic {
//     fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {}
// }
// Use a custom parser for the macro arguments: `group($group), binding($binding), $name : $ty`
pub(crate) struct UniformArgs {
    pub group: syn::LitInt,
    pub binding: syn::LitInt,

    pub name: syn::Ident,
    pub colon_token: Token![:],
    pub ty: syn::Type,
}

impl syn::parse::Parse for UniformArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // group
        let _group_ident = input.parse::<syn::Ident>()?;
        let content;
        let _group_paren_token = parenthesized!(content in input);
        let group = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        // binding
        let _binding_ident = input.parse::<syn::Ident>()?;
        let content;
        let _binding_paren_token = parenthesized!(content in input);
        let binding = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        let name: syn::Ident = input.parse()?;
        let colon_token: syn::Token![:] = input.parse()?;
        let ty: syn::Type = input.parse()?;

        Ok(UniformArgs {
            group,
            binding,
            name,
            colon_token,
            ty,
        })
    }
}

pub struct ItemUniform {
    pub group: syn::LitInt,
    pub binding: syn::LitInt,
    pub name: syn::Ident,
    pub colon_token: Token![:],
    pub ty: Type,
}

impl TryFrom<&syn::ItemMacro> for ItemUniform {
    type Error = Error;

    fn try_from(item_macro: &syn::ItemMacro) -> Result<Self, Self::Error> {
        // Ensure it's the "uniform" macro
        if item_macro
            .mac
            .path
            .get_ident()
            .map(|id| id != "uniform")
            .unwrap_or(true)
        {
            return UnsupportedSnafu {
                span: item_macro.span(),
                note: "Only 'uniform!' macro is supported as a uniform declaration.",
            }
            .fail();
        }

        let args = syn::parse2::<UniformArgs>(item_macro.mac.tokens.clone()).map_err(|e| {
            Error::Unsupported {
                span: item_macro.span(),
                note: format!("{e}"),
            }
        })?;

        Ok(ItemUniform {
            group: args.group,
            binding: args.binding,
            name: args.name,
            colon_token: args.colon_token,
            ty: Type::try_from(&args.ty)?,
        })
    }
}

pub struct Field {
    pub inter_stage_io: Vec<InterStageIo>,
    pub ident: Ident,
    pub colon_token: Option<Token![:]>,
    pub ty: Type,
}

impl TryFrom<&syn::Field> for Field {
    type Error = Error;

    fn try_from(value: &syn::Field) -> Result<Self, Self::Error> {
        let ident = value
            .ident
            .clone()
            .expect("only named fields are supported, and we checked for that before parsing this");
        let colon_token = value.colon_token;
        let ty = Type::try_from(&value.ty)?;
        let mut inter_stage_io = vec![];
        for attr in value.attrs.iter() {
            inter_stage_io.push(InterStageIo::try_from(attr)?);
        }
        Ok(Field {
            inter_stage_io,
            ident,
            colon_token,
            ty,
        })
    }
}

pub struct FieldsNamed {
    pub brace_token: syn::token::Brace,
    pub named: syn::punctuated::Punctuated<Field, Token![,]>,
}

impl TryFrom<&syn::FieldsNamed> for FieldsNamed {
    type Error = Error;

    fn try_from(value: &syn::FieldsNamed) -> Result<Self, Self::Error> {
        let brace_token = value.brace_token;
        let mut named = syn::punctuated::Punctuated::new();
        for pair in value.named.pairs() {
            let field = pair.value();
            let parsed = Field::try_from(*field)?;
            named.push_value(parsed);
            if let Some(comma) = pair.punct() {
                named.push_punct(**comma);
            }
        }
        Ok(FieldsNamed { brace_token, named })
    }
}

pub struct ItemStruct {
    pub struct_token: Token![struct],
    pub ident: Ident,
    pub fields: FieldsNamed,
}

impl TryFrom<&syn::ItemStruct> for ItemStruct {
    type Error = Error;

    fn try_from(value: &syn::ItemStruct) -> Result<Self, Self::Error> {
        let syn::ItemStruct {
            attrs: _,
            vis,
            struct_token,
            ident,
            generics,
            fields,
            semi_token: _,
        } = value;

        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            VisibilitySnafu {
                span: struct_token.span(),
                item: "Structs"
            }
        );
        snafu::ensure!(
            generics.lt_token.is_none(),
            UnsupportedSnafu {
                span: generics.span(),
                note: "Generics are not supported"
            }
        );
        let fields = match fields {
            syn::Fields::Named(fields_named) => fields_named,
            syn::Fields::Unnamed(fields_unnamed) => UnsupportedSnafu {
                span: fields_unnamed.span(),
                note: "WGSL only supports named fields",
            }
            .fail()?,
            syn::Fields::Unit => UnsupportedSnafu {
                span: ident.span(),
                note: "WGSL only supports named fields",
            }
            .fail()?,
        };
        let fields = FieldsNamed::try_from(fields)?;
        Ok(ItemStruct {
            struct_token: *struct_token,
            ident: ident.clone(),
            fields,
        })
    }
}

/// WGSL items that may appear in a "module" or scope.
pub enum Item {
    Const(Box<ItemConst>),
    Uniform(Box<ItemUniform>),
    Fn(ItemFn),
    Mod(ItemMod),
    Use(ItemUse),
    Struct(ItemStruct),
}

impl TryFrom<&syn::Item> for Item {
    type Error = Error;

    fn try_from(value: &syn::Item) -> Result<Self, Self::Error> {
        match value {
            syn::Item::Mod(item_mod) => Ok(Item::Mod(ItemMod::try_from(item_mod)?)),
            syn::Item::Macro(item_macro) => {
                // Check the macro ident to key which parser to use
                let maybe_ident = item_macro.mac.path.get_ident().map(|id| id.to_string());
                match maybe_ident.as_deref() {
                    Some("uniform") => {
                        Ok(Item::Uniform(Box::new(ItemUniform::try_from(item_macro)?)))
                    }
                    other => UnsupportedSnafu {
                        span: item_macro.ident.span(),
                        note: format!(
                            "Unknown macro '{other:?}' doesn't expand into WGSL\n\
                            Seen as '{item_macro:#?}'",
                        ),
                    }
                    .fail()?,
                }
            }
            syn::Item::Const(item_const) => {
                Ok(Item::Const(Box::new(ItemConst::try_from(item_const)?)))
            }
            syn::Item::Fn(item_fn) => Ok(Item::Fn(item_fn.try_into()?)),
            syn::Item::Use(syn::ItemUse {
                attrs: _,
                vis: _,
                use_token: _,
                leading_colon: _,
                tree,
                semi_token: _,
            }) => Ok(Item::Use(ItemUse::try_from(tree)?)),
            syn::Item::Struct(item_struct) => Ok(Item::Struct(ItemStruct::try_from(item_struct)?)),
            _ => UnsupportedSnafu {
                span: value.span(),
                note: format!(
                    "Unsupported item: '{}'\nSeen as '{}...'",
                    value.into_token_stream(),
                    format!("{value:?}")
                        .split('(')
                        .next()
                        .expect("split always returns at least one on a non-empty string")
                        .split('{')
                        .next()
                        .expect("same")
                ),
            }
            .fail(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::code_gen::GenerateCode;

    #[test]
    fn parse_lit_bool() {
        let lit: syn::Lit = syn::parse_str("true").unwrap();
        let lit = Lit::try_from(&lit).unwrap();
        assert_eq!("true", &lit.to_string());
    }

    #[test]
    fn parse_lit_float() {
        let lit: syn::Lit = syn::parse_str("3.1415").unwrap();
        let lit = Lit::try_from(&lit).unwrap();
        assert_eq!("3.1415", &lit.to_string());
    }

    #[test]
    fn parse_lit_int() {
        let lit: syn::Lit = syn::parse_str("666").unwrap();
        let lit = Lit::try_from(&lit).unwrap();
        assert_eq!("666", &lit.to_string());
    }

    #[test]
    fn parse_expr_binary() {
        let expr: syn::Expr = syn::parse_str("333 +  333").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("333 + 333", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_ident() {
        let expr: syn::Expr = syn::parse_str("333 + TIMES").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("333 + TIMES", &expr.to_wgsl());
    }

    #[test]
    fn parse_vec4_f32_type() {
        let ty: syn::Type = syn::parse_str("Vec4<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("vec4 < f32 >", &ty.to_wgsl());
    }

    #[test]
    fn parse_array_type() {
        let ty: syn::Type = syn::parse_str("[f32; 4]").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("array <f32 , 4 >", &ty.to_wgsl());
    }
}
