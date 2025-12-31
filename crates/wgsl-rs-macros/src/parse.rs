//! WGSL abstract syntax tree-ish.
//!
//! There's a lot of hand-waving going on here, but that's ok
//! because in practice this stuff is already type checked by Rust at the
//! time it's constructed.
//!
//! The syntax here is the subset of Rust that can be interpreted as WGSL.
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

    #[snafu(display("Encountered currently unsupported Rust syntax.\n  {note}"))]
    CurrentlyUnsupported {
        span: proc_macro2::Span,
        note: &'static str,
    },

    #[snafu(display(
        "Unsupported use of if-then-else, WGSL if statements are a control structure, not an \
         expression."
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
pub(crate) mod util {
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

/// Converts a SCREAMING_CASE or PascalCase identifier to snake_case.
pub fn to_snake_case(s: &str) -> String {
    // For SCREAMING_CASE (all uppercase with underscores), just lowercase it
    if s.chars()
        .all(|c| c.is_uppercase() || c == '_' || c.is_ascii_digit())
    {
        return s.to_lowercase();
    }

    // For PascalCase or camelCase (including acronyms like HTTPServer)
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut prev_is_upper = false;
    let mut prev_is_alnum = false;

    while let Some(ch) = chars.next() {
        let next = chars.peek().copied();

        let is_upper = ch.is_uppercase();
        let is_lower = ch.is_lowercase();
        let is_digit = ch.is_ascii_digit();

        // Decide if we need an underscore before this uppercase char.
        // - Transition from lower/digit to upper: fooBar -> foo_bar
        // - End of an acronym before a lowercase: HTTPServer -> http_server
        let boundary_before = if is_upper {
            let next_is_lower = next.map(|c| c.is_lowercase()).unwrap_or(false);
            (prev_is_alnum && !prev_is_upper) || (prev_is_upper && next_is_lower)
        } else {
            false
        };

        if boundary_before && !result.ends_with('_') {
            result.push('_');
        }

        // Normalize uppercase to lowercase; keep other chars as-is.
        if is_upper {
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }

        prev_is_upper = is_upper;
        prev_is_alnum = is_upper || is_lower || is_digit;
    }
    result
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

    /// Matrix types:
    /// mat{N}x{N}<f32>
    ///   where N is 2, 3, or 4
    /// Only f32 matrices are supported (matching WGSL).
    #[allow(dead_code)]
    Matrix {
        size: u8,
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
    let (_vec, n_suffix) = s.split_once("Vec")?;
    (n_suffix.len() == 2).then_some(())?;
    let split = n_suffix.split_at(1);
    Some(split)
}

fn split_as_mat(s: &str) -> Option<(&str, &str)> {
    let (_mat, n_suffix) = s.split_once("Mat")?;
    (n_suffix.len() == 2).then_some(())?;
    let split = n_suffix.split_at(1);
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
                            if let Some((n, suffix)) = split_as_vec(other) {
                                let elements = match n {
                                    "2" => 2,
                                    "3" => 3,
                                    "4" => 4,
                                    other_n => UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!(
                                            "Unsupported vector type '{other}'. `{other_n}` must \
                                             be one of 2, 3, or 4"
                                        ),
                                    }
                                    .fail()?,
                                };
                                let scalar_ty = match suffix {
                                    "i" => ScalarType::I32,
                                    "u" => ScalarType::U32,
                                    "f" => ScalarType::F32,
                                    "b" => ScalarType::Bool,
                                    other_suffix => UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!(
                                            "Unsupported vector type '{other}'. `{other_suffix}` \
                                             must be one of i, u, f or b"
                                        ),
                                    }
                                    .fail()?,
                                };
                                Type::Vector {
                                    elements,
                                    scalar_ty,
                                    ident: ident.clone(),
                                    scalar: None,
                                }
                            } else if let Some((n, suffix)) = split_as_mat(other) {
                                // Check for matrix alias (Mat2f, Mat3f, Mat4f)
                                let size = match n {
                                    "2" => 2,
                                    "3" => 3,
                                    "4" => 4,
                                    other_n => UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!(
                                            "Unsupported matrix type '{other}'. `{other_n}` must \
                                             be one of 2, 3, or 4"
                                        ),
                                    }
                                    .fail()?,
                                };
                                // Only f32 matrices are supported in WGSL
                                if suffix != "f" {
                                    UnsupportedSnafu {
                                        span: ident.span(),
                                        note: format!(
                                            "Unsupported matrix type '{other}'. Only f32 matrices \
                                             are supported (Mat2f, Mat3f, Mat4f)"
                                        ),
                                    }
                                    .fail()?
                                }
                                Type::Matrix {
                                    size,
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
                    // Expect this to be a vector or matrix of the form `Vec{N}<{scalar}>` or
                    // `Mat{N}<f32>`
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

                    let ident_str = ident.to_string();

                    // Check for vector types
                    let elements = match ident_str.as_str() {
                        "Vec2" => Some(2u8),
                        "Vec3" => Some(3),
                        "Vec4" => Some(4),
                        _ => None,
                    };

                    // Check for matrix types
                    let matrix_size = match ident_str.as_str() {
                        "Mat2" => Some(2u8),
                        "Mat3" => Some(3),
                        "Mat4" => Some(4),
                        _ => None,
                    };

                    let arg = args.first().expect("checked that len was 1");
                    match arg {
                        syn::GenericArgument::Type(ty) => {
                            if let Type::Scalar {
                                ty: scalar_ty,
                                ident: scalar_ident,
                            } = Type::try_from(ty)?
                            {
                                if let Some(elements) = elements {
                                    // Vector type
                                    Ok(Type::Vector {
                                        elements,
                                        scalar_ty,
                                        ident: ident.clone(),
                                        scalar: Some((*lt_token, scalar_ident, *gt_token)),
                                    })
                                } else if let Some(size) = matrix_size {
                                    // Matrix type - only f32 is supported
                                    if !matches!(scalar_ty, ScalarType::F32) {
                                        UnsupportedSnafu {
                                            span: ty.span(),
                                            note: "Only f32 matrices are supported in WGSL",
                                        }
                                        .fail()?
                                    }
                                    Ok(Type::Matrix {
                                        size,
                                        ident: ident.clone(),
                                        scalar: Some((*lt_token, scalar_ident, *gt_token)),
                                    })
                                } else {
                                    UnsupportedSnafu {
                                        span: ident.span(),
                                        note: "Unsupported generic type, must be one of Vec2, \
                                               Vec3, Vec4, Mat2, Mat3 or Mat4",
                                    }
                                    .fail()
                                }
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

/// A function path - either a simple identifier or a type-qualified path.
///
/// Used in function calls to support both `foo(args)` and `Type::method(args)`.
pub enum FnPath {
    /// Simple function call: `foo()`
    Ident(Ident),
    /// Type-qualified call: `Light::attenuate()`
    ///
    /// This is used for calling methods defined in impl blocks.
    /// In WGSL output, this becomes `Light_attenuate()`.
    TypeMethod {
        ty: Ident,
        colon2_token: Token![::],
        method: Ident,
    },
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
    /// This needs special help because we want to support indexing with u32 and
    /// i32 since WGSL supports this.
    Cast { lhs: Box<Expr>, ty: Box<Type> },
    /// A function call.
    ///
    /// Supports both simple calls like `foo(args)` and type-qualified calls
    /// like `Light::attenuate(args)` for calling impl block methods.
    FnCall {
        path: FnPath,
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
                util::some_is_unsupported(
                    args.first(),
                    "Method call syntax (receiver.method(args)) is only supported for swizzles \
                     (e.g., v.xyz()). For struct methods, use explicit path syntax: \
                     Type::method(receiver, args)",
                )?;
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

                    let syn_path = &expr_path.path;
                    let fn_path = if let Some(ident) = syn_path.get_ident() {
                        // Simple function call: foo(args)
                        FnPath::Ident(ident.clone())
                    } else if syn_path.segments.len() == 2 {
                        // Type::method call: Light::attenuate(args)
                        let ty = syn_path.segments[0].ident.clone();
                        let method = syn_path.segments[1].ident.clone();
                        // Check for no generics on segments
                        for seg in &syn_path.segments {
                            if !matches!(seg.arguments, syn::PathArguments::None) {
                                return UnsupportedSnafu {
                                    span: seg.arguments.span(),
                                    note: "generic arguments in function paths are not supported",
                                }
                                .fail();
                            }
                        }
                        FnPath::TypeMethod {
                            ty,
                            colon2_token: Token![::](syn_path.segments[0].ident.span()),
                            method,
                        }
                    } else {
                        return UnsupportedSnafu {
                            span: syn_path.span(),
                            note: "only simple function calls or Type::method calls are supported",
                        }
                        .fail();
                    };

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
                        path: fn_path,
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

// TODO: BuiltIn and Location should be built when a vertex or fragment shader
// are annotated with certain attributes:
//
// ```rust
// #[vertex(return(location(0)))]
// fn my_vertex() -> Vec4f {...}
//
// #[fragment(return(builtin(...)))]
// fn my_fragment() -> Vec4f {...}
// ```
//
// Then we can remove this #[allow(dead_code)]
#[allow(dead_code)]
pub enum ReturnTypeAnnotation {
    None,
    BuiltIn(Ident),
    DefaultBuiltInPosition,
    Location { ident: Ident, lit: Lit },
    DefaultLocation,
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
    /// If `mutability` is `Some`, this is a `var` binding, otherwise this is a
    /// `let` binding.
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
                        let span = at.span().join(subpat.span()).expect("same file");
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

/// WGSL built-in annotations for shader inputs and outputs.
pub enum BuiltIn {
    VertexIndex(Ident),
    InstanceIndex(Ident),
    Position(Ident),
    FrontFacing(Ident),
    FragDepth(Ident),
    SampleIndex(Ident),
    SampleMask(Ident),
    LocalInvocationId(Ident),
    LocalInvocationIndex(Ident),
    GlobalInvocationId(Ident),
    WorkgroupId(Ident),
    NumWorkgroups(Ident),
    SubgroupInvocationId(Ident),
    SubgroupSize(Ident),
    PrimitiveIndex(Ident),
    SubgroupId(Ident),
    NumSubgroups(Ident),
}

impl TryFrom<&Ident> for BuiltIn {
    type Error = Error;

    fn try_from(ident: &Ident) -> Result<Self, Self::Error> {
        Ok(match ident.to_string().as_str() {
            "vertex_index" => BuiltIn::VertexIndex(ident.clone()),
            "instance_index" => BuiltIn::InstanceIndex(ident.clone()),
            "position" => BuiltIn::Position(ident.clone()),
            "front_facing" => BuiltIn::FrontFacing(ident.clone()),
            "frag_depth" => BuiltIn::FragDepth(ident.clone()),
            "sample_index" => BuiltIn::SampleIndex(ident.clone()),
            "sample_mask" => BuiltIn::SampleMask(ident.clone()),
            "local_invocation_id" => BuiltIn::LocalInvocationId(ident.clone()),
            "local_invocation_index" => BuiltIn::LocalInvocationIndex(ident.clone()),
            "global_invocation_id" => BuiltIn::GlobalInvocationId(ident.clone()),
            "workgroup_id" => BuiltIn::WorkgroupId(ident.clone()),
            "num_workgroups" => BuiltIn::NumWorkgroups(ident.clone()),
            "subgroup_invocation_id" => BuiltIn::SubgroupInvocationId(ident.clone()),
            "subgroup_size" => BuiltIn::SubgroupSize(ident.clone()),
            "primitive_index" => BuiltIn::PrimitiveIndex(ident.clone()),
            "subgroup_id" => BuiltIn::SubgroupId(ident.clone()),
            "num_subgroups" => BuiltIn::NumSubgroups(ident.clone()),
            other => UnsupportedSnafu {
                span: ident.span(),
                note: format!("'{other}' is not a known builtin"),
            }
            .fail()?,
        })
    }
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

/// A shader stage input is a datum provided to the shader stage from upstream
/// in the pipeline.
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
    BuiltIn {
        ident: Ident,
        paren_token: syn::token::Paren,
        inner: BuiltIn,
    },
    /// <https://gpuweb.github.io/gpuweb/wgsl/#location-attr>
    Location {
        ident: Ident,
        paren_token: syn::token::Paren,
        inner: syn::LitInt,
    },
    /// <https://gpuweb.github.io/gpuweb/wgsl/#blend-src-attr>
    /// Strictly output
    ///
    /// Contains a literal value of 0 or 1.
    BlendSrc {
        // "blend_src"
        ident: Ident,
        paren_token: syn::token::Paren,
        // 0 or 1
        lit: syn::LitInt,
    },
    Interpolate {
        ident: Ident,
        paren_token: syn::token::Paren,
        inner: Interpolate,
    },
    /// Strictly output, placed in addition to a `@builtin(position)` output.
    Invariant(Ident),
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

        fn as_paren(delim: &syn::MacroDelimiter) -> Result<syn::token::Paren, Error> {
            if let syn::MacroDelimiter::Paren(p) = delim {
                Ok(*p)
            } else {
                UnsupportedSnafu {
                    span: delim.span().open(),
                    note: "Wrong delimiter",
                }
                .fail()?
            }
        }

        match ident.to_string().as_str() {
            "builtin" => {
                let list = value.meta.require_list()?;
                let inner_ident: Ident = syn::parse2(list.tokens.clone())?;
                let built_in = BuiltIn::try_from(&inner_ident)?;
                Ok(InterStageIo::BuiltIn {
                    ident: ident.clone(),
                    paren_token: as_paren(&list.delimiter)?,
                    inner: built_in,
                })
            }
            "location" => {
                let list = value.meta.require_list()?;
                let lit: syn::LitInt = syn::parse2(list.tokens.clone())?;
                Ok(InterStageIo::Location {
                    ident: ident.clone(),
                    paren_token: as_paren(&list.delimiter)?,
                    inner: lit,
                })
            }
            "blend_src" => {
                // #[blend_source()]
                let list = value.meta.require_list()?;
                let lit: syn::LitInt = syn::parse2(list.tokens.clone())?;
                Ok(InterStageIo::BlendSrc {
                    ident: ident.clone(),
                    paren_token: as_paren(&list.delimiter)?,
                    lit,
                })
            }
            "interpolate" => {
                // #[interpolate(ty, sampling?)]
                let list = value.meta.require_list()?;
                let tokens = list.tokens.clone();
                let interpolate = syn::parse2(tokens)?;
                Ok(InterStageIo::Interpolate {
                    ident: ident.clone(),
                    paren_token: as_paren(&list.delimiter)?,
                    inner: interpolate,
                })
            }
            "invariant" => {
                // #[invariant]
                Ok(InterStageIo::Invariant(ident.clone()))
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

/// Workgroup size for compute shaders.
///
/// In WGSL: `@workgroup_size(x, y, z)` where y and z default to 1.
pub struct WorkgroupSize {
    pub ident: Ident,
    pub paren_token: syn::token::Paren,
    pub x: syn::LitInt,
    pub y: Option<(syn::Token![,], syn::LitInt)>,
    pub z: Option<(syn::Token![,], syn::LitInt)>,
}

#[derive(Default)]
pub enum FnAttrs {
    #[default]
    None,
    Vertex(Ident),
    Fragment(Ident),
    Compute {
        ident: Ident,
        workgroup_size: WorkgroupSize,
    },
}

impl TryFrom<&Vec<syn::Attribute>> for FnAttrs {
    type Error = Error;

    fn try_from(value: &Vec<syn::Attribute>) -> Result<Self, Self::Error> {
        // First pass: find the shader stage attribute (vertex, fragment, compute)
        let mut stage: Option<(&str, Ident)> = None;

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
                    "vertex" => return Ok(FnAttrs::Vertex(ident.clone())),
                    "fragment" => return Ok(FnAttrs::Fragment(ident.clone())),
                    "compute" => {
                        stage = Some(("compute", ident.clone()));
                    }
                    // Skip workgroup_size and other attributes in this pass
                    "workgroup_size" => {}
                    other => UnsupportedSnafu {
                        span: ident.span(),
                        note: format!("'{other}' is not a supported annotation"),
                    }
                    .fail()?,
                }
            }
        }

        // If we found a compute stage, look for workgroup_size
        if let Some(("compute", compute_ident)) = stage {
            // Second pass: find workgroup_size
            for attr in value.iter() {
                if matches!(attr.style, syn::AttrStyle::Inner(_)) {
                    continue;
                }

                if let Some(ident) = attr.path().get_ident()
                    && ident == "workgroup_size"
                {
                    let list = attr.meta.require_list().map_err(|_| Error::Unsupported {
                        span: ident.span(),
                        note: "workgroup_size requires arguments: #[workgroup_size(x)] or \
                               #[workgroup_size(x, y)] or #[workgroup_size(x, y, z)]"
                            .to_string(),
                    })?;

                    let paren_token = if let syn::MacroDelimiter::Paren(p) = &list.delimiter {
                        *p
                    } else {
                        return UnsupportedSnafu {
                            span: ident.span(),
                            note: "workgroup_size must use parentheses",
                        }
                        .fail();
                    };

                    // Parse the workgroup size values (1-3 integers separated by commas)
                    let tokens = list.tokens.clone();
                    let parsed: WorkgroupSizeArgs =
                        syn::parse2(tokens).map_err(|e| Error::Unsupported {
                            span: ident.span(),
                            note: format!("Failed to parse workgroup_size: {e}"),
                        })?;

                    return Ok(FnAttrs::Compute {
                        ident: compute_ident,
                        workgroup_size: WorkgroupSize {
                            ident: ident.clone(),
                            paren_token,
                            x: parsed.x,
                            y: parsed.y,
                            z: parsed.z,
                        },
                    });
                }
            }

            // Compute without workgroup_size
            return UnsupportedSnafu {
                span: compute_ident.span(),
                note: "compute shader requires #[workgroup_size(x)] attribute",
            }
            .fail();
        }

        Ok(FnAttrs::None)
    }
}

/// Helper struct to parse workgroup_size arguments
struct WorkgroupSizeArgs {
    x: syn::LitInt,
    y: Option<(syn::Token![,], syn::LitInt)>,
    z: Option<(syn::Token![,], syn::LitInt)>,
}

impl syn::parse::Parse for WorkgroupSizeArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let x: syn::LitInt = input.parse()?;

        let y = if input.peek(syn::Token![,]) {
            let comma: syn::Token![,] = input.parse()?;
            let y_val: syn::LitInt = input.parse()?;
            Some((comma, y_val))
        } else {
            None
        };

        let z = if y.is_some() && input.peek(syn::Token![,]) {
            let comma: syn::Token![,] = input.parse()?;
            let z_val: syn::LitInt = input.parse()?;
            Some((comma, z_val))
        } else {
            None
        };

        Ok(WorkgroupSizeArgs { x, y, z })
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
                    ..
                } = ty.as_ref()
                {
                    *annotation = match fn_attrs {
                        FnAttrs::Vertex(_) => ReturnTypeAnnotation::DefaultBuiltInPosition,
                        FnAttrs::Fragment(_) => ReturnTypeAnnotation::DefaultLocation,
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

impl ItemFn {
    /// Convert an impl item function to an ItemFn.
    ///
    /// This is similar to `TryFrom<&syn::ItemFn>` but handles the slightly
    /// different structure of `syn::ImplItemFn`.
    pub fn try_from_impl_fn(value: &syn::ImplItemFn) -> Result<Self, Error> {
        let syn::ImplItemFn {
            attrs,
            vis,
            defaultness,
            sig,
            block,
        } = value;

        util::some_is_unsupported(
            defaultness.as_ref(),
            "default fns are not supported in WGSL",
        )?;

        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            VisibilitySnafu {
                span: sig.span(),
                item: "Impl methods"
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
                    ..
                } = ty.as_ref()
                {
                    *annotation = match fn_attrs {
                        FnAttrs::Vertex(_) => ReturnTypeAnnotation::DefaultBuiltInPosition,
                        FnAttrs::Fragment(_) => ReturnTypeAnnotation::DefaultLocation,
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
            block: Block::try_from(block)?,
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

/// A uniform declaration.
///
/// ```rust,ignore
/// uniform!(group(0), binding(0), FRAME: u32);
/// ```
///
/// ```wgsl
/// @group(0) @binding(0) var<uniform> FRAME: u32;
/// ```
pub(crate) struct ItemUniform {
    pub group_ident: Ident,
    pub group_paren_token: syn::token::Paren,
    pub group: syn::LitInt,

    pub binding_ident: Ident,
    pub binding_paren_token: syn::token::Paren,
    pub binding: syn::LitInt,

    pub name: syn::Ident,
    pub colon_token: Token![:],
    pub ty: Type,

    // We keep the Rust type around
    pub rust_ty: syn::Type,
}

impl syn::parse::Parse for ItemUniform {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // group
        let group_ident = input.parse::<syn::Ident>()?;
        let content;
        let group_paren_token = parenthesized!(content in input);
        let group = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        // binding
        let binding_ident = input.parse::<syn::Ident>()?;
        let content;
        let binding_paren_token = parenthesized!(content in input);
        let binding = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        let name: syn::Ident = input.parse()?;
        let colon_token: syn::Token![:] = input.parse()?;
        let rust_ty: syn::Type = input.parse()?;
        let ty = Type::try_from(&rust_ty)?;

        Ok(ItemUniform {
            group,
            binding,
            name,
            colon_token,
            ty,
            rust_ty,
            group_ident,
            group_paren_token,
            binding_ident,
            binding_paren_token,
        })
    }
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

        syn::parse2::<ItemUniform>(item_macro.mac.tokens.clone()).map_err(|e| Error::Unsupported {
            span: item_macro.span(),
            note: format!("{e}"),
        })
    }
}

/// Access mode for storage buffers.
#[derive(Clone, Copy, Default)]
pub enum StorageAccess {
    /// `var<storage, read>` - read-only (default when access mode is omitted)
    #[default]
    Read,
    /// `var<storage, read_write>` - read-write
    ReadWrite,
}

/// A storage buffer declaration.
///
/// ```rust,ignore
/// // Read-only (implicit):
/// storage!(group(0), binding(0), DATA: [f32; 256]);
/// // Read-only (explicit):
/// storage!(group(0), binding(0), read_only, DATA: [f32; 256]);
/// // Read-write (explicit):
/// storage!(group(0), binding(0), read_write, DATA: [f32; 256]);
/// ```
///
/// ```wgsl
/// @group(0) @binding(0) var<storage, read> DATA: array<f32, 256>;
/// @group(0) @binding(0) var<storage, read> DATA: array<f32, 256>;
/// @group(0) @binding(0) var<storage, read_write> DATA: array<f32, 256>;
/// ```
pub(crate) struct ItemStorage {
    pub group_ident: Ident,
    pub group_paren_token: syn::token::Paren,
    pub group: syn::LitInt,

    pub binding_ident: Ident,
    pub binding_paren_token: syn::token::Paren,
    pub binding: syn::LitInt,

    pub access: StorageAccess,

    pub name: syn::Ident,
    pub colon_token: Token![:],
    pub ty: Type,

    // We keep the Rust type around
    pub rust_ty: syn::Type,
}

impl syn::parse::Parse for ItemStorage {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // group
        let group_ident = input.parse::<syn::Ident>()?;
        let content;
        let group_paren_token = parenthesized!(content in input);
        let group = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        // binding
        let binding_ident = input.parse::<syn::Ident>()?;
        let content;
        let binding_paren_token = parenthesized!(content in input);
        let binding = content.parse()?;
        input.parse::<syn::Token![,]>()?;

        // Check for optional access mode (read_only or read_write)
        // If the next token is an ident that's "read_write" or "read_only", consume it
        let access = if input.peek(syn::Ident) {
            let lookahead = input.fork();
            let ident: syn::Ident = lookahead.parse()?;
            if ident == "read_write" {
                // Consume from the real stream
                let _: syn::Ident = input.parse()?;
                input.parse::<syn::Token![,]>()?;
                StorageAccess::ReadWrite
            } else if ident == "read_only" {
                // Consume from the real stream
                let _: syn::Ident = input.parse()?;
                input.parse::<syn::Token![,]>()?;
                StorageAccess::Read
            } else {
                // Not an access mode, leave it for the name
                StorageAccess::Read
            }
        } else {
            StorageAccess::Read
        };

        let name: syn::Ident = input.parse()?;
        let colon_token: syn::Token![:] = input.parse()?;
        let rust_ty: syn::Type = input.parse()?;
        let ty = Type::try_from(&rust_ty)?;

        Ok(ItemStorage {
            group,
            binding,
            access,
            name,
            colon_token,
            ty,
            rust_ty,
            group_ident,
            group_paren_token,
            binding_ident,
            binding_paren_token,
        })
    }
}

impl TryFrom<&syn::ItemMacro> for ItemStorage {
    type Error = Error;

    fn try_from(item_macro: &syn::ItemMacro) -> Result<Self, Self::Error> {
        // Ensure it's the "storage" macro
        if item_macro
            .mac
            .path
            .get_ident()
            .map(|id| id != "storage")
            .unwrap_or(true)
        {
            return UnsupportedSnafu {
                span: item_macro.span(),
                note: "Only 'storage!' macro is supported as a storage declaration.",
            }
            .fail();
        }

        syn::parse2::<ItemStorage>(item_macro.mac.tokens.clone()).map_err(|e| Error::Unsupported {
            span: item_macro.span(),
            note: format!("{e}"),
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

/// An impl block for a struct.
///
/// Methods are just regular functions with explicit receiver parameters - no
/// `self` support. They get name-mangled to `StructName_method` in WGSL output.
///
/// # Example
///
/// ```rust,ignore
/// impl Light {
///     pub fn attenuate(light: Light, distance: f32) -> f32 {
///         light.intensity / (distance * distance)
///     }
/// }
/// ```
///
/// Becomes in WGSL:
///
/// ```wgsl
/// fn Light_attenuate(light: Light, distance: f32) -> f32 {
///     return light.intensity / (distance * distance);
/// }
/// ```
pub struct ItemImpl {
    pub impl_token: Token![impl],
    pub self_ty: Ident,
    pub brace_token: syn::token::Brace,
    pub items: Vec<ItemFn>,
}

impl TryFrom<&syn::ItemImpl> for ItemImpl {
    type Error = Error;

    fn try_from(value: &syn::ItemImpl) -> Result<Self, Self::Error> {
        let syn::ItemImpl {
            attrs: _,
            defaultness,
            unsafety,
            impl_token,
            generics,
            trait_,
            self_ty,
            brace_token,
            items,
        } = value;

        // Reject unsupported syntax
        util::some_is_unsupported(
            defaultness.as_ref(),
            "default impls are not supported in WGSL",
        )?;
        util::some_is_unsupported(unsafety.as_ref(), "unsafe impls are not supported in WGSL")?;

        // Reject generics
        if !generics.params.is_empty() {
            return UnsupportedSnafu {
                span: generics.span(),
                note: "generic impl blocks are not supported in WGSL",
            }
            .fail();
        }

        // Reject trait impls
        if let Some((_, trait_path, _)) = trait_ {
            return UnsupportedSnafu {
                span: trait_path.span(),
                note: "trait impls are not supported in WGSL. Use `impl StructName { ... }` \
                       instead",
            }
            .fail();
        }

        // Get the struct name (self_ty must be a simple ident)
        let self_ty_ident = match self_ty.as_ref() {
            syn::Type::Path(type_path) => type_path
                .path
                .get_ident()
                .context(UnsupportedSnafu {
                    span: type_path.span(),
                    note: "impl block type must be a simple identifier",
                })?
                .clone(),
            other => {
                return UnsupportedSnafu {
                    span: other.span(),
                    note: "impl block type must be a simple struct name",
                }
                .fail();
            }
        };

        // Parse methods
        let mut parsed_items = Vec::new();
        for item in items {
            match item {
                syn::ImplItem::Fn(impl_fn) => {
                    let item_fn = ItemFn::try_from_impl_fn(impl_fn)?;
                    parsed_items.push(item_fn);
                }
                other => {
                    return UnsupportedSnafu {
                        span: other.span(),
                        note: "only functions are supported in impl blocks",
                    }
                    .fail();
                }
            }
        }

        Ok(ItemImpl {
            impl_token: *impl_token,
            self_ty: self_ty_ident,
            brace_token: *brace_token,
            items: parsed_items,
        })
    }
}

/// WGSL items that may appear in a "module" or scope.
pub enum Item {
    Const(Box<ItemConst>),
    Uniform(Box<ItemUniform>),
    Storage(Box<ItemStorage>),
    Fn(Box<ItemFn>),
    Mod(ItemMod),
    Use(ItemUse),
    Struct(ItemStruct),
    Impl(ItemImpl),
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
                    Some("storage") => {
                        Ok(Item::Storage(Box::new(ItemStorage::try_from(item_macro)?)))
                    }
                    other => UnsupportedSnafu {
                        span: item_macro.ident.span(),
                        note: format!(
                            "Unknown macro '{other:?}' doesn't expand into WGSL\nSeen as \
                             '{item_macro:#?}'",
                        ),
                    }
                    .fail()?,
                }
            }
            syn::Item::Const(item_const) => {
                Ok(Item::Const(Box::new(ItemConst::try_from(item_const)?)))
            }
            syn::Item::Fn(item_fn) => Ok(Item::Fn(Box::new(item_fn.try_into()?))),
            syn::Item::Use(syn::ItemUse {
                attrs: _,
                vis: _,
                use_token: _,
                leading_colon: _,
                tree,
                semi_token: _,
            }) => Ok(Item::Use(ItemUse::try_from(tree)?)),
            syn::Item::Struct(item_struct) => Ok(Item::Struct(ItemStruct::try_from(item_struct)?)),
            syn::Item::Impl(item_impl) => Ok(Item::Impl(ItemImpl::try_from(item_impl)?)),
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
        assert_eq!("333+333", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_ident() {
        let expr: syn::Expr = syn::parse_str("333 + TIMES").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("333+TIMES", &expr.to_wgsl());
    }

    #[test]
    fn parse_vec4_f32_type() {
        let ty: syn::Type = syn::parse_str("Vec4<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("vec4<f32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_array_type() {
        let ty: syn::Type = syn::parse_str("[f32; 4]").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("array<f32, 4>", &ty.to_wgsl());
    }

    #[test]
    fn parse_storage_read_only_implicit() {
        let storage: ItemStorage =
            syn::parse_str("group(0), binding(0), DATA: [f32; 256]").unwrap();
        assert!(matches!(storage.access, StorageAccess::Read));
        assert_eq!("DATA", storage.name.to_string());
        assert_eq!("0", storage.group.to_string());
        assert_eq!("0", storage.binding.to_string());
    }

    #[test]
    fn parse_storage_read_only_explicit() {
        let storage: ItemStorage =
            syn::parse_str("group(0), binding(1), read_only, DATA: [f32; 256]").unwrap();
        assert!(matches!(storage.access, StorageAccess::Read));
        assert_eq!("DATA", storage.name.to_string());
        assert_eq!("1", storage.binding.to_string());
    }

    #[test]
    fn parse_storage_read_write() {
        let storage: ItemStorage =
            syn::parse_str("group(1), binding(2), read_write, OUTPUT: [u32; 128]").unwrap();
        assert!(matches!(storage.access, StorageAccess::ReadWrite));
        assert_eq!("OUTPUT", storage.name.to_string());
        assert_eq!("1", storage.group.to_string());
        assert_eq!("2", storage.binding.to_string());
    }

    #[test]
    fn storage_to_wgsl_read() {
        let storage: ItemStorage =
            syn::parse_str("group(0), binding(0), DATA: [f32; 256]").unwrap();
        let wgsl = storage.to_wgsl();
        assert!(wgsl.contains("@group(0)"));
        assert!(wgsl.contains("@binding(0)"));
        assert!(wgsl.contains("var<storage, read>"));
        assert!(wgsl.contains("DATA"));
    }

    #[test]
    fn storage_to_wgsl_read_write() {
        let storage: ItemStorage =
            syn::parse_str("group(0), binding(1), read_write, OUTPUT: [f32; 256]").unwrap();
        let wgsl = storage.to_wgsl();
        assert!(wgsl.contains("var<storage, read_write>"));
        assert!(wgsl.contains("OUTPUT"));
    }

    #[test]
    fn parse_workgroup_size_1d() {
        let args: WorkgroupSizeArgs = syn::parse_str("64").unwrap();
        assert_eq!("64", args.x.to_string());
        assert!(args.y.is_none());
        assert!(args.z.is_none());
    }

    #[test]
    fn parse_workgroup_size_2d() {
        let args: WorkgroupSizeArgs = syn::parse_str("8, 8").unwrap();
        assert_eq!("8", args.x.to_string());
        assert!(args.y.is_some());
        let (_, y_val) = args.y.unwrap();
        assert_eq!("8", y_val.to_string());
        assert!(args.z.is_none());
    }

    #[test]
    fn parse_workgroup_size_3d() {
        let args: WorkgroupSizeArgs = syn::parse_str("4, 4, 4").unwrap();
        assert_eq!("4", args.x.to_string());
        assert!(args.y.is_some());
        assert!(args.z.is_some());
        let (_, y_val) = args.y.unwrap();
        let (_, z_val) = args.z.unwrap();
        assert_eq!("4", y_val.to_string());
        assert_eq!("4", z_val.to_string());
    }

    // Tests for to_snake_case function
    #[test]
    fn to_snake_case_screaming_case() {
        // SCREAMING_CASE (all uppercase with underscores) should just be lowercased
        assert_eq!(to_snake_case("FRAME"), "frame");
        assert_eq!(to_snake_case("FRAME_BUFFER"), "frame_buffer");
        assert_eq!(to_snake_case("MY_CONSTANT"), "my_constant");
        assert_eq!(to_snake_case("INPUT_DATA"), "input_data");
    }

    #[test]
    fn to_snake_case_screaming_case_with_numbers() {
        // SCREAMING_CASE with numbers
        assert_eq!(to_snake_case("BUFFER_1"), "buffer_1");
        assert_eq!(to_snake_case("DATA_2D"), "data_2d");
        assert_eq!(to_snake_case("VEC3_POSITION"), "vec3_position");
    }

    #[test]
    fn to_snake_case_pascal_case() {
        // PascalCase should convert to snake_case
        assert_eq!(to_snake_case("MyBuffer"), "my_buffer");
        assert_eq!(to_snake_case("FrameBuffer"), "frame_buffer");
        assert_eq!(to_snake_case("InputData"), "input_data");
        assert_eq!(to_snake_case("SimpleType"), "simple_type");
    }

    #[test]
    fn to_snake_case_camel_case() {
        // camelCase should convert to snake_case
        assert_eq!(to_snake_case("myBuffer"), "my_buffer");
        assert_eq!(to_snake_case("frameBuffer"), "frame_buffer");
        assert_eq!(to_snake_case("inputData"), "input_data");
    }

    #[test]
    fn to_snake_case_mixed_case_consecutive_caps() {
        // Mixed case with consecutive capitals
        assert_eq!(to_snake_case("HTTPServer"), "http_server");
        assert_eq!(to_snake_case("XMLParser"), "xml_parser");
        assert_eq!(to_snake_case("URLPath"), "url_path");
    }

    #[test]
    fn to_snake_case_already_snake_case() {
        // Already snake_case should remain unchanged
        assert_eq!(to_snake_case("my_buffer"), "my_buffer");
        assert_eq!(to_snake_case("frame_buffer"), "frame_buffer");
        assert_eq!(to_snake_case("input_data"), "input_data");
        assert_eq!(to_snake_case("simple_type"), "simple_type");
    }

    #[test]
    fn to_snake_case_single_character() {
        // Single character inputs
        assert_eq!(to_snake_case("A"), "a");
        assert_eq!(to_snake_case("a"), "a");
        assert_eq!(to_snake_case("Z"), "z");
        assert_eq!(to_snake_case("1"), "1");
    }

    #[test]
    fn to_snake_case_empty_string() {
        // Empty string should remain empty
        assert_eq!(to_snake_case(""), "");
    }

    #[test]
    fn to_snake_case_with_numbers() {
        // Strings with numbers in various positions
        assert_eq!(to_snake_case("Buffer2D"), "buffer2_d");
        assert_eq!(to_snake_case("Vec3Position"), "vec3_position");
        assert_eq!(to_snake_case("MyType123"), "my_type123");
    }

    #[test]
    fn to_snake_case_single_uppercase() {
        // Single uppercase word
        assert_eq!(to_snake_case("FRAME"), "frame");
        assert_eq!(to_snake_case("Buffer"), "buffer");
    }

    #[test]
    fn to_snake_case_multiple_underscores() {
        // Multiple underscores in SCREAMING_CASE
        assert_eq!(to_snake_case("MY__BUFFER"), "my__buffer");
        assert_eq!(to_snake_case("INPUT___DATA"), "input___data");
    }

    #[test]
    fn to_snake_case_leading_underscore() {
        // Leading underscores
        assert_eq!(to_snake_case("_PRIVATE"), "_private");
        assert_eq!(to_snake_case("_MyBuffer"), "_my_buffer");
    }

    #[test]
    fn to_snake_case_trailing_underscore() {
        // Trailing underscores
        assert_eq!(to_snake_case("BUFFER_"), "buffer_");
        assert_eq!(to_snake_case("MyBuffer_"), "my_buffer_");
    }

    // Matrix type parsing tests
    #[test]
    fn parse_mat4_f32_type() {
        let ty: syn::Type = syn::parse_str("Mat4<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat4x4<f32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat4f_type() {
        let ty: syn::Type = syn::parse_str("Mat4f").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat4x4f", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat3_f32_type() {
        let ty: syn::Type = syn::parse_str("Mat3<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat3x3<f32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat3f_type() {
        let ty: syn::Type = syn::parse_str("Mat3f").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat3x3f", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat2_f32_type() {
        let ty: syn::Type = syn::parse_str("Mat2<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat2x2<f32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat2f_type() {
        let ty: syn::Type = syn::parse_str("Mat2f").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("mat2x2f", &ty.to_wgsl());
    }

    #[test]
    fn parse_mat4_i32_type_fails() {
        let ty: syn::Type = syn::parse_str("Mat4<i32>").unwrap();
        let result = Type::try_from(&ty);
        assert!(result.is_err());
    }

    #[test]
    fn parse_mat4i_type_fails() {
        let ty: syn::Type = syn::parse_str("Mat4i").unwrap();
        let result = Type::try_from(&ty);
        assert!(result.is_err());
    }

    // Impl block parsing tests
    #[test]
    fn parse_impl_block() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub fn attenuate(light: Light, distance: f32) -> f32 {
                    light.intensity / (distance * distance)
                }
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Impl(item_impl) => {
                assert_eq!("Light", item_impl.self_ty.to_string());
                assert_eq!(1, item_impl.items.len());
                assert_eq!("attenuate", item_impl.items[0].ident.to_string());
            }
            _ => panic!("Expected Item::Impl"),
        }
    }

    #[test]
    fn parse_impl_block_multiple_methods() {
        let item: syn::Item = syn::parse_quote! {
            impl Point {
                pub fn new(x: f32, y: f32) -> Point {
                    Point { x: x, y: y }
                }
                pub fn distance(p1: Point, p2: Point) -> f32 {
                    let dx = p1.x - p2.x;
                    let dy = p1.y - p2.y;
                    (dx * dx + dy * dy)
                }
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Impl(item_impl) => {
                assert_eq!("Point", item_impl.self_ty.to_string());
                assert_eq!(2, item_impl.items.len());
                assert_eq!("new", item_impl.items[0].ident.to_string());
                assert_eq!("distance", item_impl.items[1].ident.to_string());
            }
            _ => panic!("Expected Item::Impl"),
        }
    }

    #[test]
    fn parse_impl_rejects_traits() {
        let item: syn::Item = syn::parse_quote! {
            impl SomeTrait for Light {
                pub fn foo() {}
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("trait impls are not supported"),
            "Expected error about trait impls, got: {}",
            err
        );
    }

    #[test]
    fn parse_impl_rejects_generics() {
        let item: syn::Item = syn::parse_quote! {
            impl<T> Light {
                pub fn foo() {}
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("generic impl blocks are not supported"),
            "Expected error about generics, got: {}",
            err
        );
    }

    #[test]
    fn parse_impl_rejects_self_receiver() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub fn foo(&self) {}
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("self"),
            "Expected error about self, got: {}",
            err
        );
    }

    // Type::method call parsing tests
    #[test]
    fn parse_type_method_call() {
        let expr: syn::Expr = syn::parse_quote! {
            Light::attenuate(light, 2.0)
        };
        let expr = Expr::try_from(&expr).unwrap();
        match expr {
            Expr::FnCall {
                path: FnPath::TypeMethod { ty, method, .. },
                ..
            } => {
                assert_eq!(ty.to_string(), "Light");
                assert_eq!(method.to_string(), "attenuate");
            }
            _ => panic!("Expected Expr::FnCall with TypeMethod path"),
        }
    }

    #[test]
    fn parse_simple_function_call() {
        let expr: syn::Expr = syn::parse_quote! {
            sin(x)
        };
        let expr = Expr::try_from(&expr).unwrap();
        match expr {
            Expr::FnCall {
                path: FnPath::Ident(ident),
                ..
            } => {
                assert_eq!(ident.to_string(), "sin");
            }
            _ => panic!("Expected Expr::FnCall with Ident path"),
        }
    }

    // WGSL code generation tests for impl blocks
    #[test]
    fn impl_block_generates_mangled_functions() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub fn attenuate(light: Light, distance: f32) -> f32 {
                    light.intensity / (distance * distance)
                }
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("fn Light_attenuate"),
            "Expected 'fn Light_attenuate' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn type_method_call_generates_mangled_name() {
        let expr: syn::Expr = syn::parse_quote! {
            Light::attenuate(light, 2.0)
        };
        let expr = Expr::try_from(&expr).unwrap();
        let wgsl = expr.to_wgsl();
        assert_eq!(wgsl, "Light_attenuate(light, 2.0)");
    }
}
