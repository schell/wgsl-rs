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

/// A binary operator: `+` `-` `*` `/` `%` `==` `!=` `<` `<=` `>` `>=` `&&` `||`
/// `&` `|` `^` `<<` `>>`.
pub enum BinOp {
    // Arithmetic
    Add(Token![+]),
    Sub(Token![-]),
    Mul(Token![*]),
    Div(Token![/]),
    Rem(Token![%]),
    // Comparison
    Eq(Token![==]),
    Ne(Token![!=]),
    Lt(Token![<]),
    Le(Token![<=]),
    Gt(Token![>]),
    Ge(Token![>=]),
    // Logical
    And(Token![&&]),
    Or(Token![||]),
    // Bitwise
    BitAnd(Token![&]),
    BitOr(Token![|]),
    BitXor(Token![^]),
    Shl(Token![<<]),
    Shr(Token![>>]),
}

impl TryFrom<&syn::BinOp> for BinOp {
    type Error = Error;

    fn try_from(value: &syn::BinOp) -> Result<Self, Self::Error> {
        Ok(match value {
            // Arithmetic
            syn::BinOp::Add(t) => Self::Add(*t),
            syn::BinOp::Sub(t) => Self::Sub(*t),
            syn::BinOp::Mul(t) => Self::Mul(*t),
            syn::BinOp::Div(t) => Self::Div(*t),
            syn::BinOp::Rem(t) => Self::Rem(*t),
            // Comparison
            syn::BinOp::Eq(t) => Self::Eq(*t),
            syn::BinOp::Ne(t) => Self::Ne(*t),
            syn::BinOp::Lt(t) => Self::Lt(*t),
            syn::BinOp::Le(t) => Self::Le(*t),
            syn::BinOp::Gt(t) => Self::Gt(*t),
            syn::BinOp::Ge(t) => Self::Ge(*t),
            // Logical
            syn::BinOp::And(t) => Self::And(*t),
            syn::BinOp::Or(t) => Self::Or(*t),
            // Bitwise
            syn::BinOp::BitAnd(t) => Self::BitAnd(*t),
            syn::BinOp::BitOr(t) => Self::BitOr(*t),
            syn::BinOp::BitXor(t) => Self::BitXor(*t),
            syn::BinOp::Shl(t) => Self::Shl(*t),
            syn::BinOp::Shr(t) => Self::Shr(*t),
            // Compound assignments are not supported (yet)
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
            // Arithmetic
            BinOp::Add(_) => "+",
            BinOp::Sub(_) => "-",
            BinOp::Mul(_) => "*",
            BinOp::Div(_) => "/",
            BinOp::Rem(_) => "%",
            // Comparison
            BinOp::Eq(_) => "==",
            BinOp::Ne(_) => "!=",
            BinOp::Lt(_) => "<",
            BinOp::Le(_) => "<=",
            BinOp::Gt(_) => ">",
            BinOp::Ge(_) => ">=",
            // Logical
            BinOp::And(_) => "&&",
            BinOp::Or(_) => "||",
            // Bitwise
            BinOp::BitAnd(_) => "&",
            BinOp::BitOr(_) => "|",
            BinOp::BitXor(_) => "^",
            BinOp::Shl(_) => "<<",
            BinOp::Shr(_) => ">>",
        };
        f.write_str(s)
    }
}

/// A compound assignment operator: `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`,
/// `^=`, `<<=`, `>>=`
#[allow(clippy::enum_variant_names)]
pub enum CompoundOp {
    AddAssign(Token![+=]),
    SubAssign(Token![-=]),
    MulAssign(Token![*=]),
    DivAssign(Token![/=]),
    RemAssign(Token![%=]),
    BitAndAssign(Token![&=]),
    BitOrAssign(Token![|=]),
    BitXorAssign(Token![^=]),
    ShlAssign(Token![<<=]),
    ShrAssign(Token![>>=]),
}

impl TryFrom<&syn::BinOp> for CompoundOp {
    type Error = Error;

    fn try_from(value: &syn::BinOp) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::BinOp::AddAssign(t) => Self::AddAssign(*t),
            syn::BinOp::SubAssign(t) => Self::SubAssign(*t),
            syn::BinOp::MulAssign(t) => Self::MulAssign(*t),
            syn::BinOp::DivAssign(t) => Self::DivAssign(*t),
            syn::BinOp::RemAssign(t) => Self::RemAssign(*t),
            syn::BinOp::BitAndAssign(t) => Self::BitAndAssign(*t),
            syn::BinOp::BitOrAssign(t) => Self::BitOrAssign(*t),
            syn::BinOp::BitXorAssign(t) => Self::BitXorAssign(*t),
            syn::BinOp::ShlAssign(t) => Self::ShlAssign(*t),
            syn::BinOp::ShrAssign(t) => Self::ShrAssign(*t),
            other => UnsupportedSnafu {
                span: other.span(),
                note: format!(
                    "'{}' is not a compound assignment operator.",
                    other.into_token_stream()
                ),
            }
            .fail()?,
        })
    }
}

/// Returns true if the binary operator is a compound assignment operator.
fn is_compound_assign_op(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
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
    /// A type-qualified path like `Light::CONSTANT`.
    ///
    /// Used for accessing associated constants defined in impl blocks.
    /// In WGSL output, this becomes `Light_CONSTANT`.
    TypePath {
        ty: Ident,
        colon2_token: Token![::],
        member: Ident,
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

                if let Some(ident) = path.get_ident() {
                    // Simple identifier: `foo`
                    Self::Ident(ident.clone())
                } else if path.segments.len() == 2 {
                    // Type::member path: `Light::CONSTANT`
                    let ty = path.segments[0].ident.clone();
                    let member = path.segments[1].ident.clone();

                    // Check no generics on segments
                    for seg in &path.segments {
                        if !matches!(seg.arguments, syn::PathArguments::None) {
                            return UnsupportedSnafu {
                                span: seg.arguments.span(),
                                note: "generic arguments in type paths are not supported",
                            }
                            .fail();
                        }
                    }

                    Self::TypePath {
                        ty,
                        colon2_token: Token![::](path.segments[0].ident.span()),
                        member,
                    }
                } else {
                    return UnsupportedSnafu {
                        span: path.span(),
                        note: format!(
                            "only simple identifiers or Type::member paths are supported, saw '{}'",
                            path.into_token_stream()
                        ),
                    }
                    .fail();
                }
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
    /// Simple assignment: `lhs = rhs;`
    Assignment {
        lhs: Expr,
        eq_token: Token![=],
        rhs: Expr,
        semi_token: Token![;],
    },
    /// Compound assignment: `lhs += rhs;`, `lhs -= rhs;`, etc.
    CompoundAssignment {
        lhs: Expr,
        op: CompoundOp,
        rhs: Expr,
        semi_token: Token![;],
    },
    /// While loop: `while condition { ... }`
    While {
        while_token: Token![while],
        condition: Expr,
        body: Block,
    },
    Expr {
        expr: Expr,
        /// If `None`, this expression is a return statement
        semi_token: Option<Token![;]>,
    },
    /// If statement (with optional else/else-if chains)
    If(Box<StmtIf>),
    /// Break statement: `break;`
    Break {
        break_token: Token![break],
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
            syn::Stmt::Expr(expr, semi_token) => {
                // Check for assignment expressions
                match expr {
                    syn::Expr::Assign(syn::ExprAssign {
                        attrs: _,
                        left,
                        eq_token,
                        right,
                    }) => {
                        let semi_token = semi_token.ok_or_else(|| Error::Unsupported {
                            span: expr.span(),
                            note: "Assignment statements must end with a semicolon".to_string(),
                        })?;
                        Ok(Stmt::Assignment {
                            lhs: Expr::try_from(left.as_ref())?,
                            eq_token: *eq_token,
                            rhs: Expr::try_from(right.as_ref())?,
                            semi_token,
                        })
                    }
                    // In syn 2.0, compound assignment like `x += y` is parsed as
                    // Expr::Binary with BinOp being a compound assignment operator
                    syn::Expr::Binary(syn::ExprBinary {
                        attrs: _,
                        left,
                        op,
                        right,
                    }) if is_compound_assign_op(op) => {
                        let semi_token = semi_token.ok_or_else(|| Error::Unsupported {
                            span: expr.span(),
                            note: "Compound assignment statements must end with a semicolon"
                                .to_string(),
                        })?;
                        Ok(Stmt::CompoundAssignment {
                            lhs: Expr::try_from(left.as_ref())?,
                            op: CompoundOp::try_from(op)?,
                            rhs: Expr::try_from(right.as_ref())?,
                            semi_token,
                        })
                    }
                    // While loop: `while condition { ... }`
                    syn::Expr::While(syn::ExprWhile {
                        attrs: _,
                        label,
                        while_token,
                        cond,
                        body,
                    }) => {
                        util::some_is_unsupported(
                            label.as_ref(),
                            "Labels on while loops are not supported in WGSL",
                        )?;
                        Ok(Stmt::While {
                            while_token: *while_token,
                            condition: Expr::try_from(cond.as_ref())?,
                            body: Block::try_from(body)?,
                        })
                    }
                    // If statements are control flow statements in WGSL
                    syn::Expr::If(expr_if) => Ok(Stmt::If(Box::new(StmtIf::try_from(expr_if)?))),
                    // Break statement: `break;`
                    syn::Expr::Break(syn::ExprBreak {
                        attrs: _,
                        break_token,
                        label,
                        expr,
                    }) => {
                        util::some_is_unsupported(
                            label.as_ref(),
                            "Labels on break statements are not supported in WGSL",
                        )?;
                        util::some_is_unsupported(
                            expr.as_ref(),
                            "Break with values is not supported in WGSL",
                        )?;
                        Ok(Stmt::Break {
                            break_token: *break_token,
                        })
                    }
                    _ => Ok(Stmt::Expr {
                        expr: Expr::try_from(expr)?,
                        semi_token: *semi_token,
                    }),
                }
            }
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

/// WGSL if statement.
///
/// Unlike Rust, WGSL `if` is a statement, not an expression.
/// This means you cannot write `let x = if cond { a } else { b }` in WGSL.
pub struct StmtIf {
    pub if_token: Token![if],
    pub condition: Box<Expr>,
    pub then_block: Block,
    pub else_branch: Option<ElseBranch>,
}

/// The else branch of an if statement.
pub struct ElseBranch {
    pub else_token: Token![else],
    pub body: ElseBody,
}

/// The body of an else branch - either a block or another if statement.
pub enum ElseBody {
    Block(Block),
    If(Box<StmtIf>),
}

impl TryFrom<&syn::ExprIf> for StmtIf {
    type Error = Error;

    fn try_from(value: &syn::ExprIf) -> Result<Self, Self::Error> {
        let condition = Box::new(Expr::try_from(value.cond.as_ref())?);
        let then_block = Block::try_from(&value.then_branch)?;

        let else_branch = if let Some((else_token, else_expr)) = &value.else_branch {
            let body = match else_expr.as_ref() {
                syn::Expr::Block(syn::ExprBlock { block, .. }) => {
                    ElseBody::Block(Block::try_from(block)?)
                }
                syn::Expr::If(else_if) => ElseBody::If(Box::new(StmtIf::try_from(else_if)?)),
                other => {
                    return UnsupportedSnafu {
                        span: other.span(),
                        note: "else branch must be a block or another if",
                    }
                    .fail();
                }
            };
            Some(ElseBranch {
                else_token: *else_token,
                body,
            })
        } else {
            None
        };

        Ok(StmtIf {
            if_token: value.if_token,
            condition,
            then_block,
            else_branch,
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

impl ItemConst {
    /// Convert an impl item constant to an ItemConst.
    ///
    /// This is similar to `TryFrom<&syn::ItemConst>` but handles the slightly
    /// different structure of `syn::ImplItemConst`.
    pub fn try_from_impl_const(value: &syn::ImplItemConst) -> Result<Self, Error> {
        let syn::ImplItemConst {
            attrs: _,
            vis,
            defaultness,
            const_token,
            ident,
            generics,
            colon_token,
            ty,
            eq_token,
            expr,
            semi_token,
        } = value;

        util::some_is_unsupported(
            defaultness.as_ref(),
            "default constants are not supported in WGSL",
        )?;

        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            VisibilitySnafu {
                span: const_token.span(),
                item: "Impl constants"
            }
        );

        // Reject generics on constants
        if !generics.params.is_empty() {
            return UnsupportedSnafu {
                span: generics.span(),
                note: "generic constants are not supported in WGSL",
            }
            .fail();
        }

        Ok(ItemConst {
            const_token: *const_token,
            ident: ident.clone(),
            colon_token: *colon_token,
            ty: Type::try_from(ty)?,
            eq_token: *eq_token,
            expr: Expr::try_from(expr)?,
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

/// A single variant of an enum.
///
/// Only unit variants are supported (no tuple or struct variants).
pub struct EnumVariant {
    pub ident: Ident,
    /// Optional explicit discriminant value, e.g., `First = 5`.
    pub discriminant: Option<(Token![=], syn::LitInt)>,
}

/// An enum declaration.
///
/// Only unit-variant enums with `#[repr(u32)]` are supported.
/// Each variant becomes a `const` in WGSL with the enum name as prefix.
///
/// # Example
///
/// ```rust,ignore
/// #[repr(u32)]
/// pub enum State {
///     Idle,
///     Running,
///     Stopped = 10,
/// }
/// ```
///
/// Becomes in WGSL:
///
/// ```wgsl
/// const State_Idle: u32 = 0u;
/// const State_Running: u32 = 1u;
/// const State_Stopped: u32 = 10u;
/// ```
pub struct ItemEnum {
    pub enum_token: Token![enum],
    pub ident: Ident,
    pub _brace_token: syn::token::Brace,
    pub variants: Vec<EnumVariant>,
}

impl TryFrom<&syn::ItemEnum> for ItemEnum {
    type Error = Error;

    fn try_from(value: &syn::ItemEnum) -> Result<Self, Self::Error> {
        let syn::ItemEnum {
            attrs,
            vis,
            enum_token,
            ident,
            generics,
            brace_token,
            variants,
        } = value;

        // Check visibility
        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            VisibilitySnafu {
                span: enum_token.span(),
                item: "Enums"
            }
        );

        // Reject generics
        if !generics.params.is_empty() {
            return UnsupportedSnafu {
                span: generics.span(),
                note: "generic enums are not supported in WGSL",
            }
            .fail();
        }

        // Check for #[repr(u32)] attribute
        let has_repr_u32 = attrs.iter().any(|attr| {
            if attr.path().is_ident("repr")
                && let Ok(inner) = attr.parse_args::<syn::Ident>()
            {
                return inner == "u32";
            }
            false
        });

        if !has_repr_u32 {
            return UnsupportedSnafu {
                span: enum_token.span(),
                note: "enums must have #[repr(u32)] attribute for WGSL compatibility",
            }
            .fail();
        }

        // Parse variants
        let mut parsed_variants = Vec::new();
        for variant in variants {
            // Reject tuple and struct variants
            match &variant.fields {
                syn::Fields::Unit => {}
                syn::Fields::Unnamed(fields) => {
                    return UnsupportedSnafu {
                        span: fields.span(),
                        note: "tuple variants are not supported in WGSL enums, only unit variants",
                    }
                    .fail();
                }
                syn::Fields::Named(fields) => {
                    return UnsupportedSnafu {
                        span: fields.span(),
                        note: "struct variants are not supported in WGSL enums, only unit variants",
                    }
                    .fail();
                }
            }

            // Parse discriminant if present
            let discriminant = if let Some((eq, expr)) = &variant.discriminant {
                // Only support integer literals
                match expr {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(lit_int),
                        ..
                    }) => Some((*eq, lit_int.clone())),
                    _ => {
                        return UnsupportedSnafu {
                            span: expr.span(),
                            note: "only integer literal discriminants are supported in WGSL enums",
                        }
                        .fail();
                    }
                }
            } else {
                None
            };

            parsed_variants.push(EnumVariant {
                ident: variant.ident.clone(),
                discriminant,
            });
        }

        Ok(ItemEnum {
            enum_token: *enum_token,
            ident: ident.clone(),
            _brace_token: *brace_token,
            variants: parsed_variants,
        })
    }
}

/// An item that can appear inside an impl block.
///
/// Currently supports functions and constants.
pub enum ImplItem {
    /// A function defined in an impl block.
    Fn(ItemFn),
    /// A constant defined in an impl block.
    Const(ItemConst),
}

/// An impl block for a struct.
///
/// Methods and constants are name-mangled to `StructName_member` in WGSL
/// output. Methods are just regular functions with explicit receiver parameters
/// - no `self` support.
///
/// # Example
///
/// ```rust,ignore
/// impl Light {
///     pub const DEFAULT_INTENSITY: f32 = 1.0;
///
///     pub fn attenuate(light: Light, distance: f32) -> f32 {
///         light.intensity / (distance * distance)
///     }
/// }
/// ```
///
/// Becomes in WGSL:
///
/// ```wgsl
/// const Light_DEFAULT_INTENSITY: f32 = 1.0;
///
/// fn Light_attenuate(light: Light, distance: f32) -> f32 {
///     return light.intensity / (distance * distance);
/// }
/// ```
pub struct ItemImpl {
    pub _impl_token: Token![impl],
    pub self_ty: Ident,
    pub _brace_token: syn::token::Brace,
    pub items: Vec<ImplItem>,
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

        // Parse impl items (functions and constants)
        let mut parsed_items = Vec::new();
        for item in items {
            match item {
                syn::ImplItem::Fn(impl_fn) => {
                    let item_fn = ItemFn::try_from_impl_fn(impl_fn)?;
                    parsed_items.push(ImplItem::Fn(item_fn));
                }
                syn::ImplItem::Const(impl_const) => {
                    let item_const = ItemConst::try_from_impl_const(impl_const)?;
                    parsed_items.push(ImplItem::Const(item_const));
                }
                other => {
                    return UnsupportedSnafu {
                        span: other.span(),
                        note: "only functions and constants are supported in impl blocks",
                    }
                    .fail();
                }
            }
        }

        Ok(ItemImpl {
            _impl_token: *impl_token,
            self_ty: self_ty_ident,
            _brace_token: *brace_token,
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
    Enum(ItemEnum),
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
            syn::Item::Enum(item_enum) => Ok(Item::Enum(ItemEnum::try_from(item_enum)?)),
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

    // Remainder operator
    #[test]
    fn parse_expr_binary_rem() {
        let expr: syn::Expr = syn::parse_str("10 % 3").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("10%3", &expr.to_wgsl());
    }

    // Comparison operators
    #[test]
    fn parse_expr_binary_eq() {
        let expr: syn::Expr = syn::parse_str("a == b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a==b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_ne() {
        let expr: syn::Expr = syn::parse_str("a != b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a!=b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_lt() {
        let expr: syn::Expr = syn::parse_str("a < b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a<b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_le() {
        let expr: syn::Expr = syn::parse_str("a <= b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a<=b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_gt() {
        let expr: syn::Expr = syn::parse_str("a > b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a>b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_ge() {
        let expr: syn::Expr = syn::parse_str("a >= b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a>=b", &expr.to_wgsl());
    }

    // Logical operators
    #[test]
    fn parse_expr_binary_and() {
        let expr: syn::Expr = syn::parse_str("a && b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a&&b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_or() {
        let expr: syn::Expr = syn::parse_str("a || b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a||b", &expr.to_wgsl());
    }

    // Bitwise operators
    #[test]
    fn parse_expr_binary_bitand() {
        let expr: syn::Expr = syn::parse_str("a & b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a&b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_bitor() {
        let expr: syn::Expr = syn::parse_str("a | b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a|b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_bitxor() {
        let expr: syn::Expr = syn::parse_str("a ^ b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a^b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_shl() {
        let expr: syn::Expr = syn::parse_str("a << b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a<<b", &expr.to_wgsl());
    }

    #[test]
    fn parse_expr_binary_shr() {
        let expr: syn::Expr = syn::parse_str("a >> b").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("a>>b", &expr.to_wgsl());
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
                match &item_impl.items[0] {
                    ImplItem::Fn(item_fn) => {
                        assert_eq!("attenuate", item_fn.ident.to_string());
                    }
                    _ => panic!("Expected ImplItem::Fn"),
                }
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
                match &item_impl.items[0] {
                    ImplItem::Fn(item_fn) => assert_eq!("new", item_fn.ident.to_string()),
                    _ => panic!("Expected ImplItem::Fn"),
                }
                match &item_impl.items[1] {
                    ImplItem::Fn(item_fn) => assert_eq!("distance", item_fn.ident.to_string()),
                    _ => panic!("Expected ImplItem::Fn"),
                }
            }
            _ => panic!("Expected Item::Impl"),
        }
    }

    #[test]
    fn parse_impl_block_with_const() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub const INTENSITY: f32 = 1.0;
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Impl(item_impl) => {
                assert_eq!("Light", item_impl.self_ty.to_string());
                assert_eq!(1, item_impl.items.len());
                match &item_impl.items[0] {
                    ImplItem::Const(item_const) => {
                        assert_eq!("INTENSITY", item_const.ident.to_string());
                    }
                    _ => panic!("Expected ImplItem::Const"),
                }
            }
            _ => panic!("Expected Item::Impl"),
        }
    }

    #[test]
    fn parse_impl_block_with_const_and_fn() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub const DEFAULT_INTENSITY: f32 = 1.0;
                pub fn attenuate(light: Light, distance: f32) -> f32 {
                    light.intensity / (distance * distance)
                }
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Impl(item_impl) => {
                assert_eq!("Light", item_impl.self_ty.to_string());
                assert_eq!(2, item_impl.items.len());
                match &item_impl.items[0] {
                    ImplItem::Const(item_const) => {
                        assert_eq!("DEFAULT_INTENSITY", item_const.ident.to_string());
                    }
                    _ => panic!("Expected ImplItem::Const"),
                }
                match &item_impl.items[1] {
                    ImplItem::Fn(item_fn) => {
                        assert_eq!("attenuate", item_fn.ident.to_string());
                    }
                    _ => panic!("Expected ImplItem::Fn"),
                }
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

    // Type::CONSTANT path parsing tests
    #[test]
    fn parse_type_path_expr() {
        let expr: syn::Expr = syn::parse_quote! { Light::INTENSITY };
        let expr = Expr::try_from(&expr).unwrap();
        match expr {
            Expr::TypePath { ty, member, .. } => {
                assert_eq!(ty.to_string(), "Light");
                assert_eq!(member.to_string(), "INTENSITY");
            }
            _ => panic!("Expected Expr::TypePath"),
        }
    }

    #[test]
    fn type_path_generates_mangled_name() {
        let expr: syn::Expr = syn::parse_quote! { Light::INTENSITY };
        let expr = Expr::try_from(&expr).unwrap();
        let wgsl = expr.to_wgsl();
        assert_eq!(wgsl, "Light_INTENSITY");
    }

    // WGSL code generation tests for impl constants
    #[test]
    fn impl_const_generates_mangled_name() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub const INTENSITY: f32 = 1.0;
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("const Light_INTENSITY"),
            "Expected 'const Light_INTENSITY' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn impl_const_and_fn_generate_mangled_names() {
        let item: syn::Item = syn::parse_quote! {
            impl Light {
                pub const DEFAULT: f32 = 1.0;
                pub fn get(l: Light) -> f32 { l.x }
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("const Light_DEFAULT"),
            "Expected 'const Light_DEFAULT' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("fn Light_get"),
            "Expected 'fn Light_get' in WGSL output, got: {}",
            wgsl
        );
    }

    // Enum parsing and code generation tests
    #[test]
    fn parse_enum_basic() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum State {
                Idle,
                Running,
                Stopped,
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Enum(item_enum) => {
                assert_eq!("State", item_enum.ident.to_string());
                assert_eq!(3, item_enum.variants.len());
                assert_eq!("Idle", item_enum.variants[0].ident.to_string());
                assert_eq!("Running", item_enum.variants[1].ident.to_string());
                assert_eq!("Stopped", item_enum.variants[2].ident.to_string());
            }
            _ => panic!("Expected Item::Enum"),
        }
    }

    #[test]
    fn parse_enum_with_explicit_discriminants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum Priority {
                Low = 1,
                Medium = 5,
                High = 10,
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Enum(item_enum) => {
                assert_eq!(3, item_enum.variants.len());
                assert!(item_enum.variants[0].discriminant.is_some());
                assert!(item_enum.variants[1].discriminant.is_some());
                assert!(item_enum.variants[2].discriminant.is_some());
                let (_, lit) = item_enum.variants[0].discriminant.as_ref().unwrap();
                assert_eq!("1", lit.to_string());
            }
            _ => panic!("Expected Item::Enum"),
        }
    }

    #[test]
    fn parse_enum_mixed_discriminants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum Mixed {
                A,
                B = 10,
                C,
            }
        };
        let item = Item::try_from(&item).unwrap();
        match item {
            Item::Enum(item_enum) => {
                assert!(item_enum.variants[0].discriminant.is_none());
                assert!(item_enum.variants[1].discriminant.is_some());
                assert!(item_enum.variants[2].discriminant.is_none());
            }
            _ => panic!("Expected Item::Enum"),
        }
    }

    #[test]
    fn enum_rejects_missing_repr() {
        let item: syn::Item = syn::parse_quote! {
            pub enum NoRepr {
                A,
                B,
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("repr(u32)"),
            "Expected error about repr(u32), got: {}",
            err
        );
    }

    #[test]
    fn enum_rejects_wrong_repr() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(i32)]
            pub enum WrongRepr {
                A,
                B,
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("repr(u32)"),
            "Expected error about repr(u32), got: {}",
            err
        );
    }

    #[test]
    fn enum_rejects_tuple_variants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum WithTuple {
                A,
                B(u32),
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("tuple variants"),
            "Expected error about tuple variants, got: {}",
            err
        );
    }

    #[test]
    fn enum_rejects_struct_variants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum WithStruct {
                A,
                B { x: u32 },
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("struct variants"),
            "Expected error about struct variants, got: {}",
            err
        );
    }

    #[test]
    fn enum_rejects_generics() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum Generic<T> {
                A,
                B,
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("generic"),
            "Expected error about generics, got: {}",
            err
        );
    }

    #[test]
    fn enum_rejects_non_public() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            enum Private {
                A,
                B,
            }
        };
        let result = Item::try_from(&item);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("public"),
            "Expected error about visibility, got: {}",
            err
        );
    }

    #[test]
    fn enum_generates_wgsl_constants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum State {
                Idle,
                Running,
                Stopped,
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("const State_Idle: u32 = 0u;"),
            "Expected 'const State_Idle: u32 = 0u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const State_Running: u32 = 1u;"),
            "Expected 'const State_Running: u32 = 1u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const State_Stopped: u32 = 2u;"),
            "Expected 'const State_Stopped: u32 = 2u;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn enum_generates_wgsl_with_explicit_discriminants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum Priority {
                Low = 1,
                Medium = 5,
                High = 10,
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("const Priority_Low: u32 = 1u;"),
            "Expected 'const Priority_Low: u32 = 1u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Priority_Medium: u32 = 5u;"),
            "Expected 'const Priority_Medium: u32 = 5u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Priority_High: u32 = 10u;"),
            "Expected 'const Priority_High: u32 = 10u;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn enum_generates_wgsl_with_mixed_discriminants() {
        let item: syn::Item = syn::parse_quote! {
            #[repr(u32)]
            pub enum Mixed {
                A,
                B = 10,
                C,
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("const Mixed_A: u32 = 0u;"),
            "Expected 'const Mixed_A: u32 = 0u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Mixed_B: u32 = 10u;"),
            "Expected 'const Mixed_B: u32 = 10u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Mixed_C: u32 = 11u;"),
            "Expected 'const Mixed_C: u32 = 11u;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn enum_variant_usage_via_type_path() {
        // This test verifies that EnumName::Variant works via the existing TypePath
        // mechanism
        let expr: syn::Expr = syn::parse_quote! { State::Running };
        let expr = Expr::try_from(&expr).unwrap();
        let wgsl = expr.to_wgsl();
        assert_eq!(wgsl, "State_Running");
    }

    #[test]
    fn parse_break_statement() {
        let stmt: syn::Stmt = syn::parse_quote! { break; };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert_eq!(wgsl, "break;");
    }

    #[test]
    fn parse_break_in_while_loop() {
        let stmt: syn::Stmt = syn::parse_quote! {
            while i < 10 {
                if i >= 5 {
                    break;
                }
                i += 1;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("break;"),
            "Expected 'break;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn break_with_label_rejected() {
        let stmt: syn::Stmt = syn::parse_quote! { break 'outer; };
        let result = Stmt::try_from(&stmt);
        assert!(
            result.is_err(),
            "Expected break with label to be rejected, but it succeeded"
        );
    }

    #[test]
    fn break_with_value_rejected() {
        let stmt: syn::Stmt = syn::parse_quote! { break 42; };
        let result = Stmt::try_from(&stmt);
        assert!(
            result.is_err(),
            "Expected break with value to be rejected, but it succeeded"
        );
    }
}
