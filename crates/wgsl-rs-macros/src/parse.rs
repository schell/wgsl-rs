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
use proc_macro2::Span;
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

    #[snafu(display("{}", warning.name))]
    SuppressableWarning { warning: Warning },

    #[snafu(display(
        "Cannot define function '{name}': this name is reserved for the WGSL builtin '{wgsl_name}'"
    ))]
    ReservedBuiltinName {
        span: proc_macro2::Span,
        name: String,
        wgsl_name: &'static str,
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
            Error::SuppressableWarning { warning } => warning
                .spans
                .first()
                .copied()
                .unwrap_or_else(Span::call_site),
            Error::ReservedBuiltinName { span, .. } => *span,
        }
    }
}

impl From<Error> for syn::Error {
    fn from(e: Error) -> Self {
        syn::Error::new(e.span(), format!("Parsing error: '{e}'"))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum WarningName {
    /// For-loops in WGSL only support ascending iteration (i++).
    /// Non-literal bounds may cause unexpected behavior if the range is
    /// descending.
    NonLiteralLoopBounds,
    /// Match statement patterns that are not integer literals cannot be
    /// verified at compile-time to be valid WGSL case selectors.
    NonLiteralMatchStatementPatterns,
}

impl std::fmt::Display for WarningName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarningName::NonLiteralLoopBounds => f.write_str(
                "for-loops with non-literal bounds cannot be verified at compile-time. \
                 Additionally, only ascending iteration is supported. If you are sure this loop's \
                 iterator is ascending, add #[wgsl_allow(non_literal_loop_bounds)] to the \
                 for-loop to suppress this error.",
            ),
            WarningName::NonLiteralMatchStatementPatterns => f.write_str(
                "match statement patterns that are not integer literals cannot be verified at \
                 compile-time to be valid WGSL case selectors. Add \
                 #[wgsl_allow(non_literal_match_statement_patterns)] to the match statement to \
                 suppress this warning.",
            ),
        }
    }
}

impl TryFrom<&syn::Ident> for WarningName {
    type Error = Error;

    fn try_from(ident: &syn::Ident) -> Result<Self, Self::Error> {
        match ident.to_string().as_str() {
            "non_literal_loop_bounds" => Ok(WarningName::NonLiteralLoopBounds),
            "non_literal_match_statement_patterns" => {
                Ok(WarningName::NonLiteralMatchStatementPatterns)
            }
            other => UnsupportedSnafu {
                span: ident.span(),
                note: format!("Unknown warning name '{other}'"),
            }
            .fail(),
        }
    }
}

#[derive(Debug)]
/// A warning that can be emitted during WGSL code generation.
pub struct Warning {
    pub name: WarningName,
    /// Spans that generated the warning
    pub spans: Vec<Span>,
}

/// Emits a warning diagnostic on nightly, no-op on stable.
///
/// On nightly, this uses `proc_macro::Diagnostic` to emit a proper compiler
/// warning. If called outside of a proc macro context (e.g., in unit tests),
/// this is a no-op.
pub(crate) fn emit_warning(_warning: &Warning) {
    #[cfg(nightly)]
    {
        use proc_macro::{Diagnostic, Level};

        // proc_macro::Span is only available during actual proc macro expansion.
        // In unit tests, attempting to use it will panic. We catch this and silently
        // ignore the warning in test contexts.
        let result = std::panic::catch_unwind(|| {
            let span = _warning
                .spans
                .first()
                .map(|s| s.unwrap())
                .unwrap_or_else(proc_macro::Span::call_site);

            let help_msg = match _warning.name {
                WarningName::NonLiteralLoopBounds => {
                    "Add #[wgsl_allow(non_literal_loop_bounds)] to suppress this warning"
                }
                WarningName::NonLiteralMatchStatementPatterns => {
                    "Add #[wgsl_allow(non_literal_match_statement_patterns)] to suppress this \
                     warning"
                }
            };

            Diagnostic::spanned(span, Level::Warning, format!("{}", _warning.name))
                .help(help_msg)
                .emit();
        });

        // Silently ignore if we're not in a proc macro context (e.g., unit tests)
        let _ = result;
    }
}

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

    #[allow(dead_code, reason = "Used in development")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl ScalarType {
    /// Returns the WGSL name for this scalar type.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            ScalarType::I32 => "i32",
            ScalarType::U32 => "u32",
            ScalarType::F32 => "f32",
            ScalarType::Bool => "bool",
        }
    }
}

/// WGSL address spaces for pointer and variable types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Workgroup is used for workgroup variables, not pointer types (yet)
pub enum AddressSpace {
    /// The `function` address space - local function variables.
    Function,
    /// The `private` address space - module-scope private variables.
    Private,
    /// The `workgroup` address space - shared within a compute shader
    /// workgroup. Variables in this address space are shared between
    /// invocations in the same workgroup.
    Workgroup,
}

/// Sampled texture dimensionality/kind.
/// These correspond to WGSL's texture_* types that are generic over a sample
/// type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    /// texture_1d<T>
    Texture1D,
    /// texture_2d<T>
    Texture2D,
    /// texture_2d_array<T>
    Texture2DArray,
    /// texture_3d<T>
    Texture3D,
    /// texture_cube<T>
    TextureCube,
    /// texture_cube_array<T>
    TextureCubeArray,
    /// texture_multisampled_2d<T>
    TextureMultisampled2D,
}

impl TextureKind {
    /// Returns the WGSL type name for this texture kind.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            TextureKind::Texture1D => "texture_1d",
            TextureKind::Texture2D => "texture_2d",
            TextureKind::Texture2DArray => "texture_2d_array",
            TextureKind::Texture3D => "texture_3d",
            TextureKind::TextureCube => "texture_cube",
            TextureKind::TextureCubeArray => "texture_cube_array",
            TextureKind::TextureMultisampled2D => "texture_multisampled_2d",
        }
    }

    /// Parse a Rust type name into a TextureKind.
    pub fn from_rust_name(name: &str) -> Option<Self> {
        match name {
            "Texture1D" => Some(TextureKind::Texture1D),
            "Texture2D" => Some(TextureKind::Texture2D),
            "Texture2DArray" => Some(TextureKind::Texture2DArray),
            "Texture3D" => Some(TextureKind::Texture3D),
            "TextureCube" => Some(TextureKind::TextureCube),
            "TextureCubeArray" => Some(TextureKind::TextureCubeArray),
            "TextureMultisampled2D" => Some(TextureKind::TextureMultisampled2D),
            _ => None,
        }
    }
}

/// Depth texture dimensionality/kind.
/// These correspond to WGSL's texture_depth_* types which have no type
/// parameter (they implicitly use f32 for depth values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDepthKind {
    /// texture_depth_2d
    Depth2D,
    /// texture_depth_2d_array
    Depth2DArray,
    /// texture_depth_cube
    DepthCube,
    /// texture_depth_cube_array
    DepthCubeArray,
    /// texture_depth_multisampled_2d
    DepthMultisampled2D,
}

impl TextureDepthKind {
    /// Returns the WGSL type name for this depth texture kind.
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            TextureDepthKind::Depth2D => "texture_depth_2d",
            TextureDepthKind::Depth2DArray => "texture_depth_2d_array",
            TextureDepthKind::DepthCube => "texture_depth_cube",
            TextureDepthKind::DepthCubeArray => "texture_depth_cube_array",
            TextureDepthKind::DepthMultisampled2D => "texture_depth_multisampled_2d",
        }
    }

    /// Parse a Rust type name into a TextureDepthKind.
    pub fn from_rust_name(name: &str) -> Option<Self> {
        match name {
            "TextureDepth2D" => Some(TextureDepthKind::Depth2D),
            "TextureDepth2DArray" => Some(TextureDepthKind::Depth2DArray),
            "TextureDepthCube" => Some(TextureDepthKind::DepthCube),
            "TextureDepthCubeArray" => Some(TextureDepthKind::DepthCubeArray),
            "TextureDepthMultisampled2D" => Some(TextureDepthKind::DepthMultisampled2D),
            _ => None,
        }
    }
}

/// Helper struct for parsing `ptr!(address_space, Type)` macro arguments.
struct PtrMacroArgs {
    address_space: Ident,
    _comma: Token![,],
    store_type: syn::Type,
}

impl Parse for PtrMacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(PtrMacroArgs {
            address_space: input.parse()?,
            _comma: input.parse()?,
            store_type: input.parse()?,
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

    /// Runtime-sized array type: RuntimeArray<T>
    /// Transpiles to array<T> in WGSL (no size parameter)
    RuntimeArray {
        ident: Ident,
        lt_token: Token![<],
        elem: Box<Type>,
        gt_token: Token![>],
    },

    /// Atomic type: atomic<T> where T is i32 or u32
    /// Used for thread-safe atomic operations in workgroup/storage address
    /// spaces. Only the atomic builtin functions can operate on atomic
    /// types.
    Atomic {
        ident: Ident,
        lt_token: Token![<],
        elem: Box<Type>,
        gt_token: Token![>],
    },

    /// Struct type: eg. MyStruct
    Struct { ident: Ident },

    /// Pointer type: ptr<address_space, T>
    /// Created from `ptr!(address_space, T)` macro invocations.
    ///
    /// In WGSL, pointers allow passing mutable references to functions.
    /// Only `function` and `private` address spaces are supported.
    Ptr {
        address_space: AddressSpace,
        elem: Box<Type>,
        /// Span of the original macro for error reporting
        span: proc_macro2::Span,
    },

    /// Sampler type: sampler
    /// Used for texture sampling operations.
    /// This is a handle to a GPU sampler object that controls how textures are
    /// sampled.
    Sampler { ident: Ident },

    /// Comparison sampler type: sampler_comparison
    /// Used for depth texture comparison sampling operations.
    /// Returns a comparison result rather than a filtered sample.
    SamplerComparison { ident: Ident },

    /// Sampled texture types: texture_1d<T>, texture_2d<T>, etc.
    /// T must be f32, i32, or u32 (the sample type).
    Texture {
        kind: TextureKind,
        sampled_type: ScalarType,
        ident: Ident,
    },

    /// Depth texture types: texture_depth_2d, etc.
    /// No type parameter (implicitly f32 for depth values).
    TextureDepth {
        kind: TextureDepthKind,
        ident: Ident,
    },
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
                        // Sampler types
                        "Sampler" => Type::Sampler {
                            ident: ident.clone(),
                        },
                        "SamplerComparison" => Type::SamplerComparison {
                            ident: ident.clone(),
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
                            } else if let Some(kind) = TextureDepthKind::from_rust_name(other) {
                                // Depth texture types (no type parameter)
                                Type::TextureDepth {
                                    kind,
                                    ident: ident.clone(),
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

                    // Check for RuntimeArray
                    let is_runtime_array = ident_str == "RuntimeArray";

                    // Check for Atomic
                    let is_atomic = ident_str == "Atomic";

                    let arg = args.first().expect("checked that len was 1");
                    match arg {
                        syn::GenericArgument::Type(ty) => {
                            // Handle RuntimeArray<T> first - it can take any element type
                            if is_runtime_array {
                                let elem_type = Type::try_from(ty)?;
                                return Ok(Type::RuntimeArray {
                                    ident: ident.clone(),
                                    lt_token: *lt_token,
                                    elem: Box::new(elem_type),
                                    gt_token: *gt_token,
                                });
                            }

                            // Handle Atomic<T> - only i32 or u32 allowed
                            if is_atomic {
                                let elem_type = Type::try_from(ty)?;
                                match &elem_type {
                                    Type::Scalar {
                                        ty: ScalarType::I32 | ScalarType::U32,
                                        ..
                                    } => {
                                        return Ok(Type::Atomic {
                                            ident: ident.clone(),
                                            lt_token: *lt_token,
                                            elem: Box::new(elem_type),
                                            gt_token: *gt_token,
                                        });
                                    }
                                    _ => {
                                        return UnsupportedSnafu {
                                            span: ty.span(),
                                            note: "Atomic<T> requires T to be i32 or u32",
                                        }
                                        .fail();
                                    }
                                }
                            }

                            // Handle sampled texture types: Texture1D<T>, Texture2D<T>, etc.
                            // T must be f32, i32, or u32
                            if let Some(texture_kind) = TextureKind::from_rust_name(&ident_str) {
                                let elem_type = Type::try_from(ty)?;
                                match &elem_type {
                                    Type::Scalar {
                                        ty:
                                            sampled_type @ (ScalarType::F32
                                            | ScalarType::I32
                                            | ScalarType::U32),
                                        ..
                                    } => {
                                        return Ok(Type::Texture {
                                            kind: texture_kind,
                                            sampled_type: *sampled_type,
                                            ident: ident.clone(),
                                        });
                                    }
                                    _ => {
                                        return UnsupportedSnafu {
                                            span: ty.span(),
                                            note: "Sampled texture type parameter must be f32, \
                                                   i32, or u32",
                                        }
                                        .fail();
                                    }
                                }
                            }

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
                                               Vec3, Vec4, Mat2, Mat3, Mat4, RuntimeArray, \
                                               Atomic, or a texture type (Texture1D, Texture2D, \
                                               etc.)",
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
        } else if let syn::Type::Macro(syn::TypeMacro { mac }) = ty {
            // Handle macro types like ptr!(function, i32)
            let macro_name = mac.path.get_ident().map(|i| i.to_string());

            if macro_name.as_deref() == Some("ptr") {
                // Parse the macro tokens: address_space, Type
                let tokens = mac.tokens.clone();
                let parsed: PtrMacroArgs = syn::parse2(tokens)?;

                let address_space = match parsed.address_space.to_string().as_str() {
                    "function" => AddressSpace::Function,
                    "private" => AddressSpace::Private,
                    other => {
                        return UnsupportedSnafu {
                            span: parsed.address_space.span(),
                            note: format!(
                                "unsupported address space '{}', only 'function' and 'private' \
                                 are supported",
                                other
                            ),
                        }
                        .fail();
                    }
                };

                let elem = Type::try_from(&parsed.store_type)?;

                Ok(Type::Ptr {
                    address_space,
                    elem: Box::new(elem),
                    span: mac.span(),
                })
            } else {
                UnsupportedSnafu {
                    span: mac.span(),
                    note: format!(
                        "unsupported macro '{}!' in type position, only 'ptr!' is supported",
                        macro_name.unwrap_or_else(|| "unknown".to_string())
                    ),
                }
                .fail()
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
    /// Dereference operator: `*ptr`
    /// Used to access the value pointed to by a pointer.
    Deref(Token![*]),
}

impl TryFrom<&syn::UnOp> for UnOp {
    type Error = Error;

    fn try_from(value: &syn::UnOp) -> Result<Self, Self::Error> {
        Ok(match value {
            syn::UnOp::Not(t) => UnOp::Not(*t),
            syn::UnOp::Neg(t) => UnOp::Neg(*t),
            syn::UnOp::Deref(t) => UnOp::Deref(*t),
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
    /// A reference expression like `&expr`.
    ///
    /// In WGSL, this is used to create pointers, particularly for builtin
    /// functions like `arrayLength` that require pointer arguments.
    Reference {
        and_token: Token![&],
        expr: Box<Expr>,
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
            // Match expressions are not supported - only match statements
            syn::Expr::Match(expr_match) => {
                return UnsupportedSnafu {
                    span: expr_match.match_token.span,
                    note: "match expressions are not supported in WGSL. WGSL switch is a \
                           statement, not an expression. Use match as a statement instead (not in \
                           let bindings or return positions).",
                }
                .fail();
            }
            // Reference expressions like `&expr` - used for pointers in WGSL
            syn::Expr::Reference(syn::ExprReference {
                attrs: _,
                and_token,
                mutability,
                expr,
            }) => {
                // WGSL doesn't distinguish between `&` and `&mut` at the syntax level
                // (mutability is determined by the address space and access mode)
                let _ = mutability;
                Self::Reference {
                    and_token: *and_token,
                    expr: Box::new(Expr::try_from(expr.as_ref())?),
                }
            }
            // Handle get_mut!(IDENT) - strip the macro, return just the ident for WGSL
            syn::Expr::Macro(syn::ExprMacro { attrs: _, mac }) => {
                let trigger_unsupported = || {
                    UnsupportedSnafu {
                        span: mac.path.span(),
                        note: format!(
                            "unsupported macro '{}!' in expression position, only mutate! is \
                             supported",
                            mac.path.to_token_stream()
                        ),
                    }
                    .fail()
                };
                // Some macros have no meaning in WGSL, and we will simply strip them
                let noop_macros = ["get_mut", "get"];
                if let Some(macro_ident) = mac.path.get_ident() {
                    let macro_ident_str = macro_ident.to_string();
                    if noop_macros.contains(&macro_ident_str.as_str()) {
                        // Parse the tokens inside the macro as a single identifier
                        let ident: Ident = syn::parse2(mac.tokens.clone()).map_err(|e| {
                            UnsupportedSnafu {
                                span: mac.path.span(),
                                note: format!(
                                    "{macro_ident_str}! expects a single identifier, got: {e}"
                                ),
                            }
                            .build()
                        })?;
                        Self::Ident(ident)
                    } else {
                        return trigger_unsupported();
                    }
                } else {
                    return trigger_unsupported();
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

impl Expr {
    /// Returns true if this expression is a literal value.
    pub fn is_literal(&self) -> bool {
        matches!(self, Expr::Lit(_))
    }

    /// Returns the span of this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::Lit(lit) => match lit {
                Lit::Bool(b) => b.span(),
                Lit::Float(f) => f.span(),
                Lit::Int(i) => i.span(),
            },
            Expr::Ident(ident) => ident.span(),
            Expr::Array { bracket_token, .. } => bracket_token.span.join(),
            Expr::Paren { paren_token, .. } => paren_token.span.join(),
            Expr::Binary { lhs, rhs, .. } => {
                lhs.span().join(rhs.span()).unwrap_or_else(|| lhs.span())
            }
            Expr::Unary { op, expr } => {
                let op_span = match op {
                    UnOp::Not(t) => t.span,
                    UnOp::Neg(t) => t.span,
                    UnOp::Deref(t) => t.span,
                };
                op_span.join(expr.span()).unwrap_or(op_span)
            }
            Expr::ArrayIndexing {
                lhs, bracket_token, ..
            } => lhs
                .span()
                .join(bracket_token.span.join())
                .unwrap_or_else(|| lhs.span()),
            Expr::Swizzle { lhs, swizzle, .. } => lhs
                .span()
                .join(swizzle.span())
                .unwrap_or_else(|| lhs.span()),
            Expr::Cast { lhs, ty } => {
                let ty_span = match ty.as_ref() {
                    Type::Scalar { ident, .. } => ident.span(),
                    Type::Vector { ident, .. } => ident.span(),
                    Type::Matrix { ident, .. } => ident.span(),
                    Type::Array { bracket_token, .. } => bracket_token.span.join(),
                    Type::RuntimeArray { ident, .. } => ident.span(),
                    Type::Atomic { ident, .. } => ident.span(),
                    Type::Struct { ident } => ident.span(),
                    Type::Ptr { span, .. } => *span,
                    Type::Sampler { ident } => ident.span(),
                    Type::SamplerComparison { ident } => ident.span(),
                    Type::Texture { ident, .. } => ident.span(),
                    Type::TextureDepth { ident, .. } => ident.span(),
                };
                lhs.span().join(ty_span).unwrap_or_else(|| lhs.span())
            }
            Expr::FnCall {
                path, paren_token, ..
            } => {
                let path_span = match path {
                    FnPath::Ident(ident) => ident.span(),
                    FnPath::TypeMethod { ty, .. } => ty.span(),
                };
                path_span.join(paren_token.span.join()).unwrap_or(path_span)
            }
            Expr::Struct {
                ident, brace_token, ..
            } => ident
                .span()
                .join(brace_token.span.join())
                .unwrap_or_else(|| ident.span()),
            Expr::FieldAccess { base, field, .. } => base
                .span()
                .join(field.span())
                .unwrap_or_else(|| base.span()),
            Expr::TypePath { ty, member, .. } => {
                ty.span().join(member.span()).unwrap_or_else(|| ty.span())
            }
            Expr::Reference { and_token, expr } => and_token
                .span
                .join(expr.span())
                .unwrap_or_else(|| and_token.span),
        }
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
    /// Loop statement: `loop { ... }`
    ///
    /// WGSL-specific infinite loop without a condition.
    /// Note: `continuing { ... }` blocks are not supported.
    Loop {
        loop_token: Token![loop],
        body: Block,
    },
    /// Any expression.
    Expr {
        expr: Expr,
        /// If `None`, this expression is considered a return statement.
        semi_token: Option<Token![;]>,
    },
    /// If statement (with optional else/else-if chains)
    If(Box<StmtIf>),
    /// Break statement: `break;`
    Break {
        break_token: Token![break],
        semi_token: Token![;],
    },
    /// Continue statement: `continue;`
    Continue {
        continue_token: Token![continue],
        semi_token: Token![;],
    },
    /// Return statement with optional expression: `return;` or `return expr;`
    Return {
        return_token: Token![return],
        expr: Option<Expr>,
        semi_token: Token![;],
    },
    /// A for-loop statement.
    For(Box<ForLoop>),
    /// Switch statement (from Rust match)
    Switch(Box<StmtSwitch>),
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
                match expr {
                    // Handle for-loops
                    syn::Expr::ForLoop(for_loop) => {
                        Ok(Stmt::For(Box::new(ForLoop::try_from(for_loop)?)))
                    }
                    // Handle assignments
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
                    // Loop statement: `loop { ... }`
                    syn::Expr::Loop(syn::ExprLoop {
                        attrs: _,
                        label,
                        loop_token,
                        body,
                    }) => {
                        util::some_is_unsupported(
                            label.as_ref(),
                            "Labels on loops are not supported in WGSL",
                        )?;
                        Ok(Stmt::Loop {
                            loop_token: *loop_token,
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
                        expr: break_expr,
                    }) => {
                        util::some_is_unsupported(
                            label.as_ref(),
                            "Labels on break statements are not supported in WGSL",
                        )?;
                        util::some_is_unsupported(
                            break_expr.as_ref(),
                            "Break with values is not supported in WGSL",
                        )?;
                        let semi_token = semi_token.ok_or_else(|| Error::Unsupported {
                            span: expr.span(),
                            note: "Break statements must end with a semicolon".to_string(),
                        })?;
                        Ok(Stmt::Break {
                            break_token: *break_token,
                            semi_token,
                        })
                    }
                    // Continue statement: `continue;`
                    syn::Expr::Continue(syn::ExprContinue {
                        attrs: _,
                        continue_token,
                        label,
                    }) => {
                        util::some_is_unsupported(
                            label.as_ref(),
                            "Labels on continue statements are not supported in WGSL",
                        )?;
                        let semi_token = semi_token.ok_or_else(|| Error::Unsupported {
                            span: expr.span(),
                            note: "Continue statements must end with a semicolon".to_string(),
                        })?;
                        Ok(Stmt::Continue {
                            continue_token: *continue_token,
                            semi_token,
                        })
                    }
                    // Return statement: `return expr;` or `return;`
                    syn::Expr::Return(syn::ExprReturn {
                        attrs: _,
                        return_token,
                        expr: return_expr,
                    }) => {
                        let semi_token = semi_token.ok_or_else(|| Error::Unsupported {
                            span: expr.span(),
                            note: "Return statements must end with a semicolon".to_string(),
                        })?;
                        let expr_opt = if let Some(e) = return_expr {
                            Some(Expr::try_from(e.as_ref())?)
                        } else {
                            None
                        };
                        Ok(Stmt::Return {
                            return_token: *return_token,
                            expr: expr_opt,
                            semi_token,
                        })
                    }
                    // Match statement → Switch statement
                    syn::Expr::Match(expr_match) => {
                        Ok(Stmt::Switch(Box::new(StmtSwitch::try_from(expr_match)?)))
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

/// A for-loop statement.
///
/// Supports Rust range-based for-loops:
/// - `for i in from..to { ... }` (exclusive upper bound)
/// - `for i in from..=to { ... }` (inclusive upper bound)
///
/// Transpiles to WGSL:
/// - `for (var i = from; i < to; i++)` (exclusive)
/// - `for (var i = from; i <= to; i++)` (inclusive)
///
/// # Limitations
/// - Only ascending iteration is supported (WGSL uses `i++`)
/// - Labels are not supported
/// - Only range expressions are supported (no arbitrary iterators)
/// - The loop variable cannot be `_`
pub struct ForLoop {
    pub for_token: Token![for],
    /// The loop variable identifier
    pub ident: Ident,
    /// Optional explicit type annotation for the loop variable
    // Note: Rust doesn't support type annotations in for-loop patterns like `for i:
    // u32 in 0..10`. The type is inferred from the range expression.
    // We keep the ty field in ForLoop for potential future use or manual
    // construction, but it will always be None when parsing from Rust source.
    pub ty: Option<(Token![:], Type)>,
    pub _in_token: Token![in],
    /// The starting value of the range
    pub from: Expr,
    /// If true, the range is inclusive (`..=`), otherwise exclusive (`..`)
    pub inclusive: bool,
    /// The span of the range operator for error reporting
    pub range_span: Span,
    /// The ending value of the range
    pub to: Expr,
    /// The loop body
    pub body: Block,
}

impl TryFrom<&syn::ExprForLoop> for ForLoop {
    type Error = Error;

    fn try_from(value: &syn::ExprForLoop) -> Result<Self, Self::Error> {
        let syn::ExprForLoop {
            attrs,
            label,
            for_token,
            pat,
            in_token,
            expr,
            body,
        } = value;

        // Parse allowed warnings from attributes on the for-loop itself
        let allowed = parse_wgsl_allow(attrs)?;

        // Reject labels
        if let Some(label) = label {
            return UnsupportedSnafu {
                span: label.span(),
                note: "Labels on for-loops are not supported in WGSL",
            }
            .fail();
        }

        // Parse the pattern to get ident and optional type
        let (ident, ty) = parse_for_loop_pattern(pat)?;

        // Parse the range expression
        let (from, inclusive, range_span, to) = parse_range_expr(expr)?;

        // Check for non-literal bounds and emit warning/error if not suppressed
        let mut non_literal_spans = vec![];
        if !from.is_literal() {
            non_literal_spans.push(from.span());
        }
        if !to.is_literal() {
            non_literal_spans.push(to.span());
        }

        if !non_literal_spans.is_empty() && !allowed.contains(&WarningName::NonLiteralLoopBounds) {
            let warning = Warning {
                name: WarningName::NonLiteralLoopBounds,
                spans: non_literal_spans,
            };

            if cfg!(nightly) {
                // Emit the warning and continue parsing on nightly
                emit_warning(&warning);
            } else {
                return Err(Error::SuppressableWarning { warning });
            }
        }

        // Parse the body
        let body = Block::try_from(body)?;

        Ok(ForLoop {
            for_token: *for_token,
            ident,
            ty,
            _in_token: *in_token,
            from,
            inclusive,
            range_span,
            to,
            body,
        })
    }
}

/// Parse `#[wgsl_allow(warning1, warning2, ...)]` attributes.
/// Returns the list of warning names that should be suppressed.
fn parse_wgsl_allow(attrs: &[syn::Attribute]) -> Result<Vec<WarningName>, Error> {
    let mut allowed = vec![];
    for attr in attrs {
        if attr.path().is_ident("wgsl_allow") {
            let list = attr.meta.require_list()?;
            let idents = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Ident, Token![,]>::parse_terminated,
            )?;
            for ident in idents {
                allowed.push(WarningName::try_from(&ident)?);
            }
        }
    }
    Ok(allowed)
}

/// Parse a for-loop pattern to extract the identifier and optional type
/// annotation.
fn parse_for_loop_pattern(pat: &syn::Pat) -> Result<(Ident, Option<(Token![:], Type)>), Error> {
    match pat {
        syn::Pat::Ident(syn::PatIdent {
            attrs: _,
            by_ref,
            mutability: _,
            ident,
            subpat,
        }) => {
            // Reject `ref` bindings
            if let Some(by_ref) = by_ref {
                return UnsupportedSnafu {
                    span: by_ref.span(),
                    note: "WGSL does not support 'ref' in for-loop patterns",
                }
                .fail();
            }

            // Reject subpatterns
            if let Some((at, _)) = subpat {
                return UnsupportedSnafu {
                    span: at.span(),
                    note: "WGSL does not support '@' patterns in for-loops",
                }
                .fail();
            }

            // Reject wildcard patterns
            if ident == "_" {
                return UnsupportedSnafu {
                    span: ident.span(),
                    note: "WGSL for-loops require a named loop variable, not '_'",
                }
                .fail();
            }

            Ok((ident.clone(), None))
        }
        syn::Pat::Type(syn::PatType {
            attrs: _,
            pat,
            colon_token,
            ty,
        }) => {
            // Recursively parse the inner pattern
            let (ident, _) = parse_for_loop_pattern(pat)?;
            let parsed_ty = Type::try_from(ty.as_ref())?;
            Ok((ident, Some((*colon_token, parsed_ty))))
        }
        syn::Pat::Wild(wild) => UnsupportedSnafu {
            span: wild.span(),
            note: "WGSL for-loops require a named loop variable, not '_'",
        }
        .fail(),
        other => UnsupportedSnafu {
            span: other.span(),
            note: format!(
                "Unsupported pattern in for-loop: '{}'. Only simple identifiers are supported.",
                other.into_token_stream()
            ),
        }
        .fail(),
    }
}

/// Parse a range expression to extract from, to, and whether it's inclusive.
fn parse_range_expr(expr: &syn::Expr) -> Result<(Expr, bool, Span, Expr), Error> {
    match expr {
        syn::Expr::Range(syn::ExprRange {
            attrs: _,
            start,
            limits,
            end,
        }) => {
            let from = start.as_ref().ok_or_else(|| Error::Unsupported {
                span: expr.span(),
                note: "For-loops require a start value (e.g., `0..10`, not `..10`)".to_string(),
            })?;

            let to = end.as_ref().ok_or_else(|| Error::Unsupported {
                span: expr.span(),
                note: "For-loops require an end value (e.g., `0..10`, not `0..`)".to_string(),
            })?;

            let (inclusive, range_span) = match limits {
                syn::RangeLimits::HalfOpen(dot2) => (false, dot2.span()),
                syn::RangeLimits::Closed(dot2eq) => (true, dot2eq.span()),
            };

            let from = Expr::try_from(from.as_ref())?;
            let to = Expr::try_from(to.as_ref())?;

            Ok((from, inclusive, range_span, to))
        }
        other => UnsupportedSnafu {
            span: other.span(),
            note: format!(
                "For-loops only support range expressions (e.g., `0..10` or `0..=9`). Saw: '{}'",
                other.into_token_stream()
            ),
        }
        .fail(),
    }
}

impl TryFrom<&syn::ExprMatch> for StmtSwitch {
    type Error = Error;

    fn try_from(value: &syn::ExprMatch) -> Result<Self, Self::Error> {
        let syn::ExprMatch {
            attrs,
            match_token,
            expr,
            brace_token,
            arms,
        } = value;

        // Parse allowed warnings from attributes
        let allowed = parse_wgsl_allow(attrs)?;

        // Parse the selector expression
        let selector = Box::new(Expr::try_from(expr.as_ref())?);

        // Track non-literal pattern spans for warning
        let mut non_literal_spans = vec![];
        let mut has_explicit_default = false;

        // Parse each arm
        let mut switch_arms = vec![];
        for arm in arms {
            let (selectors, arm_non_literal_spans, is_default) = parse_match_arm_pattern(&arm.pat)?;
            non_literal_spans.extend(arm_non_literal_spans);

            if is_default {
                has_explicit_default = true;
            }

            // Reject guard clauses
            if let Some((if_token, _)) = &arm.guard {
                return UnsupportedSnafu {
                    span: if_token.span(),
                    note: "guard clauses in match arms are not supported in WGSL switch statements",
                }
                .fail();
            }

            // Parse the body - must be a block
            let body = match &*arm.body {
                syn::Expr::Block(syn::ExprBlock { block, .. }) => Block::try_from(block)?,
                other => {
                    return UnsupportedSnafu {
                        span: other.span(),
                        note: "match arm bodies must be blocks in WGSL. Use `pattern => { ... }` \
                               syntax.",
                    }
                    .fail();
                }
            };

            switch_arms.push(SwitchArm {
                selectors,
                fat_arrow_span: arm.fat_arrow_token.span(),
                body,
            });
        }

        // Emit warning for non-literal patterns if not suppressed
        if !non_literal_spans.is_empty()
            && !allowed.contains(&WarningName::NonLiteralMatchStatementPatterns)
        {
            let warning = Warning {
                name: WarningName::NonLiteralMatchStatementPatterns,
                spans: non_literal_spans,
            };

            if cfg!(nightly) {
                emit_warning(&warning);
            } else {
                return Err(Error::SuppressableWarning { warning });
            }
        }

        Ok(StmtSwitch {
            match_token: *match_token,
            selector,
            brace_token: *brace_token,
            arms: switch_arms,
            has_explicit_default,
        })
    }
}

/// Parse a match arm pattern into case selectors.
/// Returns (selectors, non_literal_spans, has_default).
fn parse_match_arm_pattern(pat: &syn::Pat) -> Result<(Vec<CaseSelector>, Vec<Span>, bool), Error> {
    let mut selectors = vec![];
    let mut non_literal_spans = vec![];
    let mut has_default = false;

    parse_pattern_recursive(
        pat,
        &mut selectors,
        &mut non_literal_spans,
        &mut has_default,
    )?;

    Ok((selectors, non_literal_spans, has_default))
}

fn parse_pattern_recursive(
    pat: &syn::Pat,
    selectors: &mut Vec<CaseSelector>,
    non_literal_spans: &mut Vec<Span>,
    has_default: &mut bool,
) -> Result<(), Error> {
    match pat {
        // Wildcard: `_` → default
        syn::Pat::Wild(wild) => {
            *has_default = true;
            selectors.push(CaseSelector::Default(wild.underscore_token.span));
        }

        // Literal pattern: `0`, `42`, etc.
        syn::Pat::Lit(syn::PatLit { attrs: _, lit }) => {
            let lit = Lit::try_from(lit)?;
            selectors.push(CaseSelector::Literal(lit));
        }

        // Or-pattern: `1 | 2 | 3`
        syn::Pat::Or(syn::PatOr {
            attrs: _,
            leading_vert: _,
            cases,
        }) => {
            for case in cases {
                parse_pattern_recursive(case, selectors, non_literal_spans, has_default)?;
            }
        }

        // Path pattern: `MY_CONST` or `State::Running`
        syn::Pat::Path(syn::PatPath {
            attrs: _,
            qself,
            path,
        }) => {
            util::some_is_unsupported(qself.as_ref(), "QSelf is unsupported in patterns")?;

            // Convert to Expr and mark as non-literal
            let expr = Expr::try_from(&syn::Expr::Path(syn::ExprPath {
                attrs: vec![],
                qself: qself.clone(),
                path: path.clone(),
            }))?;
            non_literal_spans.push(path.span());
            selectors.push(CaseSelector::Expr(expr));
        }

        // Identifier pattern: `x` - could be a const binding
        syn::Pat::Ident(syn::PatIdent {
            attrs: _,
            by_ref,
            mutability,
            ident,
            subpat,
        }) => {
            // Reject ref/mut bindings
            util::some_is_unsupported(
                by_ref.as_ref(),
                "ref bindings not supported in switch patterns",
            )?;
            util::some_is_unsupported(
                mutability.as_ref(),
                "mut bindings not supported in switch patterns",
            )?;
            // subpat is Option<(Token![@], Box<Pat>)> - check the @ token
            if let Some((at_token, _)) = subpat {
                return UnsupportedSnafu {
                    span: at_token.span(),
                    note: "subpatterns (@) not supported in switch patterns",
                }
                .fail();
            }

            // Treat as identifier expression (could be a const)
            let expr = Expr::Ident(ident.clone());
            non_literal_spans.push(ident.span());
            selectors.push(CaseSelector::Expr(expr));
        }

        // Unsupported patterns
        syn::Pat::Range(range) => {
            return UnsupportedSnafu {
                span: range.span(),
                note: "range patterns are not supported in WGSL switch statements",
            }
            .fail();
        }

        syn::Pat::Struct(s) => {
            return UnsupportedSnafu {
                span: s.span(),
                note: "struct patterns are not supported in WGSL switch statements",
            }
            .fail();
        }

        syn::Pat::Tuple(t) => {
            return UnsupportedSnafu {
                span: t.span(),
                note: "tuple patterns are not supported in WGSL switch statements",
            }
            .fail();
        }

        syn::Pat::TupleStruct(ts) => {
            return UnsupportedSnafu {
                span: ts.span(),
                note: "tuple struct patterns are not supported in WGSL switch statements",
            }
            .fail();
        }

        syn::Pat::Slice(s) => {
            return UnsupportedSnafu {
                span: s.span(),
                note: "slice patterns are not supported in WGSL switch statements",
            }
            .fail();
        }

        other => {
            return UnsupportedSnafu {
                span: other.span(),
                note: format!(
                    "unsupported pattern in switch statement: '{}'",
                    other.to_token_stream()
                ),
            }
            .fail();
        }
    }

    Ok(())
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

/// WGSL switch statement (transpiled from Rust match).
///
/// WGSL switch statements require:
/// - Selector must be a concrete integer type (i32 or u32)
/// - Case selectors must be const-expressions
/// - Exactly one default clause (auto-generated if missing)
pub struct StmtSwitch {
    pub match_token: Token![match],
    pub selector: Box<Expr>,
    pub brace_token: syn::token::Brace,
    pub arms: Vec<SwitchArm>,
    /// Whether a default arm was explicitly provided
    pub has_explicit_default: bool,
}

/// A single arm in a switch statement.
pub struct SwitchArm {
    /// The case selectors (may contain Default for wildcard patterns)
    pub selectors: Vec<CaseSelector>,
    /// Span of the `=>` token (used for code generation)
    pub fat_arrow_span: Span,
    /// The arm body (always a block)
    pub body: Block,
}

/// A case selector value in a switch arm.
pub enum CaseSelector {
    /// Integer literal: `0`, `1`, `42u32`, etc.
    Literal(Lit),
    /// Named constant, enum variant, or other expression.
    /// Examples: `MY_CONST`, `State::Running`
    /// These emit a warning unless suppressed.
    Expr(Expr),
    /// The default case from `_` pattern.
    Default(Span),
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
                    // Skip workgroup_size in this pass (handled separately for compute)
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

        ensure_ident_is_not_shadowing_builtin(&sig.ident)?;

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

        ensure_ident_is_not_shadowing_builtin(&sig.ident)?;

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

fn ensure_ident_is_not_shadowing_builtin(ident: &Ident) -> Result<(), Error> {
    let fn_name = ident.to_string();
    // Check if function name conflicts with a WGSL builtin
    if let Some((_rust_name, wgsl_name)) = crate::builtins::is_reserved_builtin(&fn_name) {
        ReservedBuiltinNameSnafu {
            span: ident.span(),
            name: fn_name,
            wgsl_name,
        }
        .fail()
    } else {
        Ok(())
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

/// A sampler declaration.
///
/// ```rust,ignore
/// sampler!(group(0), binding(1), MY_SAMPLER: Sampler);
/// sampler!(group(0), binding(2), MY_CMP_SAMPLER: SamplerComparison);
/// ```
///
/// ```wgsl
/// @group(0) @binding(1) var MY_SAMPLER: sampler;
/// @group(0) @binding(2) var MY_CMP_SAMPLER: sampler_comparison;
/// ```
pub(crate) struct ItemSampler {
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
    #[expect(dead_code, reason = "Will be used eventually")]
    pub rust_ty: syn::Type,
}

impl syn::parse::Parse for ItemSampler {
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

        // Validate that the type is a sampler type
        match &ty {
            Type::Sampler { .. } | Type::SamplerComparison { .. } => {}
            _ => {
                return Err(syn::Error::new(
                    rust_ty.span(),
                    "sampler! macro requires a Sampler or SamplerComparison type",
                ));
            }
        }

        Ok(ItemSampler {
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

impl TryFrom<&syn::ItemMacro> for ItemSampler {
    type Error = Error;

    fn try_from(item_macro: &syn::ItemMacro) -> Result<Self, Self::Error> {
        // Ensure it's the "sampler" macro
        if item_macro
            .mac
            .path
            .get_ident()
            .map(|id| id != "sampler")
            .unwrap_or(true)
        {
            return UnsupportedSnafu {
                span: item_macro.span(),
                note: "Only 'sampler!' macro is supported as a sampler declaration.",
            }
            .fail();
        }

        syn::parse2::<ItemSampler>(item_macro.mac.tokens.clone()).map_err(|e| Error::Unsupported {
            span: item_macro.span(),
            note: format!("{e}"),
        })
    }
}

/// Texture variable declaration parsed from `texture!` macro.
///
/// Supports both sampled textures (with type parameter) and depth textures:
///
/// ```rust,ignore
/// texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
/// texture!(group(1), binding(0), SHADOW_MAP: TextureDepth2D);
/// ```
///
/// ```wgsl
/// @group(0) @binding(0) var DIFFUSE_TEX: texture_2d<f32>;
/// @group(1) @binding(0) var SHADOW_MAP: texture_depth_2d;
/// ```
pub(crate) struct ItemTexture {
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
    #[expect(dead_code, reason = "Might be used later")]
    pub rust_ty: syn::Type,
}

impl syn::parse::Parse for ItemTexture {
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

        // Validate that the type is a texture type
        match &ty {
            Type::Texture { .. } | Type::TextureDepth { .. } => {}
            _ => {
                return Err(syn::Error::new(
                    rust_ty.span(),
                    "texture! macro requires a texture type (Texture2D<f32>, TextureDepth2D, etc.)",
                ));
            }
        }

        Ok(ItemTexture {
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

impl TryFrom<&syn::ItemMacro> for ItemTexture {
    type Error = Error;

    fn try_from(item_macro: &syn::ItemMacro) -> Result<Self, Self::Error> {
        // Ensure it's the "texture" macro
        if item_macro
            .mac
            .path
            .get_ident()
            .map(|id| id != "texture")
            .unwrap_or(true)
        {
            return UnsupportedSnafu {
                span: item_macro.span(),
                note: "Only 'texture!' macro is supported as a texture declaration.",
            }
            .fail();
        }

        syn::parse2::<ItemTexture>(item_macro.mac.tokens.clone()).map_err(|e| Error::Unsupported {
            span: item_macro.span(),
            note: format!("{e}"),
        })
    }
}

/// Workgroup variable declaration parsed from `workgroup!(NAME: TYPE)`.
///
/// Transpiles to:
/// ```wgsl
/// var<workgroup> NAME: TYPE;
/// ```
///
/// Workgroup variables are shared between all invocations in a compute shader
/// workgroup. They can only be used in compute shaders.
pub(crate) struct ItemWorkgroup {
    pub name: syn::Ident,
    pub colon_token: Token![:],
    pub ty: Type,
    /// We keep the Rust type around for the Rust-side expansion
    pub rust_ty: syn::Type,
}

impl syn::parse::Parse for ItemWorkgroup {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: syn::Ident = input.parse()?;
        let colon_token: syn::Token![:] = input.parse()?;
        let rust_ty: syn::Type = input.parse()?;
        let ty = Type::try_from(&rust_ty)?;

        Ok(ItemWorkgroup {
            name,
            colon_token,
            ty,
            rust_ty,
        })
    }
}

impl TryFrom<&syn::ItemMacro> for ItemWorkgroup {
    type Error = Error;

    fn try_from(item_macro: &syn::ItemMacro) -> Result<Self, Self::Error> {
        // Ensure it's the "workgroup" macro
        if item_macro
            .mac
            .path
            .get_ident()
            .map(|id| id != "workgroup")
            .unwrap_or(true)
        {
            return UnsupportedSnafu {
                span: item_macro.span(),
                note: "Only 'workgroup!' macro is supported as a workgroup declaration.",
            }
            .fail();
        }

        syn::parse2::<ItemWorkgroup>(item_macro.mac.tokens.clone()).map_err(|e| {
            Error::Unsupported {
                span: item_macro.span(),
                note: format!("{e}"),
            }
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
    Workgroup(Box<ItemWorkgroup>),
    Sampler(Box<ItemSampler>),
    Texture(Box<ItemTexture>),
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
                    Some("workgroup") => Ok(Item::Workgroup(Box::new(ItemWorkgroup::try_from(
                        item_macro,
                    )?))),
                    Some("sampler") => {
                        Ok(Item::Sampler(Box::new(ItemSampler::try_from(item_macro)?)))
                    }
                    Some("texture") => {
                        Ok(Item::Texture(Box::new(ItemTexture::try_from(item_macro)?)))
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
            wgsl.contains("alias State = u32;"),
            "Expected 'alias State = u32;' in WGSL output, got: {}",
            wgsl
        );
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
            wgsl.contains("alias Priority = u32;"),
            "Expected 'alias Priority = u32;' in WGSL output, got: {}",
            wgsl
        );
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
            wgsl.contains("alias Mixed = u32;"),
            "Expected 'alias Mixed = u32;' in WGSL output, got: {}",
            wgsl
        );
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
    fn enum_and_struct_types_in_arrays() {
        // Test that both enum and struct types can be used in array types.
        // The enum should generate an alias that makes it usable as a type.
        let item_mod: syn::ItemMod = syn::parse_quote! {
            mod test_module {
                #[repr(u32)]
                pub enum Priority {
                    Low,
                    High,
                }

                pub struct Task {
                    pub priority: u32,
                    pub id: u32,
                }

                pub fn use_arrays(
                    priorities: [Priority; 10],
                    tasks: [Task; 10],
                ) {}
            }
        };
        let item_mod = ItemMod::try_from(&item_mod).unwrap();
        let wgsl = item_mod.to_wgsl();

        // Enum should generate alias and constants
        assert!(
            wgsl.contains("alias Priority = u32;"),
            "Expected 'alias Priority = u32;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Priority_Low: u32 = 0u;"),
            "Expected 'const Priority_Low: u32 = 0u;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("const Priority_High: u32 = 1u;"),
            "Expected 'const Priority_High: u32 = 1u;' in WGSL output, got: {}",
            wgsl
        );

        // Struct should be generated normally
        assert!(
            wgsl.contains("struct Task"),
            "Expected 'struct Task' in WGSL output, got: {}",
            wgsl
        );

        // Array types should use the type names
        assert!(
            wgsl.contains("array<Priority, 10>"),
            "Expected 'array<Priority, 10>' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("array<Task, 10>"),
            "Expected 'array<Task, 10>' in WGSL output, got: {}",
            wgsl
        );
    }

    // Loop statement tests
    #[test]
    fn parse_loop_basic() {
        let stmt: syn::Stmt = syn::parse_quote! {
            loop {
                let x: u32 = 1u;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        match stmt {
            Stmt::Loop { .. } => {}
            _ => panic!("Expected Stmt::Loop"),
        }
    }

    #[test]
    fn parse_loop_with_assignments() {
        let stmt: syn::Stmt = syn::parse_quote! {
            loop {
                let mut x: u32 = 0u;
                x = x + 1u;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        match stmt {
            Stmt::Loop { .. } => {}
            _ => panic!("Expected Stmt::Loop"),
        }
    }

    #[test]
    fn parse_loop_nested() {
        let stmt: syn::Stmt = syn::parse_quote! {
            loop {
                loop {
                    let x: u32 = 0u;
                }
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        match stmt {
            Stmt::Loop { .. } => {}
            _ => panic!("Expected Stmt::Loop"),
        }
    }

    #[test]
    fn parse_loop_rejects_label() {
        let stmt: syn::Stmt = syn::parse_quote! {
            'outer: loop {
                let x: u32 = 0u;
            }
        };
        let result = Stmt::try_from(&stmt);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("Labels on loops are not supported"),
            "Expected error about labels, got: {}",
            err
        );
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

    #[test]
    fn for_loop_exclusive_range_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            for i in 0..10 {
                let x = i;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("for (var i = 0; i < 10; i++)"),
            "Expected 'for (var i = 0; i < 10; i++)' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn for_loop_inclusive_range_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            for i in 0..=9 {
                let x = i;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("for (var i = 0; i <= 9; i++)"),
            "Expected 'for (var i = 0; i <= 9; i++)' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn for_loop_with_expressions_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            #[wgsl_allow(non_literal_loop_bounds)]
            for i in start..end {
                let x = i;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("for (var i = start; i < end; i++)"),
            "Expected 'for (var i = start; i < end; i++)' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn for_loop_rejects_labels() {
        let expr: syn::Expr = syn::parse_quote! {
            'outer: for i in 0..10 {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("Labels") || err.contains("label"),
            "Expected error about labels, got: {}",
            err
        );
    }

    #[test]
    fn for_loop_rejects_wildcard_pattern() {
        let expr: syn::Expr = syn::parse_quote! {
            for _ in 0..10 {
                let x = 0;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("_") || err.contains("named"),
            "Expected error about wildcard pattern, got: {}",
            err
        );
    }

    #[test]
    fn for_loop_rejects_unbounded_start() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in ..10 {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("start"),
            "Expected error about missing start, got: {}",
            err
        );
    }

    #[test]
    fn for_loop_rejects_unbounded_end() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in 0.. {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_err());
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("end"),
            "Expected error about missing end, got: {}",
            err
        );
    }

    // On stable, non-literal loop bounds return an error (SuppressableWarning)
    #[test]
    #[cfg(not(nightly))]
    fn for_loop_error_with_non_literal_bounds() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in start..end {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_err(), "Expected error for non-literal bounds");
        let err = result.err().unwrap();
        assert!(
            matches!(err, Error::SuppressableWarning { .. }),
            "Expected SuppressableWarning, got: {:?}",
            err
        );
    }

    // On nightly, non-literal loop bounds emit a warning diagnostic and parsing
    // succeeds. The emit_warning function gracefully handles being called
    // outside of a proc macro context (like in unit tests) by catching the
    // panic and continuing.
    #[test]
    #[cfg(nightly)]
    fn for_loop_warning_with_non_literal_bounds_nightly() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in start..end {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        // On nightly, parsing succeeds (warning would be emitted during real macro
        // expansion)
        let result = ForLoop::try_from(for_loop);
        assert!(
            result.is_ok(),
            "Expected success on nightly, got: {:?}",
            result.err()
        );
    }

    #[test]
    fn for_loop_suppressed_with_wgsl_allow() {
        let expr: syn::Expr = syn::parse_quote! {
            #[wgsl_allow(non_literal_loop_bounds)]
            for i in start..end {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_ok(), "Expected success with wgsl_allow");
    }

    #[test]
    fn for_loop_no_error_with_literal_bounds() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in 0..10 {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let result = ForLoop::try_from(for_loop);
        assert!(result.is_ok(), "Expected success for literal bounds");
    }

    #[test]
    fn for_loop_parses_correctly() {
        let expr: syn::Expr = syn::parse_quote! {
            for i in 0..10 {
                let x = i;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let for_loop = ForLoop::try_from(for_loop).unwrap();
        assert_eq!("i", for_loop.ident.to_string());
        assert!(!for_loop.inclusive);
        assert!(for_loop.ty.is_none());
    }

    #[test]
    fn for_loop_inclusive_parses_correctly() {
        let expr: syn::Expr = syn::parse_quote! {
            for j in 1..=100 {
                let y = j;
            }
        };
        let for_loop = match &expr {
            syn::Expr::ForLoop(f) => f,
            _ => panic!("Expected ForLoop"),
        };
        let for_loop = ForLoop::try_from(for_loop).unwrap();
        assert_eq!("j", for_loop.ident.to_string());
        assert!(for_loop.inclusive);
    }

    #[test]
    fn for_loop_nested_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            for i in 0..4 {
                for j in 0..4 {
                    let x = i + j;
                }
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("for (var i = 0; i < 4; i++)"),
            "Expected outer loop in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("for (var j = 0; j < 4; j++)"),
            "Expected inner loop in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn parse_continue_statement() {
        let stmt: syn::Stmt = syn::parse_quote! { continue; };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert_eq!(wgsl, "continue;");
    }

    #[test]
    fn continue_with_label_rejected() {
        let stmt: syn::Stmt = syn::parse_quote! { continue 'outer; };
        let result = Stmt::try_from(&stmt);
        assert!(
            result.is_err(),
            "Expected continue with label to be rejected, but it succeeded"
        );
    }

    #[test]
    fn loop_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            loop {
                let x: u32 = 1u;
            }
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.starts_with("loop"),
            "Expected 'loop' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("let x"),
            "Expected 'let x' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn loop_generates_wgsl_in_function() {
        let item: syn::Item = syn::parse_quote! {
            pub fn test_loop() {
                let mut x: u32 = 0u;
                loop {
                    x += 1u;
                }
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("loop {"),
            "Expected 'loop {{' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("x += 1u;"),
            "Expected 'x += 1u;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn explicit_return_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            return 42;
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert_eq!(
            wgsl.trim(),
            "return 42;",
            "Expected 'return 42;', got: {}",
            wgsl
        );
    }

    #[test]
    fn explicit_return_with_expression_generates_wgsl() {
        let stmt: syn::Stmt = syn::parse_quote! {
            return x + y;
        };
        let stmt = Stmt::try_from(&stmt).unwrap();
        let wgsl = stmt.to_wgsl();
        assert!(
            wgsl.contains("return"),
            "Expected 'return' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("x+y") || wgsl.contains("x + y"),
            "Expected 'x+y' or 'x + y' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn explicit_return_in_function_generates_wgsl() {
        let item: syn::Item = syn::parse_quote! {
            pub fn test_return(x: i32) -> i32 {
                if x > 0 {
                    return x;
                }
                return 0;
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("return x;"),
            "Expected 'return x;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("return 0;"),
            "Expected 'return 0;' in WGSL output, got: {}",
            wgsl
        );
    }

    #[test]
    fn implicit_return_still_works() {
        let item: syn::Item = syn::parse_quote! {
            pub fn test_implicit(x: i32) -> i32 {
                x + 1
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("return x+1;") || wgsl.contains("return x + 1;"),
            "Expected 'return x+1;' or 'return x + 1;' in WGSL output for implicit return, got: {}",
            wgsl
        );
    }

    #[test]
    fn explicit_and_implicit_return_can_coexist() {
        let item: syn::Item = syn::parse_quote! {
            pub fn test_mixed(x: i32) -> i32 {
                if x < 0 {
                    return 0;
                }
                x * 2
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("return 0;"),
            "Expected explicit 'return 0;' in WGSL output, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("return x*2;") || wgsl.contains("return x * 2;"),
            "Expected implicit return converted to 'return x*2;' or 'return x * 2;' in WGSL \
             output, got: {}",
            wgsl
        );
    }

    #[test]
    fn explicit_return_without_expression_generates_wgsl() {
        let item: syn::Item = syn::parse_quote! {
            pub fn void_return() {
                return;
            }
        };
        let item = Item::try_from(&item).unwrap();
        let wgsl = item.to_wgsl();
        assert!(
            wgsl.contains("return;"),
            "Expected 'return;' in WGSL output for void return, got: {}",
            wgsl
        );
    }

    #[test]
    fn return_without_semicolon_rejected() {
        let result: Result<syn::Stmt, _> = syn::parse_str("return 42");
        assert!(
            result.is_err(),
            "Expected return without semicolon to be rejected during parsing"
        );
    }

    #[test]
    fn builtin_function_call_translates_to_wgsl_name() {
        // Test that snake_case builtin function calls are translated to camelCase WGSL
        let expr: syn::Expr = syn::parse_quote! {
            count_leading_zeros(x)
        };
        let expr = Expr::try_from(&expr).unwrap();
        let wgsl = expr.to_wgsl();
        assert_eq!(
            wgsl, "countLeadingZeros(x)",
            "Expected snake_case builtin to be translated to camelCase"
        );
    }

    #[test]
    fn non_builtin_function_call_unchanged() {
        // Test that non-builtin function calls are not modified
        let expr: syn::Expr = syn::parse_quote! {
            my_custom_function(a, b)
        };
        let expr = Expr::try_from(&expr).unwrap();
        let wgsl = expr.to_wgsl();
        assert_eq!(
            wgsl, "my_custom_function(a, b)",
            "Expected non-builtin function name to remain unchanged"
        );
    }

    #[test]
    fn defining_function_with_snake_case_builtin_name_rejected() {
        let item: syn::Item = syn::parse_quote! {
            pub fn count_leading_zeros(x: u32) -> u32 {
                x
            }
        };
        let result = Item::try_from(&item);
        match result {
            Ok(_) => panic!("Expected error when defining function with reserved builtin name"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("count_leading_zeros") && msg.contains("countLeadingZeros"),
                    "Error message should mention both Rust and WGSL names, got: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn defining_function_with_camel_case_builtin_name_rejected() {
        let item: syn::Item = syn::parse_quote! {
            pub fn countLeadingZeros(x: u32) -> u32 {
                x
            }
        };
        let result = Item::try_from(&item);
        match result {
            Ok(_) => {
                panic!("Expected error when defining function with reserved WGSL builtin name")
            }
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("countLeadingZeros"),
                    "Error message should mention the WGSL name, got: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn parse_type_atomic_i32() {
        let ty: syn::Type = syn::parse_str("Atomic<i32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("atomic<i32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_type_atomic_u32() {
        let ty: syn::Type = syn::parse_str("Atomic<u32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("atomic<u32>", &ty.to_wgsl());
    }

    #[test]
    fn parse_type_atomic_f32_fails() {
        let ty: syn::Type = syn::parse_str("Atomic<f32>").unwrap();
        let result = Type::try_from(&ty);
        assert!(result.is_err(), "Atomic<f32> should fail to parse");
        // Check error message contains expected text
        if let Err(err) = result {
            let msg = err.to_string();
            assert!(
                msg.contains("i32 or u32"),
                "Error message should mention allowed types, got: {}",
                msg
            );
        }
    }

    #[test]
    fn parse_type_atomic_bool_fails() {
        let ty: syn::Type = syn::parse_str("Atomic<bool>").unwrap();
        let result = Type::try_from(&ty);
        assert!(result.is_err());
    }

    #[test]
    fn parse_workgroup_simple() {
        let workgroup: ItemWorkgroup = syn::parse_str("COUNTER: u32").unwrap();
        assert_eq!("COUNTER", workgroup.name.to_string());
    }

    #[test]
    fn parse_workgroup_atomic() {
        let workgroup: ItemWorkgroup = syn::parse_str("SHARED: Atomic<u32>").unwrap();
        assert_eq!("SHARED", workgroup.name.to_string());
        let wgsl = workgroup.to_wgsl();
        assert!(
            wgsl.contains("var<workgroup>"),
            "Expected var<workgroup> in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("atomic<u32>"),
            "Expected atomic<u32> in WGSL, got: {}",
            wgsl
        );
    }

    #[test]
    fn parse_workgroup_array() {
        let workgroup: ItemWorkgroup = syn::parse_str("TEMP_DATA: [f32; 64]").unwrap();
        assert_eq!("TEMP_DATA", workgroup.name.to_string());
        let wgsl = workgroup.to_wgsl();
        assert!(
            wgsl.contains("var<workgroup>"),
            "Expected var<workgroup>, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("array<f32, 64>"),
            "Expected array<f32, 64>, got: {}",
            wgsl
        );
    }

    #[test]
    fn workgroup_to_wgsl() {
        let workgroup: ItemWorkgroup = syn::parse_str("FLAGS: Atomic<i32>").unwrap();
        let wgsl = workgroup.to_wgsl();
        assert_eq!("var<workgroup> FLAGS: atomic<i32>;", wgsl.trim());
    }

    #[test]
    fn parse_sampler_basic() {
        let sampler: ItemSampler =
            syn::parse_str("group(0), binding(1), TEX_SAMPLER: Sampler").unwrap();
        assert_eq!("TEX_SAMPLER", sampler.name.to_string());
        assert!(matches!(sampler.ty, Type::Sampler { .. }));
    }

    #[test]
    fn parse_sampler_comparison() {
        let sampler: ItemSampler =
            syn::parse_str("group(0), binding(2), SHADOW_SAMPLER: SamplerComparison").unwrap();
        assert_eq!("SHADOW_SAMPLER", sampler.name.to_string());
        assert!(matches!(sampler.ty, Type::SamplerComparison { .. }));
    }

    #[test]
    fn sampler_to_wgsl() {
        let sampler: ItemSampler =
            syn::parse_str("group(0), binding(1), MY_SAMPLER: Sampler").unwrap();
        let wgsl = sampler.to_wgsl();
        assert!(
            wgsl.contains("@group(0)"),
            "Expected @group(0) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("@binding(1)"),
            "Expected @binding(1) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("var MY_SAMPLER"),
            "Expected 'var MY_SAMPLER' in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains(": sampler;"),
            "Expected ': sampler;' in WGSL, got: {}",
            wgsl
        );
    }

    #[test]
    fn sampler_comparison_to_wgsl() {
        let sampler: ItemSampler =
            syn::parse_str("group(1), binding(3), CMP_SAMPLER: SamplerComparison").unwrap();
        let wgsl = sampler.to_wgsl();
        assert!(
            wgsl.contains("@group(1)"),
            "Expected @group(1) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("@binding(3)"),
            "Expected @binding(3) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains(": sampler_comparison;"),
            "Expected ': sampler_comparison;' in WGSL, got: {}",
            wgsl
        );
    }

    #[test]
    fn sampler_rejects_non_sampler_type() {
        let result = syn::parse_str::<ItemSampler>("group(0), binding(0), BAD: u32");
        assert!(
            result.is_err(),
            "Expected error when using non-sampler type, but parsing succeeded"
        );
    }

    // ========== Texture Type Tests ==========

    #[test]
    fn parse_texture_type_texture2d_f32() {
        let ty: syn::Type = syn::parse_str("Texture2D<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert!(
            matches!(ty, Type::Texture { .. }),
            "Expected Type::Texture variant"
        );
        assert_eq!("texture_2d<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture2d_i32() {
        let ty: syn::Type = syn::parse_str("Texture2D<i32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_2d<i32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture2d_u32() {
        let ty: syn::Type = syn::parse_str("Texture2D<u32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_2d<u32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture1d() {
        let ty: syn::Type = syn::parse_str("Texture1D<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_1d<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture2d_array() {
        let ty: syn::Type = syn::parse_str("Texture2DArray<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_2d_array<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture3d() {
        let ty: syn::Type = syn::parse_str("Texture3D<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_3d<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture_cube() {
        let ty: syn::Type = syn::parse_str("TextureCube<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_cube<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture_cube_array() {
        let ty: syn::Type = syn::parse_str("TextureCubeArray<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_cube_array<f32>", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_type_texture_multisampled_2d() {
        let ty: syn::Type = syn::parse_str("TextureMultisampled2D<f32>").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_multisampled_2d<f32>", ty.to_wgsl());
    }

    // ========== Depth Texture Type Tests ==========

    #[test]
    fn parse_texture_depth_2d() {
        let ty: syn::Type = syn::parse_str("TextureDepth2D").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert!(
            matches!(ty, Type::TextureDepth { .. }),
            "Expected Type::TextureDepth variant"
        );
        assert_eq!("texture_depth_2d", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_depth_2d_array() {
        let ty: syn::Type = syn::parse_str("TextureDepth2DArray").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_depth_2d_array", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_depth_cube() {
        let ty: syn::Type = syn::parse_str("TextureDepthCube").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_depth_cube", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_depth_cube_array() {
        let ty: syn::Type = syn::parse_str("TextureDepthCubeArray").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_depth_cube_array", ty.to_wgsl());
    }

    #[test]
    fn parse_texture_depth_multisampled_2d() {
        let ty: syn::Type = syn::parse_str("TextureDepthMultisampled2D").unwrap();
        let ty = Type::try_from(&ty).unwrap();
        assert_eq!("texture_depth_multisampled_2d", ty.to_wgsl());
    }

    // ========== ItemTexture Parsing Tests ==========

    #[test]
    fn parse_texture_item_basic() {
        let texture: ItemTexture =
            syn::parse_str("group(0), binding(1), DIFFUSE_TEX: Texture2D<f32>").unwrap();
        assert_eq!("DIFFUSE_TEX", texture.name.to_string());
        assert!(matches!(texture.ty, Type::Texture { .. }));
    }

    #[test]
    fn parse_texture_item_depth() {
        let texture: ItemTexture =
            syn::parse_str("group(1), binding(0), SHADOW_MAP: TextureDepth2D").unwrap();
        assert_eq!("SHADOW_MAP", texture.name.to_string());
        assert!(matches!(texture.ty, Type::TextureDepth { .. }));
    }

    #[test]
    fn texture_to_wgsl() {
        let texture: ItemTexture =
            syn::parse_str("group(0), binding(2), MY_TEXTURE: Texture2D<f32>").unwrap();
        let wgsl = texture.to_wgsl();
        assert!(
            wgsl.contains("@group(0)"),
            "Expected @group(0) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("@binding(2)"),
            "Expected @binding(2) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("var MY_TEXTURE"),
            "Expected 'var MY_TEXTURE' in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains(": texture_2d<f32>;"),
            "Expected ': texture_2d<f32>;' in WGSL, got: {}",
            wgsl
        );
    }

    #[test]
    fn texture_depth_to_wgsl() {
        let texture: ItemTexture =
            syn::parse_str("group(1), binding(3), DEPTH_TEX: TextureDepth2D").unwrap();
        let wgsl = texture.to_wgsl();
        assert!(
            wgsl.contains("@group(1)"),
            "Expected @group(1) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("@binding(3)"),
            "Expected @binding(3) in WGSL, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains(": texture_depth_2d;"),
            "Expected ': texture_depth_2d;' in WGSL, got: {}",
            wgsl
        );
    }

    #[test]
    fn texture_rejects_non_texture_type() {
        let result = syn::parse_str::<ItemTexture>("group(0), binding(0), BAD: u32");
        assert!(
            result.is_err(),
            "Expected error when using non-texture type, but parsing succeeded"
        );
    }

    #[test]
    fn texture_rejects_sampler_type() {
        let result = syn::parse_str::<ItemTexture>("group(0), binding(0), BAD: Sampler");
        assert!(
            result.is_err(),
            "Expected error when using Sampler in texture! macro, but parsing succeeded"
        );
    }
}
