//! WGSL abstract syntax tree-ish.
//!
//! There's a lot of hand-waving going on here, but that's ok
//! because in practice this stuff is already type checked by Rust at the
//! time it's constructed.
//!
//! The syntax here is the subset of Rust that can be interpreted as WGSL.
use quote::{ToTokens, quote};
use snafu::prelude::*;
use syn::{Ident, Token, spanned::Spanned};

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
        "Unsupported use of if-then-else, WGSL if statements are a control structure, not an expression."
    ))]
    UnsupportedIfThen { span: proc_macro2::Span },

    #[snafu(display("All WGSL functions must be public"))]
    FnVisibility { span: proc_macro2::Span },

    #[snafu(display("In progress:\n{message}"))]
    InProgress {
        span: proc_macro2::Span,
        message: String,
    },
}

impl Error {
    pub fn span(&self) -> &proc_macro2::Span {
        match self {
            Error::Unsupported { span, .. } => span,
            Error::CurrentlyUnsupported { span, .. } => span,
            Error::UnsupportedIfThen { span } => span,
            Error::InProgress { span, message: _ } => span,
            Error::FnVisibility { span } => span,
        }
    }
}

impl From<Error> for syn::Error {
    fn from(e: Error) -> Self {
        syn::Error::new(*e.span(), format!("{e}"))
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

/// Concrete scalar types.
/// * i32
/// * u32
/// * f32
/// * bool
pub struct Type {
    pub ident: Ident,
}

impl TryFrom<&syn::Type> for Type {
    type Error = Error;

    fn try_from(ty: &syn::Type) -> Result<Self, Self::Error> {
        let span = ty.span();
        if let syn::Type::Path(type_path) = ty {
            util::some_is_unsupported(
                type_path.qself.as_ref(),
                "QSelf not allowed in scalar type",
            )?;
            let ident = type_path
                .path
                .get_ident()
                .context(UnsupportedSnafu {
                    span: type_path.span(),
                    note: "Not an identifier for a scalar type",
                })?
                .clone();
            Ok(match ident.to_string().as_str() {
                "i32" => Type { ident },
                "u32" => Type { ident },
                "f32" => Type { ident },
                "bool" => Type { ident },
                other => UnsupportedSnafu {
                    span,
                    note: format!("Unknown type '{other}'."),
                }
                .fail()?,
            })
        } else {
            UnsupportedSnafu {
                span,
                note: format!("Type is not a path: '{}'", ty.into_token_stream()),
            }
            .fail()
        }
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.ident.to_tokens(tokens);
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

impl ToTokens for Lit {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Lit::Bool(lit_bool) => lit_bool.to_tokens(tokens),
            Lit::Float(lit_float) => lit_float.to_tokens(tokens),
            Lit::Int(lit_int) => lit_int.to_tokens(tokens),
        }
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

impl ToTokens for BinOp {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            BinOp::Add(t) => t.to_tokens(tokens),
            BinOp::Sub(t) => t.to_tokens(tokens),
            BinOp::Mul(t) => t.to_tokens(tokens),
            BinOp::Div(t) => t.to_tokens(tokens),
        }
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
    // /// A postfix operator, like array access notation "thing[2]".
    // PostfixOp { op: String, lhs: Box<Expr> },
    // /// A function call
    FnCall {
        lhs: Ident,
        paren_token: syn::token::Paren,
        params: syn::punctuated::Punctuated<Expr, syn::Token![,]>,
    },
}

impl TryFrom<&syn::Expr> for Expr {
    type Error = Error;

    fn try_from(value: &syn::Expr) -> Result<Self, Self::Error> {
        let span = value.span();

        Ok(match value {
            syn::Expr::Lit(syn::ExprLit { attrs: _, lit }) => Self::Lit(Lit::try_from(lit)?),
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
            other => InProgressSnafu {
                span,
                message: format!("{other:#?}"),
            }
            .fail()?,
        })
    }
}

impl ToTokens for Expr {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Expr::Lit(lit) => lit.to_tokens(tokens),
            Expr::Ident(id) => id.to_tokens(tokens),
            Expr::Paren { paren_token, inner } => {
                paren_token.surround(tokens, |tokens| inner.to_tokens(tokens));
            }
            Expr::Binary { lhs, op, rhs } => {
                lhs.to_tokens(tokens);
                op.to_tokens(tokens);
                rhs.to_tokens(tokens);
            }
            Expr::FnCall {
                lhs,
                paren_token,
                params,
            } => {
                lhs.to_tokens(tokens);
                paren_token.surround(tokens, |tokens| {
                    for pair in params.pairs() {
                        let expr = pair.value();
                        expr.to_tokens(tokens);
                        if let Some(comma) = pair.punct() {
                            comma.to_tokens(tokens);
                        }
                    }
                });
            }
        }
    }
}

// /// A WGSL statement.
// pub struct Stmt {

// }

// pub struct FnArg {
//     pub ident: syn::Ident,
//     pub colon: syn::Token![:],
//     pub ty: Type,
// }

pub enum ReturnType {
    Default,
    Type(Token![->], Type),
}

impl TryFrom<&syn::ReturnType> for ReturnType {
    type Error = Error;

    fn try_from(ret: &syn::ReturnType) -> Result<Self, Self::Error> {
        match ret {
            syn::ReturnType::Default => Ok(ReturnType::Default),
            syn::ReturnType::Type(arrow, ty) => {
                let scalar = Type::try_from(ty.as_ref())?;
                Ok(ReturnType::Type(*arrow, scalar))
            }
        }
    }
}

impl ToTokens for ReturnType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ReturnType::Default => {}
            ReturnType::Type(arrow, ty) => {
                arrow.to_tokens(tokens);
                ty.to_tokens(tokens);
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

impl ToTokens for Local {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        // let or var
        if self.mutability.is_some() {
            // This is a "var"
            quote! { var }.to_tokens(tokens);
        } else {
            self.let_token.to_tokens(tokens);
        }
        self.ident.to_tokens(tokens);
        if let Some((colon_token, ty)) = &self.ty {
            colon_token.to_tokens(tokens);
            ty.to_tokens(tokens);
        }
        if let Some(init) = &self.init {
            init.eq_token.to_tokens(tokens);
            init.expr.to_tokens(tokens);
        }
        self.semi_token.to_tokens(tokens);
    }
}

pub enum Stmt {
    Local(Local),
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
            syn::Stmt::Local(local) => Ok(Stmt::Local(Local::try_from(local)?)),
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

impl ToTokens for Stmt {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Stmt::Local(local) => local.to_tokens(tokens),
            Stmt::Expr { expr, semi_token } => {
                if let Some(semi) = semi_token {
                    expr.to_tokens(tokens);
                    semi.to_tokens(tokens);
                } else {
                    quote! { return #expr; }.to_tokens(tokens);
                }
            }
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

impl ToTokens for Block {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let brace_token = &self.brace_token;
        let stmts = &self.stmt;
        brace_token.surround(tokens, |inner| {
            for stmt in stmts {
                stmt.to_tokens(inner);
            }
        });
    }
}

pub struct FnArg {
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
                    let ty = Type::try_from(pat_type.ty.as_ref())?;

                    Ok(FnArg {
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

impl ToTokens for FnArg {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let Self {
            ident,
            colon_token,
            ty,
        } = self;
        ident.to_tokens(tokens);
        colon_token.to_tokens(tokens);
        ty.to_tokens(tokens);
    }
}

pub struct ItemFn {
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
            attrs: _,
            vis,
            sig,
            block,
        } = value;
        snafu::ensure!(
            matches!(vis, syn::Visibility::Public(_)),
            FnVisibilitySnafu { span: vis.span() }
        );

        let mut inputs = syn::punctuated::Punctuated::new();
        for pair in sig.inputs.pairs() {
            let input = pair.value();
            let arg = FnArg::try_from(*input)?;
            inputs.push_value(arg);
            if let Some(comma) = pair.punct() {
                inputs.push_punct(**comma);
            }
        }

        Ok(ItemFn {
            fn_token: sig.fn_token,
            ident: sig.ident.clone(),
            paren_token: sig.paren_token,
            inputs,
            return_type: ReturnType::try_from(&sig.output)?,
            block: Block::try_from(block.as_ref())?,
        })
    }
}

impl ToTokens for ItemFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ItemFn {
            fn_token,
            ident,
            paren_token,
            inputs,
            return_type,
            block,
        } = self;
        fn_token.to_tokens(tokens);
        ident.to_tokens(tokens);
        paren_token.surround(tokens, |tokens| {
            inputs.to_tokens(tokens);
        });
        return_type.to_tokens(tokens);
        block.to_tokens(tokens);
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

impl ToTokens for ItemConst {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ItemConst {
            const_token,
            ident,
            colon_token,
            ty,
            eq_token,
            expr,
            semi_token,
        } = self;
        const_token.to_tokens(tokens);
        ident.to_tokens(tokens);
        colon_token.to_tokens(tokens);
        ty.to_tokens(tokens);
        eq_token.to_tokens(tokens);
        expr.to_tokens(tokens);
        semi_token.to_tokens(tokens);
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

impl ToTokens for ItemMod {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for item in &self.content {
            item.to_tokens(tokens);
        }
    }
}

impl ItemMod {
    pub fn imports(&self) -> Vec<proc_macro2::TokenStream> {
        let mut imports = vec![];
        for item in self.content.iter() {
            if let Item::Use(use_item) = item {
                for path in use_item.modules.iter() {
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
            syn::UseTree::Glob(_) => Self { modules: vec![] },
            syn::UseTree::Group(use_group) => UnsupportedSnafu {
                span: use_group.span(),
                note: "Grouped use statements are not supported.",
            }
            .fail()?,
        })
    }
}

/// WGSL items that may appear in a "module" or scope.
pub enum Item {
    Const(ItemConst),
    Fn(ItemFn),
    Mod(ItemMod),
    Use(ItemUse),
}

impl TryFrom<&syn::Item> for Item {
    type Error = Error;

    fn try_from(value: &syn::Item) -> Result<Self, Self::Error> {
        match value {
            syn::Item::Mod(item_mod) => Ok(Item::Mod(ItemMod::try_from(item_mod)?)),
            syn::Item::Const(syn::ItemConst {
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
            }) => Ok(Item::Const(ItemConst {
                const_token: *const_token,
                ident: ident.clone(),
                colon_token: *colon_token,
                ty: Type::try_from(ty.as_ref())?,
                eq_token: *eq_token,
                expr: Expr::try_from(expr.as_ref())?,
                semi_token: *semi_token,
            })),
            syn::Item::Fn(item_fn) => Ok(Item::Fn(item_fn.try_into()?)),
            syn::Item::Use(syn::ItemUse {
                attrs: _,
                vis: _,
                use_token: _,
                leading_colon: _,
                tree,
                semi_token: _,
            }) => Ok(Item::Use(ItemUse::try_from(tree)?)),
            _ => UnsupportedSnafu {
                span: value.span(),
                note: format!("Unsupported item: '{}'", value.into_token_stream()),
            }
            .fail(),
        }
    }
}

/// Converts to WGSL tokens.
impl ToTokens for Item {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Item::Mod(item_mod) => item_mod.to_tokens(tokens),
            Item::Const(item_const) => item_const.to_tokens(tokens),
            Item::Fn(item_fn) => item_fn.to_tokens(tokens),
            Item::Use(_item_use) => {
                // Skip as "use" does not produce WGSL.
                //
                // Instead "use" is used by the `wgsl` macro to include
                // imports of other WGSL code.
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
        assert_eq!("333 + 333", &expr.into_token_stream().to_string());
    }

    #[test]
    fn parse_expr_binary_ident() {
        let expr: syn::Expr = syn::parse_str("333 + TIMES").unwrap();
        let expr = Expr::try_from(&expr).unwrap();
        assert_eq!("333 + TIMES", &expr.into_token_stream().to_string());
    }
}
