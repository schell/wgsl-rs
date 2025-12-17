use std::ops::{Deref, DerefMut};

use proc_macro2::{Ident, LineColumn, Span, TokenStream};
use quote::{ToTokens, format_ident};
use syn::spanned::Spanned;

use crate::{
    code_gen::SourceMapping,
    parse::{BinOp, Lit, Type, UnOp},
};

enum Line {
    IndentInc,
    IndentDec,
    Source(String),
}

#[derive(Default)]
pub struct GeneratedWgslCode {
    pub lines: Vec<Line>,
    pub line: String,
    pub source_map: Vec<SourceMapping>,
}

struct Surrounded<'a> {
    rust_span: Span,
    open: &'a str,
    close: &'a str,
    use_newline_after_open: bool,
    use_newline_after_close: bool,
    increase_indentation: bool,
}

impl Default for Surrounded<'_> {
    fn default() -> Self {
        Self {
            rust_span: Span::call_site(),
            open: "",
            close: "",
            use_newline_after_open: false,
            use_newline_after_close: false,
            increase_indentation: false,
        }
    }
}

struct Sequenced<'a> {
    rust_span: Span,
    delim: &'a str,
    use_newlines: bool,
}

impl Default for Sequenced<'_> {
    fn default() -> Self {
        Self {
            rust_span: Span::call_site(),
            delim: Default::default(),
            use_newlines: Default::default(),
        }
    }
}

impl GeneratedWgslCode {
    /// Returns the line and column of the current WGSL source "cursor".
    pub fn next_wgsl_line_column(&self) -> LineColumn {
        LineColumn {
            line: self.lines.len() + 1,
            column: self.line.len() + 1,
        }
    }

    /// Returns the line and column of the last token written.
    pub fn last_wgsl_line_column(&self) -> LineColumn {
        if self.lines.is_empty() && self.line.is_empty() {
            LineColumn {
                line: self.lines.len(),
                column: self.line.len(),
            }
        } else {
            LineColumn {
                line: self.lines.len() + 1,
                column: self.line.len(),
            }
        }
    }

    /// Write a bit of WGSL code that is the "leaf" or "atom" of the tree.
    fn write_atom(&mut self, to_tokens: &(impl Spanned + ToTokens)) {
        let rust_span = to_tokens.span();
        let wgsl_start = self.next_wgsl_line_column();
        let tokens = to_tokens.into_token_stream();
        self.line.push_str(&tokens.to_string());
        let wgsl_end = self.last_wgsl_line_column();
        self.source_map.push(SourceMapping {
            rust_span: (rust_span.start(), rust_span.end()),
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    /// Create a new line.
    fn newline(&mut self) {
        self.lines
            .push(Line::Source(std::mem::take(&mut self.line)));
    }

    /// Increment the indentation level.
    fn inc_indent(&mut self) {
        self.lines.push(Line::IndentInc);
    }

    /// Decrement the identation level.
    fn dec_indent(&mut self) {
        self.lines.push(Line::IndentDec);
    }

    /// Perform the inner function with incremented indentation, then reset.
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.newline();
        self.inc_indent();
        f(self);
        self.newline();
        self.dec_indent();
    }

    /// Insert one space.
    fn space(&mut self) {
        self.line.push(' ');
    }

    /// Insert four spaces.
    fn spaces(&mut self) {
        self.space();
        self.space();
        self.space();
        self.space();
    }

    /// Surround an inner function with delimiters.
    fn write_surrounded(&mut self, surrounded: Surrounded<'_>, f: impl FnOnce(&mut Self)) {
        let rust_span = surrounded.rust_span;
        let wgsl_start = self.next_wgsl_line_column();
        let (open, close) = (surrounded.open, surrounded.close);
        self.line.push_str(open);
        if surrounded.use_newline_after_open
            && surrounded.use_newline_after_close
            && surrounded.increase_indentation
        {
            self.indented(f);
            self.line.push_str(close);
        } else {
            if surrounded.use_newline_after_open {
                self.newline();
            }
            if surrounded.increase_indentation {
                self.inc_indent();
            }
            f(self);
            if surrounded.increase_indentation {
                self.dec_indent();
            }
            self.line.push_str(close);
            if surrounded.use_newline_after_close {
                self.newline();
            }
        }
        let wgsl_end = self.last_wgsl_line_column();
        self.source_map.push(SourceMapping {
            rust_span: (rust_span.start(), rust_span.end()),
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    fn append(&mut self, node: impl Into<GeneratedWgslCode>) {
        let GeneratedWgslCode {
            lines,
            line,
            source_map,
        } = node.into();
        self.lines.extend(lines);
        self.line.push_str(&line);
        self.source_map.extend(source_map);
    }

    fn write_sequenced(&mut self, sequenced: Sequenced<'_>, items: Vec<GeneratedWgslCode>) {
        let len = items.len();
        for (i, item) in items.into_iter().enumerate() {
            self.append(item);
            if i > 0 && i < len - 1 {
                self.line.push_str(sequenced.delim);
                if sequenced.use_newlines {
                    self.newline();
                } else {
                    self.space();
                }
            }
        }
    }

    /// Construct the WGSL source code and return it as a list of lines.
    pub fn source_lines(&self) -> Vec<String> {
        let mut indent = 0;
        self.lines
            .iter()
            .flat_map(|line| match line {
                Line::IndentInc => {
                    indent += 1;
                    None
                }

                Line::IndentDec => {
                    indent -= 1;
                    None
                }
                Line::Source(line) => {
                    let padding = "    ".repeat(indent);
                    Some(format!("{padding}{line}"))
                }
            })
            .collect()
    }

    /// Construct the WGSL source code and return it as one contiguous string.
    pub fn source(&self) -> String {
        self.source_lines().join("\n")
    }
}

trait GenerateCode {
    fn write_code(&self, code: &mut GeneratedWgslCode);

    fn to_wgsl(&self) -> String {
        let mut code = GeneratedWgslCode::default();
        self.write_code(&mut code);
        code.source()
    }
}

impl<T: GenerateCode> From<&T> for GeneratedWgslCode {
    fn from(value: &T) -> Self {
        let mut code = Self::default();
        value.write_code(&mut code);
        code
    }
}

impl GenerateCode for Lit {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Lit::Bool(lit_bool) => code.write_atom(lit_bool),
            Lit::Float(lit_float) => code.write_atom(lit_float),
            Lit::Int(lit_int) => code.write_atom(lit_int),
        }
    }
}

impl GenerateCode for Type {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Type::Scalar { ty: _, ident } => {
                code.write_atom(ident);
            }
            Type::Vector {
                elements: _,
                scalar_ty: _,
                ident,
                scalar,
            } => {
                let ident = quote::format_ident!("{}", ident.to_string().to_lowercase());
                code.write_atom(&ident);
                if let Some((lt, scalar, rt)) = scalar {
                    code.write_atom(lt);
                    code.write_atom(scalar);
                    code.write_atom(rt);
                }
            }
            Type::Array {
                bracket_token,
                elem,
                semi_token,
                len,
            } => {
                let span: Span = bracket_token.span.span();
                let array = Ident::new("array", span.into());
                code.write_atom(&array);
                code.write_surrounded(
                    Surrounded {
                        rust_span: span,
                        open: "<",
                        close: ">",
                        ..Default::default()
                    },
                    |code| {
                        code.write_sequenced(
                            Sequenced {
                                rust_span: span,
                                delim: ",",
                                ..Default::default()
                            },
                            vec![elem.as_ref().into(), len.into()],
                        );
                    },
                );
            }
            Type::Struct { ident } => code.write_atom(ident),
        }
    }
}

impl GenerateCode for BinOp {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            BinOp::Add(t) => code.write_atom(t),
            BinOp::Sub(t) => code.write_atom(t),
            BinOp::Mul(t) => code.write_atom(t),
            BinOp::Div(t) => code.write_atom(t),
        }
    }
}

impl GenerateCode for UnOp {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            UnOp::Not(t) => code.write_atom(t),
            UnOp::Neg(t) => code.write_atom(t),
        }
    }
}

impl ToTokens for FieldValue {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.member.to_tokens(tokens);
        if let Some(colon_token) = &self.colon_token {
            colon_token.to_tokens(tokens);
        }
        self.expr.to_tokens(tokens);
    }
}

impl ToTokens for Expr {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Expr::Lit(lit) => lit.to_tokens(tokens),
            Expr::Ident(id) => id.to_tokens(tokens),
            Expr::Array(elems) => {
                quote! { array }.to_tokens(tokens);
                let paren = syn::token::Paren::default();
                paren.surround(tokens, |inner| {
                    for pair in elems.pairs() {
                        let expr = pair.value();
                        expr.to_tokens(inner);
                        if let Some(comma) = pair.punct() {
                            comma.to_tokens(inner);
                        }
                    }
                });
            }
            Expr::Paren { paren_token, inner } => {
                paren_token.surround(tokens, |tokens| inner.to_tokens(tokens));
            }
            Expr::Binary { lhs, op, rhs } => {
                lhs.to_tokens(tokens);
                op.to_tokens(tokens);
                rhs.to_tokens(tokens);
            }
            Expr::Unary { op, expr } => {
                op.to_tokens(tokens);
                expr.to_tokens(tokens);
            }
            Expr::ArrayIndexing {
                lhs,
                bracket_token,
                index,
            } => {
                lhs.to_tokens(tokens);
                bracket_token.surround(tokens, |inner| {
                    index.to_tokens(inner);
                });
            }
            Expr::Swizzle {
                lhs,
                dot_token,
                swizzle,
            } => {
                lhs.to_tokens(tokens);
                dot_token.to_tokens(tokens);
                swizzle.to_tokens(tokens);
            }
            Expr::FieldAccess {
                base,
                dot_token,
                field,
            } => {
                base.to_tokens(tokens);
                dot_token.to_tokens(tokens);
                field.to_tokens(tokens);
            }
            Expr::Cast { lhs, ty } => quote! { #ty(#lhs) }.to_tokens(tokens),
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
            Expr::Struct {
                ident,
                brace_token,
                fields,
            } => {
                ident.to_tokens(tokens);
                brace_token.surround(tokens, |inner| {
                    fields.to_tokens(inner);
                });
            }
        }
    }
}

impl ToTokens for ReturnTypeAnnotation {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ReturnTypeAnnotation::None => {}
            ReturnTypeAnnotation::BuiltIn(ident) => quote! { @builtin(#ident) }.to_tokens(tokens),
            ReturnTypeAnnotation::Location(lit) => quote! { @location(#lit) }.to_tokens(tokens),
        }
    }
}

impl ToTokens for ReturnType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            ReturnType::Default => {}
            ReturnType::Type {
                arrow,
                annotation,
                ty,
            } => {
                arrow.to_tokens(tokens);
                annotation.to_tokens(tokens);
                ty.to_tokens(tokens);
            }
        }
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

impl ToTokens for Stmt {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Stmt::Local(local) => local.to_tokens(tokens),
            Stmt::Const(item_const) => item_const.to_tokens(tokens),
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

impl ToTokens for BuiltIn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ident = match self {
            BuiltIn::VertexIndex => "vertex_index",
            BuiltIn::InstanceIndex => "instance_index",
            BuiltIn::Position => "position",
            BuiltIn::FrontFacing => "front_facing",
            BuiltIn::FragDepth => "frag_depth",
            BuiltIn::SampleIndex => "sample_index",
            BuiltIn::SampleMask => "sample_mask",
            BuiltIn::LocalInvocationId => "local_invocation_id",
            BuiltIn::LocalInvocationIndex => "local_invocation_index",
            BuiltIn::GlobalInvocationId => "global_invocation_id",
            BuiltIn::WorkgroupId => "workgroup_id",
            BuiltIn::NumWorkgroups => "num_workgroups",
            BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
            BuiltIn::SubgroupSize => "subgroup_size",
            BuiltIn::PrimitiveIndex => "primitive_index",
            BuiltIn::SubgroupId => "subgroup_id",
            BuiltIn::NumSubgroups => "num_subgroups",
        };
        let ident = quote::format_ident!("{ident}");
        ident.to_tokens(tokens);
    }
}

impl ToTokens for InterpolationType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ident = match self {
            InterpolationType::Perspective(ident) => ident,
            InterpolationType::Linear(ident) => ident,
            InterpolationType::Flat(ident) => ident,
        };
        ident.to_tokens(tokens)
    }
}

impl ToTokens for InterpolationSampling {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ident = match self {
            InterpolationSampling::Center(ident) => ident,
            InterpolationSampling::Centroid(ident) => ident,
            InterpolationSampling::Sample(ident) => ident,
            InterpolationSampling::First(ident) => ident,
            InterpolationSampling::Either(ident) => ident,
        };
        ident.to_tokens(tokens)
    }
}

impl ToTokens for Interpolate {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let Self {
            ty,
            comma_token,
            sampling,
        } = self;
        quote! { @interpolate }.to_tokens(tokens);
        syn::token::Paren::default().surround(tokens, |tokens| {
            ty.to_tokens(tokens);
            comma_token.to_tokens(tokens);
            sampling.to_tokens(tokens);
        });
    }
}

impl ToTokens for InterStageIo {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            InterStageIo::BuiltIn(built_in) => quote! { @builtin(#built_in) },
            InterStageIo::Location(loc) => quote! { @location(#loc) },
            InterStageIo::BlendSrc(src) => quote! { @blend_src(#src) },
            InterStageIo::Interpolate(lerp) => quote! { #lerp },
            InterStageIo::Invariant => quote! { @invariant },
        }
        .to_tokens(tokens)
    }
}

impl ToTokens for FnArg {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let Self {
            inter_stage_io,
            ident,
            colon_token,
            ty,
        } = self;
        for inter_stage_io in inter_stage_io.iter() {
            inter_stage_io.to_tokens(tokens);
        }
        ident.to_tokens(tokens);
        colon_token.to_tokens(tokens);
        ty.to_tokens(tokens);
    }
}

impl ToTokens for FnAttrs {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            FnAttrs::None => {}
            FnAttrs::Vertex => quote! { @vertex }.to_tokens(tokens),
            FnAttrs::Fragment => quote! { @fragment }.to_tokens(tokens),
        }
    }
}

impl ToTokens for ItemFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ItemFn {
            fn_attrs,
            fn_token,
            ident,
            paren_token,
            inputs,
            return_type,
            block,
        } = self;
        fn_attrs.to_tokens(tokens);
        fn_token.to_tokens(tokens);
        ident.to_tokens(tokens);
        paren_token.surround(tokens, |tokens| {
            inputs.to_tokens(tokens);
        });
        return_type.to_tokens(tokens);
        block.to_tokens(tokens);
    }
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

impl ToTokens for ItemUniform {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let group = &self.group;
        let binding = &self.binding;
        let name = &self.name;
        let ty = &self.ty;

        // WGSL uniform declaration (example, adjust as needed for your codegen)
        quote! {
            @group(#group) @binding(#binding) var<uniform> #name: #ty;
        }
        .to_tokens(tokens);
    }
}

impl ToTokens for Field {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for io in self.inter_stage_io.iter() {
            io.to_tokens(tokens);
        }
        self.ident.to_tokens(tokens);
        if let Some(colon_token) = &self.colon_token {
            colon_token.to_tokens(tokens);
        }
        self.ty.to_tokens(tokens);
    }
}

impl ToTokens for FieldsNamed {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let brace_token = &self.brace_token;
        let named = &self.named;
        brace_token.surround(tokens, |inner| {
            named.to_tokens(inner);
        });
    }
}
