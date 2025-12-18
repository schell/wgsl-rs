use proc_macro2::{Ident, LineColumn, Span};
use quote::{ToTokens, quote, quote_spanned};
use syn::spanned::Spanned;

use crate::parse::*;

pub struct SourceMapping {
    pub rust_span: (LineColumn, LineColumn),
    pub wgsl_span: (LineColumn, LineColumn),
}

pub enum Line {
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

impl Surrounded<'_> {
    pub fn block(outer_span: Span) -> Self {
        Surrounded {
            rust_span: outer_span,
            open: "{",
            close: "}",
            use_newline_after_open: true,
            use_newline_after_close: true,
            increase_indentation: true,
        }
    }

    pub fn parens() -> Self {
        Surrounded {
            open: "(",
            close: ")",
            ..Default::default()
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.rust_span = span;
        self
    }
}

#[derive(Default)]
struct Sequenced<'a> {
    delim: &'a str,
    use_newlines: bool,
    use_delimiter_on_last_line: bool,
}

impl Sequenced<'_> {
    pub fn comma() -> Sequenced<'static> {
        Sequenced {
            delim: ",",
            use_newlines: false,
            use_delimiter_on_last_line: false,
        }
    }

    pub fn newlines() -> Sequenced<'static> {
        Sequenced {
            delim: "",
            use_newlines: true,
            use_delimiter_on_last_line: false,
        }
    }

    pub fn comma_with_newlines() -> Sequenced<'static> {
        Sequenced {
            delim: ",",
            use_newlines: true,
            use_delimiter_on_last_line: true,
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

    /// Write a string to the current line.
    fn write_str(&mut self, rust_span: Span, s: &str) {
        let wgsl_start = self.next_wgsl_line_column();
        self.line.push_str(s);
        let wgsl_end = self.last_wgsl_line_column();
        self.source_map.push(SourceMapping {
            rust_span: (rust_span.start(), rust_span.end()),
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    /// Write a bit of WGSL code that is the "leaf" or "atom" of the tree.
    fn write_atom(&mut self, to_tokens: &(impl Spanned + ToTokens)) {
        let rust_span = to_tokens.span();
        let tokens = to_tokens.into_token_stream();
        self.write_str(rust_span, &tokens.to_string());
    }

    /// Write an annotation.
    fn write_annotation(&mut self, annotation: &str) {
        self.line.push('@');
        self.line.push_str(annotation);
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
        if surrounded.use_newline_after_open && surrounded.increase_indentation {
            self.indented(f);
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
        }
        self.line.push_str(close);
        if surrounded.use_newline_after_close {
            self.newline();
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

    fn write_sequenced(
        &mut self,
        sequenced: Sequenced<'_>,
        items: impl IntoIterator<Item = impl Into<GeneratedWgslCode>>,
    ) {
        let items = items.into_iter().collect::<Vec<_>>();
        let len = items.len();
        for (i, item) in items.into_iter().enumerate() {
            self.append(item);
            let is_last = i == len - 1;
            if !is_last || sequenced.use_delimiter_on_last_line {
                self.line.push_str(sequenced.delim);
            }

            if sequenced.use_newlines && !is_last {
                self.newline();
            } else if !is_last {
                self.space();
            }
        }
    }

    pub fn last_line_is_empty(&self) -> bool {
        for line in self.lines.iter().rev() {
            match line {
                Line::IndentInc | Line::IndentDec => continue,
                Line::Source(src) => return src.is_empty(),
            }
        }
        true
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

pub trait GenerateCode {
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

impl<T: GenerateCode> GenerateCode for &Option<T> {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        if let Some(item) = self.as_ref() {
            item.write_code(code);
        }
    }
}

impl GenerateCode for Ident {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(self);
    }
}

impl GenerateCode for proc_macro2::TokenStream {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(self);
    }
}

impl GenerateCode for syn::LitInt {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(self);
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
                code.write_str(span, "array");
                code.write_surrounded(
                    Surrounded {
                        rust_span: span,
                        open: "<",
                        close: ">",
                        ..Default::default()
                    },
                    |code| {
                        elem.write_code(code);
                        code.write_str(semi_token.span, ", ");
                        len.write_code(code);
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

impl GenerateCode for FieldValue {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        self.member.write_code(code);
        if let Some(colon_token) = &self.colon_token {
            code.write_atom(colon_token);
        }
        self.expr.write_code(code);
    }
}

impl GenerateCode for Expr {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Expr::Lit(lit) => lit.write_code(code),
            Expr::Ident(id) => id.write_code(code),
            Expr::Array {
                bracket_token,
                elems,
            } => {
                quote! { array }.write_code(code);
                let indented = elems.len() > 4;
                code.write_surrounded(
                    Surrounded {
                        rust_span: bracket_token.span.join(),
                        open: "(",
                        close: ")",
                        use_newline_after_open: indented,
                        use_newline_after_close: indented,
                        increase_indentation: indented,
                    },
                    |code| {
                        code.write_sequenced(
                            Sequenced {
                                delim: ",",
                                use_newlines: indented,
                                use_delimiter_on_last_line: indented,
                            },
                            elems.iter(),
                        );
                    },
                );
            }
            Expr::Paren { paren_token, inner } => {
                code.write_surrounded(
                    Surrounded {
                        rust_span: paren_token.span.join(),
                        open: "(",
                        close: ")",
                        ..Default::default()
                    },
                    |code| {
                        inner.write_code(code);
                    },
                );
            }
            Expr::Binary { lhs, op, rhs } => {
                lhs.write_code(code);
                op.write_code(code);
                rhs.write_code(code);
            }
            Expr::Unary { op, expr } => {
                op.write_code(code);
                expr.write_code(code);
            }
            Expr::ArrayIndexing {
                lhs,
                bracket_token,
                index,
            } => {
                lhs.write_code(code);
                code.write_surrounded(
                    Surrounded {
                        rust_span: bracket_token.span.join(),
                        open: "[",
                        close: "]",
                        ..Default::default()
                    },
                    |code| {
                        index.write_code(code);
                    },
                );
            }
            Expr::Swizzle {
                lhs,
                dot_token,
                swizzle,
            } => {
                lhs.write_code(code);
                code.write_atom(dot_token);
                swizzle.write_code(code);
            }
            Expr::FieldAccess {
                base,
                dot_token,
                field,
            } => {
                base.write_code(code);
                code.write_atom(dot_token);
                field.write_code(code);
            }
            Expr::Cast { lhs, ty } => {
                ty.write_code(code);
                code.write_surrounded(
                    Surrounded {
                        open: "(",
                        close: ")",
                        ..Default::default()
                    },
                    |code| {
                        lhs.write_code(code);
                    },
                );
            }
            Expr::FnCall {
                lhs,
                paren_token,
                params,
            } => {
                lhs.write_code(code);
                let indented = params.len() > 4;
                code.write_surrounded(
                    Surrounded {
                        rust_span: paren_token.span.join(),
                        open: "(",
                        close: ")",
                        use_newline_after_open: indented,
                        use_newline_after_close: indented,
                        increase_indentation: indented,
                    },
                    |code| {
                        code.write_sequenced(
                            Sequenced {
                                delim: ",",
                                use_newlines: indented,
                                use_delimiter_on_last_line: indented,
                            },
                            params.iter(),
                        );
                    },
                );
            }
            Expr::Struct {
                ident,
                brace_token,
                fields,
            } => {
                ident.write_code(code);
                code.write_surrounded(Surrounded::block(brace_token.span.join()), |code| {
                    code.write_sequenced(Sequenced::comma_with_newlines(), fields.iter())
                });
            }
        }
    }
}

impl GenerateCode for ReturnTypeAnnotation {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            ReturnTypeAnnotation::None => {}
            ReturnTypeAnnotation::BuiltIn(ident) => {
                code.write_annotation("builtin");
                code.write_surrounded(
                    Surrounded {
                        open: "(",
                        close: ")",
                        ..Default::default()
                    },
                    |code| {
                        ident.write_code(code);
                    },
                );
                code.space();
            }
            ReturnTypeAnnotation::Location(lit) => {
                code.write_annotation("location");
                code.write_surrounded(
                    Surrounded {
                        open: "(",
                        close: ")",
                        ..Default::default()
                    },
                    |code| {
                        lit.write_code(code);
                    },
                );
                code.space();
            }
        }
    }
}

impl GenerateCode for ReturnType {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            ReturnType::Default => {}
            ReturnType::Type {
                arrow,
                annotation,
                ty,
            } => {
                code.write_atom(arrow);
                code.space();
                annotation.write_code(code);
                ty.write_code(code);
                code.space();
            }
        }
    }
}

impl GenerateCode for Local {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        // let or var
        if self.mutability.is_some() {
            // This is a "var"
            code.write_atom(&Ident::new("var", self.let_token.span()));
        } else {
            code.write_atom(&self.let_token);
        }
        code.space();
        self.ident.write_code(code);
        if let Some((colon_token, ty)) = &self.ty {
            code.write_atom(colon_token);
            code.space();
            ty.write_code(code);
        }
        if let Some(init) = &self.init {
            code.space();
            code.write_atom(&init.eq_token);
            code.space();
            init.expr.write_code(code);
        }
        code.write_atom(&self.semi_token);
    }
}

impl GenerateCode for Stmt {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Stmt::Local(local) => local.write_code(code),
            Stmt::Const(item_const) => item_const.as_ref().write_code(code),
            Stmt::Expr { expr, semi_token } => {
                if let Some(semi) = semi_token {
                    expr.write_code(code);
                    code.write_atom(semi);
                } else {
                    code.write_str(Span::call_site(), "return");
                    code.space();
                    expr.write_code(code);
                    code.write_atom(&<syn::Token![;]>::default());
                }
            }
        }
    }
}

impl GenerateCode for Block {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let brace_token = &self.brace_token;
        let stmts = &self.stmt;
        code.write_surrounded(Surrounded::block(brace_token.span.join()), |code| {
            code.write_sequenced(Sequenced::newlines(), stmts);
        });
        code.newline();
    }
}

impl GenerateCode for BuiltIn {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
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
        ident.write_code(code);
    }
}

impl GenerateCode for InterpolationType {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ident = match self {
            InterpolationType::Perspective(ident) => ident,
            InterpolationType::Linear(ident) => ident,
            InterpolationType::Flat(ident) => ident,
        };
        ident.write_code(code)
    }
}

impl GenerateCode for InterpolationSampling {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ident = match self {
            InterpolationSampling::Center(ident) => ident,
            InterpolationSampling::Centroid(ident) => ident,
            InterpolationSampling::Sample(ident) => ident,
            InterpolationSampling::First(ident) => ident,
            InterpolationSampling::Either(ident) => ident,
        };
        ident.write_code(code)
    }
}

impl GenerateCode for Interpolate {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let Self {
            ty,
            comma_token,
            sampling,
        } = self;
        code.write_annotation("interpolate");
        code.write_surrounded(Surrounded::parens(), |code| {
            ty.write_code(code);
            code.write_atom(comma_token);
            sampling.write_code(code);
        });
    }
}

impl GenerateCode for InterStageIo {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            InterStageIo::BuiltIn(built_in) => {
                code.write_annotation("builtin");
                code.write_surrounded(Surrounded::parens(), |code| built_in.write_code(code));
            }
            InterStageIo::Location(loc) => {
                code.write_annotation("location");
                code.write_surrounded(Surrounded::parens(), |code| loc.write_code(code));
            }
            InterStageIo::BlendSrc(src) => {
                code.write_annotation("blend_src");
                code.write_surrounded(Surrounded::parens(), |code| src.write_code(code));
            }
            InterStageIo::Interpolate(lerp) => lerp.write_code(code),
            InterStageIo::Invariant => code.write_annotation("invariant"),
        }
        code.space();
    }
}

impl GenerateCode for FnArg {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let Self {
            inter_stage_io,
            ident,
            colon_token,
            ty,
        } = self;
        for inter_stage_io in inter_stage_io.iter() {
            inter_stage_io.write_code(code);
        }
        ident.write_code(code);
        code.write_atom(colon_token);
        ty.write_code(code);
    }
}

impl GenerateCode for FnAttrs {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            FnAttrs::None => {}
            FnAttrs::Vertex => {
                code.write_annotation("vertex");
                code.space();
            }
            FnAttrs::Fragment => {
                code.write_annotation("fragment");
                code.space();
            }
        }
    }
}

impl GenerateCode for ItemFn {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ItemFn {
            fn_attrs,
            fn_token,
            ident,
            paren_token,
            inputs,
            return_type,
            block,
        } = self;

        if !code.last_line_is_empty() {
            code.newline();
        }

        fn_attrs.write_code(code);
        code.write_atom(fn_token);
        code.space();

        ident.write_code(code);
        code.write_surrounded(
            Surrounded::parens().with_span(paren_token.span.join()),
            |code| code.write_sequenced(Sequenced::comma(), inputs.iter()),
        );
        code.space();

        return_type.write_code(code);
        block.write_code(code);
    }
}

impl GenerateCode for ItemConst {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ItemConst {
            const_token,
            ident,
            colon_token,
            ty,
            eq_token,
            expr,
            semi_token,
        } = self;
        code.write_atom(const_token);
        code.space();
        ident.write_code(code);
        code.write_atom(colon_token);
        code.space();
        ty.write_code(code);
        code.space();
        code.write_atom(eq_token);
        code.space();
        expr.write_code(code);
        code.write_atom(semi_token);
        code.newline();
    }
}

impl GenerateCode for ItemUniform {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let Self {
            group,
            binding,
            name,
            colon_token,
            ty,
        } = self;

        // @group(#group) @binding(#binding) var<uniform> #name: #ty;

        code.write_annotation("group");
        code.write_surrounded(Surrounded::parens(), |code| {
            group.write_code(code);
        });
        code.space();

        code.write_annotation("binding");
        code.write_surrounded(Surrounded::parens(), |code| {
            binding.write_code(code);
        });
        code.space();

        code.write_str(name.span(), "var<uniform>");
        code.space();

        name.write_code(code);
        code.write_atom(colon_token);
        ty.write_code(code);
        code.write_atom(&<syn::Token![;]>::default());

        code.newline();
    }
}

impl GenerateCode for Field {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        for io in self.inter_stage_io.iter() {
            io.write_code(code);
        }
        self.ident.write_code(code);
        code.write_atom(&self.colon_token);
        self.ty.write_code(code);
    }
}

impl GenerateCode for FieldsNamed {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_surrounded(Surrounded::block(self.brace_token.span.join()), |code| {
            code.write_sequenced(Sequenced::comma_with_newlines(), self.named.iter());
        });
    }
}

impl GenerateCode for ItemStruct {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ItemStruct {
            struct_token,
            ident,
            fields,
        } = self;

        if !code.last_line_is_empty() {
            code.newline();
        }

        code.write_atom(struct_token);
        code.space();
        ident.write_code(code);
        code.space();
        code.write_surrounded(Surrounded::block(fields.brace_token.span.join()), |code| {
            code.write_sequenced(Sequenced::comma_with_newlines(), fields.named.iter());
        });
    }
}

impl GenerateCode for ItemMod {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        for item in &self.content {
            item.write_code(code);
        }
    }
}

impl GenerateCode for Item {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Item::Mod(item_mod) => item_mod.write_code(code),
            Item::Uniform(item_uniform) => item_uniform.write_code(code),
            Item::Const(item_const) => item_const.write_code(code),
            Item::Fn(item_fn) => item_fn.write_code(code),
            Item::Use(_item_use) => {
                // Skip as "use" does not produce WGSL.
                //
                // Instead "use" is used by the `wgsl` macro to include
                // imports of other WGSL code.
            }
            Item::Struct(item_struct) => item_struct.write_code(code),
        }
    }
}
