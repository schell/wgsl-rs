//! Performs code generation and formatting.
use proc_macro2::{Ident, LineColumn, Span};
use quote::ToTokens;
use syn::spanned::Spanned;

use crate::parse::*;

#[derive(Clone)]
pub enum RustAtom {
    Tokens {
        tokens: proc_macro2::TokenStream,
        span: Span,
    },
    String {
        string: String,
        span: Span,
    },
}

impl std::fmt::Debug for RustAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokens { tokens, span } => f
                .debug_tuple("Tokens")
                .field(&tokens.to_string())
                .field(&(span.start(), span.end()))
                .finish(),
            Self::String { string, span } => f
                .debug_struct("String")
                .field("string", string)
                .field("span", &(span.start(), span.end()))
                .finish(),
        }
    }
}

impl RustAtom {
    pub fn span(&self) -> Span {
        match self {
            Self::Tokens { tokens: _, span } => *span,
            Self::String { string: _, span } => *span,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SourceMapping {
    pub rust_atom: RustAtom,
    pub wgsl_span: (LineColumn, LineColumn),
}

// Helper to compare line/column positions
fn lc_is_before_or_equal(a: &LineColumn, b: &LineColumn) -> bool {
    a.line < b.line || (a.line == b.line && a.column <= b.column)
}

impl SourceMapping {
    /// Returns the number of WGSL lines spanned
    pub fn wgsl_lines_spanned(&self) -> usize {
        self.wgsl_span.1.line - self.wgsl_span.0.line
    }

    /// Returns the number of WGSL columns spanned
    pub fn wgsl_columns_spanned(&self) -> usize {
        // We have to make sure that we only compare columns on one line,
        // otherwise it doesn't really mean anything unless we actually
        // count the bytes in the source code
        if self.wgsl_span.0.line == self.wgsl_span.1.line {
            self.wgsl_span.1.column - self.wgsl_span.0.column
        } else {
            0
        }
    }

    /// Returns whether this mapping contains the input WGSL line column.
    pub fn wgsl_contains(&self, line_column: LineColumn) -> bool {
        let (start, end) = self.wgsl_span;

        // line_column must be: start <= line_column <= end
        lc_is_before_or_equal(&start, &line_column) && lc_is_before_or_equal(&line_column, &end)
    }

    /// Returns the size of the span by lines and columns
    pub fn wgsl_size(&self) -> (usize, usize) {
        (self.wgsl_lines_spanned(), self.wgsl_columns_spanned())
    }
}

pub enum Line {
    IndentInc,
    IndentDec,
    Source(String),
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

#[derive(Default)]
pub struct GeneratedWgslCode {
    pub lines: Vec<Line>,
    pub line: String,
    pub source_map: Vec<SourceMapping>,
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
    ///
    /// This is used for situations where we don't have verbatim Rust tokens to
    /// write.
    fn write_str(&mut self, rust_span: Span, s: &str) {
        let wgsl_start = self.next_wgsl_line_column();
        self.line.push_str(s);
        let wgsl_end = self.last_wgsl_line_column();
        self.source_map.push(SourceMapping {
            rust_atom: RustAtom::String {
                string: s.to_owned(),
                span: rust_span,
            },
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    /// Write a bit of WGSL code that is the "leaf" or "atom" of the tree.
    fn write_atom(&mut self, to_tokens: &impl ToTokens) {
        let tokens = to_tokens.into_token_stream();
        let span = tokens.span();
        let wgsl_start = self.next_wgsl_line_column();
        self.line.push_str(&tokens.to_string());
        let wgsl_end = self.last_wgsl_line_column();

        self.source_map.push(SourceMapping {
            rust_atom: RustAtom::Tokens { tokens, span },
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    /// Write an annotation.
    fn write_annotation(&mut self, ident: &Ident) {
        self.line.push('@');
        self.write_atom(ident);
    }

    /// Create a new line.
    fn newline(&mut self) {
        self.lines
            .push(Line::Source(std::mem::take(&mut self.line)));
    }

    /// If the current line is empty the last completed line is popped off the
    /// list and put back as the line to be appended to.
    fn collapse_empty_trailing_line(&mut self) {
        if self.line.is_empty()
            && let Some(line) = self.lines.pop()
        {
            if let Line::Source(line) = line {
                self.line = line;
            } else {
                self.lines.push(line);
            }
        }
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
            rust_atom: RustAtom::String {
                span: rust_span,
                string: format!("{}{}", surrounded.open, surrounded.close),
            },
            wgsl_span: (wgsl_start, wgsl_end),
        });
    }

    fn append(&mut self, node: impl Into<GeneratedWgslCode>) {
        let GeneratedWgslCode {
            lines,
            line,
            mut source_map,
        } = node.into();
        let num_lines = self.lines.len().max(1) - 1;
        for mapping in source_map.iter_mut() {
            mapping.wgsl_span.0.line += num_lines;
            mapping.wgsl_span.1.line += num_lines;
        }

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

    #[cfg(test)]
    /// Construct the WGSL source code and return it as one contiguous string.
    pub fn source(&self) -> String {
        let mut result = self.source_lines().join("\n");
        // Include any uncommitted content on the current line
        if !self.line.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&self.line);
        }
        result
    }

    /// Returns the mapping that exactly matches the given WGSL span, if any.
    /// Falls back to finding the smallest mapping that contains the span.
    pub fn mapping_for_wgsl_span(
        &self,
        start: LineColumn,
        end: LineColumn,
    ) -> Option<&SourceMapping> {
        // First, try to find an exact match
        if let Some(mapping) = self
            .source_map
            .iter()
            .find(|m| m.wgsl_span.0 == start && m.wgsl_span.1 == end)
        {
            return Some(mapping);
        }

        // Fall back to finding the smallest mapping that contains both start and end
        self.source_map
            .iter()
            .filter(|m| m.wgsl_contains(start) && m.wgsl_contains(end))
            .min_by_key(|m| m.wgsl_size())
    }

    /// Returns all Rust spans that contain the WGSL line column.
    pub fn all_mappings_containing_wgsl_lc(
        &self,
        wgsl_lc: LineColumn,
    ) -> impl Iterator<Item = &'_ SourceMapping> {
        let mut mappings = self
            .source_map
            .iter()
            .filter(move |mapping| mapping.wgsl_contains(wgsl_lc))
            .collect::<Vec<_>>();
        mappings.sort_by_key(|a| a.wgsl_size());
        mappings.into_iter()
    }
}

pub trait GenerateCode {
    fn write_code(&self, code: &mut GeneratedWgslCode);

    #[cfg(test)]
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

impl GenerateCode for FnPath {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            FnPath::Ident(ident) => ident.write_code(code),
            FnPath::TypeMethod {
                ty,
                colon2_token,
                method,
            } => {
                // Light::attenuate -> Light_attenuate
                ty.write_code(code);
                code.write_str(colon2_token.span(), "_");
                method.write_code(code);
            }
        }
    }
}

impl GenerateCode for proc_macro2::TokenStream {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(self);
    }
}

impl GenerateCode for syn::LitInt {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        // Convert Rust integer literal suffixes to WGSL suffixes:
        // - u32, usize -> u
        // - i32, isize -> i
        // - no suffix -> no suffix (abstract int)
        let base = self.base10_digits();
        let suffix = self.suffix();
        let wgsl_suffix = match suffix {
            "u32" | "usize" => "u",
            "i32" | "isize" => "i",
            "" => "",
            // For any other suffix, just pass it through
            other => other,
        };
        let wgsl_lit = format!("{}{}", base, wgsl_suffix);
        code.write_str(self.span(), &wgsl_lit);
    }
}

impl GenerateCode for Lit {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Lit::Bool(lit_bool) => code.write_atom(lit_bool),
            Lit::Float(lit_float) => code.write_atom(lit_float),
            // Use the GenerateCode impl for LitInt to handle suffix conversion
            Lit::Int(lit_int) => lit_int.write_code(code),
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
                code.write_str(ident.span(), &ident.to_string().to_lowercase().to_string());
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
            Type::Matrix {
                size,
                ident,
                scalar,
            } => {
                // WGSL format: mat{N}x{N}<f32> or mat{N}x{N}f
                code.write_str(ident.span(), &format!("mat{}x{}", size, size));
                if let Some((lt, scalar_ident, gt)) = scalar {
                    code.write_atom(lt);
                    code.write_atom(scalar_ident);
                    code.write_atom(gt);
                } else {
                    code.write_str(ident.span(), "f");
                }
            }
        }
    }
}

impl GenerateCode for BinOp {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            // Arithmetic
            BinOp::Add(t) => code.write_atom(t),
            BinOp::Sub(t) => code.write_atom(t),
            BinOp::Mul(t) => code.write_atom(t),
            BinOp::Div(t) => code.write_atom(t),
            BinOp::Rem(t) => code.write_atom(t),
            // Comparison
            BinOp::Eq(t) => code.write_atom(t),
            BinOp::Ne(t) => code.write_atom(t),
            BinOp::Lt(t) => code.write_atom(t),
            BinOp::Le(t) => code.write_atom(t),
            BinOp::Gt(t) => code.write_atom(t),
            BinOp::Ge(t) => code.write_atom(t),
            // Logical
            BinOp::And(t) => code.write_atom(t),
            BinOp::Or(t) => code.write_atom(t),
            // Bitwise
            BinOp::BitAnd(t) => code.write_atom(t),
            BinOp::BitOr(t) => code.write_atom(t),
            BinOp::BitXor(t) => code.write_atom(t),
            BinOp::Shl(t) => code.write_atom(t),
            BinOp::Shr(t) => code.write_atom(t),
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

impl GenerateCode for CompoundOp {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            CompoundOp::AddAssign(t) => code.write_atom(t),
            CompoundOp::SubAssign(t) => code.write_atom(t),
            CompoundOp::MulAssign(t) => code.write_atom(t),
            CompoundOp::DivAssign(t) => code.write_atom(t),
            CompoundOp::RemAssign(t) => code.write_atom(t),
            CompoundOp::BitAndAssign(t) => code.write_atom(t),
            CompoundOp::BitOrAssign(t) => code.write_atom(t),
            CompoundOp::BitXorAssign(t) => code.write_atom(t),
            CompoundOp::ShlAssign(t) => code.write_atom(t),
            CompoundOp::ShrAssign(t) => code.write_atom(t),
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
                code.write_str(Span::call_site(), "array");
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
                path,
                paren_token,
                params,
            } => {
                path.write_code(code);
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
                code.write_surrounded(
                    Surrounded::parens().with_span(brace_token.span.join()),
                    |code| {
                        for pair in fields.pairs() {
                            pair.value().expr.write_code(code);
                            if let Some(p) = pair.punct() {
                                code.write_atom(p);
                                code.space();
                            }
                        }
                    },
                );
            }
            Expr::TypePath {
                ty,
                colon2_token,
                member,
            } => {
                // Light::CONSTANT -> Light_CONSTANT
                ty.write_code(code);
                code.write_str(colon2_token.span(), "_");
                member.write_code(code);
            }
        }
    }
}

impl GenerateCode for ReturnTypeAnnotation {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            ReturnTypeAnnotation::None => {}
            ReturnTypeAnnotation::BuiltIn(ident) => {
                code.write_annotation(ident);
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
            ReturnTypeAnnotation::Location { ident, lit } => {
                code.write_annotation(ident);
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
            ReturnTypeAnnotation::DefaultBuiltInPosition => {
                code.write_str(Span::call_site(), "@builtin(position)");
                code.space();
            }
            ReturnTypeAnnotation::DefaultLocation => {
                code.write_str(Span::call_site(), "@location(0)");
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
            Stmt::Assignment {
                lhs,
                eq_token,
                rhs,
                semi_token,
            } => {
                lhs.write_code(code);
                code.space();
                code.write_atom(eq_token);
                code.space();
                rhs.write_code(code);
                code.write_atom(semi_token);
            }
            Stmt::CompoundAssignment {
                lhs,
                op,
                rhs,
                semi_token,
            } => {
                lhs.write_code(code);
                code.space();
                op.write_code(code);
                code.space();
                rhs.write_code(code);
                code.write_atom(semi_token);
            }
            Stmt::While {
                while_token,
                condition,
                body,
            } => {
                code.write_atom(while_token);
                code.space();
                condition.write_code(code);
                code.space();
                body.write_code(code);
            }
            Stmt::Expr { expr, semi_token } => {
                if let Some(semi) = semi_token {
                    expr.write_code(code);
                    code.write_atom(semi);
                } else {
                    code.write_str(Span::call_site(), "return");
                    code.space();
                    expr.write_code(code);
                    // Collapse any empty trailing line so the semitoken is written on the end of
                    // the last line with code on it.
                    code.collapse_empty_trailing_line();
                    code.write_atom(&<syn::Token![;]>::default());
                }
            }
            Stmt::If(stmt_if) => stmt_if.write_code(code),
            Stmt::Break { break_token } => {
                code.write_atom(break_token);
                code.write_atom(&<syn::Token![;]>::default());
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

/// Helper to write a block for if statements, controlling trailing newline.
///
/// When `has_else` is true, we want `} else {` on the same line, so we
/// collapse the trailing newline after the closing brace.
fn write_block_for_if(block: &Block, code: &mut GeneratedWgslCode, has_else: bool) {
    let brace_token = &block.brace_token;
    let stmts = &block.stmt;
    code.write_surrounded(Surrounded::block(brace_token.span.join()), |code| {
        code.write_sequenced(Sequenced::newlines(), stmts);
    });

    if has_else {
        // Don't add newline - we want "} else {" on same line
        code.collapse_empty_trailing_line();
        code.space();
    } else {
        code.newline();
    }
}

impl GenerateCode for StmtIf {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(&self.if_token);
        code.space();
        self.condition.write_code(code);
        code.space();

        // Write the then block, controlling newline based on else presence
        write_block_for_if(&self.then_block, code, self.else_branch.is_some());

        if let Some(else_branch) = &self.else_branch {
            else_branch.write_code(code);
        }
    }
}

impl GenerateCode for ElseBranch {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        code.write_atom(&self.else_token);
        code.space();
        self.body.write_code(code);
    }
}

impl GenerateCode for ElseBody {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            ElseBody::Block(block) => {
                // Final else block - use normal block formatting with trailing newline
                block.write_code(code);
            }
            ElseBody::If(if_stmt) => {
                // else if - recursively write the if statement
                if_stmt.write_code(code);
            }
        }
    }
}

impl GenerateCode for BuiltIn {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ident = match self {
            BuiltIn::VertexIndex(ident) => ident,
            BuiltIn::InstanceIndex(ident) => ident,
            BuiltIn::Position(ident) => ident,
            BuiltIn::FrontFacing(ident) => ident,
            BuiltIn::FragDepth(ident) => ident,
            BuiltIn::SampleIndex(ident) => ident,
            BuiltIn::SampleMask(ident) => ident,
            BuiltIn::LocalInvocationId(ident) => ident,
            BuiltIn::LocalInvocationIndex(ident) => ident,
            BuiltIn::GlobalInvocationId(ident) => ident,
            BuiltIn::WorkgroupId(ident) => ident,
            BuiltIn::NumWorkgroups(ident) => ident,
            BuiltIn::SubgroupInvocationId(ident) => ident,
            BuiltIn::SubgroupSize(ident) => ident,
            BuiltIn::PrimitiveIndex(ident) => ident,
            BuiltIn::SubgroupId(ident) => ident,
            BuiltIn::NumSubgroups(ident) => ident,
        };
        code.write_atom(ident);
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
        ty.write_code(code);
        code.write_atom(comma_token);
        sampling.write_code(code);
    }
}

impl GenerateCode for InterStageIo {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            InterStageIo::BuiltIn {
                ident,
                paren_token,
                inner,
            } => {
                code.write_annotation(ident);
                code.write_surrounded(
                    Surrounded::parens().with_span(paren_token.span.join()),
                    |code| inner.write_code(code),
                );
            }
            InterStageIo::Location {
                ident,
                paren_token,
                inner,
            } => {
                code.write_annotation(ident);
                code.write_surrounded(
                    Surrounded::parens().with_span(paren_token.span.join()),
                    |code| inner.write_code(code),
                );
            }
            InterStageIo::BlendSrc {
                ident,
                paren_token,
                lit,
            } => {
                code.write_annotation(ident);
                code.write_surrounded(
                    Surrounded::parens().with_span(paren_token.span.join()),
                    |code| lit.write_code(code),
                );
            }
            InterStageIo::Interpolate {
                ident,
                paren_token,
                inner,
            } => {
                code.write_annotation(ident);
                code.write_surrounded(
                    Surrounded::parens().with_span(paren_token.span.join()),
                    |code| inner.write_code(code),
                );
            }
            InterStageIo::Invariant(ident) => code.write_annotation(ident),
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
            FnAttrs::Vertex(ident) => {
                code.write_annotation(ident);
                code.space();
            }
            FnAttrs::Fragment(ident) => {
                code.write_annotation(ident);
                code.space();
            }
            FnAttrs::Compute {
                ident,
                workgroup_size,
            } => {
                // @compute
                code.write_annotation(ident);
                code.space();
                // @workgroup_size(x, y?, z?)
                code.write_annotation(&workgroup_size.ident);
                code.write_surrounded(
                    Surrounded::parens().with_span(workgroup_size.paren_token.span.join()),
                    |code| {
                        workgroup_size.x.write_code(code);
                        if let Some((comma, y)) = &workgroup_size.y {
                            code.write_atom(comma);
                            code.space();
                            y.write_code(code);
                        }
                        if let Some((comma, z)) = &workgroup_size.z {
                            code.write_atom(comma);
                            code.space();
                            z.write_code(code);
                        }
                    },
                );
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
            group_ident,
            group_paren_token,
            group,
            binding_ident,
            binding_paren_token,
            binding,
            name,
            colon_token,
            ty,
            rust_ty: _,
        } = self;

        // @group(#group) @binding(#binding) var<uniform> #name: #ty;

        code.write_annotation(group_ident);
        code.write_surrounded(
            Surrounded::parens().with_span(group_paren_token.span.join()),
            |code| {
                group.write_code(code);
            },
        );
        code.space();

        code.write_annotation(binding_ident);
        code.write_surrounded(
            Surrounded::parens().with_span(binding_paren_token.span.join()),
            |code| {
                binding.write_code(code);
            },
        );
        code.space();

        code.write_str(Span::call_site(), "var<uniform>");
        code.space();

        name.write_code(code);
        code.write_atom(colon_token);
        ty.write_code(code);
        code.write_atom(&<syn::Token![;]>::default());

        code.newline();
    }
}

impl GenerateCode for ItemStorage {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let Self {
            group_ident,
            group_paren_token,
            group,
            binding_ident,
            binding_paren_token,
            binding,
            access,
            name,
            colon_token,
            ty,
            rust_ty: _,
        } = self;

        // @group(#group) @binding(#binding) var<storage, read|read_write> #name: #ty;

        code.write_annotation(group_ident);
        code.write_surrounded(
            Surrounded::parens().with_span(group_paren_token.span.join()),
            |code| {
                group.write_code(code);
            },
        );
        code.space();

        code.write_annotation(binding_ident);
        code.write_surrounded(
            Surrounded::parens().with_span(binding_paren_token.span.join()),
            |code| {
                binding.write_code(code);
            },
        );
        code.space();

        // var<storage, read> or var<storage, read_write>
        let access_mode = match access {
            StorageAccess::Read => "read",
            StorageAccess::ReadWrite => "read_write",
        };
        code.write_str(Span::call_site(), &format!("var<storage, {access_mode}>"));
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

impl GenerateCode for ItemImpl {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ItemImpl {
            _impl_token: _,
            self_ty,
            _brace_token: _,
            items,
        } = self;

        // Write each item with mangled name: StructName_member
        for item in items {
            match item {
                ImplItem::Const(item_const) => {
                    let ItemConst {
                        const_token,
                        ident,
                        colon_token,
                        ty,
                        eq_token,
                        expr,
                        semi_token,
                    } = item_const;

                    if !code.last_line_is_empty() {
                        code.newline();
                    }

                    code.write_atom(const_token);
                    code.space();

                    // Write mangled name: StructName_CONSTANT
                    self_ty.write_code(code);
                    code.write_str(ident.span(), "_");
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
                ImplItem::Fn(item_fn) => {
                    let ItemFn {
                        fn_attrs,
                        fn_token,
                        ident,
                        paren_token,
                        inputs,
                        return_type,
                        block,
                    } = item_fn;

                    if !code.last_line_is_empty() {
                        code.newline();
                    }

                    fn_attrs.write_code(code);
                    code.write_atom(fn_token);
                    code.space();

                    // Write mangled name: StructName_method
                    self_ty.write_code(code);
                    code.write_str(ident.span(), "_");
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
        }
    }
}

impl GenerateCode for ItemEnum {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        let ItemEnum {
            enum_token,
            ident: enum_ident,
            _brace_token: _,
            variants,
        } = self;

        let mut current_discriminant: u32 = 0;

        for variant in variants {
            // If explicit discriminant, use it; otherwise use current value
            let value = if let Some((_, lit_int)) = &variant.discriminant {
                let val = lit_int
                    .base10_parse::<u32>()
                    // This may silently swallow an error if the literal isn't a u32, but
                    // that case is unreachable because enums defined within a `#[wgsl]` module
                    // must be `#[repr(u32)]`, so Rust would catch this before we get here.
                    .unwrap_or(current_discriminant);
                current_discriminant = val;
                val
            } else {
                current_discriminant
            };

            // Generate: const EnumName_VariantName: u32 = Nu;
            code.write_str(enum_token.span(), "const");
            code.space();
            enum_ident.write_code(code);
            code.write_str(variant.ident.span(), "_");
            variant.ident.write_code(code);
            code.write_str(variant.ident.span(), ": u32 = ");
            code.write_str(variant.ident.span(), &format!("{value}u"));
            code.write_str(variant.ident.span(), ";");
            code.newline();

            // Increment for next variant
            current_discriminant += 1;
        }
    }
}

impl GenerateCode for Item {
    fn write_code(&self, code: &mut GeneratedWgslCode) {
        match self {
            Item::Mod(item_mod) => item_mod.write_code(code),
            Item::Uniform(item_uniform) => item_uniform.write_code(code),
            Item::Storage(item_storage) => item_storage.write_code(code),
            Item::Const(item_const) => item_const.write_code(code),
            Item::Fn(item_fn) => item_fn.write_code(code),
            Item::Use(_item_use) => {
                // Skip as "use" does not produce WGSL.
                //
                // Instead "use" is used by the `wgsl` macro to include
                // imports of other WGSL code.
            }
            Item::Struct(item_struct) => item_struct.write_code(code),
            Item::Impl(item_impl) => item_impl.write_code(code),
            Item::Enum(item_enum) => item_enum.write_code(code),
        }
    }
}
