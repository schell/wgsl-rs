//! Formats a WGSL token tree, poorly.

use proc_macro2::TokenStream;

enum Line {
    IndentInc,
    IndentDec,
    String(String),
}

#[derive(Default)]
struct Formatter {
    lines: Vec<Line>,
    line: String,
    last_token: Option<proc_macro2::TokenTree>,
}

impl Formatter {
    fn write_end_of_line(&mut self, ending: &str) {
        self.line.push_str(ending);
        self.lines
            .push(Line::String(std::mem::take(&mut self.line)));
    }

    fn last_was_ident(&self) -> bool {
        matches!(self.last_token, Some(proc_macro2::TokenTree::Ident(_)))
    }
    fn last_was_lit(&self) -> bool {
        matches!(self.last_token, Some(proc_macro2::TokenTree::Literal(_)))
    }
    fn last_was_closing_punc(&self) -> bool {
        match self.last_token.as_ref() {
            Some(proc_macro2::TokenTree::Punct(p)) => matches!(p.as_char(), ':' | '>'),
            Some(proc_macro2::TokenTree::Group(g)) => match g.delimiter() {
                proc_macro2::Delimiter::Parenthesis => true,
                proc_macro2::Delimiter::Brace => false,
                proc_macro2::Delimiter::Bracket => false,
                proc_macro2::Delimiter::None => false,
            },
            _ => false,
        }
    }

    fn indent(&mut self) {
        self.lines.push(Line::IndentInc);
    }

    fn outdent(&mut self) {
        self.lines.push(Line::IndentDec);
    }

    fn add_token(&mut self, tt: proc_macro2::TokenTree) {
        match &tt {
            proc_macro2::TokenTree::Group(group) => {
                let delim = group.delimiter();
                let (open, close, is_indented) = match delim {
                    proc_macro2::Delimiter::Parenthesis => ("(", ")", false),
                    proc_macro2::Delimiter::Brace => (" {", "}", true),
                    proc_macro2::Delimiter::Bracket => ("[", "]", false),
                    proc_macro2::Delimiter::None => ("", "", false),
                };

                if is_indented {
                    self.write_end_of_line(open);
                    self.indent();
                } else {
                    self.line.push_str(open);
                }
                self.last_token = None;
                for group_tt in group.stream() {
                    self.add_token(group_tt);
                }
                if is_indented {
                    self.outdent();
                    self.write_end_of_line(close);
                } else {
                    self.line.push_str(close);
                }
            }
            proc_macro2::TokenTree::Ident(ident) => {
                if self.last_was_ident() || self.last_was_lit() || self.last_was_closing_punc() {
                    self.line.push(' ');
                }
                self.line.push_str(&ident.to_string());
            }
            proc_macro2::TokenTree::Punct(punct) => {
                let ch = punct.as_char();
                if ch == ';' {
                    self.write_end_of_line(";");
                } else if ch == ',' {
                    self.line.push_str(", ");
                } else if ch == '=' {
                    self.line.push_str(" = ");
                } else if matches!(ch, '@' | '-') {
                    if self.last_was_closing_punc() {
                        self.line.push(' ');
                    }
                    self.line.push(ch);
                } else {
                    self.line.push(ch);
                }
            }
            proc_macro2::TokenTree::Literal(literal) => {
                self.line.push_str(&literal.to_string());
            }
        }
        self.last_token = Some(tt);
    }
}

pub fn format_wgsl(token_stream: TokenStream) -> Vec<String> {
    let mut fmt = Formatter::default();
    for tt in token_stream {
        fmt.add_token(tt);
    }

    let mut indent = 0;
    fmt.lines
        .into_iter()
        .flat_map(|line| match line {
            Line::IndentInc => {
                indent += 1;
                None
            }

            Line::IndentDec => {
                indent -= 1;
                None
            }
            Line::String(line) => {
                let padding = "    ".repeat(indent);
                Some(format!("{padding}{line}"))
            }
        })
        .collect()
}
