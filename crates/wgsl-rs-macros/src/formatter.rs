//! Formats a WGSL token tree, poorly.

use proc_macro2::TokenStream;

pub fn format_wgsl(tt: TokenStream) -> String {
    fn format_inner(tt: TokenStream, indent: usize) -> String {
        let mut out = String::new();
        let mut last_was_ident = false;
        let mut last_was_literal = false;

        for token in tt.into_iter() {
            match token {
                proc_macro2::TokenTree::Group(group) => {
                    let delim = group.delimiter();
                    let (open, close) = match delim {
                        proc_macro2::Delimiter::Parenthesis => ("(", ")"),
                        proc_macro2::Delimiter::Brace => ("{", "}"),
                        proc_macro2::Delimiter::Bracket => ("[", "]"),
                        proc_macro2::Delimiter::None => ("", ""),
                    };
                    if open == "{" {
                        out.push(' ');
                        out.push_str(open);
                        out.push('\n');
                        out.push_str(&"    ".repeat(indent + 1));
                        let inner = format_inner(group.stream(), indent + 1);
                        out.push_str(inner.trim_end());
                        out.push('\n');
                        out.push_str(&"    ".repeat(indent));
                        out.push_str(close);
                    } else {
                        out.push_str(open);
                        out.push_str(&format_inner(group.stream(), indent));
                        out.push_str(close);
                    }
                    last_was_ident = false;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Ident(ident) => {
                    if last_was_ident || last_was_literal {
                        out.push(' ');
                    }
                    out.push_str(&ident.to_string());
                    last_was_ident = true;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Punct(punct) => {
                    let ch = punct.as_char();
                    if ch == ';' {
                        out.push(ch);
                        out.push('\n');
                        out.push_str(&"    ".repeat(indent));
                    } else if ch == ',' {
                        out.push(ch);
                        out.push(' ');
                    } else if ch == '=' {
                        out.push(' ');
                        out.push(ch);
                        out.push(' ');
                    } else {
                        out.push(ch);
                    }
                    last_was_ident = false;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Literal(literal) => {
                    if last_was_ident || last_was_literal {
                        out.push(' ');
                    }
                    out.push_str(&literal.to_string());
                    last_was_ident = false;
                    last_was_literal = true;
                }
            }
        }
        out
    }

    format_inner(tt, 0)
}
