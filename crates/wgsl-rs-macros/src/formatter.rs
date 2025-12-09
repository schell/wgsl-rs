//! Formats a WGSL token tree, poorly.

use proc_macro2::TokenStream;

pub fn format_wgsl(tt: TokenStream) -> Vec<String> {
    fn format_inner(tt: TokenStream, indent: usize) -> Vec<String> {
        let mut lines = vec![String::new()];
        let mut line = String::new();
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
                        lines.last_mut().unwrap().push(' ');
                        lines.last_mut().unwrap().push_str(open);
                        lines.push("".to_string());
                        lines
                            .last_mut()
                            .unwrap()
                            .push_str(&"    ".repeat(indent + 1));
                        let inner = format_inner(group.stream(), indent + 1);
                        for l in inner {
                            if !l.is_empty() {
                                lines.push(format!(
                                    "{}{}",
                                    "    ".repeat(indent + 1),
                                    l.trim_end()
                                ));
                            }
                        }
                        lines.push(format!("{}{}", "    ".repeat(indent), close));
                    } else {
                        lines.last_mut().unwrap().push_str(open);
                        let inner = format_inner(group.stream(), indent);
                        for l in inner {
                            lines.last_mut().unwrap().push_str(&l);
                        }
                        lines.last_mut().unwrap().push_str(close);
                    }
                    last_was_ident = false;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Ident(ident) => {
                    if last_was_ident || last_was_literal {
                        lines.last_mut().unwrap().push(' ');
                    }
                    lines.last_mut().unwrap().push_str(&ident.to_string());
                    last_was_ident = true;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Punct(punct) => {
                    let ch = punct.as_char();
                    if ch == ';' {
                        lines.last_mut().unwrap().push(ch);
                        lines.push("".to_string());
                        lines.last_mut().unwrap().push_str(&"    ".repeat(indent));
                    } else if ch == ',' {
                        lines.last_mut().unwrap().push(ch);
                        lines.last_mut().unwrap().push(' ');
                    } else if ch == '=' {
                        lines.last_mut().unwrap().push(' ');
                        lines.last_mut().unwrap().push(ch);
                        lines.last_mut().unwrap().push(' ');
                    } else {
                        lines.last_mut().unwrap().push(ch);
                    }
                    last_was_ident = false;
                    last_was_literal = false;
                }
                proc_macro2::TokenTree::Literal(literal) => {
                    if last_was_ident || last_was_literal {
                        lines.last_mut().unwrap().push(' ');
                    }
                    lines.last_mut().unwrap().push_str(&literal.to_string());
                    last_was_ident = false;
                    last_was_literal = true;
                }
            }
        }
        lines
    }

    let mut lines = format_inner(tt, 0);
    // Remove any empty trailing lines
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    lines
}
