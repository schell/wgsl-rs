//! Formats a WGSL token tree, poorly.

use std::sync::{Arc, LazyLock, Mutex, RwLock};

use proc_macro2::{LineColumn, Span, TokenStream};
use quote::ToTokens;
use syn::{Token, spanned::Spanned};

use crate::parse::{Item, ItemMod, ItemStruct};

mod formatter;

pub struct SourceMapping {
    pub rust_span: (LineColumn, LineColumn),
    pub wgsl_span: (LineColumn, LineColumn),
}

// struct Generator<'a, T> {
//     pub lines: Vec<Line>,
//     pub line: String,
//     pub last_node: Option<&'a T>,
// }

// pub struct Generator {
//     formatter: Formatter,
//     source_map: Vec<SourceMapping>,
// }

// impl Generator {
//     pub fn span(
//         &mut self,
//         f: impl FnOnce(&mut Generator) -> proc_macro2::Span,
//     ) -> proc_macro2::Span {
//         let wgsl_start = self.formatter.next_wgsl_line_column();
//         let span = f(self);
//         let wgsl_end = self.formatter.last_wgsl_line_column();
//         self.source_map.push(SourceMapping {
//             rust_span: (span.start(), span.end()),
//             wgsl_span: (wgsl_start, wgsl_end),
//         });
//         span
//     }

//     pub fn emit(&mut self, span: Span, node: &impl ToTokens) -> Span {
//         self.span(|g| {
//             g.formatter.write_node(node);
//             span
//         })
//     }

//     pub fn surround(&mut self, span: Span, open: char, close: char, )
// }

// fn concat_spans(spans: impl IntoIterator<Item = Span>) -> Span {
//     let mut span = None::<Span>;
//     for right_span in spans.into_iter() {
//         if let Some(left_span) = span.take() {
//             span = Some(left_span.join(right_span).unwrap());
//         }
//     }
//     span.unwrap_or_else(|| Span::call_site())
// }

trait ToWgsl {
    fn to_wgsl(&self, tokens: &mut TokenStream) -> impl Iterator<Item = SourceMapping>;

    // fn write_wgsl(&self, generator: &mut Generator) -> ! {
    //     let rust_span = ast.span();
    //     let rust_span = (rust_span.start(), rust_span.end());
    //     let wgsl_start = generator.current_wgsl_line_column();
    //     ast.to_wgsl(self);
    //     let wgsl_end = self.current_wgsl_line_column();
    //     self
    // }
}

// #[repr(transparent)]
// struct Atom<'a, T>(&'a T);

// impl<'a, T: Spanned + ToTokens> ToWgsl for Atom<'a, T> {
//     fn to_wgsl(&self, generator: &mut Generator) -> proc_macro2::Span {
//         generator.emit(self.0.span(), self.0)
//     }
// }

// impl<T: ToWgsl> ToWgsl for Vec<T> {
//     fn to_wgsl(&self, generator: &mut Generator) -> proc_macro2::Span {
//         generator.span(|g| {
//             let spans = self.iter().map(|t| t.to_wgsl(g)).collect::<Vec<_>>();
//             concat_spans(spans)
//         })
//     }
// }

// impl ToWgsl for ItemStruct {
//     fn to_wgsl(&self, generator: &mut Generator) {
//         let ItemStruct {
//             struct_token,
//             ident,
//             fields,
//         } = self;
//         fields.brace_token.surround(tokens, f);

//         generator.span(|g| {
//             concat_spans(vec![
//                 Atom(struct_token).to_wgsl(g),
//                 Atom(ident).to_wgsl(g),
//                 g.emit(fields.brace_token.span(), syn::Delimiter::Brace),
//                 fields.brace_token.surround(tokens, |tokens| {
//                     fields.named.to_tokens(tokens);
//                 }),
//             ])
//         })
//     }
// }

// impl ToWgsl for Item {
//     fn to_wgsl(&self, generator: &mut Generator) -> impl Iterator<Item = Box<dyn ToWgsl>> {
//         match self {
//             Item::Mod(item_mod) => item_mod.to_wgsl(generator),
//             Item::Uniform(item_uniform) => item_uniform.as_wgsl(generator),
//             Item::Const(item_const) => item_const.as_wgsl(generator),
//             Item::Fn(item_fn) => item_fn.as_wgsl(generator),
//             Item::Use(item_use) => {
//                 // Skip as "use" does not produce WGSL.
//                 //
//                 // Instead "use" is used by the `wgsl` macro to include
//                 // imports of other WGSL code.
//                 item_use
//                     .modules
//                     .iter()
//                     .map(|p| p.span())
//                     .fold(None, |mut ms, p| {
//                         if let Some(s) = ms.take() {
//                             Some(s.join(p.span()))
//                         } else {
//                             Some(p.span())
//                         }
//                     })
//                     .unwrap_or_else(|| Span::call_site())
//             }
//             Item::Struct(item_struct) => item_struct.to_wgsl(),
//         }
//     }
// }

// impl ToWgsl for ItemMod {
//     fn to_wgsl(&self, generator: &mut Generator) -> proc_macro2::Span {
//         generator.span(|g| self.content.to_wgsl(g))
//     }
// }

// #[derive(Default)]
// pub struct GeneratedWgsl {
//     pub source_lines: Vec<String>,
//     pub source_map: Vec<SourceMapping>,
// }

/// Generate the WGSL code and a source map back to Rust spans.
pub fn generate_wgsl(module: ItemMod) -> GeneratedWgsl {
    let mut generator = Generator::default();
    // let mut fmt = formatter::Formatter::default();
    // let mut source_map = vec![];

    let _span = module.to_wgsl(&mut generator);
    let mut indent = 0;
    let source_lines = generator
        .formatter
        .lines
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
        .collect();
    GeneratedWgsl {
        source_lines,
        source_map: generator.source_map,
    }
}
