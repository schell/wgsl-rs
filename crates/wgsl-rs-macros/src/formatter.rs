//! Formats a WGSL token tree, poorly.

use proc_macro2::TokenStream;

pub fn format_wgsl(tt: TokenStream) -> String {
    let mut out = String::new();

    for token in tt.into_iter() {
        match token {
            proc_macro2::TokenTree::Group(group) => todo!(),
            proc_macro2::TokenTree::Ident(ident) => todo!(),
            proc_macro2::TokenTree::Punct(punct) => todo!(),
            proc_macro2::TokenTree::Literal(literal) => todo!(),
        }
    }

    out
}
