//! Swizzle implementation macro.

use proc_macro::TokenStream;
use syn::parse::Parse;

/// Parses macro input like `Vec2, [x, y], [r, g]` and
/// produces swizzle function implementations of for
/// functions x, y, r, g, xx, xy, yy, yx, rr, rg, gg, gr.
/// `r` and `g` point to `x` and `y`.
struct Swizzling {
    ty: syn::Ident,
}

impl Parse for Swizzling {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty = syn::Ident::parse(input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        // Parse the first bracketed term: [x, y]
        let bracketed1;
        syn::bracketed!(bracketed1 in input);
        let _idents1: syn::punctuated::Punctuated<syn::Ident, syn::Token![,]> =
            bracketed1.parse_terminated(syn::Ident::parse)?;

        // Optionally parse a comma and second bracketed term: [r, g]
        let _idents2 = if input.peek(syn::Token![,]) {
            let _comma2: syn::Token![,] = input.parse()?;
            let bracketed2;
            syn::bracketed!(bracketed2 in input);
            Some(
                bracketed2.parse_terminated(syn::Ident::parse)?
            )
        } else {
            None
        };

        Err(syn::Error::new(input.span(), format!("{input:#?}")))
    }
}

pub fn swizzle(token_stream: TokenStream) -> TokenStream {
    let swizzling: Swizzling = syn::parse_macro_input!(token_stream);
    quote::quote! {}.into()
}
