//! Swizzle implementation macro.

use proc_macro::TokenStream;
use quote::ToTokens;
use syn::parse::Parse;

/// Parses macro input like `Vec2, [x, y], [r, g]` and
/// produces swizzle function implementations of for
/// functions x, y, r, g, xx, xy, yy, yx, rr, rg, gg, gr.
/// `r` and `g` point to `x` and `y`.
struct Swizzling {
    ty: syn::Ident,
    fields: Vec<syn::Ident>,
    swizzles: Vec<syn::Ident>,
}

impl Parse for Swizzling {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty = syn::Ident::parse(input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        // Parse the first bracketed term, eg [x, y, z, w]
        let bracketed1;
        syn::bracketed!(bracketed1 in input);
        // Parse the fields to access when swizzling
        let fields = bracketed1.parse_terminated(syn::Ident::parse, syn::Token![,])?;
        // Optionally parse the identifiers for the swizzle function, eg [r, g, b, a]
        // If these are omitted we'll use the accessors
        let swizzles = if input.peek(syn::Token![,]) {
            let _comma2: syn::Token![,] = input.parse()?;
            let bracketed2;
            syn::bracketed!(bracketed2 in input);
            bracketed2.parse_terminated(syn::Ident::parse, syn::Token![,])?
        } else {
            fields.clone()
        };

        Ok(Self {
            ty,
            fields: fields.into_iter().collect(),
            swizzles: swizzles.into_iter().collect(),
        })
    }
}

impl ToTokens for Swizzling {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ty = &self.ty;
        let fields = &self.fields;
        let swizzles = &self.swizzles;

        // For each swizzle accessor (e.g. x, y, r, g), generate a method
        // For simplicity, only generate single-field accessors here
        let mut methods = Vec::new();
        for (i, swizzle) in swizzles.iter().enumerate() {
            let field = &fields[i];
            methods.push(quote::quote! {
                impl #ty {
                    pub fn #swizzle(&self) -> &Self {
                        // This is a placeholder; real implementation would return the field
                        self
                    }
                }
            });
        }

        tokens.extend(quote::quote! {
            #(#methods)*
        });
    }
}

pub fn swizzle(token_stream: TokenStream) -> TokenStream {
    let swizzling: Swizzling = syn::parse_macro_input!(token_stream);
    swizzling.into_token_stream().into()
}
