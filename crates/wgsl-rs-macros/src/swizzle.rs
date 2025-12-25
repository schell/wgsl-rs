//! Swizzle implementation macro.

use itertools::Itertools;
use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::parse::Parse;

/// Parses macro input like `Vec2, [x, y], [r, g]` and
/// produces swizzle function implementations of for
/// functions x, y, r, g, xx, xy, yy, yx, rr, rg, gg, gr.
/// `r` and `g` point to `x` and `y`.
struct Swizzling {
    /// Type of Self, VecN
    ty: syn::Ident,
    /// Result types of swizzle constructors, starting with T, ending with VecN
    tys: Vec<syn::Ident>,
    /// Result types of swizzling operation, starting with Vec2
    down_constructors: Vec<syn::Ident>,

    fields: Vec<syn::Ident>,
    swizzles: Vec<syn::Ident>,
}

impl Parse for Swizzling {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty = syn::Ident::parse(input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        fn parse_bracketed_idents(
            input: &syn::parse::ParseStream,
        ) -> Result<Vec<syn::Ident>, syn::Error> {
            let bracketed;
            syn::bracketed!(bracketed in input);
            let punc = bracketed.parse_terminated(syn::Ident::parse, syn::Token![,])?;
            Ok(punc.into_iter().collect())
        }

        let tys = parse_bracketed_idents(&input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        let down_constructors = parse_bracketed_idents(&input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        // Parse the fields to access when swizzling, eg [x, y, z, w]
        let fields = parse_bracketed_idents(&input)?;
        let _comma = <syn::Token![,]>::parse(input)?;

        // Parse the identifiers for the swizzle function, eg [r, g, b, a]
        let swizzles = parse_bracketed_idents(&input)?;

        let fields_len = fields.len();
        let swizzles_len = swizzles.len();
        let tys_len = tys.len();
        let down_constructors_len = down_constructors.len();
        assert!(
            fields.len() == swizzles.len(),
            "fields {fields_len} != swizzles {swizzles_len}"
        );
        assert!(
            fields.len() == tys.len(),
            "fields {fields_len} != tys {tys_len}"
        );
        assert!(
            fields.len() == down_constructors.len() + 1,
            "fields {fields_len} != constructors {down_constructors_len} + 1"
        );

        Ok(Self {
            ty,
            tys,
            down_constructors,
            fields: fields.into_iter().collect(),
            swizzles: swizzles.into_iter().collect(),
        })
    }
}

impl ToTokens for Swizzling {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let Self {
            ty,
            tys,
            down_constructors,
            fields,
            swizzles,
        } = self;

        let mut methods: Vec<proc_macro2::TokenStream> = vec![];

        #[derive(Clone, PartialEq, PartialOrd, Hash, Eq)]
        struct SwizzleComponent {
            /// The field on "type" that contains the component
            field: syn::Ident,
            /// The name of the component used in the swizzle function
            name: syn::Ident,
        }

        struct Swizzle {
            /// The components of the swizzle.
            components: Vec<SwizzleComponent>,
            /// The constructor, if any.
            ///
            /// Single component swizzles simply return the component, so they
            /// don't need a constructor.
            constructor: Option<syn::Ident>,
            /// The return type of the swizzle
            return_ty: syn::Ident,
        }

        impl Swizzle {
            fn fn_ident(&self) -> syn::Ident {
                let names = self.components.iter().map(|p| p.name.clone());
                format_ident!("{}", names.map(|n| n.to_string()).join(""))
            }

            fn constructor(&self) -> proc_macro2::TokenStream {
                let mut components = self
                    .components
                    .iter()
                    .map(|p| p.field.clone())
                    .collect::<Vec<_>>();
                if let Some(constructor) = &self.constructor {
                    quote! { #constructor(#(self.inner.#components),*) }
                } else {
                    // There's only one component
                    let component = components.pop().unwrap();
                    quote! { self.inner.#component }
                }
            }

            fn return_ty(&self) -> proc_macro2::TokenStream {
                let return_ty = &self.return_ty;
                if self.constructor.is_some() {
                    quote! { #return_ty }
                } else {
                    return_ty.into_token_stream()
                }
            }
        }

        impl ToTokens for Swizzle {
            fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
                let fn_ident = self.fn_ident();
                let constructor = self.constructor();
                let return_ty = self.return_ty();

                quote! {
                    pub fn #fn_ident(&self) -> #return_ty {
                        #constructor
                    }
                }
                .to_tokens(tokens)
            }
        }

        let swizzlings = fields
            .iter()
            .zip(swizzles)
            .map(|(a, b)| SwizzleComponent {
                field: a.clone(),
                name: b.clone(),
            })
            .collect::<Vec<_>>();

        let mut methods = vec![];

        for ((n, return_ty), constructor) in (1..=swizzlings.len())
            .zip(tys)
            .zip(std::iter::once(None).chain(down_constructors.iter().map(Some)))
        {
            let input_swizzlings = vec![swizzlings.clone(); n].concat();
            let perms = input_swizzlings.into_iter().permutations(n).unique();
            for components in perms {
                let swizzle = Swizzle {
                    components,
                    constructor: constructor.cloned(),
                    return_ty: return_ty.clone(),
                };
                methods.push(swizzle.into_token_stream());
            }
        }

        tokens.extend(quote! {
            impl #ty {
                #(#methods)*
            }
        });
    }
}

pub fn swizzle(token_stream: TokenStream) -> TokenStream {
    let swizzling: Swizzling = syn::parse_macro_input!(token_stream);
    swizzling.into_token_stream().into()
}
