use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{ToTokens, quote};

mod formatter;
mod parse;
mod swizzle;
mod uniform;

#[derive(Default, FromMeta)]
#[darling(derive_syn_parse)]
struct Attrs {
    /// Present if the `wgsl` macro is of the form:
    /// #[wgsl(crate_path = path::to::crate)]
    ///
    /// Otherwise this is `None`.
    crate_path: Option<syn::Path>,
}

impl Attrs {
    fn crate_path(&self) -> syn::Path {
        self.crate_path
            .as_ref()
            .cloned()
            .unwrap_or_else(|| syn::Path {
                leading_colon: None,
                segments: syn::punctuated::Punctuated::from_iter(Some(syn::PathSegment {
                    ident: quote::format_ident!("wgsl_rs"),
                    arguments: syn::PathArguments::None,
                })),
            })
    }
}

fn go_wgsl(attr: TokenStream, mut input_mod: syn::ItemMod) -> Result<TokenStream, syn::Error> {
    // Parse Attrs from attr TokenStream
    let attrs: Attrs = syn::parse(attr)?;
    let crate_path = attrs.crate_path();

    let item = parse::ItemMod::try_from(&input_mod)?;
    // Get the modules imported into this one
    let imports = item.imports(&crate_path);
    // Emit both Rust code and WGSL stuff (code + linkage, etc)
    let wgsl = item.into_token_stream();
    // Format the token stream
    let wgsl_formatted = formatter::format_wgsl(wgsl);

    let fragment = quote! {
        pub const WGSL_MODULE: #crate_path::Module = #crate_path::Module {
            imports: &[
                #(&#imports),*
            ],
            source: &[
                #(#wgsl_formatted),*
            ]
        };
    };

    if let Some((_, content)) = input_mod.content.as_mut() {
        let fragment_item: syn::Item = syn::parse2(fragment)?;
        content.push(fragment_item);
    }

    Ok(input_mod.into_token_stream().into())
}

#[proc_macro_attribute]
pub fn wgsl(attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input_mod = syn::parse_macro_input!(token_stream as syn::ItemMod);
    match go_wgsl(attr, input_mod) {
        Ok(tokens) => tokens,
        Err(e) => e.into_compile_error().into(),
    }
}

#[proc_macro]
pub fn swizzle(token_stream: TokenStream) -> TokenStream {
    swizzle::swizzle(token_stream)
}

#[proc_macro_attribute]
pub fn vertex(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // For now we don't do any transformations except for pulling out the #[builtin(...)]s
    let mut item_fn: syn::ItemFn = syn::parse_macro_input!(token_stream);
    for arg in item_fn.sig.inputs.iter_mut() {
        if let syn::FnArg::Typed(pat_type) = arg {
            pat_type.attrs.retain(|attr| {
                if let Some(ident) = attr.path().get_ident() {
                    !matches!(ident.to_string().as_str(), "builtin")
                } else {
                    true
                }
            });
        }
    }
    item_fn.into_token_stream().into()
}

#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

#[proc_macro]
pub fn uniform(input: TokenStream) -> TokenStream {
    uniform::uniform(input)
}
