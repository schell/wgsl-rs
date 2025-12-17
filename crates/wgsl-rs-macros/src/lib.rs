use proc_macro::TokenStream;
use proc_macro2::LineColumn;
use quote::{ToTokens, quote};
use snafu::prelude::*;

#[cfg(feature = "validation")]
use crate::code_gen::GeneratedWgslCode;
use crate::parse::InterStageIo;

mod code_gen;
mod parse;
mod swizzle;
mod uniform;

#[derive(Default)]
struct Attrs {
    /// Present if the `wgsl` macro is of the form:
    /// #[wgsl(crate_path = path::to::crate)]
    ///
    /// Otherwise this is `None`.
    crate_path: Option<syn::Path>,
}

impl syn::parse::Parse for Attrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::default());
        }

        let ident: syn::Ident = input.parse()?;
        if ident.to_string().as_str() != "crate_path" {
            return Err(syn::Error::new(ident.span(), "Expected only 'crate_path'"));
        }

        let _eq: syn::Token![=] = input.parse()?;
        let path: syn::Path = input.parse()?;
        Ok(Self {
            crate_path: Some(path),
        })
    }
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

#[derive(Debug, Snafu)]
enum WgslGenError {
    RustParse {
        source: syn::Error,
    },

    WgslParse {
        source: parse::Error,
    },

    #[cfg(feature = "validation")]
    WgslValidate {
        input_mod: Box<syn::ItemMod>,
        error: syn::Error,
    },
}

impl From<syn::Error> for WgslGenError {
    fn from(source: syn::Error) -> Self {
        Self::RustParse { source }
    }
}

impl From<parse::Error> for WgslGenError {
    fn from(source: parse::Error) -> Self {
        Self::WgslParse { source }
    }
}

#[cfg(feature = "validation")]
fn validate_wgsl(code: &GeneratedWgslCode, source_lines: &[String]) -> Result<(), syn::Error> {
    // Validate the module and emit validation errors as compilation errors
    // by mapping the WGSL spans to Rust spans.
    let source = source_lines.join("\n");
    if let Err(e) = naga::front::wgsl::parse_str(&source) {
        let mut errors = e.labels().flat_map(|(naga_span, msg)| {
            let naga::SourceLocation {
                line_number,
                line_position,
                offset: _,
                length: _,
            } = naga_span.location(&source);
            let wgsl_lc = LineColumn {
                line: line_number as usize,
                column: (line_position - 1) as usize,
            };
            code.all_mappings_containing_wgsl_lc(wgsl_lc)
                .next()
                .map(move |mapping| {
                    syn::Error::new(
                        mapping.rust_atom.span(),
                        format!("WGSL validation error: {msg}"),
                    )
                })
        });
        let error = errors.next().expect("there is at least one");
        Err(errors.fold(error, |mut error, e| {
            error.combine(e);
            error
        }))
    } else {
        Ok(())
    }
}

fn gen_wgsl_module(
    crate_path: &syn::Path,
    imports: &[proc_macro2::TokenStream],
    source_lines: &[String],
) -> proc_macro2::TokenStream {
    quote! {
        pub const WGSL_MODULE: #crate_path::Module = #crate_path::Module {
            imports: &[
                #(&#imports),*
            ],
            source: &[
                #(#source_lines),*
            ]
        };
    }
}

fn go_wgsl(attr: TokenStream, mut input_mod: syn::ItemMod) -> Result<TokenStream, WgslGenError> {
    // Parse Attrs from attr TokenStream
    let attrs: Attrs = syn::parse(attr)?;
    let crate_path = attrs.crate_path();

    let wgsl_module = parse::ItemMod::try_from(&input_mod)?;
    let imports = wgsl_module.imports(&crate_path);

    let code = code_gen::generate_wgsl(wgsl_module);
    let source_lines = code.source_lines();

    let module_fragment = gen_wgsl_module(&crate_path, &imports, &source_lines);
    if let Some((_, content)) = input_mod.content.as_mut() {
        let fragment_item: syn::Item = syn::parse2(module_fragment)?;
        content.push(fragment_item);
    }

    #[cfg(feature = "validation")]
    {
        if let Err(error) = validate_wgsl(&code, &source_lines) {
            return WgslValidateSnafu { input_mod, error }.fail();
        }
    }

    Ok(input_mod.into_token_stream().into())
}

#[proc_macro_attribute]
pub fn wgsl(attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input_mod = syn::parse_macro_input!(token_stream as syn::ItemMod);
    match go_wgsl(attr, input_mod) {
        Ok(tokens) => tokens,
        Err(e) => match e {
            WgslGenError::RustParse { source } => source.to_compile_error().into(),
            WgslGenError::WgslParse { source } => {
                syn::Error::from(source).to_compile_error().into()
            }
            WgslGenError::WgslValidate { input_mod, error } => {
                let error = error.into_compile_error();
                quote! {
                    #input_mod
                    #error
                }
                .into()
            }
        },
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

/// Defines an "input" struct.
#[proc_macro_attribute]
pub fn input(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    let mut rust_struct: syn::ItemStruct = syn::parse_macro_input!(token_stream);
    if let syn::Fields::Named(syn::FieldsNamed {
        brace_token: _,
        named,
    }) = &mut rust_struct.fields
    {
        for syn::Field { attrs, .. } in named.iter_mut() {
            // Only keep the attributes that aren't from wgsl-rs
            let mut output_attrs = vec![];
            for attr in std::mem::take(attrs).into_iter() {
                if let Ok(_inter_stage_io) = InterStageIo::try_from(&attr) {
                    // Generate some linkage for this struct
                } else {
                    output_attrs.push(attr);
                }
            }
            *attrs = output_attrs;
        }
    }
    rust_struct.into_token_stream().into()
}

/// Defines an "output" struct.
#[proc_macro_attribute]
pub fn output(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    let mut rust_struct: syn::ItemStruct = syn::parse_macro_input!(token_stream);
    if let syn::Fields::Named(syn::FieldsNamed {
        brace_token: _,
        named,
    }) = &mut rust_struct.fields
    {
        for syn::Field { attrs, .. } in named.iter_mut() {
            // Only keep the attributes that aren't from wgsl-rs
            let mut output_attrs = vec![];
            for attr in std::mem::take(attrs).into_iter() {
                if let Ok(_inter_stage_io) = InterStageIo::try_from(&attr) {
                    // Generate some linkage for this struct
                } else {
                    output_attrs.push(attr);
                }
            }
            *attrs = output_attrs;
        }
    }
    rust_struct.into_token_stream().into()
}
