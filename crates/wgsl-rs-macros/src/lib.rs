#![cfg_attr(nightly, feature(proc_macro_diagnostic))]

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use snafu::prelude::*;
use syn::visit_mut::{self, VisitMut};

#[cfg(feature = "validation")]
use crate::code_gen::GeneratedWgslCode;
use crate::parse::InterStageIo;

mod code_gen;
#[cfg(feature = "linkage-wgpu")]
mod linkage;
mod parse;
mod storage;
mod swizzle;
mod uniform;

/// Visitor that strips `#[wgsl_allow(...)]` attributes from expressions.
/// These attributes are used by wgsl-rs during parsing but should not appear
/// in the emitted Rust code (since statement-level attributes require nightly).
struct StripWgslAllowAttrs;

impl VisitMut for StripWgslAllowAttrs {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        // Strip wgsl_allow attributes from any expression that has them
        strip_wgsl_allow_attrs(expr);
        // Continue visiting nested expressions
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Strips `#[wgsl_allow(...)]` attributes from an expression.
fn strip_wgsl_allow_attrs(expr: &mut syn::Expr) {
    let attrs = match expr {
        syn::Expr::ForLoop(e) => &mut e.attrs,
        syn::Expr::While(e) => &mut e.attrs,
        syn::Expr::Loop(e) => &mut e.attrs,
        syn::Expr::If(e) => &mut e.attrs,
        syn::Expr::Match(e) => &mut e.attrs,
        syn::Expr::Block(e) => &mut e.attrs,
        _ => return,
    };

    attrs.retain(|attr| !attr.path().is_ident("wgsl_allow"));
}

#[derive(Default)]
struct Attrs {
    /// Present if the `wgsl` macro is of the form:
    /// #[wgsl(crate_path = path::to::crate)]
    ///
    /// Otherwise this is `None`.
    crate_path: Option<syn::Path>,

    /// If true, skip all validation (compile-time and test-time).
    ///
    /// Set via `#[wgsl(skip_validation)]`.
    skip_validation: bool,
}

impl syn::parse::Parse for Attrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::default());
        }

        let mut attrs = Self::default();

        loop {
            let ident: syn::Ident = input.parse()?;
            match ident.to_string().as_str() {
                "crate_path" => {
                    let _eq: syn::Token![=] = input.parse()?;
                    let path: syn::Path = input.parse()?;
                    attrs.crate_path = Some(path);
                }
                "skip_validation" => {
                    attrs.skip_validation = true;
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "Unknown attribute '{other}', expected 'crate_path' or \
                             'skip_validation'"
                        ),
                    ));
                }
            }

            // Check for comma separator or end
            if input.is_empty() {
                break;
            }
            let _comma: syn::Token![,] = input.parse()?;
            if input.is_empty() {
                break;
            }
        }

        Ok(attrs)
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

    /// Converts a byte offset in the source to a LineColumn position.
    /// Returns 1-based line and column numbers to match our WGSL source map.
    fn offset_to_line_column(source: &str, offset: u32) -> proc_macro2::LineColumn {
        let mut line = 1;
        let mut column = 1;
        for (i, ch) in source.char_indices() {
            if i >= offset as usize {
                break;
            }
            if ch == '\n' {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }
        proc_macro2::LineColumn { line, column }
    }

    /// Converts a naga error message and source location into a syn::Error,
    /// mapping WGSL source positions back to Rust source spans when possible.
    fn convert_naga_error(
        code: &GeneratedWgslCode,
        source: &str,
        msg: &str,
        loc: Option<naga::SourceLocation>,
    ) -> syn::Error {
        if let Some(naga::SourceLocation { offset, length, .. }) = loc {
            let wgsl_start = offset_to_line_column(source, offset);
            let wgsl_end = offset_to_line_column(source, offset + length);

            // First, try to find a mapping that exactly matches the naga span
            if let Some(mapping) = code.mapping_for_wgsl_span(wgsl_start, wgsl_end) {
                return syn::Error::new(mapping.rust_atom.span(), msg);
            }

            // Fall back to finding any mapping that contains the start position
            if let Some(mapping) = code.all_mappings_containing_wgsl_lc(wgsl_start).next() {
                return syn::Error::new(mapping.rust_atom.span(), msg);
            }
        }
        // Fall back to call_site if we can't map the location
        syn::Error::new(proc_macro2::Span::call_site(), msg)
    }

    // First, parse the WGSL source
    let module = match naga::front::wgsl::parse_str(&source) {
        Ok(module) => module,
        Err(e) => {
            let mut errors = e.labels().map(|(naga_span, label_msg)| {
                let loc = Some(naga_span.location(&source));
                convert_naga_error(
                    code,
                    &source,
                    &format!("WGSL parse error: {label_msg}"),
                    loc,
                )
            });
            let error = errors
                .next()
                .unwrap_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), format!("{e}")));
            return Err(errors.fold(error, |mut error, e| {
                error.combine(e);
                error
            }));
        }
    };

    // Then run validation
    let validation_result = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module);

    if let Err(e) = validation_result {
        let loc = e.location(&source);
        let msg = format!("WGSL validation error: {}", e.emit_to_string(&source));
        return Err(convert_naga_error(code, &source, &msg, loc));
    }

    Ok(())
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

/// Generates a `#[test]` function that validates the concatenated WGSL source.
///
/// This is used for modules that have imports, which cannot be validated at
/// compile-time because the imported symbols aren't available during macro
/// expansion.
fn gen_validation_test(module_ident: &syn::Ident) -> proc_macro2::TokenStream {
    let error_msg = format!("WGSL validation failed for module '{module_ident}'");
    quote! {
        #[cfg(all(test, feature = "validation"))]
        #[test]
        fn __validate_wgsl() {
            WGSL_MODULE.validate().expect(#error_msg);
        }
    }
}

fn go_wgsl(attr: TokenStream, mut input_mod: syn::ItemMod) -> Result<TokenStream, WgslGenError> {
    // Parse Attrs from attr TokenStream
    let attrs: Attrs = syn::parse(attr)?;
    let crate_path = attrs.crate_path();

    let wgsl_module = parse::ItemMod::try_from(&input_mod)?;
    let imports = wgsl_module.imports(&crate_path);

    let code = code_gen::generate_wgsl(&wgsl_module);
    let source_lines = code.source_lines();
    let module_fragment = gen_wgsl_module(&crate_path, &imports, &source_lines);

    // Generate validation test for modules with imports (unless skip_validation is
    // set)
    let validation_test = if !attrs.skip_validation && !imports.is_empty() {
        gen_validation_test(&input_mod.ident)
    } else {
        quote! {}
    };

    // Generate linkage module when feature is enabled
    #[cfg(feature = "linkage-wgpu")]
    let linkage_fragment = {
        let linkage_info =
            linkage::LinkageInfo::from_item_mod(input_mod.ident.clone(), &wgsl_module);
        linkage::generate_linkage_module(&linkage_info, &source_lines)
    };

    if let Some((_, content)) = input_mod.content.as_mut() {
        let fragment_item: syn::Item = syn::parse2(module_fragment)?;
        content.push(fragment_item);

        // Add validation test function if generated
        if !validation_test.is_empty() {
            let test_item: syn::Item = syn::parse2(validation_test)?;
            content.push(test_item);
        }

        // Add linkage if the feature is set
        #[cfg(feature = "linkage-wgpu")]
        {
            let linkage_item: syn::Item = syn::parse2(linkage_fragment)?;
            content.push(linkage_item);
        }
    }

    #[cfg(feature = "validation")]
    if !attrs.skip_validation {
        // Only validate modules that don't have imports from other WGSL modules.
        // Modules with imports cannot be validated in isolation because naga doesn't
        // see the imported symbols. These modules will be validated at test-time
        // via the auto-generated __validate_wgsl() test function.
        // Note: imports from `wgsl_rs::std` are filtered out by `imports()`, so modules
        // that only import from std will still be validated at compile-time.
        if imports.is_empty()
            && let Err(error) = validate_wgsl(&code, &source_lines)
        {
            return WgslValidateSnafu { input_mod, error }.fail();
        }
    }

    // Strip #[wgsl_allow] attributes before emitting Rust code.
    // These attributes are used during parsing but must be removed from the output
    // because statement-level attributes require the unstable stmt_expr_attributes
    // feature.
    StripWgslAllowAttrs.visit_item_mod_mut(&mut input_mod);

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
    // For now we don't do any transformations except for pulling out the
    // #[builtin(...)]s
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

#[proc_macro_attribute]
pub fn compute(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    // Strip #[builtin(...)] attributes from function arguments
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
pub fn workgroup_size(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Attribute for marking function parameters with WGSL builtin identifiers.
/// This is stripped during Rust compilation but preserved in WGSL output.
#[proc_macro_attribute]
pub fn builtin(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Suppresses specific wgsl-rs warnings/errors on annotated statements.
///
/// Use this attribute to acknowledge cases where the transpiler cannot
/// guarantee correctness at compile time, but you know the code is valid.
///
/// # Available Warnings
///
/// - `non_literal_loop_bounds`: Suppresses the error for for-loops with
///   non-literal bounds (e.g., `for i in 0..n` where `n` is a variable). WGSL
///   only supports ascending iteration, so the loop may fail at runtime if the
///   range is descending.
///
/// - `non_literal_match_statement_patterns`: Suppresses the warning for match
///   statements with non-literal case selectors (e.g., constants or
///   identifiers). WGSL requires case selectors to be const-expressions, which
///   the transpiler cannot always verify.
///
/// # Examples
///
/// ```ignore
/// pub fn sum_to_n(n: i32) -> i32 {
///     let mut total = 0;
///     #[wgsl_allow(non_literal_loop_bounds)]
///     for i in 0..n {
///         total += i;
///     }
///     total
/// }
/// ```
///
/// ```ignore
/// const LOW: i32 = 0;
/// const HIGH: i32 = 1;
///
/// pub fn with_const_patterns(level: i32) -> f32 {
///     let mut result = 0.0;
///     #[wgsl_allow(non_literal_match_statement_patterns)]
///     match level {
///         LOW => { result = 0.1; }
///         HIGH => { result = 1.0; }
///         _ => {}
///     }
///     result
/// }
/// ```
///
/// # Note
///
/// This attribute is stripped from the emitted Rust code by the `#[wgsl]`
/// macro, allowing statement-level attributes to work on stable Rust.
#[proc_macro_attribute]
pub fn wgsl_allow(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

#[proc_macro]
pub fn uniform(input: TokenStream) -> TokenStream {
    uniform::uniform(input)
}

#[proc_macro]
pub fn storage(input: TokenStream) -> TokenStream {
    storage::storage(input)
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
