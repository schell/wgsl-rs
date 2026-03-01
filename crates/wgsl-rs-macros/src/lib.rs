#![cfg_attr(nightly, feature(proc_macro_diagnostic))]

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use snafu::prelude::*;
use syn::visit_mut::{self, VisitMut};

#[cfg(feature = "validation")]
use crate::code_gen::GeneratedWgslCode;
use crate::parse::InterStageIo;

mod builtins;
mod code_gen;
#[cfg(feature = "linkage-wgpu")]
mod linkage;
mod parse;
mod ptr;
mod sampler;
mod storage;
mod swizzle;
mod texture;
mod uniform;
mod workgroup;

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
    name: &syn::Ident,
    crate_path: &syn::Path,
    imports: &[proc_macro2::TokenStream],
    source_lines: &[String],
) -> proc_macro2::TokenStream {
    quote! {
        pub const WGSL_MODULE: #crate_path::Module = #crate_path::Module {
            name: stringify!(#name),
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
    let module_fragment = gen_wgsl_module(&wgsl_module.ident, &crate_path, &imports, &source_lines);

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
        linkage::generate_linkage_module(&linkage_info)
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

/// Transpiles a Rust module to WGSL.
///
/// Apply `#[wgsl]` to a `mod` item to generate a `WGSL_MODULE` constant
/// containing the transpiled WGSL source. The Rust code inside the module
/// remains fully functional and can be executed on the CPU, while the
/// generated WGSL runs on the GPU.
///
/// # Module Options
///
/// | Syntax | Description |
/// |--------|-------------|
/// | `#[wgsl]` | Transpile the module with default settings. |
/// | `#[wgsl(skip_validation)]` | Skip compile-time and test-time WGSL validation. |
/// | `#[wgsl(crate_path = path::to::crate)]` | Override the path to the `wgsl_rs` crate. |
///
/// Options can be combined: `#[wgsl(crate_path = my_crate::wgsl_rs,
/// skip_validation)]`.
///
/// # Entry Points
///
/// Mark `pub fn` items with an entry point attribute to declare shader stages.
///
/// | Attribute | WGSL Output | Notes |
/// |-----------|------------|-------|
/// | `#[vertex]` | `@vertex` | |
/// | `#[fragment]` | `@fragment` | |
/// | `#[compute]` | `@compute` | Must also have `#[workgroup_size(...)]`. |
/// | `#[workgroup_size(X)]` | `@workgroup_size(X)` | 1D workgroup. |
/// | `#[workgroup_size(X, Y)]` | `@workgroup_size(X, Y)` | 2D workgroup. |
/// | `#[workgroup_size(X, Y, Z)]` | `@workgroup_size(X, Y, Z)` | 3D workgroup. |
///
/// **Auto-inference rules:**
/// - A `#[vertex]` function returning `Vec4f` automatically gets
///   `@builtin(position)` on its return type.
/// - A `#[fragment]` function returning `Vec4f` automatically gets
///   `@location(0)` on its return type.
///
/// ```ignore
/// #[vertex]
/// pub fn vs_main(#[builtin(vertex_index)] vi: u32) -> Vec4f {
///     // return type automatically gets @builtin(position)
///     vec4f(0.0, 0.0, 0.0, 1.0)
/// }
///
/// #[compute]
/// #[workgroup_size(64)]
/// pub fn cs_main(#[builtin(global_invocation_id)] gid: Vec3u) {
///     // ...
/// }
/// ```
///
/// # Built-in Values
///
/// Use `#[builtin(...)]` on function parameters or struct fields to bind
/// to WGSL built-in values. Each emits `@builtin(name)` in the WGSL output.
///
/// | Built-in Name | Stage | Direction | Type |
/// |---------------|-------|-----------|------|
/// | `vertex_index` | vertex | input | `u32` |
/// | `instance_index` | vertex | input | `u32` |
/// | `position` | vertex / fragment | output / input | `Vec4f` |
/// | `front_facing` | fragment | input | `bool` |
/// | `frag_depth` | fragment | output | `f32` |
/// | `sample_index` | fragment | input | `u32` |
/// | `sample_mask` | fragment | input+output | `u32` |
/// | `local_invocation_id` | compute | input | `Vec3u` |
/// | `local_invocation_index` | compute | input | `u32` |
/// | `global_invocation_id` | compute | input | `Vec3u` |
/// | `workgroup_id` | compute | input | `Vec3u` |
/// | `num_workgroups` | compute | input | `Vec3u` |
/// | `primitive_index` | fragment | input | `u32` |
/// | `subgroup_invocation_id` | compute+fragment | input | `u32` |
/// | `subgroup_size` | compute+fragment | input | `u32` |
/// | `subgroup_id` | compute | input | `u32` |
/// | `num_subgroups` | compute | input | `u32` |
///
/// # Inter-Stage IO Attributes
///
/// These attributes can be placed on function parameters or struct fields
/// to control data flow between shader stages.
///
/// ## `#[location(N)]`
///
/// Assigns a user-defined IO location. Emits `@location(N)` in WGSL.
/// Location indices must be unique within a given set of inputs or outputs
/// but do not need to be contiguous or ordered.
///
/// ## `#[blend_src(N)]`
///
/// For dual-source blending. Emits `@blend_src(N)` in WGSL. Only valid
/// on output struct fields with `@location(0)`. Must have exactly two
/// entries: `@blend_src(0)` and `@blend_src(1)`.
///
/// ## `#[interpolate(type)]` / `#[interpolate(type, sampling)]`
///
/// Controls how values are interpolated between vertex and fragment stages.
/// Emits `@interpolate(type)` or `@interpolate(type, sampling)` in WGSL.
///
/// **Interpolation types:** `perspective`, `linear`, `flat`
///
/// **Sampling modes:** `center`, `centroid`, `sample`, `first`, `either`
///
/// ## `#[invariant]`
///
/// Marks a `@builtin(position)` output as invariant. Emits `@invariant`
/// in WGSL.
///
/// # IO Structs
///
/// Use `#[input]` and `#[output]` on structs to group inter-stage IO
/// fields. These attributes strip the IO annotations from the Rust output
/// (so the struct compiles as plain Rust) while preserving them in the
/// WGSL output.
///
/// ```ignore
/// #[output]
/// pub struct VertexOutput {
///     #[builtin(position)]
///     pub pos: Vec4f,
///     #[location(0)]
///     pub color: Vec4f,
///     #[location(1)]
///     #[interpolate(flat)]
///     pub material_id: u32,
/// }
/// ```
///
/// Transpiles to:
///
/// ```wgsl
/// struct VertexOutput {
///     @builtin(position) pos: vec4<f32>,
///     @location(0) color: vec4<f32>,
///     @location(1) @interpolate(flat) material_id: u32,
/// }
/// ```
///
/// # Resource Declarations
///
/// Declare GPU resources using macro invocations inside the module.
///
/// ## `uniform!`
///
/// ```ignore
/// uniform!(group(0), binding(0), MY_UNIFORM: MyStruct);
/// ```
///
/// Emits `@group(0) @binding(0) var<uniform> MY_UNIFORM: MyStruct;` in
/// WGSL. On the Rust side, generates a thread-safe static that can be
/// accessed with `get!` and `get_mut!`.
///
/// ## `storage!`
///
/// ```ignore
/// storage!(group(0), binding(1), MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_only, MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_write, MY_BUFFER: MyStruct);
/// ```
///
/// Emits `@group(G) @binding(B) var<storage, read>` or
/// `var<storage, read_write>` in WGSL. The default access mode is
/// `read_only`.
///
/// ## `sampler!`
///
/// ```ignore
/// sampler!(group(0), binding(2), TEX_SAMPLER: Sampler);
/// sampler!(group(0), binding(3), SHADOW_SAMPLER: SamplerComparison);
/// ```
///
/// Emits `@group(G) @binding(B) var NAME: sampler;` or
/// `var NAME: sampler_comparison;` in WGSL.
///
/// ## `texture!`
///
/// ```ignore
/// texture!(group(0), binding(4), DIFFUSE: Texture2D<f32>);
/// texture!(group(0), binding(5), SHADOW_MAP: TextureDepth2D);
/// ```
///
/// **Sampled textures** (with type parameter `f32`, `i32`, or `u32`):
/// `Texture1D`, `Texture2D`, `Texture2DArray`, `Texture3D`, `TextureCube`,
/// `TextureCubeArray`, `TextureMultisampled2D`.
///
/// **Depth textures** (no type parameter):
/// `TextureDepth2D`, `TextureDepth2DArray`, `TextureDepthCube`,
/// `TextureDepthCubeArray`, `TextureDepthMultisampled2D`.
///
/// ## `workgroup!`
///
/// ```ignore
/// workgroup!(SHARED_DATA: [f32; 64]);
/// workgroup!(COUNTER: Atomic<u32>);
/// ```
///
/// Emits `var<workgroup> NAME: TYPE;` in WGSL. Only usable in compute
/// shaders.
///
/// # Pointer Types
///
/// Use `ptr!(address_space, Type)` in function parameter types to declare
/// WGSL pointers. On the Rust side this expands to `&mut T`.
///
/// **Supported address spaces:** `function`, `private`, `workgroup`.
///
/// ```ignore
/// fn increment(p: ptr!(function, i32)) {
///     *p += 1;
/// }
/// ```
///
/// Emits `fn increment(p: ptr<function, i32>)` in WGSL.
///
/// # Enums
///
/// Enums must have `#[repr(u32)]` and become type aliases with constants
/// in WGSL.
///
/// ```ignore
/// #[repr(u32)]
/// pub enum Material {
///     Metal = 0,
///     Wood = 1,
///     Glass = 2,
/// }
/// ```
///
/// Transpiles to:
///
/// ```wgsl
/// alias Material = u32;
/// const Material_Metal: u32 = 0u;
/// const Material_Wood: u32 = 1u;
/// const Material_Glass: u32 = 2u;
/// ```
///
/// # Statement Macros
///
/// These macros are available inside function bodies within `#[wgsl]`
/// modules.
///
/// ## `get!` / `get_mut!`
///
/// Access module-level variables declared with `uniform!`, `storage!`, or
/// `workgroup!`. In Rust, these acquire a read or write lock on the
/// underlying thread-safe storage. In WGSL, they are stripped and the
/// variable name is emitted directly.
///
/// ```ignore
/// let value = get!(MY_UNIFORM).field;
/// get_mut!(MY_BUFFER).field = 42;
/// ```
///
/// ## `slab_read_array!` / `slab_write_array!`
///
/// Bulk read/write operations on storage buffer slabs.
///
/// ```ignore
/// slab_read_array!(get!(SLAB), offset, dest_array, count);
/// slab_write_array!(get_mut!(SLAB), offset, src_array, count);
/// ```
///
/// These expand to indexed for-loops in WGSL.
///
/// # Warning Suppression
///
/// Use `#[wgsl_allow(...)]` on loop or match expressions to suppress
/// specific transpiler warnings.
///
/// | Warning Name | Description |
/// |-------------|-------------|
/// | `non_literal_loop_bounds` | Allow `for i in 0..n` where `n` is not a literal. |
/// | `non_literal_match_statement_patterns` | Allow match arms with non-literal patterns. |
///
/// ```ignore
/// #[wgsl_allow(non_literal_loop_bounds)]
/// for i in 0..n {
///     // ...
/// }
/// ```
///
/// # Complete Example
///
/// ```ignore
/// use wgsl_rs::std::*;
///
/// #[wgsl]
/// pub mod my_shader {
///     use super::*;
///
///     pub struct Uniforms {
///         pub projection: Mat4x4f,
///         pub modelview: Mat4x4f,
///     }
///
///     uniform!(group(0), binding(0), UNIFORMS: Uniforms);
///
///     #[output]
///     pub struct VertexOutput {
///         #[builtin(position)]
///         pub clip_position: Vec4f,
///         #[location(0)]
///         pub color: Vec4f,
///     }
///
///     #[vertex]
///     pub fn vs_main(
///         #[builtin(vertex_index)] vertex_index: u32,
///         #[location(0)] position: Vec3f,
///         #[location(1)] color: Vec4f,
///     ) -> VertexOutput {
///         let u = get!(UNIFORMS);
///         let clip = u.projection * u.modelview * vec4f(position.x, position.y, position.z, 1.0);
///         VertexOutput {
///             clip_position: clip,
///             color,
///         }
///     }
///
///     #[fragment]
///     pub fn fs_main(
///         #[location(0)] color: Vec4f,
///     ) -> Vec4f {
///         color
///     }
/// }
/// ```
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

/// Marks a function as a vertex shader entry point.
///
/// Emits `@vertex` in WGSL. If the return type is `Vec4f`, the output
/// automatically gets `@builtin(position)`. Use `#[builtin(...)]` on
/// parameters to bind vertex-stage built-in inputs.
///
/// See [`wgsl`] for the full annotation reference.
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

/// Marks a function as a fragment shader entry point.
///
/// Emits `@fragment` in WGSL. If the return type is `Vec4f`, the output
/// automatically gets `@location(0)`. Use `#[builtin(...)]` on parameters
/// to bind fragment-stage built-in inputs (e.g., `position`, `front_facing`).
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn fragment(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Marks a function as a compute shader entry point.
///
/// Emits `@compute` in WGSL. Must be paired with `#[workgroup_size(...)]`.
/// Use `#[builtin(...)]` on parameters to bind compute-stage built-in
/// inputs (e.g., `global_invocation_id`, `local_invocation_id`).
///
/// See [`wgsl`] for the full annotation reference.
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

/// Specifies the workgroup dimensions for a `#[compute]` entry point.
///
/// Accepts 1, 2, or 3 integer literal arguments. Emits
/// `@workgroup_size(X)`, `@workgroup_size(X, Y)`, or
/// `@workgroup_size(X, Y, Z)` in WGSL.
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro_attribute]
pub fn workgroup_size(_attr: TokenStream, token_stream: TokenStream) -> TokenStream {
    token_stream
}

/// Binds a function parameter or struct field to a WGSL built-in value.
///
/// Emits `@builtin(name)` in WGSL. Stripped from the Rust output during
/// compilation. See [`wgsl`] for the full list of supported built-in names
/// and which shader stages they belong to.
///
/// ```ignore
/// #[vertex]
/// pub fn main(#[builtin(vertex_index)] vi: u32) -> Vec4f { /* ... */ }
/// ```
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

/// Declares a uniform buffer binding.
///
/// Emits `@group(G) @binding(B) var<uniform> NAME: TYPE;` in WGSL.
/// On the Rust side, generates a thread-safe static accessible via
/// `get!` and `get_mut!`.
///
/// ```ignore
/// uniform!(group(0), binding(0), MY_UNIFORM: MyStruct);
/// ```
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro]
pub fn uniform(input: TokenStream) -> TokenStream {
    uniform::uniform(input)
}

/// Declares a storage buffer binding.
///
/// Emits `@group(G) @binding(B) var<storage, read>` or
/// `var<storage, read_write>` in WGSL depending on the access mode.
/// The default is `read_only`.
///
/// ```ignore
/// storage!(group(0), binding(1), MY_BUFFER: MyStruct);
/// storage!(group(0), binding(1), read_write, MY_BUFFER: MyStruct);
/// ```
///
/// See [`wgsl`] for the full annotation reference.
#[proc_macro]
pub fn storage(input: TokenStream) -> TokenStream {
    storage::storage(input)
}

/// Defines a sampler or comparison sampler for texture sampling operations.
///
/// # Syntax
/// ```ignore
/// sampler!(group(G), binding(B), NAME: Sampler);
/// sampler!(group(G), binding(B), NAME: SamplerComparison);
/// ```
///
/// # Description
/// Samplers control how textures are sampled in shaders, including filtering
/// modes and address wrapping behavior. Comparison samplers are used for
/// depth texture sampling operations like shadow mapping.
///
/// # WGSL Output
/// The macro transpiles to:
/// - `@group(G) @binding(B) var NAME: sampler;` for regular samplers
/// - `@group(G) @binding(B) var NAME: sampler_comparison;` for comparison
///   samplers
///
/// # Rust Expansion
/// On the Rust side, the macro generates:
/// - A static `Sampler` or `SamplerComparison` instance
/// - A `SamplerDescriptor` constant for creating the sampler
/// - A convenience function to create the sampler
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// sampler!(group(0), binding(1), TEX_SAMPLER: Sampler);
/// sampler!(group(0), binding(2), SHADOW_SAMPLER: SamplerComparison);
///
/// #[fragment]
/// pub fn main() -> Vec4f {
///     // Use samplers for texture sampling...
///     Vec4f::ZERO
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// @group(0) @binding(1) var TEX_SAMPLER: sampler;
/// @group(0) @binding(2) var SHADOW_SAMPLER: sampler_comparison;
///
/// @fragment
/// fn main() -> vec4<f32> {
///     // Use samplers for texture sampling...
///     return vec4<f32>(0.0);
/// }
/// ```
#[proc_macro]
pub fn sampler(input: TokenStream) -> TokenStream {
    sampler::sampler(input)
}

/// Defines a texture or depth texture for sampling operations.
///
/// # Syntax
/// ```ignore
/// // Sampled textures (with type parameter: f32, i32, or u32)
/// texture!(group(G), binding(B), NAME: Texture2D<f32>);
/// texture!(group(G), binding(B), NAME: TextureCube<i32>);
///
/// // Depth textures (no type parameter)
/// texture!(group(G), binding(B), NAME: TextureDepth2D);
/// texture!(group(G), binding(B), NAME: TextureDepthCube);
/// ```
///
/// # Supported Texture Types
///
/// ## Sampled Textures
/// - `Texture1D<T>` - 1D texture
/// - `Texture2D<T>` - 2D texture
/// - `Texture2DArray<T>` - 2D texture array
/// - `Texture3D<T>` - 3D texture
/// - `TextureCube<T>` - Cube texture
/// - `TextureCubeArray<T>` - Cube texture array
/// - `TextureMultisampled2D<T>` - Multisampled 2D texture
///
/// Where `T` is one of `f32`, `i32`, or `u32`.
///
/// ## Depth Textures
/// - `TextureDepth2D` - 2D depth texture
/// - `TextureDepth2DArray` - 2D depth texture array
/// - `TextureDepthCube` - Cube depth texture
/// - `TextureDepthCubeArray` - Cube depth texture array
/// - `TextureDepthMultisampled2D` - Multisampled 2D depth texture
///
/// # WGSL Output
/// The macro transpiles to:
/// - `@group(G) @binding(B) var NAME: texture_2d<f32>;` for sampled textures
/// - `@group(G) @binding(B) var NAME: texture_depth_2d;` for depth textures
///
/// # Rust Expansion
/// On the Rust side, the macro generates:
/// - A static texture handle instance
/// - A `TextureViewDescriptor` constant for creating views
/// - A convenience function to create a texture view
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// texture!(group(0), binding(0), DIFFUSE_TEX: Texture2D<f32>);
/// texture!(group(0), binding(1), SHADOW_MAP: TextureDepth2D);
/// sampler!(group(0), binding(2), TEX_SAMPLER: Sampler);
///
/// #[fragment]
/// pub fn main() -> Vec4f {
///     // Sample the diffuse texture...
///     Vec4f::ZERO
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// @group(0) @binding(0) var DIFFUSE_TEX: texture_2d<f32>;
/// @group(0) @binding(1) var SHADOW_MAP: texture_depth_2d;
/// @group(0) @binding(2) var TEX_SAMPLER: sampler;
///
/// @fragment
/// fn main() -> vec4<f32> {
///     // Sample the diffuse texture...
///     return vec4<f32>(0.0);
/// }
/// ```
#[proc_macro]
pub fn texture(input: TokenStream) -> TokenStream {
    texture::texture(input)
}

/// Defines a workgroup-scoped variable shared between invocations in a compute
/// shader workgroup.
///
/// # Syntax
/// ```ignore
/// workgroup!(NAME: TYPE);
/// ```
///
/// # Description
/// Workgroup variables are shared between all invocations in a compute shader
/// workgroup. They can only be used in compute shaders and are useful for
/// inter-invocation communication and shared temporary storage.
///
/// # WGSL Output
/// The macro transpiles to `var<workgroup> NAME: TYPE;` in WGSL.
///
/// # Rust Expansion
/// On the Rust side, the variable is backed by a thread-safe `RwLock` to
/// simulate the shared nature of workgroup memory.
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// workgroup!(SHARED_COUNTER: Atomic<u32>);
/// workgroup!(TEMP_DATA: [f32; 64]);
///
/// #[compute]
/// #[workgroup_size(64)]
/// pub fn main(#[builtin(local_invocation_id)] local_id: Vec3u) {
///     // Access workgroup variables...
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// var<workgroup> SHARED_COUNTER: atomic<u32>;
/// var<workgroup> TEMP_DATA: array<f32, 64>;
///
/// @compute @workgroup_size(64)
/// fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
///     // Access workgroup variables...
/// }
/// ```
#[proc_macro]
pub fn workgroup(input: TokenStream) -> TokenStream {
    workgroup::workgroup(input)
}

/// Defines a WGSL pointer type for function parameters.
///
/// # Syntax
/// ```ignore
/// ptr!(address_space, Type)
/// ```
///
/// # Supported Address Spaces
/// - `function` - For pointers to local function variables
/// - `private` - For pointers to module-scope private variables
///
/// # Rust Expansion
/// The macro expands to `&mut T` in Rust, allowing the code to compile and run
/// on the CPU with mutable reference semantics.
///
/// # WGSL Output
/// During transpilation, `ptr!(function, T)` becomes `ptr<function, T>` in
/// WGSL. The access mode is always `read_write` (the only mode supported by
/// `function` and `private` address spaces) and is not written in the output.
///
/// # Example
/// ```ignore
/// use wgsl_rs::std::*;
///
/// fn increment(p: ptr!(function, i32)) {
///     *p += 1;
/// }
///
/// fn test() {
///     let mut x: i32 = 5;
///     increment(&mut x);
///     // x is now 6
/// }
/// ```
///
/// This transpiles to:
/// ```wgsl
/// fn increment(p: ptr<function, i32>) {
///     *p += 1;
/// }
/// ```
#[proc_macro]
pub fn ptr(input: TokenStream) -> TokenStream {
    ptr::ptr(input)
}

/// Marks a struct as a shader input, preserving IO attributes in WGSL.
///
/// Fields may carry `#[builtin(...)]`, `#[location(N)]`, and
/// `#[interpolate(...)]` attributes. These are emitted in the WGSL output
/// and stripped from the Rust output so the struct compiles normally.
///
/// See [`wgsl`] for the full annotation reference.
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

/// Marks a struct as a shader output, preserving IO attributes in WGSL.
///
/// Fields may carry `#[builtin(...)]`, `#[location(N)]`,
/// `#[interpolate(...)]`, `#[blend_src(N)]`, and `#[invariant]`
/// attributes. These are emitted in the WGSL output and stripped from
/// the Rust output so the struct compiles normally.
///
/// See [`wgsl`] for the full annotation reference.
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
