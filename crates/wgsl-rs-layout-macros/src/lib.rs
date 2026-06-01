//! `#[derive(Layout)]` proc-macro for WGSL struct layout computation.
//!
//! Generates `WgslLayout` and `Layout` impls for structs, computing field
//! offsets, sizes, and alignments per the WGSL specification §14.4.1.
//!
//! Supports both concrete and generic structs. Generic type parameters
//! receive a `T: WgslLayout` bound on the generated impl.
//!
//! # Errors
//!
//! This derive requires a named-field struct (`struct Foo { x: f32 }`).
//! Unit structs, tuple structs, and enums are rejected with compile errors.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type};

fn derive_layout(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_name = input.ident;
    let struct_name_span = struct_name.span();
    let generics = input.generics;

    let fields_data = match &input.data {
        Data::Struct(s) => &s.fields,
        Data::Enum(_) => {
            return Err(syn::Error::new(
                struct_name_span,
                "#[derive(Layout)] is only supported on structs, not enums",
            ));
        }
        Data::Union(_) => {
            return Err(syn::Error::new(
                struct_name_span,
                "#[derive(Layout)] is only supported on structs, not unions",
            ));
        }
    };

    let named_fields = match fields_data {
        Fields::Named(named) => named,
        Fields::Unnamed(_) | Fields::Unit => {
            return Err(syn::Error::new(
                struct_name_span,
                "#[derive(Layout)] is only supported on structs with named fields",
            ));
        }
    };

    let field_count = named_fields.named.len();

    if field_count == 0 {
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let wgsl_layout_impl = quote! {
            impl #impl_generics ::wgsl_rs_layout::WgslLayout for #struct_name #ty_generics #where_clause {
                const SIZE: usize = 0;
                const ALIGN: usize = 1;

                fn layout_read_bytes(_buf: &[u8]) -> ::std::result::Result<Self, ::wgsl_rs_layout::Error> {
                    ::std::result::Result::Ok(#struct_name {})
                }

                fn layout_write_bytes(&self, _buf: &mut [u8]) -> ::std::result::Result<(), ::wgsl_rs_layout::Error> {
                    ::std::result::Result::Ok(())
                }
            }
        };

        let layout_impl = quote! {
            impl #impl_generics ::wgsl_rs_layout::Layout for #struct_name #ty_generics #where_clause {
                const FIELDS: &'static [::wgsl_rs_layout::FieldLayout] = &[];
            }
        };

        return Ok(quote! {
            #wgsl_layout_impl
            #layout_impl
        });
    }

    let trait_bound: syn::TypeParamBound = syn::parse_quote!(::wgsl_rs_layout::WgslLayout);

    let expanded_generics = add_trait_bound(&generics, &trait_bound);
    let (impl_generics, ty_generics, where_clause) = expanded_generics.split_for_impl();

    let field_names: Vec<String> = named_fields
        .named
        .iter()
        .map(|f| f.ident.as_ref().expect("named field has ident").to_string())
        .collect();
    let field_types: Vec<&Type> = named_fields.named.iter().map(|f| &f.ty).collect();

    let layout_trait_path: syn::Path = syn::parse_quote!(::wgsl_rs_layout::WgslLayout);
    let round_up_path: syn::Path = syn::parse_quote!(::wgsl_rs_layout::round_up);

    let size_of = |ty: &Type| -> proc_macro2::TokenStream {
        quote! { <#ty as #layout_trait_path>::SIZE }
    };
    let align_of = |ty: &Type| -> proc_macro2::TokenStream {
        quote! { <#ty as #layout_trait_path>::ALIGN }
    };

    // Detect runtime array fields. Per the WGSL spec §6.2.10, a
    // runtime-sized array may only appear as the last member of a
    // struct. We detect by inspecting the type's last path segment
    // and a single generic argument.
    //
    // The heuristic is: the type is a path that ends in
    // `RuntimeArray` and has exactly one generic type argument.
    let runtime_array_indices: Vec<usize> = field_types
        .iter()
        .enumerate()
        .filter_map(|(i, ty)| if is_runtime_array(ty) { Some(i) } else { None })
        .collect();

    // Reject any runtime array that isn't the last field.
    for &i in &runtime_array_indices {
        if i != field_count - 1 {
            return Err(syn::Error::new_spanned(
                &named_fields.named[i],
                "runtime-sized array must be the last member of the struct (WGSL spec §6.2.10)",
            ));
        }
    }

    let has_runtime_array = !runtime_array_indices.is_empty();
    // If the last field is a runtime array, we exclude it from the
    // per-field FIELDS array and treat the remaining fields as the
    // struct's "prefix".
    let prefix_count = if has_runtime_array {
        field_count - 1
    } else {
        field_count
    };

    // Build inherent consts to hold offset/size/align for each
    // *prefix* field. The runtime array (if any) is handled
    // separately.
    let mut inherent_consts = Vec::new();
    for (i, ty) in field_types.iter().enumerate().take(prefix_count) {
        let offset_name =
            syn::Ident::new(&format!("__OFFSET_{}", i), proc_macro2::Span::call_site());
        let size_name = syn::Ident::new(&format!("__SIZE_{}", i), proc_macro2::Span::call_site());
        let align_name = syn::Ident::new(&format!("__ALIGN_{}", i), proc_macro2::Span::call_site());

        let size = size_of(ty);
        let align = align_of(ty);

        inherent_consts.push(quote! {
            const #size_name: usize = #size;
            const #align_name: usize = #align;
        });

        if i == 0 {
            inherent_consts.push(quote! {
                const #offset_name: usize = 0;
            });
        } else {
            let prev_offset = syn::Ident::new(
                &format!("__OFFSET_{}", i - 1),
                proc_macro2::Span::call_site(),
            );
            let prev_size = size_of(field_types[i - 1]);
            let cur_align = align_of(ty);
            inherent_consts.push(quote! {
                const #offset_name: usize = #round_up_path(
                    Self::#prev_offset + #prev_size,
                    #cur_align,
                );
            });
        }
    }

    // If there's a runtime array, emit its offset and stride consts.
    if has_runtime_array {
        let rt_i = field_count - 1;
        let rt_offset_name = syn::Ident::new(
            &format!("__OFFSET_{}", rt_i),
            proc_macro2::Span::call_site(),
        );
        let rt_size_name =
            syn::Ident::new(&format!("__SIZE_{}", rt_i), proc_macro2::Span::call_site());
        let rt_align_name =
            syn::Ident::new(&format!("__ALIGN_{}", rt_i), proc_macro2::Span::call_site());
        let rt_type = field_types[rt_i];
        let elem_type = runtime_array_elem(rt_type)
            .expect("detected as runtime array but couldn't extract element type");

        // The runtime array sits at the next aligned offset after the
        // last prefix field.
        let prev_offset = if prefix_count > 0 {
            let name = syn::Ident::new(
                &format!("__OFFSET_{}", prefix_count - 1),
                proc_macro2::Span::call_site(),
            );
            quote! { Self::#name }
        } else {
            quote! { 0 }
        };
        let prev_size = if prefix_count > 0 {
            size_of(field_types[prefix_count - 1])
        } else {
            quote! { 0 }
        };
        let cur_align = align_of(rt_type);

        inherent_consts.push(quote! {
            const #rt_size_name: usize = 0;
            const #rt_align_name: usize = #cur_align;
            const #rt_offset_name: usize = #round_up_path(
                #prev_offset + #prev_size,
                #cur_align,
            );
            const __RUNTIME_ARRAY_STRIDE: usize = #round_up_path(
                <#elem_type as #layout_trait_path>::SIZE,
                <#elem_type as #layout_trait_path>::ALIGN,
            );
        });
    }

    // Build ALIGN: max of all *prefix* field alignments. If there's a
    // runtime array, its alignment is already included since the
    // prefix's `__ALIGN_N-1` is the last prefix field and the runtime
    // array's `__ALIGN_N` is in inherent_consts above.
    let self_align = build_max_align_from_consts(field_count, has_runtime_array);

    // SIZE = roundUp(AlignOf(S), lastOffset + lastSize). For a struct
    // with a runtime array, `lastSize` is 0 (the array's SIZE), so
    // SIZE = roundUp(AlignOf(S), runtime_array_offset) — i.e. the
    // size with N=0 elements.
    let last_i = field_count - 1;
    let last_offset = syn::Ident::new(
        &format!("__OFFSET_{}", last_i),
        proc_macro2::Span::call_site(),
    );
    let last_size = syn::Ident::new(
        &format!("__SIZE_{}", last_i),
        proc_macro2::Span::call_site(),
    );

    let size_expr = quote! {
        #round_up_path(Self::#last_offset + Self::#last_size, Self::ALIGN)
    };

    // Build FIELDS array entries. Skip the runtime array field (if
    // any) so it's not represented as a 0-byte data field.
    let mut field_entry_tokens = Vec::new();
    let mut field_write_tokens = Vec::new();
    let mut field_read_tokens = Vec::new();
    let mut field_init_tokens = Vec::new();
    for (i, name) in field_names.iter().enumerate().take(prefix_count) {
        let offset_name =
            syn::Ident::new(&format!("__OFFSET_{}", i), proc_macro2::Span::call_site());
        let size_name = syn::Ident::new(&format!("__SIZE_{}", i), proc_macro2::Span::call_site());
        let align_name = syn::Ident::new(&format!("__ALIGN_{}", i), proc_macro2::Span::call_site());
        let field_ident = syn::Ident::new(name, proc_macro2::Span::call_site());

        let pad_after = if i == field_count - 1 {
            // For a non-runtime-array struct, pad_after for the last
            // field is the trailing struct padding.
            quote! { Self::SIZE - (Self::#offset_name + Self::#size_name) }
        } else {
            let next_offset_name = syn::Ident::new(
                &format!("__OFFSET_{}", i + 1),
                proc_macro2::Span::call_site(),
            );
            quote! { Self::#next_offset_name - (Self::#offset_name + Self::#size_name) }
        };

        field_entry_tokens.push(quote! {
            ::wgsl_rs_layout::FieldLayout {
                name: #name,
                offset: Self::#offset_name,
                size: Self::#size_name,
                alignment: Self::#align_name,
                pad_after: #pad_after,
            }
        });

        field_write_tokens.push(quote! {
            ::wgsl_rs_layout::WgslLayout::layout_write_bytes(
                &self.#field_ident,
                &mut buf[Self::#offset_name..],
            )?;
            let _pad_start = Self::#offset_name + Self::#size_name;
            let _pad_end = _pad_start + #pad_after;
            ::wgsl_rs_layout::zero_buffer(&mut buf[_pad_start.._pad_end], #pad_after);
        });

        field_read_tokens.push({
            let ty = field_types[i]; // Type is Clone
            quote! {
                let #field_ident = <#ty as ::wgsl_rs_layout::WgslLayout>::layout_read_bytes(
                    &buf[Self::#offset_name..],
                )?;
            }
        });

        field_init_tokens.push(field_ident);
    }

    // If there's a runtime array, also emit a read/write/init for it
    // using the existing `WgslLayout` impl on `RuntimeArray<T>`. The
    // buffer for read/write must be sized appropriately by the
    // caller, so this is best-effort: the read/write paths use the
    // struct's `SIZE` (which is the prefix size with N=0) for the
    // check, and skip the runtime-array portion.
    if has_runtime_array {
        let rt_i = field_count - 1;
        let field_ident = syn::Ident::new(&field_names[rt_i], proc_macro2::Span::call_site());
        let rt_offset = syn::Ident::new(
            &format!("__OFFSET_{}", rt_i),
            proc_macro2::Span::call_site(),
        );
        let rt_type = field_types[rt_i];

        // The runtime array's `layout_read_bytes` reads as many
        // elements as fit in the remaining buffer; we don't need to
        // verify that the buffer extends beyond `SIZE` (which is the
        // prefix size).
        field_read_tokens.push(quote! {
            let #field_ident = <#rt_type as ::wgsl_rs_layout::WgslLayout>::layout_read_bytes(
                &buf[Self::#rt_offset..],
            )?;
        });

        // Writing the runtime array uses the buffer beyond `SIZE`.
        // The array's `layout_write_bytes` iterates the elements and
        // writes each element's bytes. The caller is responsible for
        // sizing `buf` to hold the prefix + the array's element bytes.
        field_write_tokens.push(quote! {
            ::wgsl_rs_layout::WgslLayout::layout_write_bytes(
                &self.#field_ident,
                &mut buf[Self::#rt_offset..],
            )?;
        });

        field_init_tokens.push(field_ident);
    }

    // Inherent impl to hold the helper consts (accessible from trait impls via
    // Self::)
    let inherent_impl = quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#inherent_consts)*
        }
    };

    let wgsl_layout_impl = quote! {
        impl #impl_generics ::wgsl_rs_layout::WgslLayout for #struct_name #ty_generics #where_clause {
            const SIZE: usize = #size_expr;
            const ALIGN: usize = #self_align;

            fn layout_read_bytes(buf: &[u8]) -> ::std::result::Result<Self, ::wgsl_rs_layout::Error> {
                if buf.len() < Self::SIZE {
                    return ::std::result::Result::Err(::wgsl_rs_layout::Error::BufferTooSmall {
                        needed: Self::SIZE,
                        actual: buf.len(),
                    });
                }
                #(#field_read_tokens)*
                ::std::result::Result::Ok(#struct_name {
                    #(#field_init_tokens),*
                })
            }

            fn layout_write_bytes(&self, buf: &mut [u8]) -> ::std::result::Result<(), ::wgsl_rs_layout::Error> {
                if buf.len() < Self::SIZE {
                    return ::std::result::Result::Err(::wgsl_rs_layout::Error::BufferTooSmall {
                        needed: Self::SIZE,
                        actual: buf.len(),
                    });
                }
                #(#field_write_tokens)*
                ::std::result::Result::Ok(())
            }
        }
    };

    let runtime_array_impls = if has_runtime_array {
        let rt_i = field_count - 1;
        let rt_field_name = &field_names[rt_i];
        let rt_offset_name = syn::Ident::new(
            &format!("__OFFSET_{}", rt_i),
            proc_macro2::Span::call_site(),
        );
        let rt_align_name =
            syn::Ident::new(&format!("__ALIGN_{}", rt_i), proc_macro2::Span::call_site());
        quote! {
            const RUNTIME_ARRAY_STRIDE: Option<usize> = Some(Self::__RUNTIME_ARRAY_STRIDE);
            const RUNTIME_ARRAY_FIELD: ::std::option::Option<::wgsl_rs_layout::FieldLayout> =
                ::std::option::Option::Some(::wgsl_rs_layout::FieldLayout {
                    name: #rt_field_name,
                    offset: Self::#rt_offset_name,
                    size: 0,
                    alignment: Self::#rt_align_name,
                    pad_after: Self::__RUNTIME_ARRAY_STRIDE,
                });
        }
    } else {
        quote! {
            const RUNTIME_ARRAY_STRIDE: Option<usize> = ::std::option::Option::None;
            const RUNTIME_ARRAY_FIELD: ::std::option::Option<::wgsl_rs_layout::FieldLayout> =
                ::std::option::Option::None;
        }
    };

    let layout_impl = quote! {
        impl #impl_generics ::wgsl_rs_layout::Layout for #struct_name #ty_generics #where_clause {
            const FIELDS: &'static [::wgsl_rs_layout::FieldLayout] = &[
                #(#field_entry_tokens),*
            ];
            #runtime_array_impls
        }
    };

    Ok(quote! {
        #inherent_impl
        #wgsl_layout_impl
        #layout_impl
    })
}

/// Heuristic check for "this type is `RuntimeArray<...>`". Matches
/// any path whose last segment is `RuntimeArray` and has exactly one
/// generic type argument.
fn is_runtime_array(ty: &Type) -> bool {
    runtime_array_elem(ty).is_some()
}

/// If `ty` looks like a runtime array (`Path` ending in
/// `RuntimeArray<...>` with one type arg), return the inner element
/// type.
fn runtime_array_elem(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last_seg = type_path.path.segments.last()?;
    if last_seg.ident != "RuntimeArray" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    let arg = args.args.first()?;
    if let syn::GenericArgument::Type(elem_ty) = arg {
        Some(elem_ty)
    } else {
        None
    }
}

/// Build a const expression computing max of all field alignments,
/// referencing inherent consts __ALIGN_0..__ALIGN_N.
fn build_max_align_from_consts(
    field_count: usize,
    _has_runtime_array: bool,
) -> proc_macro2::TokenStream {
    if field_count == 0 {
        return quote! { 1usize };
    }
    let a0 = syn::Ident::new("__ALIGN_0", proc_macro2::Span::call_site());
    if field_count == 1 {
        return quote! { Self::#a0 };
    }
    let mut max_exprs = vec![];
    for i in 1..field_count {
        let a = syn::Ident::new(&format!("__ALIGN_{}", i), proc_macro2::Span::call_site());
        max_exprs.push(a);
    }
    quote! {{
        let mut max = Self::#a0;
        #(if Self::#max_exprs > max { max = Self::#max_exprs; })*
        max
    }}
}

fn add_trait_bound(generics: &syn::Generics, bound: &syn::TypeParamBound) -> syn::Generics {
    let mut g = generics.clone();
    for param in g.type_params_mut() {
        param.bounds.push(bound.clone());
    }
    g
}

#[proc_macro_derive(Layout)]
pub fn layout_derive(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    derive_layout(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
