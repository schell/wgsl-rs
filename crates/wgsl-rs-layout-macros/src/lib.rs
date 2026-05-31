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

    // Build inherent consts to hold offset/size/align for each field.
    // These are accessible from both trait impls via Self::.
    let mut inherent_consts = Vec::new();
    for (i, ty) in field_types.iter().enumerate() {
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

    // Build ALIGN: max of all field alignments
    let self_align = build_max_align_from_consts(field_count);

    // SIZE = roundUp(AlignOf(S), lastOffset + lastSize)
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

    // Build FIELDS array entries
    let mut field_entry_tokens = Vec::new();
    let mut field_write_tokens = Vec::new();
    let mut field_read_tokens = Vec::new();
    let mut field_init_tokens = Vec::new();
    for (i, name) in field_names.iter().enumerate() {
        let offset_name =
            syn::Ident::new(&format!("__OFFSET_{}", i), proc_macro2::Span::call_site());
        let size_name = syn::Ident::new(&format!("__SIZE_{}", i), proc_macro2::Span::call_site());
        let align_name = syn::Ident::new(&format!("__ALIGN_{}", i), proc_macro2::Span::call_site());
        let field_ident = syn::Ident::new(name, proc_macro2::Span::call_site());

        let pad_after = if i == field_count - 1 {
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

    let layout_impl = quote! {
        impl #impl_generics ::wgsl_rs_layout::Layout for #struct_name #ty_generics #where_clause {
            const FIELDS: &'static [::wgsl_rs_layout::FieldLayout] = &[
                #(#field_entry_tokens),*
            ];
        }
    };

    Ok(quote! {
        #inherent_impl
        #wgsl_layout_impl
        #layout_impl
    })
}

/// Build a const expression computing max of all field alignments,
/// referencing inherent consts __ALIGN_0..__ALIGN_N.
fn build_max_align_from_consts(field_count: usize) -> proc_macro2::TokenStream {
    if field_count == 0 {
        return quote! { 1usize };
    }
    let a0 = syn::Ident::new("__ALIGN_0", proc_macro2::Span::call_site());
    if field_count == 1 {
        return quote! { Self::#a0 };
    }
    let mut max_expr = quote! { Self::#a0 };
    for i in 1..field_count {
        let a = syn::Ident::new(&format!("__ALIGN_{}", i), proc_macro2::Span::call_site());
        max_expr = quote! { (if (#max_expr > Self::#a) { #max_expr } else { Self::#a }) };
    }
    max_expr
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
