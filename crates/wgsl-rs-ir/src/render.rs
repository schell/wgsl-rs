//! Render IR to WGSL source text.
//!
//! This module is the IR-side counterpart of the proc-macro's
//! `code_gen/formatter.rs`. It walks an [`crate::Module`] (or any
//! sub-tree) and produces a pretty-printed WGSL string with 4-space
//! indentation.
//!
//! The renderer is responsible for all WGSL-specific lowering that the IR
//! intentionally leaves un-baked:
//!
//! * Builtin Rust function names are translated to WGSL camelCase (see
//!   [`builtin_lookup`]).
//! * `Expr::Struct` field names are dropped and emitted positionally.
//! * `Impl` blocks have their methods / constants name-mangled to
//!   `StructName_member`.
//! * Enum discriminants auto-increment starting from 0.
//! * `let mut` (`Local::mutable == true`) renders as `var`; `let` renders as
//!   `let`.
//! * Vector / matrix constructors lower-case their type names.
//! * Integer suffix translation (`u32` / `usize` → `u`, `i32` / `isize` → `i`).

use crate::*;

mod builtin_lookup;

/// Render a [`Module`] to a complete WGSL source string.
pub fn render_module(module: &Module) -> String {
    let mut w = Writer::new();
    for item in &module.items {
        write_item(&mut w, item);
    }
    w.finish()
}

/// Render a slice of [`Item`]s (e.g. an instantiated generic template) to
/// a WGSL source string. No leading or trailing blank lines are inserted
/// beyond what each item produces.
pub fn render_items(items: &[Item]) -> String {
    let mut w = Writer::new();
    for item in items {
        write_item(&mut w, item);
    }
    w.finish()
}

// ===== Writer =====

struct Writer {
    /// Already-finalized lines.
    lines: Vec<String>,
    /// The current in-progress line (no trailing newline).
    cur: String,
    /// Current indentation level (in 4-space steps).
    indent: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            cur: String::new(),
            indent: 0,
        }
    }

    /// Finish, returning the joined source. Joins with `\n` and ensures a
    /// final newline.
    fn finish(mut self) -> String {
        if !self.cur.is_empty() {
            self.lines.push(std::mem::take(&mut self.cur));
        }
        // Drop trailing empty lines.
        while self.lines.last().map(|s| s.is_empty()).unwrap_or(false) {
            self.lines.pop();
        }
        let mut out = self.lines.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }

    /// Push the current line into [`lines`] and start a new one.
    fn newline(&mut self) {
        // If we're starting a fresh line and the current line is empty, we
        // still emit one (this is how blank separator lines are produced).
        let line = std::mem::take(&mut self.cur);
        self.lines.push(line);
    }

    /// Ensure that the next write begins on a fresh, properly-indented
    /// line.
    fn start_line(&mut self) {
        if !self.cur.is_empty() {
            self.newline();
        }
        for _ in 0..self.indent {
            self.cur.push_str("    ");
        }
    }

    /// Ensure there is at least one fully-blank line between the previous
    /// content and the next item to be written.
    fn blank_line(&mut self) {
        if !self.cur.is_empty() {
            self.newline();
        }
        // If the last line isn't already blank and we have prior output,
        // add one blank line.
        match self.lines.last() {
            None => {}
            Some(last) if last.is_empty() => {}
            _ => self.lines.push(String::new()),
        }
    }

    fn write(&mut self, s: &str) {
        self.cur.push_str(s);
    }

    fn space(&mut self) {
        self.cur.push(' ');
    }
}

// ===== Items =====

fn write_item(w: &mut Writer, item: &Item) {
    match item {
        Item::Const(c) => {
            w.start_line();
            write_const(w, c);
            w.newline();
        }
        Item::Uniform(u) => {
            w.start_line();
            w.write(&format!("@group({}) @binding({}) ", u.group, u.binding));
            w.write("var<uniform> ");
            w.write(&u.name);
            w.write(": ");
            write_type(w, &u.ty);
            w.write(";");
            w.newline();
        }
        Item::Storage(s) => {
            w.start_line();
            w.write(&format!("@group({}) @binding({}) ", s.group, s.binding));
            let acc = match s.access {
                StorageAccess::Read => "read",
                StorageAccess::ReadWrite => "read_write",
            };
            w.write(&format!("var<storage, {acc}> "));
            w.write(&s.name);
            w.write(": ");
            write_type(w, &s.ty);
            w.write(";");
            w.newline();
        }
        Item::Workgroup(wg) => {
            w.start_line();
            w.write("var<workgroup> ");
            w.write(&wg.name);
            w.write(": ");
            write_type(w, &wg.ty);
            w.write(";");
            w.newline();
        }
        Item::Sampler(s) => {
            w.start_line();
            w.write(&format!("@group({}) @binding({}) ", s.group, s.binding));
            w.write("var ");
            w.write(&s.name);
            w.write(": ");
            write_type(w, &s.ty);
            w.write(";");
            w.newline();
        }
        Item::Texture(t) => {
            w.start_line();
            w.write(&format!("@group({}) @binding({}) ", t.group, t.binding));
            w.write("var ");
            w.write(&t.name);
            w.write(": ");
            write_type(w, &t.ty);
            w.write(";");
            w.newline();
        }
        Item::Fn(f) => {
            w.blank_line();
            write_fn(w, f, None);
        }
        Item::Struct(s) => {
            w.blank_line();
            write_struct(w, s);
        }
        Item::Impl(i) => {
            for ii in &i.items {
                match ii {
                    ImplItem::Const(c) => {
                        let mangled = format!("{}_{}", i.self_ty, c.name);
                        let mc = ItemConst {
                            name: mangled,
                            ty: c.ty.clone(),
                            expr: c.expr.clone(),
                        };
                        w.start_line();
                        write_const(w, &mc);
                        w.newline();
                    }
                    ImplItem::Fn(f) => {
                        let mangled = format!("{}_{}", i.self_ty, f.name);
                        w.blank_line();
                        write_fn(w, f, Some(&mangled));
                    }
                }
            }
        }
        Item::Enum(e) => {
            w.start_line();
            w.write("alias ");
            w.write(&e.name);
            w.write(" = u32;");
            w.newline();
            let mut next: u32 = 0;
            for v in &e.variants {
                let value = if let Some(d) = v.discriminant {
                    next = d;
                    d
                } else {
                    next
                };
                w.start_line();
                w.write("const ");
                w.write(&e.name);
                w.write("_");
                w.write(&v.name);
                w.write(": u32 = ");
                w.write(&format!("{value}u"));
                w.write(";");
                w.newline();
                next = next.wrapping_add(1);
            }
        }
    }
}

fn write_const(w: &mut Writer, c: &ItemConst) {
    w.write("const ");
    w.write(&c.name);
    w.write(": ");
    write_type(w, &c.ty);
    w.write(" = ");
    write_expr(w, &c.expr);
    w.write(";");
}

fn write_struct(w: &mut Writer, s: &ItemStruct) {
    w.start_line();
    w.write("struct ");
    w.write(&s.name);
    w.write(" {");
    w.newline();
    w.indent += 1;
    let n = s.fields.len();
    for (i, f) in s.fields.iter().enumerate() {
        w.start_line();
        for io in &f.inter_stage_io {
            write_inter_stage_io(w, io);
            w.space();
        }
        w.write(&f.name);
        w.write(": ");
        write_type(w, &f.ty);
        if i + 1 < n {
            w.write(",");
        }
        w.newline();
    }
    w.indent -= 1;
    w.start_line();
    w.write("}");
    w.newline();
}

fn write_fn(w: &mut Writer, f: &ItemFn, override_name: Option<&str>) {
    w.start_line();
    match &f.fn_attrs {
        FnAttrs::None => {}
        FnAttrs::Vertex => {
            w.write("@vertex ");
        }
        FnAttrs::Fragment => {
            w.write("@fragment ");
        }
        FnAttrs::Compute { workgroup_size } => {
            w.write("@compute ");
            w.write("@workgroup_size(");
            w.write(&workgroup_size.x.to_string());
            if let Some(y) = workgroup_size.y {
                w.write(", ");
                w.write(&y.to_string());
            }
            if let Some(z) = workgroup_size.z {
                w.write(", ");
                w.write(&z.to_string());
            }
            w.write(") ");
        }
    }
    w.write("fn ");
    w.write(override_name.unwrap_or(&f.name));
    w.write("(");
    let n = f.inputs.len();
    if n > 4 {
        w.newline();
        w.indent += 1;
        for arg in f.inputs.iter() {
            w.start_line();
            for io in &arg.inter_stage_io {
                write_inter_stage_io(w, io);
                w.space();
            }
            w.write(&arg.name);
            w.write(": ");
            write_type(w, &arg.ty);
            // Always emit a trailing comma when arguments are spread
            // across multiple lines.
            w.write(",");
            w.newline();
        }
        w.indent -= 1;
        w.start_line();
        let _ = n;
    } else {
        for (i, arg) in f.inputs.iter().enumerate() {
            for io in &arg.inter_stage_io {
                write_inter_stage_io(w, io);
                w.space();
            }
            w.write(&arg.name);
            w.write(": ");
            write_type(w, &arg.ty);
            if i + 1 < n {
                w.write(", ");
            }
        }
    }
    w.write(")");
    match &f.return_type {
        ReturnType::Default => {}
        ReturnType::Type { annotation, ty } => {
            w.write(" -> ");
            write_return_annotation(w, annotation);
            write_type(w, ty);
        }
    }
    w.space();
    write_block(w, &f.block);
    w.newline();
}

fn write_return_annotation(w: &mut Writer, ann: &ReturnTypeAnnotation) {
    match ann {
        ReturnTypeAnnotation::None => {}
        ReturnTypeAnnotation::BuiltIn(b) => {
            w.write("@builtin(");
            w.write(builtin_name(*b));
            w.write(") ");
        }
        ReturnTypeAnnotation::Location(n) => {
            w.write(&format!("@location({n}) "));
        }
        ReturnTypeAnnotation::DefaultBuiltInPosition => {
            w.write("@builtin(position) ");
        }
        ReturnTypeAnnotation::DefaultLocation => {
            w.write("@location(0) ");
        }
    }
}

fn write_inter_stage_io(w: &mut Writer, io: &InterStageIo) {
    match io {
        InterStageIo::BuiltIn(b) => {
            w.write("@builtin(");
            w.write(builtin_name(*b));
            w.write(")");
        }
        InterStageIo::Location(n) => {
            w.write(&format!("@location({n})"));
        }
        InterStageIo::BlendSrc(n) => {
            w.write(&format!("@blend_src({n})"));
        }
        InterStageIo::Interpolate(i) => {
            w.write("@interpolate(");
            w.write(interp_type_name(i.ty));
            if let Some(s) = i.sampling {
                w.write(", ");
                w.write(interp_sampling_name(s));
            }
            w.write(")");
        }
        InterStageIo::Invariant => {
            w.write("@invariant");
        }
    }
}

fn builtin_name(b: BuiltIn) -> &'static str {
    match b {
        BuiltIn::VertexIndex => "vertex_index",
        BuiltIn::InstanceIndex => "instance_index",
        BuiltIn::Position => "position",
        BuiltIn::FrontFacing => "front_facing",
        BuiltIn::FragDepth => "frag_depth",
        BuiltIn::SampleIndex => "sample_index",
        BuiltIn::SampleMask => "sample_mask",
        BuiltIn::LocalInvocationId => "local_invocation_id",
        BuiltIn::LocalInvocationIndex => "local_invocation_index",
        BuiltIn::GlobalInvocationId => "global_invocation_id",
        BuiltIn::WorkgroupId => "workgroup_id",
        BuiltIn::NumWorkgroups => "num_workgroups",
        BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
        BuiltIn::SubgroupSize => "subgroup_size",
        BuiltIn::PrimitiveIndex => "primitive_index",
        BuiltIn::SubgroupId => "subgroup_id",
        BuiltIn::NumSubgroups => "num_subgroups",
    }
}

fn interp_type_name(t: InterpolationType) -> &'static str {
    match t {
        InterpolationType::Perspective => "perspective",
        InterpolationType::Linear => "linear",
        InterpolationType::Flat => "flat",
    }
}

fn interp_sampling_name(s: InterpolationSampling) -> &'static str {
    match s {
        InterpolationSampling::Center => "center",
        InterpolationSampling::Centroid => "centroid",
        InterpolationSampling::Sample => "sample",
        InterpolationSampling::First => "first",
        InterpolationSampling::Either => "either",
    }
}

// ===== Types =====

fn write_type(w: &mut Writer, ty: &Type) {
    match ty {
        Type::Scalar(s) => w.write(scalar_name(*s)),
        Type::Vector {
            elements,
            scalar_ty,
        } => {
            // WGSL accepts both `vec4<f32>` and the shorthand `vec4f`.
            // We always emit the shorthand when the scalar is known.
            // Without a scalar the abstract `vec4` form is used (only
            // valid in const contexts).
            w.write(&format!("vec{elements}"));
            if let Some(st) = scalar_ty {
                w.write(scalar_short(*st));
            }
        }
        Type::Matrix { size, scalar_ty } => {
            w.write(&format!("mat{size}x{size}"));
            if let Some(st) = scalar_ty {
                w.write("<");
                w.write(scalar_name(*st));
                w.write(">");
            } else {
                w.write("f");
            }
        }
        Type::Array { elem, len } => {
            w.write("array<");
            write_type(w, elem);
            w.write(", ");
            write_expr(w, len);
            w.write(">");
        }
        Type::RuntimeArray { elem } => {
            w.write("array<");
            write_type(w, elem);
            w.write(">");
        }
        Type::Atomic { elem } => {
            w.write("atomic<");
            write_type(w, elem);
            w.write(">");
        }
        Type::Struct { name, type_args: _ } => {
            // The IR may carry type_args on Struct types when describing
            // a generic instantiation pre-substitution (e.g. `Pair<T>`),
            // but at render time we expect the name to already be
            // monomorphized. Any leftover type_args are ignored.
            w.write(name);
        }
        Type::Ptr {
            address_space,
            elem,
        } => {
            w.write("ptr<");
            w.write(address_space_name(*address_space));
            w.write(", ");
            write_type(w, elem);
            w.write(">");
        }
        Type::Sampler => w.write("sampler"),
        Type::SamplerComparison => w.write("sampler_comparison"),
        Type::Texture { kind, sampled_type } => {
            w.write(texture_kind_name(*kind));
            w.write("<");
            w.write(scalar_name(*sampled_type));
            w.write(">");
        }
        Type::TextureDepth { kind } => {
            w.write(texture_depth_kind_name(*kind));
        }
        Type::TypeParam { name } => {
            // Templates leave TypeParam unresolved; we render a unique
            // placeholder. This should not normally be reached when
            // rendering instantiated modules.
            w.write(&format!("__TP{name}__"));
        }
    }
}

fn scalar_name(s: ScalarType) -> &'static str {
    match s {
        ScalarType::I32 => "i32",
        ScalarType::U32 => "u32",
        ScalarType::F32 => "f32",
        ScalarType::Bool => "bool",
    }
}

/// Single-letter scalar suffix used in vec/mat shorthand types.
fn scalar_short(s: ScalarType) -> &'static str {
    match s {
        ScalarType::I32 => "i",
        ScalarType::U32 => "u",
        ScalarType::F32 => "f",
        ScalarType::Bool => "b",
    }
}

fn address_space_name(a: AddressSpace) -> &'static str {
    match a {
        AddressSpace::Function => "function",
        AddressSpace::Private => "private",
        AddressSpace::Workgroup => "workgroup",
    }
}

fn texture_kind_name(k: TextureKind) -> &'static str {
    match k {
        TextureKind::Texture1D => "texture_1d",
        TextureKind::Texture2D => "texture_2d",
        TextureKind::Texture2DArray => "texture_2d_array",
        TextureKind::Texture3D => "texture_3d",
        TextureKind::TextureCube => "texture_cube",
        TextureKind::TextureCubeArray => "texture_cube_array",
        TextureKind::TextureMultisampled2D => "texture_multisampled_2d",
    }
}

fn texture_depth_kind_name(k: TextureDepthKind) -> &'static str {
    match k {
        TextureDepthKind::Depth2D => "texture_depth_2d",
        TextureDepthKind::Depth2DArray => "texture_depth_2d_array",
        TextureDepthKind::DepthCube => "texture_depth_cube",
        TextureDepthKind::DepthCubeArray => "texture_depth_cube_array",
        TextureDepthKind::DepthMultisampled2D => "texture_depth_multisampled_2d",
    }
}

// ===== Expressions =====

fn write_expr(w: &mut Writer, e: &Expr) {
    match e {
        Expr::Lit(l) => write_lit(w, l),
        Expr::Ident(name) => w.write(name),
        Expr::Array { elems } => {
            w.write("array(");
            for (i, x) in elems.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                write_expr(w, x);
            }
            w.write(")");
        }
        Expr::Paren(inner) => {
            w.write("(");
            write_expr(w, inner);
            w.write(")");
        }
        Expr::Binary { lhs, op, rhs } => {
            write_expr(w, lhs);
            w.space();
            w.write(binop_str(*op));
            w.space();
            write_expr(w, rhs);
        }
        Expr::Unary { op, expr } => {
            w.write(unop_str(*op));
            write_expr(w, expr);
        }
        Expr::ArrayIndexing { lhs, index } => {
            write_expr(w, lhs);
            w.write("[");
            write_expr(w, index);
            w.write("]");
        }
        Expr::Swizzle {
            lhs,
            swizzle,
            params,
        } => {
            write_expr(w, lhs);
            w.write(".");
            w.write(swizzle);
            if let Some(args) = params {
                w.write("(");
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        w.write(", ");
                    }
                    write_expr(w, a);
                }
                w.write(")");
            }
        }
        Expr::Cast { lhs, ty } => {
            write_type(w, ty);
            w.write("(");
            write_expr(w, lhs);
            w.write(")");
        }
        Expr::FnCall {
            path,
            type_args,
            params,
        } => {
            write_fn_path(w, path);
            for ta in type_args {
                w.write("_");
                write_type(w, ta);
            }
            w.write("(");
            for (i, p) in params.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                write_expr(w, p);
            }
            w.write(")");
        }
        Expr::Struct {
            name,
            type_args: _,
            fields,
        } => {
            // Leftover type_args (from a pre-substitution generic
            // construction expression) are ignored at render time — the
            // name is already monomorphized.
            w.write(name);
            w.write("(");
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                write_expr(w, &f.expr);
            }
            w.write(")");
        }
        Expr::FieldAccess { base, field } => {
            write_expr(w, base);
            w.write(".");
            w.write(field);
        }
        Expr::TypePath { ty, member } => {
            w.write(ty);
            w.write("_");
            w.write(member);
        }
        Expr::Reference(inner) => {
            w.write("&");
            write_expr(w, inner);
        }
        Expr::ZeroValueArray { elem_type, len } => {
            w.write("array<");
            write_type(w, elem_type);
            w.write(", ");
            write_expr(w, len);
            w.write(">()");
        }
    }
}

fn write_fn_path(w: &mut Writer, p: &FnPath) {
    match p {
        FnPath::Ident(name) => {
            // Builtin name translation lives here so the IR can stay
            // semantic.
            let translated = builtin_lookup::lookup(name).unwrap_or(name);
            w.write(translated);
        }
        FnPath::TypeMethod { ty, method } => {
            w.write(ty);
            w.write("_");
            w.write(method);
        }
    }
}

fn write_lit(w: &mut Writer, l: &Lit) {
    match l {
        Lit::Bool(b) => w.write(if *b { "true" } else { "false" }),
        Lit::Int { digits, suffix } => {
            let wgsl_suffix = match suffix.as_str() {
                "u32" | "usize" => "u",
                "i32" | "isize" => "i",
                "" => "",
                other => other,
            };
            w.write(digits);
            w.write(wgsl_suffix);
        }
        Lit::Float { text } => w.write(text),
    }
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

fn compound_str(op: CompoundOp) -> &'static str {
    match op {
        CompoundOp::AddAssign => "+=",
        CompoundOp::SubAssign => "-=",
        CompoundOp::MulAssign => "*=",
        CompoundOp::DivAssign => "/=",
        CompoundOp::RemAssign => "%=",
        CompoundOp::BitAndAssign => "&=",
        CompoundOp::BitOrAssign => "|=",
        CompoundOp::BitXorAssign => "^=",
        CompoundOp::ShlAssign => "<<=",
        CompoundOp::ShrAssign => ">>=",
    }
}

fn unop_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Not => "!",
        UnOp::Neg => "-",
        UnOp::Deref => "*",
    }
}

// ===== Statements / blocks =====

fn write_block(w: &mut Writer, b: &Block) {
    w.write("{");
    w.newline();
    w.indent += 1;
    for stmt in &b.stmts {
        write_stmt(w, stmt);
    }
    w.indent -= 1;
    w.start_line();
    w.write("}");
}

fn write_stmt(w: &mut Writer, s: &Stmt) {
    match s {
        Stmt::Local(l) => {
            w.start_line();
            w.write(if l.mutable { "var " } else { "let " });
            w.write(&l.name);
            if let Some(t) = &l.ty {
                w.write(": ");
                write_type(w, t);
            }
            if let Some(e) = &l.init {
                w.write(" = ");
                write_expr(w, e);
            }
            w.write(";");
            w.newline();
        }
        Stmt::Const(c) => {
            w.start_line();
            write_const(w, c);
            w.newline();
        }
        Stmt::Assignment { lhs, rhs } => {
            w.start_line();
            write_expr(w, lhs);
            w.write(" = ");
            write_expr(w, rhs);
            w.write(";");
            w.newline();
        }
        Stmt::CompoundAssignment { lhs, op, rhs } => {
            w.start_line();
            write_expr(w, lhs);
            w.space();
            w.write(compound_str(*op));
            w.space();
            write_expr(w, rhs);
            w.write(";");
            w.newline();
        }
        Stmt::While { condition, body } => {
            w.start_line();
            w.write("while ");
            write_expr(w, condition);
            w.space();
            write_block(w, body);
            w.newline();
        }
        Stmt::Loop { body } => {
            w.start_line();
            w.write("loop ");
            write_block(w, body);
            w.newline();
        }
        Stmt::Expr { expr, has_semi } => {
            w.start_line();
            if !*has_semi {
                w.write("return ");
            }
            write_expr(w, expr);
            w.write(";");
            w.newline();
        }
        Stmt::If(i) => {
            w.start_line();
            write_if(w, i);
            w.newline();
        }
        Stmt::Break => {
            w.start_line();
            w.write("break;");
            w.newline();
        }
        Stmt::Continue => {
            w.start_line();
            w.write("continue;");
            w.newline();
        }
        Stmt::Return(e) => {
            w.start_line();
            w.write("return");
            if let Some(e) = e {
                w.space();
                write_expr(w, e);
            }
            w.write(";");
            w.newline();
        }
        Stmt::For(f) => {
            w.start_line();
            w.write("for (var ");
            w.write(&f.var);
            if let Some(t) = &f.var_ty {
                w.write(": ");
                write_type(w, t);
            }
            w.write(" = ");
            write_expr(w, &f.from);
            w.write("; ");
            w.write(&f.var);
            w.write(if f.inclusive { " <= " } else { " < " });
            write_expr(w, &f.to);
            w.write("; ");
            w.write(&f.var);
            w.write("++) ");
            write_block(w, &f.body);
            w.newline();
        }
        Stmt::Switch(s) => {
            w.start_line();
            write_switch(w, s);
            w.newline();
        }
        Stmt::Block(b) => {
            w.start_line();
            write_block(w, b);
            w.newline();
        }
        Stmt::SlabRead {
            slab,
            offset,
            dest,
            size,
        } => {
            w.start_line();
            w.write("for (var _i: u32 = 0u; _i < ");
            write_expr(w, size);
            w.write("; _i++) {");
            w.newline();
            w.indent += 1;
            w.start_line();
            write_expr(w, dest);
            w.write("[_i] = ");
            write_expr(w, slab);
            w.write("[");
            write_expr(w, offset);
            w.write(" + _i];");
            w.newline();
            w.indent -= 1;
            w.start_line();
            w.write("}");
            w.newline();
        }
        Stmt::SlabWrite {
            slab,
            offset,
            src,
            size,
        } => {
            w.start_line();
            w.write("for (var _i: u32 = 0u; _i < ");
            match size {
                Some(sz) => write_expr(w, sz),
                None => {
                    w.write("arrayLength(&");
                    write_expr(w, slab);
                    w.write(")");
                }
            }
            w.write("; _i++) {");
            w.newline();
            w.indent += 1;
            w.start_line();
            write_expr(w, slab);
            w.write("[");
            write_expr(w, offset);
            w.write(" + _i] = ");
            write_expr(w, src);
            w.write("[_i];");
            w.newline();
            w.indent -= 1;
            w.start_line();
            w.write("}");
            w.newline();
        }
        Stmt::Discard => {
            w.start_line();
            w.write("discard;");
            w.newline();
        }
    }
}

fn write_if(w: &mut Writer, i: &StmtIf) {
    w.write("if ");
    write_expr(w, &i.condition);
    w.space();
    write_block(w, &i.then_block);
    if let Some(eb) = &i.else_branch {
        w.write(" else ");
        match eb {
            ElseBranch::Block(b) => write_block(w, b),
            ElseBranch::If(inner) => write_if(w, inner),
        }
    }
}

fn write_switch(w: &mut Writer, s: &StmtSwitch) {
    w.write("switch ");
    write_expr(w, &s.selector);
    w.write(" {");
    w.newline();
    w.indent += 1;
    for arm in &s.arms {
        w.start_line();
        let mut has_default = false;
        let mut case_selectors: Vec<&CaseSelector> = Vec::new();
        for sel in &arm.selectors {
            match sel {
                CaseSelector::Default => {
                    has_default = true;
                }
                _ => case_selectors.push(sel),
            }
        }
        if !case_selectors.is_empty() {
            w.write("case ");
            for (i, sel) in case_selectors.iter().enumerate() {
                if i > 0 {
                    w.write(", ");
                }
                match sel {
                    CaseSelector::Literal(l) => write_lit(w, l),
                    CaseSelector::Expr(e) => write_expr(w, e),
                    CaseSelector::Default => unreachable!(),
                }
            }
            if has_default {
                w.write(", default");
            }
            w.write(": ");
        } else if has_default {
            w.write("default: ");
        }
        write_block(w, &arm.body);
        w.newline();
    }
    if !s.has_explicit_default {
        w.start_line();
        w.write("default: { }");
        w.newline();
    }
    w.indent -= 1;
    w.start_line();
    w.write("}");
}
