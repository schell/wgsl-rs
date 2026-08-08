//! IR type definitions.
//!
//! These types mirror the parse tree of `wgsl-rs-macros` but use owned data
//! (`String`, `Vec<T>`, plain numeric/bool literals) so they can live at
//! runtime without any dependency on `syn` or `proc-macro2`.

use std::borrow::Cow;

/// An attribute preserved from Rust source on an IR item.
///
/// Not rendered in WGSL output — exists for extension inspection.
/// E.g., `#[derive(SlabItem, Clone)]` → `Attribute { path: "derive", args:
/// vec!["SlabItem", "Clone"] }` E.g., `#[repr(C)]` → `Attribute { path: "repr",
/// args: vec!["C"] }` E.g., `#[inline]` → `Attribute { path: "inline", args:
/// vec![] }`
#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    /// The attribute path (e.g., "derive", "repr", "crabslab::slab_item").
    pub path: String,
    /// Arguments to the attribute, split on commas.
    /// E.g., `["SlabItem", "Clone"]` for `#[derive(SlabItem, Clone)]`,
    /// `["C"]` for `#[repr(C)]`, `[]` for `#[inline]`.
    pub args: Vec<String>,
}

/// A complete WGSL module: a name and an ordered list of top-level items.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub name: &'static str,
    pub items: Vec<Item>,
    /// Attributes preserved from Rust source on the module itself.
    pub attrs: Vec<Attribute>,
}

/// A block of statements `{ ... }`.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

// ===== Scalar / address space / texture kinds =====

/// WGSL scalar types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I32,
    U32,
    F32,
    Bool,
}

/// WGSL address spaces relevant for pointer types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
}

/// Storage buffer access mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageAccess {
    Read,
    ReadWrite,
}

/// Sampled texture kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Texture1D,
    Texture2D,
    Texture2DArray,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    TextureMultisampled2D,
}

/// Depth texture kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDepthKind {
    Depth2D,
    Depth2DArray,
    DepthCube,
    DepthCubeArray,
    DepthMultisampled2D,
}

/// Storage texture kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureStorageKind {
    Storage1D,
    Storage2D,
    Storage2DArray,
    Storage3D,
}

/// Storage texture access modes. WGSL §14.2.
///
/// Unlike [`StorageAccess`] (used for storage buffers, which only allow
/// `read` and `read_write`), storage textures also support `write`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageTextureAccess {
    Read,
    Write,
    ReadWrite,
}

/// Texel formats for storage textures. WGSL §6.6.1.
///
/// Each variant maps 1:1 to a `wgpu::TextureFormat` of the same name. The
/// `requires_tier1` method reports whether the `texture_formats_tier1`
/// language extension must be enabled to use the format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelFormat {
    // ===== Core formats (no extension required) =====
    Rgba8unorm,
    Rgba8snorm,
    Rgba8uint,
    Rgba8sint,
    Rgba16uint,
    Rgba16sint,
    Rgba16float,
    R32uint,
    R32sint,
    R32float,
    Rg32uint,
    Rg32sint,
    Rg32float,
    Rgba32uint,
    Rgba32sint,
    Rgba32float,
    Bgra8unorm,
    // ===== Tier-1 extension formats =====
    Rgba16unorm,
    Rgba16snorm,
    Rg8unorm,
    Rg8snorm,
    Rg8uint,
    Rg8sint,
    Rg16unorm,
    Rg16snorm,
    Rg16uint,
    Rg16sint,
    Rg16float,
    R8unorm,
    R8snorm,
    R8uint,
    R8sint,
    R16unorm,
    R16snorm,
    R16uint,
    R16sint,
    R16float,
    Rgb10a2unorm,
    Rgb10a2uint,
    Rg11b10ufloat,
}

impl TexelFormat {
    /// Whether this format requires the `texture_formats_tier1` language
    /// extension (WGSL §6.6.1, last column of the storage texel formats
    /// table).
    pub fn requires_tier1(&self) -> bool {
        matches!(
            self,
            TexelFormat::Rgba16unorm
                | TexelFormat::Rgba16snorm
                | TexelFormat::Rg8unorm
                | TexelFormat::Rg8snorm
                | TexelFormat::Rg8uint
                | TexelFormat::Rg8sint
                | TexelFormat::Rg16unorm
                | TexelFormat::Rg16snorm
                | TexelFormat::Rg16uint
                | TexelFormat::Rg16sint
                | TexelFormat::Rg16float
                | TexelFormat::R8unorm
                | TexelFormat::R8snorm
                | TexelFormat::R8uint
                | TexelFormat::R8sint
                | TexelFormat::R16unorm
                | TexelFormat::R16snorm
                | TexelFormat::R16uint
                | TexelFormat::R16sint
                | TexelFormat::R16float
                | TexelFormat::Rgb10a2unorm
                | TexelFormat::Rgb10a2uint
                | TexelFormat::Rg11b10ufloat
        )
    }

    /// The shader-side scalar type (`f32`, `u32`, or `i32`) that
    /// `textureLoad` returns / `textureStore` accepts for this format.
    /// Derived from the channel format table in WGSL §6.6.1.
    pub fn shader_scalar(&self) -> ScalarType {
        match self {
            TexelFormat::Rgba8unorm
            | TexelFormat::Rgba8snorm
            | TexelFormat::Rgba16float
            | TexelFormat::Rgba16unorm
            | TexelFormat::Rgba16snorm
            | TexelFormat::R32float
            | TexelFormat::Rg32float
            | TexelFormat::Rgba32float
            | TexelFormat::Bgra8unorm
            | TexelFormat::Rg8unorm
            | TexelFormat::Rg8snorm
            | TexelFormat::Rg16unorm
            | TexelFormat::Rg16snorm
            | TexelFormat::Rg16float
            | TexelFormat::R8unorm
            | TexelFormat::R8snorm
            | TexelFormat::R16unorm
            | TexelFormat::R16snorm
            | TexelFormat::R16float
            | TexelFormat::Rgb10a2unorm
            | TexelFormat::Rg11b10ufloat => ScalarType::F32,
            TexelFormat::Rgba8uint
            | TexelFormat::Rgba16uint
            | TexelFormat::R32uint
            | TexelFormat::Rg32uint
            | TexelFormat::Rgba32uint
            | TexelFormat::Rg8uint
            | TexelFormat::Rg16uint
            | TexelFormat::R8uint
            | TexelFormat::R16uint
            | TexelFormat::Rgb10a2uint => ScalarType::U32,
            TexelFormat::Rgba8sint
            | TexelFormat::Rgba16sint
            | TexelFormat::R32sint
            | TexelFormat::Rg32sint
            | TexelFormat::Rgba32sint
            | TexelFormat::Rg8sint
            | TexelFormat::Rg16sint
            | TexelFormat::R8sint
            | TexelFormat::R16sint => ScalarType::I32,
        }
    }

    /// The WGSL enumerant name (lowercase, as written in WGSL source).
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            TexelFormat::Rgba8unorm => "rgba8unorm",
            TexelFormat::Rgba8snorm => "rgba8snorm",
            TexelFormat::Rgba8uint => "rgba8uint",
            TexelFormat::Rgba8sint => "rgba8sint",
            TexelFormat::Rgba16uint => "rgba16uint",
            TexelFormat::Rgba16sint => "rgba16sint",
            TexelFormat::Rgba16float => "rgba16float",
            TexelFormat::R32uint => "r32uint",
            TexelFormat::R32sint => "r32sint",
            TexelFormat::R32float => "r32float",
            TexelFormat::Rg32uint => "rg32uint",
            TexelFormat::Rg32sint => "rg32sint",
            TexelFormat::Rg32float => "rg32float",
            TexelFormat::Rgba32uint => "rgba32uint",
            TexelFormat::Rgba32sint => "rgba32sint",
            TexelFormat::Rgba32float => "rgba32float",
            TexelFormat::Bgra8unorm => "bgra8unorm",
            TexelFormat::Rgba16unorm => "rgba16unorm",
            TexelFormat::Rgba16snorm => "rgba16snorm",
            TexelFormat::Rg8unorm => "rg8unorm",
            TexelFormat::Rg8snorm => "rg8snorm",
            TexelFormat::Rg8uint => "rg8uint",
            TexelFormat::Rg8sint => "rg8sint",
            TexelFormat::Rg16unorm => "rg16unorm",
            TexelFormat::Rg16snorm => "rg16snorm",
            TexelFormat::Rg16uint => "rg16uint",
            TexelFormat::Rg16sint => "rg16sint",
            TexelFormat::Rg16float => "rg16float",
            TexelFormat::R8unorm => "r8unorm",
            TexelFormat::R8snorm => "r8snorm",
            TexelFormat::R8uint => "r8uint",
            TexelFormat::R8sint => "r8sint",
            TexelFormat::R16unorm => "r16unorm",
            TexelFormat::R16snorm => "r16snorm",
            TexelFormat::R16uint => "r16uint",
            TexelFormat::R16sint => "r16sint",
            TexelFormat::R16float => "r16float",
            TexelFormat::Rgb10a2unorm => "rgb10a2unorm",
            TexelFormat::Rgb10a2uint => "rgb10a2uint",
            TexelFormat::Rg11b10ufloat => "rg11b10ufloat",
        }
    }

    /// Parse a WGSL texel format name (lowercase) into a `TexelFormat`.
    /// Returns `None` if the name is not a recognized storage texel format.
    pub fn from_wgsl_name(name: &str) -> Option<Self> {
        Some(match name {
            "rgba8unorm" => TexelFormat::Rgba8unorm,
            "rgba8snorm" => TexelFormat::Rgba8snorm,
            "rgba8uint" => TexelFormat::Rgba8uint,
            "rgba8sint" => TexelFormat::Rgba8sint,
            "rgba16uint" => TexelFormat::Rgba16uint,
            "rgba16sint" => TexelFormat::Rgba16sint,
            "rgba16float" => TexelFormat::Rgba16float,
            "r32uint" => TexelFormat::R32uint,
            "r32sint" => TexelFormat::R32sint,
            "r32float" => TexelFormat::R32float,
            "rg32uint" => TexelFormat::Rg32uint,
            "rg32sint" => TexelFormat::Rg32sint,
            "rg32float" => TexelFormat::Rg32float,
            "rgba32uint" => TexelFormat::Rgba32uint,
            "rgba32sint" => TexelFormat::Rgba32sint,
            "rgba32float" => TexelFormat::Rgba32float,
            "bgra8unorm" => TexelFormat::Bgra8unorm,
            "rgba16unorm" => TexelFormat::Rgba16unorm,
            "rgba16snorm" => TexelFormat::Rgba16snorm,
            "rg8unorm" => TexelFormat::Rg8unorm,
            "rg8snorm" => TexelFormat::Rg8snorm,
            "rg8uint" => TexelFormat::Rg8uint,
            "rg8sint" => TexelFormat::Rg8sint,
            "rg16unorm" => TexelFormat::Rg16unorm,
            "rg16snorm" => TexelFormat::Rg16snorm,
            "rg16uint" => TexelFormat::Rg16uint,
            "rg16sint" => TexelFormat::Rg16sint,
            "rg16float" => TexelFormat::Rg16float,
            "r8unorm" => TexelFormat::R8unorm,
            "r8snorm" => TexelFormat::R8snorm,
            "r8uint" => TexelFormat::R8uint,
            "r8sint" => TexelFormat::R8sint,
            "r16unorm" => TexelFormat::R16unorm,
            "r16snorm" => TexelFormat::R16snorm,
            "r16uint" => TexelFormat::R16uint,
            "r16sint" => TexelFormat::R16sint,
            "r16float" => TexelFormat::R16float,
            "rgb10a2unorm" => TexelFormat::Rgb10a2unorm,
            "rgb10a2uint" => TexelFormat::Rgb10a2uint,
            "rg11b10ufloat" => TexelFormat::Rg11b10ufloat,
            _ => return None,
        })
    }
}

impl StorageTextureAccess {
    /// The WGSL enumerant name (lowercase).
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            StorageTextureAccess::Read => "read",
            StorageTextureAccess::Write => "write",
            StorageTextureAccess::ReadWrite => "read_write",
        }
    }

    /// Parse a WGSL access mode name (lowercase) into a
    /// `StorageTextureAccess`. Returns `None` if unrecognized.
    pub fn from_wgsl_name(name: &str) -> Option<Self> {
        Some(match name {
            "read" => StorageTextureAccess::Read,
            "write" => StorageTextureAccess::Write,
            "read_write" => StorageTextureAccess::ReadWrite,
            _ => return None,
        })
    }
}

impl TextureStorageKind {
    /// The WGSL type name (e.g. `texture_storage_2d`).
    pub fn wgsl_name(&self) -> &'static str {
        match self {
            TextureStorageKind::Storage1D => "texture_storage_1d",
            TextureStorageKind::Storage2D => "texture_storage_2d",
            TextureStorageKind::Storage2DArray => "texture_storage_2d_array",
            TextureStorageKind::Storage3D => "texture_storage_3d",
        }
    }
}

// ===== Type =====

/// WGSL type expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// A scalar type such as `i32`, `u32`, `f32`, `bool`.
    Scalar(ScalarType),
    /// A vector type such as `vec3<f32>` or `vec3f`. When `scalar_ty` is
    /// `None`, the rendered output uses the abstract form.
    Vector {
        elements: u8,
        /// The scalar element type. `None` means abstract / unspecified.
        scalar_ty: Option<ScalarType>,
    },
    /// A matrix type such as `mat4x4<f32>` / `mat4x4f` (square) or
    /// `mat2x3<f32>` / `mat2x3f` (non-square). `columns` is the number of
    /// columns (the first dimension in WGSL's `matCxR<T>`), `rows` is the
    /// number of rows. WGSL allows any `C` and `R` in `{2, 3, 4}`.
    Matrix {
        columns: u8,
        rows: u8,
        scalar_ty: Option<ScalarType>,
    },
    /// A fixed-size array `array<T, N>`.
    Array { elem: Box<Type>, len: Expr },
    /// A runtime-sized array `array<T>`.
    RuntimeArray { elem: Box<Type> },
    /// `atomic<T>`.
    Atomic { elem: Box<Type> },
    /// A user-defined struct, possibly generic with type arguments.
    Struct { name: String, type_args: Vec<Type> },
    /// A pointer type `ptr<address_space, T>`.
    Ptr {
        address_space: AddressSpace,
        elem: Box<Type>,
    },
    /// A `sampler`.
    Sampler,
    /// A `sampler_comparison`.
    SamplerComparison,
    /// A sampled texture, e.g. `texture_2d<f32>`.
    Texture {
        kind: TextureKind,
        sampled_type: ScalarType,
    },
    /// A depth texture, e.g. `texture_depth_2d`.
    TextureDepth { kind: TextureDepthKind },
    /// A storage texture, e.g. `texture_storage_2d<rgba8unorm, write>`.
    /// WGSL §6.6.5.
    TextureStorage {
        kind: TextureStorageKind,
        format: TexelFormat,
        access: StorageTextureAccess,
    },
    /// A type parameter referenced by name. These are replaced by concrete
    /// types via [`crate::substitute_types`] before rendering.
    TypeParam { name: String },
    /// A `PhantomData<T>` marker field. Retained in the IR so that
    /// extensions can observe which type parameter each phantom slot
    /// binds (e.g. `struct Foo<T, A> { x: f32, t: PhantomData<T>,
    /// a: PhantomData<A> }`). The renderer omits phantom fields from
    /// the emitted WGSL; this variant should never reach
    /// [`crate::render::render_items`]'s type writer.
    Phantom { elem: Box<Type> },
}

// ===== Literals / operators =====

/// A literal value.
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    Bool(bool),
    /// An integer literal, with the original text and an optional Rust-style
    /// suffix. The suffix matters because the renderer translates `u32` /
    /// `usize` to WGSL's `u`, and `i32` / `isize` to WGSL's `i`.
    Int {
        digits: String,
        suffix: String,
    },
    /// A float literal, stored as the original text. WGSL accepts the same
    /// textual forms (with optional `f` suffix).
    Float {
        text: String,
    },
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Compound assignment operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompoundOp {
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

/// Unary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    Neg,
    Deref,
}

// ===== Expressions =====

/// A function call path: either a free function name or a `Type::method`
/// path (which is mangled to `Type_method` at render time).
#[derive(Clone, Debug, PartialEq)]
pub enum FnPath {
    Ident(String),
    TypeMethod { ty: String, method: String },
}

/// A struct expression field: `name: expr` or shorthand `name`.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldValue {
    pub member: String,
    pub expr: Expr,
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Lit(Lit),
    Ident(String),
    Array {
        elems: Vec<Expr>,
    },
    Paren(Box<Expr>),
    Binary {
        lhs: Box<Expr>,
        op: BinOp,
        rhs: Box<Expr>,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    ArrayIndexing {
        lhs: Box<Expr>,
        index: Box<Expr>,
    },
    /// Vector swizzle / component access. `params` is `Some(args)` when the
    /// swizzle is actually a method-style call (e.g. matrix access via
    /// `m.x(i)` style); when `None` the swizzle is a plain field access.
    Swizzle {
        lhs: Box<Expr>,
        swizzle: String,
        params: Option<Vec<Expr>>,
    },
    /// `T(expr)` cast / construction.
    Cast {
        lhs: Box<Expr>,
        ty: Box<Type>,
    },
    FnCall {
        path: FnPath,
        type_args: Vec<Type>,
        params: Vec<Expr>,
    },
    /// A struct construction expression. Fields are kept by name in the IR;
    /// the renderer drops the names and emits positional arguments.
    Struct {
        name: String,
        type_args: Vec<Type>,
        fields: Vec<FieldValue>,
    },
    FieldAccess {
        base: Box<Expr>,
        field: String,
    },
    /// `Type::MEMBER` — for associated constants. The renderer emits this
    /// as `Type_MEMBER`.
    TypePath {
        ty: String,
        member: String,
    },
    Reference(Box<Expr>),
    /// `[T; N]()` — a zero-initialized array literal. Renders as
    /// `array<T, N>()`.
    ZeroValueArray {
        elem_type: Box<Type>,
        len: Box<Expr>,
    },
}

// ===== Statements =====

/// A `let` / `var` / `const` initializer.
#[derive(Clone, Debug, PartialEq)]
pub struct Local {
    /// `true` when this should render as `var` (Rust `let mut`); `false`
    /// for `let` (Rust `let`).
    pub mutable: bool,
    pub name: String,
    pub ty: Option<Type>,
    pub init: Option<Expr>,
}

/// A `for` loop lowered from a Rust `for i in from..to` (or `..=to`).
#[derive(Clone, Debug, PartialEq)]
pub struct ForLoop {
    pub var: String,
    pub var_ty: Option<Type>,
    pub from: Expr,
    pub to: Expr,
    /// `true` for `..=` (inclusive).
    pub inclusive: bool,
    pub body: Block,
}

/// An `if` statement, possibly with an `else` branch.
#[derive(Clone, Debug, PartialEq)]
pub struct StmtIf {
    pub condition: Expr,
    pub then_block: Block,
    pub else_branch: Option<ElseBranch>,
}

/// An `else` branch.
#[derive(Clone, Debug, PartialEq)]
pub enum ElseBranch {
    Block(Block),
    If(Box<StmtIf>),
}

/// A `match` / `switch` statement.
#[derive(Clone, Debug, PartialEq)]
pub struct StmtSwitch {
    pub selector: Expr,
    pub arms: Vec<SwitchArm>,
    /// Whether the original source contained an explicit default arm.
    pub has_explicit_default: bool,
}

/// One arm of a switch / match.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchArm {
    pub selectors: Vec<CaseSelector>,
    pub body: Block,
}

/// A case selector for a switch arm.
#[derive(Clone, Debug, PartialEq)]
pub enum CaseSelector {
    Literal(Lit),
    Expr(Expr),
    Default,
}

/// A statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Local(Local),
    /// A `const` item declared inside a function body.
    Const(ItemConst),
    Assignment {
        lhs: Expr,
        rhs: Expr,
    },
    CompoundAssignment {
        lhs: Expr,
        op: CompoundOp,
        rhs: Expr,
    },
    While {
        condition: Expr,
        body: Block,
    },
    Loop {
        body: Block,
    },
    /// An expression statement. When `has_semi` is `false`, the renderer
    /// treats this as an implicit `return expr;` (Rust-style trailing
    /// expression in a function body).
    Expr {
        expr: Expr,
        has_semi: bool,
    },
    If(StmtIf),
    Break,
    Continue,
    Return(Option<Expr>),
    For(ForLoop),
    Switch(StmtSwitch),
    Block(Block),
    /// Slab read: copy `size` elements from `slab[offset..]` into `dest`.
    SlabRead {
        slab: Expr,
        offset: Expr,
        dest: Expr,
        size: Expr,
    },
    /// Slab write: copy elements from `src` into `slab[offset..]`. When
    /// `size` is `None`, the loop bound is `arrayLength(&slab)`.
    SlabWrite {
        slab: Expr,
        offset: Expr,
        src: Expr,
        size: Option<Expr>,
    },
    Discard,
}

// ===== Function attrs / args / return =====

/// Workgroup size for a `@compute` shader.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: Option<u32>,
    pub z: Option<u32>,
}

/// Function-level attributes (entry point markers).
#[derive(Clone, Debug, PartialEq)]
pub enum FnAttrs {
    None,
    Vertex,
    Fragment,
    Compute { workgroup_size: WorkgroupSize },
}

/// A WGSL builtin attribute name (used inside `@builtin(...)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    SampleIndex,
    SampleMask,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SubgroupInvocationId,
    SubgroupSize,
    PrimitiveIndex,
    SubgroupId,
    NumSubgroups,
}

/// `@interpolate(...)` interpolation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterpolationType {
    Perspective,
    Linear,
    Flat,
}

/// `@interpolate(_, sampling)` sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InterpolationSampling {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

/// Body of an `@interpolate` attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interpolate {
    pub ty: InterpolationType,
    pub sampling: Option<InterpolationSampling>,
}

/// An inter-stage IO attribute on a function argument or struct field.
#[derive(Clone, Debug, PartialEq)]
pub enum InterStageIo {
    BuiltIn(BuiltIn),
    Location(u32),
    BlendSrc(u32),
    Interpolate(Interpolate),
    Invariant,
}

/// A return type annotation: `@builtin(position)`, `@location(0)`, etc.
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnTypeAnnotation {
    None,
    BuiltIn(BuiltIn),
    Location(u32),
    DefaultBuiltInPosition,
    DefaultLocation,
}

/// Return type of a function.
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnType {
    Default,
    Type {
        annotation: ReturnTypeAnnotation,
        ty: Type,
    },
}

/// A function argument.
#[derive(Clone, Debug, PartialEq)]
pub struct FnArg {
    pub inter_stage_io: Vec<InterStageIo>,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

// ===== Items =====

/// A `const` item.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemConst {
    pub name: String,
    pub ty: Type,
    pub expr: Expr,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A `@group(N) @binding(M) var<uniform>` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemUniform {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A `@group(N) @binding(M) var<storage, ...>` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStorage {
    pub group: u32,
    pub binding: u32,
    pub access: StorageAccess,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A `var<workgroup>` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemWorkgroup {
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A `@group(N) @binding(M) var ... : sampler[_comparison]` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemSampler {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A `@group(N) @binding(M) var ... : texture_*` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemTexture {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemFn {
    /// Type parameters declared on the function. After monomorphization
    /// these are usually empty; non-empty values appear only on generic
    /// templates (which need substitution before rendering).
    pub type_params: Vec<String>,
    /// `const` parameters declared on the function (e.g. `["N"]` for
    /// `fn foo<const N: u32>`). After monomorphization these are usually
    /// empty; non-empty values appear only on generic templates, where
    /// the corresponding `Expr::Ident(name)` nodes in array lengths are
    /// substituted with concrete `Expr::Lit` values via
    /// [`crate::substitute_consts`] before rendering.
    pub const_params: Vec<String>,
    pub fn_attrs: FnAttrs,
    /// The function's identifier. For fresh (non-monomorphized) functions
    /// this is a `Cow::Borrowed` of a `stringify!`-emitted `'static` literal
    /// — safe to borrow for FFI boundaries (wgpu, etc.) without copying.
    /// For monomorphized instances (e.g. `id` → `id_f32`) the rename
    /// pass produces a `Cow::Owned` containing a runtime-computed name;
    /// the owning `String` lives in this `Cow` and is freed when the
    /// owning `ItemFn` is dropped.
    pub name: Cow<'static, str>,
    pub inputs: Vec<FnArg>,
    pub return_type: ReturnType,
    pub block: Block,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A struct field.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub inter_stage_io: Vec<InterStageIo>,
    pub name: String,
    pub ty: Type,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A struct definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStruct {
    pub type_params: Vec<String>,
    /// `const` parameters declared on the struct (e.g. `["N"]` for
    /// `struct Grid<const N: u32>`).
    pub const_params: Vec<String>,
    pub name: String,
    pub fields: Vec<Field>,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// An item inside an `impl` block.
#[derive(Clone, Debug, PartialEq)]
pub enum ImplItem {
    Fn(ItemFn),
    Const(ItemConst),
}

/// An `impl` block. Methods and associated constants are name-mangled to
/// `StructName_member` at render time.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemImpl {
    pub type_params: Vec<String>,
    /// `const` parameters declared on the impl block.
    pub const_params: Vec<String>,
    pub self_ty: String,
    pub items: Vec<ImplItem>,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// One variant of an enum.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    /// Optional explicit discriminant value. When `None`, the renderer
    /// auto-increments from the previous value (starting at 0).
    pub discriminant: Option<u32>,
}

/// An enum definition. Renders as a `u32` alias plus per-variant `const`s.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemEnum {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    /// Attributes preserved from Rust source.
    pub attrs: Vec<Attribute>,
}

/// A top-level WGSL module item.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Const(ItemConst),
    Uniform(ItemUniform),
    Storage(ItemStorage),
    Workgroup(ItemWorkgroup),
    Sampler(ItemSampler),
    Texture(ItemTexture),
    Fn(ItemFn),
    Struct(ItemStruct),
    Impl(ItemImpl),
    Enum(ItemEnum),
}

impl Module {
    /// Renders this IR module to its WGSL source text.
    ///
    /// This is the canonical (and only) WGSL emitter in the project. The
    /// IR may contain `Type::TypeParam`s (e.g. a template module that has
    /// not yet been monomorphized); in that case `render_module` emits
    /// `__TP{name}__` placeholders, which are not valid WGSL. Callers that
    /// need valid WGSL should ensure the IR is concrete (no `Type::TypeParam`s)
    /// before calling — for a `wgsl_rs::Source` template, use the
    /// macro-emitted `instantiate::<…>()` to obtain a concrete IR module.
    ///
    /// For methods that need access to `wgsl-rs` types (e.g. `WgpuLinkage`),
    /// see the `wgsl_rs::linkage::wgpu::IrModuleExt` extension trait.
    pub fn wgsl_source(&self) -> String {
        crate::render_module(self)
    }
}
