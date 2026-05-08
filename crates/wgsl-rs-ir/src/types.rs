//! IR type definitions.
//!
//! These types mirror the parse tree of `wgsl-rs-macros` but use owned data
//! (`String`, `Vec<T>`, plain numeric/bool literals) so they can live at
//! runtime without any dependency on `syn` or `proc-macro2`.

/// A complete WGSL module: a name and an ordered list of top-level items.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub name: String,
    pub items: Vec<Item>,
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
    /// A type parameter referenced by name. These are replaced by concrete
    /// types via [`crate::substitute_types`] before rendering.
    TypeParam { name: String },
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
}

// ===== Items =====

/// A `const` item.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemConst {
    pub name: String,
    pub ty: Type,
    pub expr: Expr,
}

/// A `@group(N) @binding(M) var<uniform>` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemUniform {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
}

/// A `@group(N) @binding(M) var<storage, ...>` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStorage {
    pub group: u32,
    pub binding: u32,
    pub access: StorageAccess,
    pub name: String,
    pub ty: Type,
}

/// A `var<workgroup>` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemWorkgroup {
    pub name: String,
    pub ty: Type,
}

/// A `@group(N) @binding(M) var ... : sampler[_comparison]` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemSampler {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
}

/// A `@group(N) @binding(M) var ... : texture_*` linkage.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemTexture {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Type,
}

/// A function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemFn {
    /// Type parameters declared on the function. After monomorphization
    /// these are usually empty; non-empty values appear only on generic
    /// templates (which need substitution before rendering).
    pub type_params: Vec<String>,
    pub fn_attrs: FnAttrs,
    pub name: String,
    pub inputs: Vec<FnArg>,
    pub return_type: ReturnType,
    pub block: Block,
}

/// A struct field.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub inter_stage_io: Vec<InterStageIo>,
    pub name: String,
    pub ty: Type,
}

/// A struct definition.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStruct {
    pub type_params: Vec<String>,
    pub name: String,
    pub fields: Vec<Field>,
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
    pub self_ty: String,
    pub items: Vec<ImplItem>,
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
