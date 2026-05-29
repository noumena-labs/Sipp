//! CLI definitions for the xtask build orchestrator.

use clap::{Args, Parser, Subcommand, ValueEnum};

const TOP_LEVEL_HELP: &str = "\
CogentLM's developer automation lives under focused command groups.

Start with:
  cargo xtask build --help
  cargo xtask build all
  cargo xtask build node --backend cpu
  cargo xtask build python --backend cuda";

const BUILD_HELP: &str = "\
Build CogentLM targets from the workspace root.

Examples:
  cargo xtask build all
  cargo xtask build core
  cargo xtask build wasm
  cargo xtask build node --backend cpu
  cargo xtask build node --backend all
  cargo xtask build python --backend cuda

Notes:
  `build all` builds every target family with default CPU bindings.
  It does not build every Node/Python backend variant.";

const BACKEND_HELP: &str = "\
Backend values:
  cpu     Portable default backend
  cuda    NVIDIA CUDA backend; requires a local CUDA Toolkit
  metal   Apple Metal backend on macOS
  vulkan  Vulkan backend; xtask bootstraps the Vulkan SDK when needed
  all     Host-supported backend set for the binding target";

/// Top-level xtask command-line arguments.
#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "CogentLM developer automation")]
#[command(long_about = TOP_LEVEL_HELP)]
#[command(after_long_help = "Run `cargo xtask <command> --help` for detailed guidance.")]
#[command(arg_required_else_help = true)]
pub struct Cli {
    /// Command group to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level xtask command groups.
#[derive(Subcommand)]
pub enum Commands {
    /// Build CogentLM targets and language bindings.
    #[command(long_about = BUILD_HELP)]
    #[command(after_long_help = BACKEND_HELP)]
    #[command(arg_required_else_help = true)]
    Build {
        /// Build target to execute.
        #[command(subcommand)]
        target: BuildCommands,
    },
}

/// Supported build targets.
#[derive(Subcommand)]
pub enum BuildCommands {
    /// Build core, WASM, Python CPU, and Node CPU targets.
    #[command(long_about = "\
Build every target family in sequence:
  1. Native Rust workspace
  2. Browser WASM/WebGPU package
  3. Python bindings with the default CPU backend
  4. Node bindings with the default CPU backend

This command does not build every backend variant. Use
`cargo xtask build node --backend all` or
`cargo xtask build python --backend all` when you need fat binding artifacts.")]
    All,

    /// Build the native Rust workspace crates.
    #[command(long_about = "\
Build the native Rust workspace in release mode, excluding the xtask crate.

Equivalent build step:
  cargo build --release --workspace --exclude xtask")]
    Core,

    /// Build browser WASM/WebGPU artifacts and TypeScript wrappers.
    #[command(long_about = "\
Build browser artifacts with Emscripten and stage the NPM browser package.

The pipeline builds both single-threaded and pthread WASM outputs, then
compiles and stages the TypeScript package wrappers.")]
    Wasm,

    /// Build Python bindings.
    #[command(long_about = "\
Build Python bindings with uv and maturin.

Examples:
  cargo xtask build python
  cargo xtask build python --backend cpu
  cargo xtask build python --backend vulkan
  cargo xtask build python --backend all

The default backend is CPU. `--backend all` builds host-supported native
variants and packages them into a backend-fat wheel.")]
    #[command(after_long_help = BACKEND_HELP)]
    Python(BackendArgs),

    /// Build Node.js bindings.
    #[command(long_about = "\
Build Node.js N-API bindings with Bun and napi-rs.

Examples:
  cargo xtask build node
  cargo xtask build node --backend cpu
  cargo xtask build node --backend cuda
  cargo xtask build node --backend all

The default backend is CPU. `--backend all` expands to the backend set
supported by the host operating system.")]
    #[command(after_long_help = BACKEND_HELP)]
    Node(BackendArgs),
}

/// Shared backend selection flags for binding builds.
#[derive(Args)]
pub struct BackendArgs {
    /// Computation backend to compile against.
    #[arg(long, short, value_enum)]
    pub backend: Option<Backend>,
}

/// Hardware backend selected for binding builds.
#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum Backend {
    /// Standard CPU computation fallback.
    Cpu,
    /// NVIDIA CUDA hardware acceleration.
    Cuda,
    /// Apple Metal native acceleration.
    Metal,
    /// Vulkan cross-platform GPU acceleration.
    Vulkan,
    /// Build all supported binding backends for the host OS.
    All,
}

impl Backend {
    /// Converts the backend into the lowercase feature/build tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::All => "all",
        }
    }
}
