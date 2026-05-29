//! CLI definitions for the xtask build orchestrator.

use clap::{Parser, Subcommand, ValueEnum};

/// Top-level xtask command-line arguments.
#[derive(Parser)]
#[command(name = "xtask", about = "CogentLM Build Orchestrator")]
pub struct Cli {
    /// Command to execute.
    #[command(subcommand)]
    pub command: Commands,
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

/// Supported xtask subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Build all targets (wasm, node, python).
    BuildAll,
    /// Build the core native Rust crates.
    BuildCore,
    /// Build the WASM/WebGPU bindings using Emscripten.
    BuildWasm,
    /// Build Python bindings [BACKEND] = {cpu | cuda | metal | vulkan | all}.
    BuildPython {
        /// The computation backend to compile against.
        #[arg(long, short)]
        backend: Option<Backend>,
    },
    /// Build Node bindings [BACKEND] = {cpu | cuda | metal | vulkan | all}.
    BuildNode {
        /// The computation backend to compile against.
        #[arg(long, short)]
        backend: Option<Backend>,
    },
}
