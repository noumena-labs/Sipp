//! CLI definitions for the xtask build orchestrator.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::fmt;
use std::path::PathBuf;

const TOP_LEVEL_HELP: &str = "\
CogentLM's developer automation lives under focused command groups.

Start with:
  ./setup.sh
  .\\setup.ps1
  cargo xtask build --help
  cargo xtask build all
  cargo xtask doctor
  cargo xtask toolchain status
  cargo xtask clean --dry-run
  cargo xtask run --help
  cargo xtask test --help
  cargo xtask build node --backend cpu
  cargo xtask build python --backend cuda
  cargo xtask build cli --backend all
  cargo xtask setup";

const BUILD_HELP: &str = "\
Build CogentLM targets from the workspace root.

Examples:
  cargo xtask build all
  cargo xtask build core
  cargo xtask build wasm
  cargo xtask build node --backend cpu
  cargo xtask build node --backend all
  cargo xtask build python --backend cuda
  cargo xtask build cli --backend all

Notes:
  `build all` builds every target family with default CPU native outputs.
  It does not build every Node/Python/CLI backend variant.";

const BACKEND_HELP: &str = "\
Backend values:
  cpu     Portable default backend
  cuda    NVIDIA CUDA backend; requires a local CUDA Toolkit
  metal   Apple Metal backend on macOS
  vulkan  Vulkan backend; xtask bootstraps the Vulkan SDK when needed
  all     Host-supported backend set for the target";

const RUN_HELP: &str = "\
Run long-lived apps and non-test diagnostics from the workspace root.

Examples:
  cargo xtask run apps build examples
  cargo xtask run apps serve benchmark --port 5173
  cargo xtask run llama backend-ops --backend cpu --mode support
  cargo xtask run llama backend-ops --backend cuda --mode perf --op MUL_MAT

Notes:
  Test execution, smoke checks, and coverage live under `cargo xtask test`.
  `run apps serve` is intentionally long-running and starts a Vite server.";

const RUN_APPS_HELP: &str = "\
Build or serve individual browser apps.

Examples:
  cargo xtask run apps build examples
  cargo xtask run apps build simulation
  cargo xtask run apps serve examples
  cargo xtask run apps serve benchmark --mode preview --port 4173

The apps group has no app-level `all` command. Build and serve commands target
one app at a time. App tests live under `cargo xtask test run --suite app-ts`.";

const RUN_LLAMA_HELP: &str = "\
Build standalone llama.cpp targets and run backend operation checks.

Examples:
  cargo xtask run llama backend-ops
  cargo xtask run llama backend-ops --backend cpu
  cargo xtask run llama backend-ops --backend vulkan --mode support
  cargo xtask run llama backend-ops --backend cuda --mode perf --op MUL_MAT

Correctness mode lives under `cargo xtask test run --suite llama-backend-ops`.
The run command defaults to support probing.";

const TEST_HELP: &str = "\
List, run, and verify cataloged workspace tests.

Examples:
  cargo xtask test list
  cargo xtask test list --category whitebox --cases --search router --format json
  cargo xtask test run
  cargo xtask test run --category whitebox
  cargo xtask test run --suite rust-crates --package cogentlm-core
  cargo xtask test run --suite node-package --backend cpu
  cargo xtask test verify --category whitebox
  cargo xtask test verify --changed

Model-backed tests default to the setup sample model cache under .build/models
when --model is omitted.

`test run` executes suites and writes .build/test and .build/coverage artifacts.
`test verify` analyzes existing artifacts and test structure without running tests.

Run `cargo xtask test list` to see suite requirements.";

/// Top-level xtask command-line arguments.
#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "CogentLM developer automation")]
#[command(long_about = TOP_LEVEL_HELP)]
#[command(after_long_help = "Run `cargo xtask <command> --help` for detailed guidance.")]
#[command(arg_required_else_help = true)]
pub struct Cli {
    /// Stream subprocess output directly instead of capturing it behind spinners.
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Disable decorative terminal banners.
    #[arg(long, global = true)]
    pub no_banner: bool,

    /// Disable bounded inline rendering and use traditional line output.
    #[arg(long, global = true)]
    pub plain: bool,

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

    /// Remove generated build outputs and dependency installs.
    #[command(long_about = "\
Clean generated artifacts from the workspace.

Examples:
  cargo xtask clean --dry-run
  cargo xtask clean
  cargo xtask clean --purge
  cargo xtask clean --toolchains

By default, clean removes build outputs and generated app/package output while
preserving downloaded toolchains and dependency installs. Use `--purge` to also
remove workspace node_modules directories.")]
    Clean(CleanArgs),

    /// Run long-lived apps and non-test diagnostics.
    #[command(long_about = RUN_HELP)]
    #[command(arg_required_else_help = true)]
    Run {
        /// Run target to execute.
        #[command(subcommand)]
        command: RunCommands,
    },

    /// Run workspace tests and smoke checks.
    #[command(long_about = TEST_HELP)]
    #[command(after_long_help = BACKEND_HELP)]
    #[command(arg_required_else_help = true)]
    Test {
        /// Test target to execute.
        #[command(subcommand)]
        command: TestCommands,
    },

    /// Inspect or bootstrap xtask-managed toolchains.
    #[command(arg_required_else_help = true)]
    #[command(long_about = "\
Inspect or install xtask-managed toolchains.

Examples:
  cargo xtask toolchain status
  cargo xtask toolchain install uv
  cargo xtask toolchain install all

CUDA is externally installed and is reported by status/doctor, but xtask never
installs or deletes it.")]
    Toolchain {
        /// Toolchain operation to run.
        #[command(subcommand)]
        command: ToolchainCommands,
    },

    /// Check local developer build readiness.
    #[command(long_about = "\
Inspect local build readiness without installing or deleting anything.

Examples:
  cargo xtask doctor
  cargo xtask doctor --target wasm
  cargo xtask doctor --target node --backend vulkan

Doctor fails for missing core prerequisites and warns for optional GPU/backend
readiness so developers can decide what they need for their target.")]
    #[command(after_long_help = BACKEND_HELP)]
    Doctor(DoctorArgs),

    /// Guide first-time local setup and install the cached clm launcher.
    #[command(long_about = "\
Guide first-time local setup for CogentLM.

Examples:
  ./setup.sh
  .\\setup.ps1
  cargo xtask setup
  cargo xtask setup --profile browser
  cargo xtask setup --profile full --yes
  cargo xtask setup --profile bindings --no-downloads --no-splash

Interactive setup shows a short COGENTLM splash, checks local readiness, asks
before downloading toolchains or sample files, and can install the shorter
`clm` launcher under .build/bin.")]
    Setup(SetupArgs),
}

/// Supported build targets.
#[derive(Subcommand)]
pub enum BuildCommands {
    /// Build core, WASM, Python CPU, Node CPU, and CLI CPU targets.
    #[command(long_about = "\
Build every target family in sequence:
  1. Native Rust workspace
  2. Browser WASM/WebGPU package
  3. Python bindings with the default CPU backend
  4. Node bindings with the default CPU backend
  5. CLI distribution with the default CPU backend

This command does not build every backend variant. Use
`cargo xtask build node --backend all` or
`cargo xtask build python --backend all` or
`cargo xtask build cli --backend all` when you need fat native artifacts.")]
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

    /// Build the Rust CLI distribution directory.
    #[command(long_about = "\
Build the Rust CLI and stage a runnable distribution directory.

Examples:
  cargo xtask build cli
  cargo xtask build cli --backend cpu
  cargo xtask build cli --backend vulkan
  cargo xtask build cli --backend all

The CLI build uses llama.cpp dynamic backend loading. The staged artifact
contains the cogentlm executable, base runtime libraries, and any selected
ggml backend plugins in .build/artifacts/cli.")]
    #[command(after_long_help = BACKEND_HELP)]
    Cli(BackendArgs),
}

/// Developer run workflows.
#[derive(Subcommand)]
pub enum RunCommands {
    /// Build or serve browser apps.
    #[command(long_about = RUN_APPS_HELP)]
    #[command(arg_required_else_help = true)]
    Apps {
        /// App workflow to run.
        #[command(subcommand)]
        command: RunAppsCommands,
    },

    /// Build and run standalone llama.cpp diagnostics.
    #[command(long_about = RUN_LLAMA_HELP)]
    #[command(arg_required_else_help = true)]
    Llama {
        /// llama.cpp workflow to run.
        #[command(subcommand)]
        command: RunLlamaCommands,
    },
}

/// Workspace test workflows.
#[derive(Subcommand)]
pub enum TestCommands {
    /// List known test suites and optionally discover test cases.
    List(TestListArgs),

    /// Run cataloged tests selected by suite or category.
    #[command(after_long_help = BACKEND_HELP)]
    Run(TestRunArgs),

    /// Verify test structure and existing coverage artifacts.
    Verify(TestVerifyArgs),
}

/// Options for listing test suites and cases.
#[derive(Args)]
pub struct TestListArgs {
    /// Category to include in the listing.
    #[arg(long, value_enum, default_value = "all")]
    pub category: TestCategoryFilter,

    /// Suite id to include. Repeat to include multiple suites.
    #[arg(long, value_enum)]
    pub suite: Vec<TestSuiteId>,

    /// Include individual test cases where they can be discovered cheaply.
    #[arg(long)]
    pub cases: bool,

    /// Search suite metadata and discoverable case names or paths.
    #[arg(long)]
    pub search: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value = "text")]
    pub format: TestListFormat,
}

/// Options for running cataloged tests.
#[derive(Args)]
pub struct TestRunArgs {
    /// Category to run.
    #[arg(long, value_enum, default_value = "all")]
    pub category: TestCategoryFilter,

    /// Suite id to run. Repeat to run multiple suites.
    #[arg(long, value_enum)]
    pub suite: Vec<TestSuiteId>,

    /// Rust package filter for the `rust-crates` suite.
    #[arg(long)]
    pub package: Option<String>,

    /// Backend passed to backend-aware suites.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// GGUF model for model-backed suites. Defaults to .build/models.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Do not download the default sample model when model-backed suites have no --model.
    #[arg(long)]
    pub offline: bool,
}

/// Options for test and coverage verification.
#[derive(Args)]
pub struct TestVerifyArgs {
    /// Category to verify.
    #[arg(long, value_enum, default_value = "all")]
    pub category: TestCategoryFilter,

    /// Suite id to verify. Repeat to include multiple suites.
    #[arg(long, value_enum)]
    pub suite: Vec<TestSuiteId>,

    /// Validate that changed source files have matching catalog-owned test changes.
    #[arg(long)]
    pub changed: bool,
}

/// Category filter for test listing, running, and verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestCategoryFilter {
    /// Include white-box and interface suites.
    All,
    /// Include implementation-oriented white-box suites.
    Whitebox,
    /// Include public interface black-box suites.
    Interface,
}

impl TestCategoryFilter {
    /// Stable category filter label.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestCategoryFilter::All => "all",
            TestCategoryFilter::Whitebox => "whitebox",
            TestCategoryFilter::Interface => "interface",
        }
    }
}

/// Output format for `test list`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestListFormat {
    /// Human-readable table.
    Text,
    /// JSON array.
    Json,
}

/// Test suite selector.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, ValueEnum)]
pub enum TestSuiteId {
    /// xtask CLI and orchestration unit tests.
    Xtask,
    /// Rust unit tests for core workspace crates.
    RustCrates,
    /// Rust unit tests for language binding crates.
    RustBindings,
    /// Browser package TypeScript tests.
    PackageTs,
    /// Browser app TypeScript tests.
    AppTs,
    /// Rust public API integration tests.
    RustPublicApi,
    /// CLI black-box integration tests.
    Cli,
    /// Node package interface tests.
    NodePackage,
    /// Python package interface tests.
    PythonPackage,
    /// Browser runtime smoke tests.
    BrowserRuntime,
    /// Model-backed smoke tests.
    ModelSmoke,
    /// llama.cpp backend operation tests.
    LlamaBackendOps,
}

impl TestSuiteId {
    /// Stable CLI and JSON suite label.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestSuiteId::Xtask => "xtask",
            TestSuiteId::RustCrates => "rust-crates",
            TestSuiteId::RustBindings => "rust-bindings",
            TestSuiteId::PackageTs => "package-ts",
            TestSuiteId::AppTs => "app-ts",
            TestSuiteId::RustPublicApi => "rust-public-api",
            TestSuiteId::Cli => "cli",
            TestSuiteId::NodePackage => "node-package",
            TestSuiteId::PythonPackage => "python-package",
            TestSuiteId::BrowserRuntime => "browser-runtime",
            TestSuiteId::ModelSmoke => "model-smoke",
            TestSuiteId::LlamaBackendOps => "llama-backend-ops",
        }
    }
}

/// Browser app run workflows.
#[derive(Subcommand)]
pub enum RunAppsCommands {
    /// Build one browser app.
    Build(RunAppBuildArgs),

    /// Start one long-running Vite dev or preview server.
    Serve(RunAppServeArgs),
}

/// Options for building a browser app.
#[derive(Args)]
pub struct RunAppBuildArgs {
    /// App to build.
    #[arg(value_enum)]
    pub app: AppName,
}

/// Options for serving a browser app.
#[derive(Args)]
pub struct RunAppServeArgs {
    /// App to serve.
    #[arg(value_enum)]
    pub app: AppName,

    /// Vite server mode to run.
    #[arg(long, value_enum, default_value = "dev")]
    pub mode: AppServeMode,

    /// Host passed through to Vite.
    #[arg(long)]
    pub host: Option<String>,

    /// Port passed through to Vite.
    #[arg(long)]
    pub port: Option<u16>,

    /// Start the server without first building browser package artifacts.
    #[arg(long)]
    pub no_build: bool,
}

/// Browser apps known to xtask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AppName {
    /// Browser avatar app.
    Avatar,
    /// Browser benchmark app.
    Benchmark,
    /// Browser examples app.
    Examples,
    /// Browser proactive UI app.
    ProactiveUi,
    /// Browser simulation app.
    Simulation,
}

impl AppName {
    /// Directory name under `apps`.
    pub(crate) fn slug(&self) -> &'static str {
        match self {
            AppName::Avatar => "avatar",
            AppName::Benchmark => "benchmark",
            AppName::Examples => "examples",
            AppName::ProactiveUi => "proactive-ui",
            AppName::Simulation => "simulation",
        }
    }
}

/// Long-running browser app server mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AppServeMode {
    /// Vite development server.
    Dev,
    /// Vite preview server for built output.
    Preview,
}

impl AppServeMode {
    /// Command label used in console output.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            AppServeMode::Dev => "dev",
            AppServeMode::Preview => "preview",
        }
    }
}

/// Standalone llama.cpp run workflows.
#[derive(Subcommand)]
pub enum RunLlamaCommands {
    /// Build and run llama.cpp test-backend-ops.
    BackendOps(RunLlamaBackendOpsArgs),
}

/// Options for llama.cpp backend operation tests.
#[derive(Args)]
pub struct RunLlamaBackendOpsArgs {
    /// Backend to compile and exercise.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// Diagnostic test-backend-ops mode.
    #[arg(long, value_enum, default_value = "support")]
    pub mode: LlamaBackendOpsMode,

    /// Operation filter passed as `-o`.
    #[arg(long)]
    pub op: Option<String>,

    /// Parameter regex passed as `-p`.
    #[arg(long)]
    pub params: Option<String>,

    /// test-backend-ops output format.
    #[arg(long, value_enum, default_value = "console")]
    pub output: LlamaBackendOpsOutput,
}

/// llama.cpp test-backend-ops mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum LlamaBackendOpsMode {
    /// Compare backend correctness against CPU reference.
    Test,
    /// Compare gradients against finite differences.
    Grad,
    /// Run performance measurements.
    Perf,
    /// Probe operation support.
    Support,
}

impl LlamaBackendOpsMode {
    /// CLI argument expected by test-backend-ops.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            LlamaBackendOpsMode::Test => "test",
            LlamaBackendOpsMode::Grad => "grad",
            LlamaBackendOpsMode::Perf => "perf",
            LlamaBackendOpsMode::Support => "support",
        }
    }
}

/// llama.cpp test-backend-ops output format.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum LlamaBackendOpsOutput {
    /// Human-readable console output.
    Console,
    /// CSV output.
    Csv,
    /// SQL output.
    Sql,
}

impl LlamaBackendOpsOutput {
    /// CLI argument expected by test-backend-ops.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            LlamaBackendOpsOutput::Console => "console",
            LlamaBackendOpsOutput::Csv => "csv",
            LlamaBackendOpsOutput::Sql => "sql",
        }
    }
}

/// Cleanup options for generated workspace artifacts.
#[derive(Args)]
pub struct CleanArgs {
    /// Also delete root/app/package/node binding node_modules directories.
    #[arg(long)]
    pub purge: bool,

    /// Also delete xtask-managed toolchains under `.build/toolchain`.
    #[arg(long)]
    pub toolchains: bool,

    /// Print paths that would be deleted without removing anything.
    #[arg(long)]
    pub dry_run: bool,
}

/// Toolchain management operations.
#[derive(Subcommand)]
pub enum ToolchainCommands {
    /// Print installed/missing status for all relevant developer toolchains.
    Status,

    /// Bootstrap an xtask-managed toolchain component.
    Install {
        /// Component to install.
        #[arg(value_enum)]
        component: ToolchainComponent,
    },
}

/// Toolchain components that xtask can bootstrap.
#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ToolchainComponent {
    /// Install every xtask-managed toolchain.
    All,
    /// Install the hermetic uv executable.
    Uv,
    /// Install Ninja on platforms where xtask manages it.
    Ninja,
    /// Install and activate Emscripten SDK.
    Emsdk,
    /// Install the hermetic Vulkan SDK.
    Vulkan,
}

/// Readiness-check options for `doctor`.
#[derive(Args)]
pub struct DoctorArgs {
    /// Build target to focus the check on.
    #[arg(long, value_enum, default_value = "all")]
    pub target: DoctorTarget,

    /// Backend to include in binding readiness checks.
    #[arg(long, value_enum, default_value = "all")]
    pub backend: Backend,
}

/// Setup guide options.
#[derive(Args)]
pub struct SetupArgs {
    /// Setup profile to use without asking for the target workflow.
    #[arg(long, value_enum)]
    pub profile: Option<SetupProfile>,

    /// Accept recommended setup actions without prompting.
    #[arg(long)]
    pub yes: bool,

    /// Do not download toolchains, dependencies, or sample models.
    #[arg(long)]
    pub no_downloads: bool,

    /// Skip the animated setup splash.
    #[arg(long)]
    pub no_splash: bool,
}

/// Developer setup profiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SetupProfile {
    /// Browser package and app development.
    Browser,
    /// Native Node/Python binding development.
    Bindings,
    /// Full workspace development, including browser and native bindings.
    Full,
}

impl SetupProfile {
    /// Lowercase label used in command output.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            SetupProfile::Browser => "browser",
            SetupProfile::Bindings => "bindings",
            SetupProfile::Full => "full",
        }
    }
}

impl fmt::Display for SetupProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            SetupProfile::Browser => "Browser apps and WebAssembly",
            SetupProfile::Bindings => "Node/Python bindings",
            SetupProfile::Full => "Full workspace",
        })
    }
}

/// Target scope for doctor checks.
#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum DoctorTarget {
    /// Check the full developer build surface.
    All,
    /// Check native Rust workspace prerequisites.
    Core,
    /// Check WASM/browser package prerequisites.
    Wasm,
    /// Check Node binding prerequisites.
    Node,
    /// Check Python binding prerequisites.
    Python,
}

/// Shared backend selection flags for native target builds.
#[derive(Args)]
pub struct BackendArgs {
    /// Computation backend to compile against.
    #[arg(long, short, value_enum)]
    pub backend: Option<Backend>,
}

/// Hardware backend selected for native target builds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
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
