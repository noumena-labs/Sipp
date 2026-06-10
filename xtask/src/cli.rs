//! CLI definitions for the xtask build orchestrator.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::fmt;
use std::path::PathBuf;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/cli_tests.rs"]
mod cli_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

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
  cargo xtask run examples serve browser
  cargo xtask run examples serve gateway-local --model .build/models/model.gguf
  cargo xtask run examples gateway rust
  cargo xtask run tools serve playground
  cargo xtask test --help
  cargo xtask test unit group full
  cargo xtask test smoke group examples --backend cpu
  cargo xtask build node --backend cpu
  cargo xtask build python --backend cuda
  cargo xtask build cli --backend all
  cargo xtask build gateway-server --backend cpu
  cargo xtask run gateway-server check --config apps/gateway-server/config/development.toml
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
  cargo xtask build gateway-server --backend vulkan

Notes:
  `build all` builds every target family with default CPU native outputs.
  It does not build every Node/Python/CLI/Gateway backend variant.";

const BACKEND_HELP: &str = "\
Backend values:
  cpu     Portable default backend
  cuda    NVIDIA CUDA backend; requires a local CUDA Toolkit
  metal   Apple Metal backend on macOS
  vulkan  Vulkan backend; xtask bootstraps the Vulkan SDK when needed
  all     Host-supported backend set for the target";

const RUN_HELP: &str = "\
Run long-lived demos and non-test diagnostics from the workspace root.

Examples:
  cargo xtask run demos build chat
  cargo xtask run demos serve avatar --port 5173
  cargo xtask run examples serve browser --port 5173
  cargo xtask run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
  cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
  cargo xtask run examples gateway rust --case query
  cargo xtask run examples gateway web --port 5173
  cargo xtask run tools build playground
  cargo xtask run tools serve playground --mode preview --port 4173
  cargo xtask run gateway-server serve --config apps/gateway-server/config/development.toml --backend cpu
  cargo xtask run llama backend-ops --backend cpu --mode support
  cargo xtask run llama backend-ops --backend cuda --mode perf --op MUL_MAT

Notes:
  Test execution, smoke checks, and coverage live under `cargo xtask test`.
  Serve commands are intentionally long-running and start Vite servers.
  Playground validation lives under `cargo xtask test smoke suite playground-browser`.";

const RUN_GATEWAY_SERVER_HELP: &str = "\
Check or serve the source-built standalone gateway-server.

Examples:
  cargo xtask run gateway-server check --config apps/gateway-server/config/development.toml
  cargo xtask run gateway-server serve --config apps/gateway-server/config/development.toml --backend cpu
  cargo xtask run gateway-server serve --config apps/gateway-server/config/production.toml --backend vulkan

`check` builds the staged gateway binary and validates TOML without loading
targets or reading secrets. `serve` builds the staged binary, loads the
selected config, and runs it from .build/artifacts/gateway-server.

Use raw Docker Compose commands for container workflows; start from the
checked-in apps/gateway-server/*.yml.example templates.";

const RUN_DEMOS_HELP: &str = "\
Build or serve individual browser demos.

Examples:
  cargo xtask run demos build chat
  cargo xtask run demos build simulation
  cargo xtask run demos serve chat
  cargo xtask run demos serve avatar --mode preview --port 4173

The demos group has no demo-level `all` command. Build and serve commands
target one demo at a time. Demo tests live under `cargo xtask test unit suite demos`.";

const RUN_LLAMA_HELP: &str = "\
Build standalone llama.cpp targets and run backend operation checks.

Examples:
  cargo xtask run llama backend-ops
  cargo xtask run llama backend-ops --backend cpu
  cargo xtask run llama backend-ops --backend vulkan --mode support
  cargo xtask run llama backend-ops --backend cuda --mode perf --op MUL_MAT

Correctness mode lives under `cargo xtask test smoke suite llama-backend-ops`.
The run command defaults to support probing.";

const RUN_EXAMPLES_HELP: &str = "\
Serve onboarding examples.

Examples:
  cargo xtask run examples serve browser
  cargo xtask run examples serve browser --mode preview --port 4173
  cargo xtask run examples serve gateway-local --model .build/models/model.gguf --bind 127.0.0.1:8787
  cargo xtask run examples serve gateway-openai --bind 127.0.0.1:8787
  cargo xtask run examples gateway rust --case query
  cargo xtask run examples gateway node --case chat
  cargo xtask run examples gateway python --case embed
  cargo xtask run examples gateway web

The browser example lives under examples/web and mirrors the public CogentClient
query, chat, embed, and gateway examples. Gateway commands start a real gateway
process from examples/gateway-style configs. The `gateway` workflow starts the
local gateway and a selected Rust, Node, Python, or web client in one terminal;
when --model is omitted it uses the cached sample model under .build/models.
`gateway-openai` requires OPENAI_API_KEY. Browser example smoke lives under
`cargo xtask test smoke suite example-browser`.";

const RUN_TOOLS_HELP: &str = "\
Build or serve developer tools.

Examples:
  cargo xtask run tools build playground
  cargo xtask run tools serve playground
  cargo xtask run tools serve playground --mode preview --port 4173

The browser playground lives under tools/playground. Playground smoke stays in
the test namespace: `cargo xtask test smoke suite playground-browser`.";

const SMOKE_HELP: &str = "\
Run holistic smoke checks through explicit suite and group namespaces.

Examples:
  cargo xtask test smoke suite example-node --backend cpu --case query
  cargo xtask test smoke suite example-gateway --backend cpu --case chat
  cargo xtask test smoke suite example-browser --case chat
  cargo xtask test smoke suite playground-browser --require-webgpu
  cargo xtask test smoke group examples --backend cpu
  cargo xtask test smoke group local-model --backend cpu
  cargo xtask test smoke group full --backend cpu

Use `suite` for one concrete smoke target and `group` for a named bundle.
Playground serving lives under `cargo xtask run tools serve playground`.";

const SMOKE_SUITE_HELP: &str = "\
Run exactly one smoke suite.

Suites and code locations:
  cli                 staged CLI built from apps/cli
  example-rust        examples/rust query/chat/embed binaries
  example-node        examples/node query.mjs/chat.mjs/embed.mjs
  example-python      examples/python query.py/chat.py/embed.py
  example-gateway     embedded local gateway proxy plus local/gateway clients
  example-browser     examples/web query/chat/embed pages through Playwright
  playground-browser  tools/playground runtime smoke through Playwright
  llama-backend-ops   third_party/llama.cpp test-backend-ops";

const SMOKE_GROUP_HELP: &str = "\
Run a named bundle of smoke suites.

Groups:
  examples     Rust, Node, Python, gateway, and browser onboarding examples
  local-model  CLI plus Rust, Node, and Python local model smoke
  full         every smoke suite, including playground, gateway, and llama checks";

const UNIT_HELP: &str = "\
Run deterministic tests through explicit suite and group namespaces.

Examples:
  cargo xtask test unit suite xtask
  cargo xtask test unit suite rust-crates --package cogentlm-core
  cargo xtask test unit suite node-package --backend cpu
  cargo xtask test unit group whitebox
  cargo xtask test unit group interface
  cargo xtask test unit group full

Use `suite` for one concrete unit test suite and `group` for a named bundle.";

const UNIT_SUITE_HELP: &str = "\
Run exactly one deterministic unit suite.

Suites and code locations:
  xtask             xtask CLI and orchestration tests under xtask/src/tests
  rust-crates       core workspace crate unit tests under crates/ and lib/rust
  rust-bindings     Rust tests for Node, Python, and WASM binding crates
  browser-package   browser package TypeScript tests under lib/web/tests
  demos             browser demo TypeScript tests under demos/
  api               crate-level public API integration tests
  cli               CLI black-box integration tests
  node-package      deterministic Node package API tests
  python-package    deterministic Python package API tests";

const UNIT_GROUP_HELP: &str = "\
Run a named bundle of deterministic unit suites.

Groups:
  whitebox   internal code-flow suites: xtask, rust-crates, rust-bindings, browser-package, demos
  interface  public API and binding package suites: api, cli, node-package, python-package
  full       every deterministic unit suite";

const TEST_HELP: &str = "\
List, run, and verify cataloged workspace tests.

Examples:
  cargo xtask test list
  cargo xtask test list --group unit --layer interface --cases --search router --format json
  cargo xtask test unit group full
  cargo xtask test unit group whitebox
  cargo xtask test unit suite rust-crates --package cogentlm-core
  cargo xtask test unit suite node-package --backend cpu
  cargo xtask test smoke suite example-node --backend cpu
  cargo xtask test smoke suite playground-browser
  cargo xtask test smoke group local-model --backend cpu
  cargo xtask test verify --changed
  cargo xtask test verify --target public-docs

Model-backed smoke tests default to the setup sample model cache under
.build/models when --model is omitted.

`test unit suite` runs one deterministic suite. `test unit group` runs a named bundle.
`test smoke suite` runs one smoke suite. `test smoke group` runs a named bundle.
`test unit` and `test smoke` execute suites and write .build/test artifacts.
Coverage-capable unit suites also write .build/coverage artifacts.
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

    /// Disable bounded inline rendering when compact output would otherwise be active.
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

By default, clean removes build outputs and generated demo/package output while
preserving downloaded toolchains and dependency installs. Use `--purge` to also
remove workspace node_modules directories.")]
    Clean(CleanArgs),

    /// Run long-lived demos and non-test diagnostics.
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

    /// Inspect, bootstrap, or configure toolchains.
    #[command(arg_required_else_help = true)]
    #[command(long_about = "\
Inspect, bootstrap, or configure developer toolchains.

Examples:
  cargo xtask toolchain status
  cargo xtask toolchain install uv
  cargo xtask toolchain install all
  cargo xtask toolchain setup cuda

CUDA is externally installed and is reported by status/doctor, but xtask never
installs or deletes it. Use `toolchain setup cuda` to configure which CUDA
architectures the build pipeline compiles for.")]
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

    /// Build and serve the documentation book.
    #[command(long_about = "\
Build or serve the mdBook documentation with mermaid diagram support.

Examples:
  cargo xtask docs build
  cargo xtask docs serve

The command installs mdbook and mdbook-mermaid when missing, extracts the
bundled mermaid JavaScript assets into theme/, then builds or serves the
book with hot-reload.")]
    #[command(arg_required_else_help = true)]
    Docs {
        /// Docs workflow to run.
        #[command(subcommand)]
        command: DocsCommands,
    },
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

    /// Build the standalone gateway-server distribution directory.
    #[command(long_about = "\
Build the gateway-server binary and stage a runnable distribution directory.

Examples:
  cargo xtask build gateway-server
  cargo xtask build gateway-server --backend cpu
  cargo xtask build gateway-server --backend cuda
  cargo xtask build gateway-server --backend vulkan
  cargo xtask build gateway-server --backend all

The gateway-server build uses llama.cpp dynamic backend loading. The staged
artifact contains cogentlm-gateway, base runtime libraries, and selected ggml
backend plugins in .build/artifacts/gateway-server.")]
    #[command(after_long_help = BACKEND_HELP)]
    GatewayServer(BackendArgs),
}

/// Developer run workflows.
#[derive(Subcommand)]
pub enum RunCommands {
    /// Check or serve the source-built standalone gateway-server.
    #[command(long_about = RUN_GATEWAY_SERVER_HELP)]
    #[command(after_long_help = BACKEND_HELP)]
    GatewayServer(RunGatewayServerArgs),

    /// Build or serve browser demos.
    #[command(long_about = RUN_DEMOS_HELP)]
    #[command(arg_required_else_help = true)]
    Demos {
        /// Demo workflow to run.
        #[command(subcommand)]
        command: RunDemosCommands,
    },

    /// Serve onboarding examples.
    #[command(long_about = RUN_EXAMPLES_HELP)]
    #[command(arg_required_else_help = true)]
    Examples {
        /// Example workflow to run.
        #[command(subcommand)]
        command: RunExamplesCommands,
    },

    /// Build or serve developer tools.
    #[command(long_about = RUN_TOOLS_HELP)]
    #[command(arg_required_else_help = true)]
    Tools {
        /// Tool workflow to run.
        #[command(subcommand)]
        command: RunToolsCommands,
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

/// Source gateway-server workflows.
#[derive(Args)]
pub struct RunGatewayServerArgs {
    /// Gateway-server source workflow to run.
    #[command(subcommand)]
    pub command: RunGatewayServerCommand,
}

/// Source gateway-server command variants.
#[derive(Subcommand)]
pub enum RunGatewayServerCommand {
    /// Validate a TOML config without loading targets or reading secrets.
    Check(RunGatewayServerSourceArgs),
    /// Build and run the gateway-server from a TOML config.
    Serve(RunGatewayServerSourceArgs),
}

/// Shared source gateway-server options.
#[derive(Args)]
pub struct RunGatewayServerSourceArgs {
    /// Path to the gateway-server TOML config.
    #[arg(long, default_value = "apps/gateway-server/config/development.toml")]
    pub config: PathBuf,

    /// Native backend variant to compile into the staged gateway distribution.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,
}

/// Workspace test workflows.
#[derive(Subcommand)]
pub enum TestCommands {
    /// List known test suites and optionally discover test cases.
    List(TestListArgs),

    /// Run deterministic code-flow and API-layer tests.
    #[command(long_about = UNIT_HELP)]
    #[command(arg_required_else_help = true)]
    Unit(TestUnitArgs),

    /// Run holistic integration smoke tests.
    #[command(long_about = SMOKE_HELP)]
    #[command(arg_required_else_help = true)]
    Smoke(TestSmokeArgs),

    /// Verify test structure and existing coverage artifacts.
    Verify(TestVerifyArgs),
}

/// Options for listing test suites and cases.
#[derive(Args)]
pub struct TestListArgs {
    /// Suite group to include in the listing.
    #[arg(long, value_enum, default_value = "all")]
    pub group: TestGroupFilter,

    /// Unit layer to include in the listing.
    #[arg(long, value_enum)]
    pub layer: Option<TestUnitLayer>,

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

/// Options for deterministic unit test workflows.
#[derive(Args)]
pub struct TestUnitArgs {
    /// Unit namespace to run.
    #[command(subcommand)]
    pub command: TestUnitCommands,
}

/// Unit command namespaces.
#[derive(Subcommand)]
pub enum TestUnitCommands {
    /// Run exactly one deterministic unit suite.
    #[command(long_about = UNIT_SUITE_HELP)]
    #[command(arg_required_else_help = true)]
    Suite(TestUnitSuiteArgs),
    /// Run a named bundle of deterministic unit suites.
    #[command(long_about = UNIT_GROUP_HELP)]
    #[command(arg_required_else_help = true)]
    Group(TestUnitGroupArgs),
}

/// Options for one deterministic unit suite.
#[derive(Args)]
pub struct TestUnitSuiteArgs {
    /// Concrete unit suite to run.
    #[command(subcommand)]
    pub target: TestUnitSuiteTarget,
}

/// Concrete deterministic unit suite targets.
#[derive(Subcommand)]
pub enum TestUnitSuiteTarget {
    /// Run xtask CLI and orchestration unit tests.
    Xtask,
    /// Run Rust unit tests for core workspace crates.
    #[command(name = "rust-crates")]
    RustCrates(TestUnitRustArgs),
    /// Run Rust unit tests for language binding crates.
    #[command(name = "rust-bindings")]
    RustBindings,
    /// Run browser package TypeScript tests.
    #[command(name = "browser-package")]
    BrowserPackage,
    /// Run browser demo TypeScript tests.
    Demos,
    /// Run crate-level public API integration tests.
    Api,
    /// Run CLI black-box integration tests.
    Cli,
    /// Run deterministic Node package API tests.
    #[command(name = "node-package")]
    #[command(after_long_help = BACKEND_HELP)]
    NodePackage(TestUnitBackendArgs),
    /// Run deterministic Python package API tests.
    #[command(name = "python-package")]
    #[command(after_long_help = BACKEND_HELP)]
    PythonPackage(TestUnitBackendArgs),
}

/// Options for one deterministic unit group.
#[derive(Args)]
pub struct TestUnitGroupArgs {
    /// Unit group to run.
    #[command(subcommand)]
    pub target: TestUnitGroupTarget,
}

/// Named deterministic unit suite groups.
#[derive(Subcommand)]
pub enum TestUnitGroupTarget {
    /// Run all white-box unit suites.
    Whitebox,
    /// Run all public API/interface unit suites.
    Interface,
    /// Run every deterministic unit suite.
    Full,
}

/// Options for Rust unit tests.
#[derive(Args)]
pub struct TestUnitRustArgs {
    /// Rust package filter for the `rust-crates` suite.
    #[arg(long)]
    pub package: Option<String>,
}

/// Backend options for deterministic binding unit tests.
#[derive(Args)]
pub struct TestUnitBackendArgs {
    /// Backend passed to backend-aware suites.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,
}

/// Options for smoke test workflows.
#[derive(Args)]
pub struct TestSmokeArgs {
    /// Smoke namespace to run.
    #[command(subcommand)]
    pub command: TestSmokeCommands,
}

/// Smoke command namespaces.
#[derive(Subcommand)]
pub enum TestSmokeCommands {
    /// Run exactly one smoke suite.
    #[command(long_about = SMOKE_SUITE_HELP)]
    #[command(arg_required_else_help = true)]
    Suite(TestSmokeSuiteArgs),
    /// Run a named bundle of smoke suites.
    #[command(long_about = SMOKE_GROUP_HELP)]
    #[command(arg_required_else_help = true)]
    Group(TestSmokeGroupArgs),
}

/// Options for one smoke suite.
#[derive(Args)]
pub struct TestSmokeSuiteArgs {
    /// Concrete smoke suite to run.
    #[command(subcommand)]
    pub target: TestSmokeSuiteTarget,
}

/// Concrete smoke suite targets.
#[derive(Subcommand)]
pub enum TestSmokeSuiteTarget {
    /// Run model-backed CLI smoke from apps/cli.
    #[command(after_long_help = BACKEND_HELP)]
    Cli(TestSmokeModelArgs),
    /// Run model-backed Rust query/chat/embed examples under examples/rust.
    #[command(name = "example-rust")]
    #[command(after_long_help = BACKEND_HELP)]
    ExampleRust(TestSmokeCaseArgs),
    /// Run model-backed Node query/chat/embed examples under examples/node.
    #[command(name = "example-node")]
    #[command(after_long_help = BACKEND_HELP)]
    ExampleNode(TestSmokeCaseArgs),
    /// Run model-backed Python query/chat/embed examples under examples/python.
    #[command(name = "example-python")]
    #[command(after_long_help = BACKEND_HELP)]
    ExamplePython(TestSmokeCaseArgs),
    /// Run embedded local gateway proxy examples and local/gateway clients.
    #[command(name = "example-gateway")]
    #[command(after_long_help = BACKEND_HELP)]
    ExampleGateway(TestSmokeCaseArgs),
    /// Run browser query/chat/embed examples under examples/web through Playwright.
    #[command(name = "example-browser")]
    ExampleBrowser(TestSmokeExampleBrowserArgs),
    /// Run playground browser runtime smoke under tools/playground.
    #[command(name = "playground-browser")]
    PlaygroundBrowser(TestSmokePlaygroundBrowserArgs),
    /// Run llama.cpp backend operation smoke.
    #[command(name = "llama-backend-ops")]
    #[command(after_long_help = BACKEND_HELP)]
    LlamaBackendOps(TestSmokeLlamaArgs),
}

/// Options for one smoke group.
#[derive(Args)]
pub struct TestSmokeGroupArgs {
    /// Smoke group to run.
    #[command(subcommand)]
    pub target: TestSmokeGroupTarget,
}

/// Named smoke suite groups.
#[derive(Subcommand)]
pub enum TestSmokeGroupTarget {
    /// Run Rust, Node, Python, and browser onboarding example smoke suites.
    #[command(after_long_help = BACKEND_HELP)]
    Examples(TestSmokeExamplesGroupArgs),
    /// Run CLI plus Rust, Node, and Python local model smoke suites.
    #[command(name = "local-model")]
    #[command(after_long_help = BACKEND_HELP)]
    LocalModel(TestSmokeModelArgs),
    /// Run every smoke suite.
    #[command(after_long_help = BACKEND_HELP)]
    Full(TestSmokeFullGroupArgs),
}

/// Model-backed smoke options.
#[derive(Args, Clone)]
pub struct TestSmokeModelArgs {
    /// Backend passed to model-backed smoke tests.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// GGUF model for model-backed suites. Defaults to .build/models.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Do not download the default sample model when model-backed suites have no --model.
    #[arg(long)]
    pub offline: bool,

    /// Prompt passed to local generation smoke tests.
    #[arg(long, default_value = "Describe browser LLM inference.")]
    pub prompt: String,

    /// Maximum generated tokens for local generation smoke tests.
    #[arg(long, default_value = "64")]
    pub max_tokens: u32,

    /// Sampling temperature for local generation smoke tests.
    #[arg(long, default_value = "0")]
    pub temperature: f32,
}

/// Model-backed smoke options with query/chat/embed case selection.
#[derive(Args, Clone)]
pub struct TestSmokeCaseArgs {
    /// Model-backed smoke options.
    #[command(flatten)]
    pub model: TestSmokeModelArgs,

    /// Example case to run. Repeat to include multiple cases.
    #[arg(long = "case", value_enum)]
    pub cases: Vec<TestSmokeCase>,
}

/// Options for browser example smoke.
#[derive(Args)]
pub struct TestSmokeExampleBrowserArgs {
    /// GGUF model for browser example smoke. Defaults to .build/models.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Do not download the default sample model when --model is omitted.
    #[arg(long)]
    pub offline: bool,

    /// Prompt passed to browser query/chat/embed examples.
    #[arg(long, default_value = "Describe browser LLM inference.")]
    pub prompt: String,

    /// Maximum generated tokens for browser query/chat examples.
    #[arg(long, default_value = "64")]
    pub max_tokens: u32,

    /// Browser example case to run. Repeat to include multiple cases.
    #[arg(long = "case", value_enum)]
    pub cases: Vec<TestSmokeCase>,

    /// Host used for the examples/web Vite server.
    #[arg(long)]
    pub host: Option<String>,

    /// Port used for the examples/web Vite server.
    #[arg(long)]
    pub port: Option<u16>,

    /// Browser example smoke timeout in milliseconds.
    #[arg(long, default_value = "30000")]
    pub timeout_ms: u64,
}

/// Options for the examples smoke group.
#[derive(Args)]
pub struct TestSmokeExamplesGroupArgs {
    /// Model-backed example options.
    #[command(flatten)]
    pub cases: TestSmokeCaseArgs,

    /// Host used for the examples/web Vite server.
    #[arg(long)]
    pub browser_host: Option<String>,

    /// Port used for the examples/web Vite server.
    #[arg(long)]
    pub browser_port: Option<u16>,

    /// Browser example smoke timeout in milliseconds.
    #[arg(long, default_value = "30000")]
    pub browser_timeout_ms: u64,
}

/// Options for the full smoke group.
#[derive(Args)]
pub struct TestSmokeFullGroupArgs {
    /// Model-backed smoke options shared by CLI and example smoke tests.
    #[command(flatten)]
    pub model: TestSmokeModelArgs,

    /// Browser example smoke timeout in milliseconds.
    #[arg(long, default_value = "30000")]
    pub example_browser_timeout_ms: u64,

    /// Playground browser smoke timeout in milliseconds.
    #[arg(long, default_value = "30000")]
    pub playground_browser_timeout_ms: u64,
}

/// Model-backed example smoke cases.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestSmokeCase {
    /// Query/text generation example.
    Query,
    /// Chat generation example.
    Chat,
    /// Embedding example.
    Embed,
}

impl TestSmokeCase {
    /// Stable smoke case label.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            TestSmokeCase::Query => "query",
            TestSmokeCase::Chat => "chat",
            TestSmokeCase::Embed => "embed",
        }
    }
}

/// Playground browser smoke options.
#[derive(Args)]
pub struct TestSmokePlaygroundBrowserArgs {
    /// Host used for the playground Vite server.
    #[arg(long)]
    pub host: Option<String>,

    /// Port used for the playground Vite server.
    #[arg(long)]
    pub port: Option<u16>,

    /// Browser smoke timeout in milliseconds.
    #[arg(long, default_value = "30000")]
    pub timeout_ms: u64,

    /// Require the Rust browser engine smoke to pass.
    #[arg(long)]
    pub require_rust_engine: bool,

    /// Require browser GGUF ingest smoke to pass.
    #[arg(long)]
    pub require_gguf_ingest: bool,

    /// Require WebGPU backend readiness.
    #[arg(long)]
    pub require_webgpu: bool,
}

/// Options for llama.cpp backend operation smoke tests.
#[derive(Args)]
pub struct TestSmokeLlamaArgs {
    /// Backend to compile and exercise.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// Diagnostic test-backend-ops mode.
    #[arg(long, value_enum, default_value = "test")]
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

/// Options for test and coverage verification.
#[derive(Args)]
pub struct TestVerifyArgs {
    /// Unit target to verify.
    #[arg(long, value_enum, default_value = "all")]
    pub target: TestVerifyTarget,

    /// Validate that changed source files have matching catalog-owned test changes.
    #[arg(long)]
    pub changed: bool,
}

/// Suite group filter for test listing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestGroupFilter {
    /// Include unit and smoke suites.
    All,
    /// Include deterministic unit suites.
    Unit,
    /// Include holistic smoke suites.
    Smoke,
}

impl TestGroupFilter {
    /// Stable group filter label.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestGroupFilter::All => "all",
            TestGroupFilter::Unit => "unit",
            TestGroupFilter::Smoke => "smoke",
        }
    }
}

/// Unit test layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestUnitLayer {
    /// Implementation-oriented white-box suites.
    Whitebox,
    /// Public API and interface suites.
    Interface,
}

impl TestUnitLayer {
    /// Stable unit layer label.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestUnitLayer::Whitebox => "whitebox",
            TestUnitLayer::Interface => "interface",
        }
    }
}

/// Coverage verification target.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum TestVerifyTarget {
    /// Verify all coverage-capable unit suites.
    All,
    /// Verify coverage-capable white-box unit suites.
    Whitebox,
    /// Verify coverage-capable interface unit suites.
    Interface,
    /// Verify xtask unit coverage.
    Xtask,
    /// Verify Rust crate unit coverage.
    Rust,
    /// Verify binding crate unit coverage.
    Bindings,
    /// Verify browser package unit coverage.
    BrowserPackage,
    /// Verify demo TypeScript unit coverage.
    Demos,
    /// Verify crate-level public API coverage.
    Api,
    /// Verify CLI black-box coverage.
    Cli,
    /// Verify Node package API coverage.
    Node,
    /// Verify Python package API coverage.
    Python,
    /// Verify curated public API documentation comments.
    PublicDocs,
}

impl TestVerifyTarget {
    /// Stable verification target label.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestVerifyTarget::All => "all",
            TestVerifyTarget::Whitebox => "whitebox",
            TestVerifyTarget::Interface => "interface",
            TestVerifyTarget::Xtask => "xtask",
            TestVerifyTarget::Rust => "rust",
            TestVerifyTarget::Bindings => "bindings",
            TestVerifyTarget::BrowserPackage => "browser-package",
            TestVerifyTarget::Demos => "demos",
            TestVerifyTarget::Api => "api",
            TestVerifyTarget::Cli => "cli",
            TestVerifyTarget::Node => "node",
            TestVerifyTarget::Python => "python",
            TestVerifyTarget::PublicDocs => "public-docs",
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
    /// Browser demo TypeScript tests.
    DemoTs,
    /// Rust public API integration tests.
    RustPublicApi,
    /// CLI black-box integration tests.
    Cli,
    /// Node package interface tests.
    NodePackage,
    /// Python package interface tests.
    PythonPackage,
    /// CLI local generation smoke tests.
    CliSmoke,
    /// Rust local generation smoke tests.
    RustSmoke,
    /// Node local generation smoke tests.
    NodeSmoke,
    /// Python local generation smoke tests.
    PythonSmoke,
    /// Real local gateway example smoke tests.
    ExampleGatewaySmoke,
    /// Browser example smoke tests.
    ExampleBrowserSmoke,
    /// Playground browser runtime smoke tests.
    PlaygroundBrowserSmoke,
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
            TestSuiteId::DemoTs => "demo-ts",
            TestSuiteId::RustPublicApi => "rust-public-api",
            TestSuiteId::Cli => "cli",
            TestSuiteId::NodePackage => "node-package",
            TestSuiteId::PythonPackage => "python-package",
            TestSuiteId::CliSmoke => "cli-smoke",
            TestSuiteId::RustSmoke => "rust-smoke",
            TestSuiteId::NodeSmoke => "node-smoke",
            TestSuiteId::PythonSmoke => "python-smoke",
            TestSuiteId::ExampleGatewaySmoke => "example-gateway-smoke",
            TestSuiteId::ExampleBrowserSmoke => "example-browser-smoke",
            TestSuiteId::PlaygroundBrowserSmoke => "playground-browser-smoke",
            TestSuiteId::LlamaBackendOps => "llama-backend-ops",
        }
    }
}

/// Browser demo run workflows.
#[derive(Subcommand)]
pub enum RunDemosCommands {
    /// Build one browser demo.
    Build(RunDemoBuildArgs),

    /// Start one long-running Vite dev or preview server.
    Serve(RunDemoServeArgs),
}

/// Options for building a browser demo.
#[derive(Args)]
pub struct RunDemoBuildArgs {
    /// Demo to build.
    #[arg(value_enum)]
    pub demo: DemoName,
}

/// Options for serving a browser demo.
#[derive(Args)]
pub struct RunDemoServeArgs {
    /// Demo to serve.
    #[arg(value_enum)]
    pub demo: DemoName,

    /// Vite server mode to run.
    #[arg(long, value_enum, default_value = "dev")]
    pub mode: DemoServeMode,

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

/// Browser demos known to xtask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum DemoName {
    /// Browser avatar demo.
    Avatar,
    /// Browser chat demo.
    Chat,
    /// Browser proactive UI demo.
    ProactiveUi,
    /// Browser simulation demo.
    Simulation,
}

impl DemoName {
    /// Directory name under `demos`.
    pub(crate) fn slug(&self) -> &'static str {
        match self {
            DemoName::Avatar => "avatar",
            DemoName::Chat => "chat",
            DemoName::ProactiveUi => "proactive-ui",
            DemoName::Simulation => "simulation",
        }
    }
}

/// Example run workflows.
#[derive(Subcommand)]
pub enum RunExamplesCommands {
    /// Start one long-running example server.
    Serve(RunExampleServeArgs),

    /// Start a local gateway and run one gateway client workflow.
    #[command(after_long_help = BACKEND_HELP)]
    Gateway(RunGatewayExampleArgs),
}

/// Options for serving an example.
#[derive(Args)]
pub struct RunExampleServeArgs {
    /// Example server to start.
    #[command(subcommand)]
    pub target: RunExampleServeTarget,
}

/// Options for running local gateway examples.
#[derive(Args)]
pub struct RunGatewayExampleArgs {
    /// Gateway client workflow to run.
    #[command(subcommand)]
    pub target: RunGatewayExampleTarget,
}

/// Gateway example client workflows.
#[derive(Subcommand)]
pub enum RunGatewayExampleTarget {
    /// Start a gateway and run an examples/rust gateway binary.
    Rust(RunGatewayExampleClientArgs),
    /// Start a gateway and run an examples/node gateway script.
    Node(RunGatewayExampleClientArgs),
    /// Start a gateway and run an examples/python gateway script.
    Python(RunGatewayExampleClientArgs),
    /// Start a gateway and serve examples/web.
    Web(RunGatewayExampleWebArgs),
}

/// Shared options for local gateway example workflows.
#[derive(Args)]
pub struct RunGatewayExampleCommonArgs {
    /// GGUF model loaded by the gateway process and local client endpoint.
    /// Defaults to the cached sample model under .build/models.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Gateway client case to run.
    #[arg(long, value_enum, default_value = "query")]
    pub case: RunGatewayExampleCase,

    /// Gateway socket address.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub bind: String,

    /// Native backend used by the gateway process.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// Gateway bearer token used for the generated local gateway config.
    #[arg(long, default_value = "dev-token")]
    pub token: String,
}

/// Options for one-shot Rust, Node, and Python gateway clients.
#[derive(Args)]
pub struct RunGatewayExampleClientArgs {
    /// Shared gateway options.
    #[command(flatten)]
    pub common: RunGatewayExampleCommonArgs,

    /// Prompt or embedding input passed to the client example.
    #[arg(long, default_value = "Write one sentence about gateway inference.")]
    pub prompt: String,

    /// Maximum generated tokens for query and chat clients.
    #[arg(long, default_value = "128")]
    pub max_tokens: u32,

    /// Sampling temperature for query and chat clients.
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
}

/// Options for serving the web gateway example.
#[derive(Args)]
pub struct RunGatewayExampleWebArgs {
    /// Shared gateway options.
    #[command(flatten)]
    pub common: RunGatewayExampleCommonArgs,

    /// Vite server mode to run.
    #[arg(long, value_enum, default_value = "dev")]
    pub mode: DemoServeMode,

    /// Host passed through to Vite.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port passed through to Vite.
    #[arg(long, default_value = "5173")]
    pub port: u16,

    /// Start the server without first building browser package artifacts.
    #[arg(long)]
    pub no_build: bool,
}

/// Gateway example cases known to xtask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RunGatewayExampleCase {
    /// Query/text generation example.
    Query,
    /// Chat generation example.
    Chat,
    /// Embedding example.
    Embed,
}

impl RunGatewayExampleCase {
    /// Stable case label.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            RunGatewayExampleCase::Query => "query",
            RunGatewayExampleCase::Chat => "chat",
            RunGatewayExampleCase::Embed => "embed",
        }
    }
}

/// Example server targets.
#[derive(Subcommand)]
pub enum RunExampleServeTarget {
    /// Start the examples/web Vite dev or preview server.
    Browser(RunBrowserExampleServeArgs),
    /// Start a local-GGUF gateway for examples/gateway and gateway clients.
    #[command(name = "gateway-local")]
    #[command(after_long_help = BACKEND_HELP)]
    GatewayLocal(RunGatewayLocalServeArgs),
    /// Start an OpenAI-backed gateway for examples/gateway and gateway clients.
    #[command(name = "gateway-openai")]
    GatewayOpenAi(RunGatewayOpenAiServeArgs),
}

/// Options for serving the browser example.
#[derive(Args)]
pub struct RunBrowserExampleServeArgs {
    /// Vite server mode to run.
    #[arg(long, value_enum, default_value = "dev")]
    pub mode: DemoServeMode,

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

/// Options for serving the local gateway example.
#[derive(Args)]
pub struct RunGatewayLocalServeArgs {
    /// GGUF model loaded by the gateway process.
    #[arg(long)]
    pub model: PathBuf,

    /// Gateway socket address.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub bind: String,

    /// Native backend used by the gateway process.
    #[arg(long, short, value_enum, default_value = "cpu")]
    pub backend: Backend,
}

/// Options for serving the OpenAI gateway example.
#[derive(Args)]
pub struct RunGatewayOpenAiServeArgs {
    /// Gateway socket address.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub bind: String,

    /// Environment variable containing the gateway bearer token.
    #[arg(long, default_value = "COGENTLM_GATEWAY_TOKEN")]
    pub token_env: String,

    /// Environment variable containing the OpenAI API key.
    #[arg(long, default_value = "OPENAI_API_KEY")]
    pub api_key_env: String,

    /// OpenAI model used by the query/chat target.
    #[arg(long, default_value = "gpt-5-mini")]
    pub chat_model: String,

    /// OpenAI embedding model used by the embed target.
    #[arg(long, default_value = "text-embedding-3-small")]
    pub embed_model: String,
}

/// Browser examples known to xtask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ExampleName {
    /// Browser examples under examples/web.
    Browser,
}

impl ExampleName {
    /// Directory name under `examples`.
    pub(crate) fn dir_name(&self) -> &'static str {
        match self {
            ExampleName::Browser => "web",
        }
    }

    /// Command label used in console output.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ExampleName::Browser => "browser",
        }
    }
}

/// Developer tool run workflows.
#[derive(Subcommand)]
pub enum RunToolsCommands {
    /// Build one developer tool.
    Build(RunToolBuildArgs),

    /// Start one long-running Vite dev or preview server.
    Serve(RunToolServeArgs),
}

/// Options for building a developer tool.
#[derive(Args)]
pub struct RunToolBuildArgs {
    /// Tool to build.
    #[arg(value_enum)]
    pub tool: ToolName,
}

/// Options for serving a developer tool.
#[derive(Args)]
pub struct RunToolServeArgs {
    /// Tool to serve.
    #[arg(value_enum)]
    pub tool: ToolName,

    /// Vite server mode to run.
    #[arg(long, value_enum, default_value = "dev")]
    pub mode: DemoServeMode,

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

/// Developer tools known to xtask.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ToolName {
    /// Browser playground under tools/playground.
    Playground,
}

impl ToolName {
    /// Directory name under `tools`.
    pub(crate) fn slug(&self) -> &'static str {
        match self {
            ToolName::Playground => "playground",
        }
    }
}

/// Long-running browser demo server mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum DemoServeMode {
    /// Vite development server.
    Dev,
    /// Vite preview server for built output.
    Preview,
}

impl DemoServeMode {
    /// Command label used in console output.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            DemoServeMode::Dev => "dev",
            DemoServeMode::Preview => "preview",
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
    /// Also delete root/demo/tool/package/node binding node_modules directories.
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

    /// Configure an external toolchain component interactively.
    Setup {
        /// Component to configure.
        #[arg(value_enum)]
        component: ToolchainSetupComponent,
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

/// Toolchain components that xtask can configure interactively.
///
/// These are external components (not downloaded by xtask) that benefit from
/// guided setup — architecture selection, SDK registration, etc.
#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ToolchainSetupComponent {
    /// Select CUDA architectures to compile for.
    Cuda,
}

/// Docs workflow commands.
#[derive(Subcommand)]
pub enum DocsCommands {
    /// Build the documentation book.
    #[command(long_about = "\
Build the mdBook documentation with mermaid diagram support.

Installs mdbook and mdbook-mermaid when missing, extracts the bundled mermaid
JavaScript assets into theme/, and runs `mdbook build`.")]
    Build,
    /// Build and serve the documentation book with live reload.
    #[command(long_about = "\
Build the mdBook documentation and serve it with hot-reload.

Same setup as `build`, then runs `mdbook serve --open` to open the book in
the default browser.")]
    Serve,
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
    /// Browser package and demo development.
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
            SetupProfile::Browser => "Browser demos and WebAssembly",
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
