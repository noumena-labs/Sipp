//! Tests the `cli` module in `xtask`.
//!
//! Covers Clap parsing, enum labels, display strings, and rejected command
//! shapes without running any selected command.

use clap::Parser;

use super::{
    Backend, BenchmarkName, Cli, Commands, DemoName, DemoServeMode, DoctorTarget, ExampleName,
    LlamaBackendOpsMode, LlamaBackendOpsOutput, RunBenchmarksCommands, RunCommands,
    RunDemosCommands, RunExamplesCommands, RunLlamaCommands, SetupProfile, TestCommands,
    TestGroupFilter, TestListFormat, TestSmokeCommands, TestSmokeGroupTarget, TestSmokeSuiteTarget,
    TestSuiteId, TestUnitCommands, TestUnitGroupTarget, TestUnitLayer, TestUnitSuiteTarget,
    ToolchainCommands, ToolchainComponent,
};

#[test]
fn build_commands_parse_backend_defaults_and_overrides() {
    let cli = Cli::parse_from(["xtask", "build", "node"]);
    let Commands::Build { target } = cli.command else {
        panic!("expected build command");
    };
    let super::BuildCommands::Node(args) = target else {
        panic!("expected node build");
    };
    assert_eq!(args.backend, None);

    let cli = Cli::parse_from(["xtask", "build", "cli", "--backend", "all"]);
    let Commands::Build { target } = cli.command else {
        panic!("expected build command");
    };
    let super::BuildCommands::Cli(args) = target else {
        panic!("expected cli build");
    };
    assert_eq!(args.backend, Some(Backend::All));
}

#[test]
fn run_demo_serve_parses_optional_host_port_and_no_build() {
    let cli = Cli::parse_from([
        "xtask",
        "run",
        "demos",
        "serve",
        "chat",
        "--mode",
        "preview",
        "--host",
        "127.0.0.1",
        "--port",
        "4173",
        "--no-build",
    ]);

    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    let RunCommands::Demos { command } = command else {
        panic!("expected demos command");
    };
    let RunDemosCommands::Serve(args) = command else {
        panic!("expected serve args");
    };
    assert_eq!(args.demo, DemoName::Chat);
    assert_eq!(args.mode, DemoServeMode::Preview);
    assert_eq!(args.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(args.port, Some(4173));
    assert!(args.no_build);
}

#[test]
fn run_examples_and_benchmarks_parse_browser_workflows() {
    let cli = Cli::parse_from([
        "xtask",
        "run",
        "examples",
        "serve",
        "browser",
        "--mode",
        "preview",
        "--port",
        "4173",
        "--no-build",
    ]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    let RunCommands::Examples { command } = command else {
        panic!("expected examples command");
    };
    let RunExamplesCommands::Serve(args) = command;
    assert_eq!(args.example, ExampleName::Browser);
    assert_eq!(args.mode, DemoServeMode::Preview);
    assert_eq!(args.port, Some(4173));
    assert!(args.no_build);

    let cli = Cli::parse_from(["xtask", "run", "benchmarks", "build", "browser"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    let RunCommands::Benchmarks { command } = command else {
        panic!("expected benchmarks command");
    };
    let RunBenchmarksCommands::Build(args) = command else {
        panic!("expected benchmark build command");
    };
    assert_eq!(args.benchmark, BenchmarkName::Browser);
}

#[test]
fn run_llama_backend_ops_parses_filters() {
    let cli = Cli::parse_from([
        "xtask",
        "run",
        "llama",
        "backend-ops",
        "--backend",
        "cuda",
        "--mode",
        "perf",
        "--op",
        "MUL_MAT",
        "--params",
        "q4",
        "--output",
        "csv",
    ]);

    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    let RunCommands::Llama { command } = command else {
        panic!("expected llama command");
    };
    let RunLlamaCommands::BackendOps(args) = command;
    assert_eq!(args.backend, Backend::Cuda);
    assert_eq!(args.mode, LlamaBackendOpsMode::Perf);
    assert_eq!(args.op.as_deref(), Some("MUL_MAT"));
    assert_eq!(args.params.as_deref(), Some("q4"));
    assert_eq!(args.output, LlamaBackendOpsOutput::Csv);
}

#[test]
fn test_list_parses_json_cases_and_search() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "list",
        "--group",
        "unit",
        "--layer",
        "interface",
        "--cases",
        "--search",
        "router",
        "--format",
        "json",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::List(args) = command else {
        panic!("expected list command");
    };
    assert_eq!(args.group, TestGroupFilter::Unit);
    assert_eq!(args.layer, Some(TestUnitLayer::Interface));
    assert!(args.cases);
    assert_eq!(args.search.as_deref(), Some("router"));
    assert_eq!(args.format, TestListFormat::Json);
}

#[test]
fn test_unit_and_smoke_targets_parse() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "unit",
        "suite",
        "rust-crates",
        "--package",
        "cogentlm-core",
    ]);
    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Unit(args) = command else {
        panic!("expected unit command");
    };
    let TestUnitCommands::Suite(args) = args.command else {
        panic!("expected unit suite command");
    };
    let TestUnitSuiteTarget::RustCrates(args) = args.target else {
        panic!("expected rust unit target");
    };
    assert_eq!(args.package.as_deref(), Some("cogentlm-core"));

    let cli = Cli::parse_from(["xtask", "test", "unit", "group", "interface"]);
    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Unit(args) = command else {
        panic!("expected unit command");
    };
    let TestUnitCommands::Group(args) = args.command else {
        panic!("expected unit group command");
    };
    assert!(matches!(args.target, TestUnitGroupTarget::Interface));

    let cli = Cli::parse_from([
        "xtask",
        "test",
        "smoke",
        "suite",
        "benchmark-browser",
        "--require-webgpu",
        "--timeout-ms",
        "45000",
    ]);
    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Smoke(args) = command else {
        panic!("expected smoke command");
    };
    let TestSmokeCommands::Suite(args) = args.command else {
        panic!("expected smoke suite command");
    };
    let TestSmokeSuiteTarget::BenchmarkBrowser(args) = args.target else {
        panic!("expected benchmark browser smoke target");
    };
    assert!(args.require_webgpu);
    assert_eq!(args.timeout_ms, 45_000);

    let cli = Cli::parse_from(["xtask", "test", "smoke", "group", "examples"]);
    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Smoke(args) = command else {
        panic!("expected smoke command");
    };
    let TestSmokeCommands::Group(args) = args.command else {
        panic!("expected smoke group command");
    };
    assert!(matches!(args.target, TestSmokeGroupTarget::Examples(_)));
}

#[test]
fn toolchain_doctor_and_setup_commands_parse() {
    let cli = Cli::parse_from(["xtask", "toolchain", "install", "vulkan"]);
    let Commands::Toolchain { command } = cli.command else {
        panic!("expected toolchain command");
    };
    let ToolchainCommands::Install { component } = command else {
        panic!("expected toolchain install command");
    };
    assert_eq!(component, ToolchainComponent::Vulkan);

    let cli = Cli::parse_from(["xtask", "doctor", "--target", "python", "--backend", "all"]);
    let Commands::Doctor(args) = cli.command else {
        panic!("expected doctor command");
    };
    assert_eq!(args.target, DoctorTarget::Python);
    assert_eq!(args.backend, Backend::All);

    let cli = Cli::parse_from([
        "xtask",
        "setup",
        "--profile",
        "bindings",
        "--yes",
        "--no-downloads",
        "--no-splash",
    ]);
    let Commands::Setup(args) = cli.command else {
        panic!("expected setup command");
    };
    assert_eq!(args.profile, Some(SetupProfile::Bindings));
    assert!(args.yes);
    assert!(args.no_downloads);
    assert!(args.no_splash);
}

#[test]
fn labels_match_cli_wire_values() {
    assert_eq!(Backend::Cpu.as_str(), "cpu");
    assert_eq!(Backend::Cuda.as_str(), "cuda");
    assert_eq!(Backend::Metal.as_str(), "metal");
    assert_eq!(Backend::Vulkan.as_str(), "vulkan");
    assert_eq!(Backend::All.as_str(), "all");
    assert_eq!(TestSuiteId::RustCrates.as_str(), "rust-crates");
    assert_eq!(
        TestSuiteId::ExampleBrowserSmoke.as_str(),
        "example-browser-smoke"
    );
    assert_eq!(
        TestSuiteId::BenchmarkBrowserSmoke.as_str(),
        "benchmark-browser-smoke"
    );
    assert_eq!(DemoName::ProactiveUi.slug(), "proactive-ui");
    assert_eq!(DemoServeMode::Dev.as_str(), "dev");
    assert_eq!(LlamaBackendOpsMode::Support.as_str(), "support");
    assert_eq!(LlamaBackendOpsOutput::Sql.as_str(), "sql");
    assert_eq!(SetupProfile::Full.as_str(), "full");
    assert_eq!(
        SetupProfile::Browser.to_string(),
        "Browser demos and WebAssembly"
    );
}

#[test]
fn invalid_enums_and_missing_subcommands_are_rejected() {
    assert!(Cli::try_parse_from(["xtask", "build", "node", "--backend", "gpu"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "run", "apps", "serve", "chat"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "run", "demos", "serve", "missing"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "toolchain", "install", "cuda"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "build"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "node"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "all"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "browser"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "whitebox"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "rust"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "node"]).is_err());
}
