//! Cataloged workspace test and coverage orchestration.

use crate::cli::{
    Backend, LlamaBackendOpsMode, LlamaBackendOpsOutput, RunLlamaBackendOpsArgs, TestAllArgs,
    TestCommands, TestCoverageArgs, TestCoverageScope, TestInterfaceArgs, TestListArgs,
    TestListCategory, TestListFormat, TestProfile, TestSuiteId, TestWhiteboxArgs,
};
use crate::output;
use crate::sample_model::{self, SampleModelOptions};
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::{apply_uv_env, setup_uv};
use crate::utils::{ensure_playwright_chromium, BuildContext};
use anyhow::{Context, Result};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use xshell::{cmd, Shell};

const DEFAULT_SMOKE_PROMPT: &str = "Describe browser LLM inference.";
const RUST_GENERATION_SMOKE_EXAMPLES: &[&str] = &["query", "chat", "embed"];
const NODE_GENERATION_SMOKE_SCRIPTS: &[&str] = &[
    "examples/query.mjs",
    "examples/chat.mjs",
    "examples/embed.mjs",
];
const PYTHON_GENERATION_SMOKE_SCRIPTS: &[&str] =
    &["examples/query.py", "examples/chat.py", "examples/embed.py"];
const APP_TEST_SUFFIX: &str = ".test.ts";
const BROWSER_PACKAGE_TEST_DIR: &str = "packages/npm/tests";
const SKIPPED_APP_TEST_DIRS: &[&str] = &[
    "node_modules",
    "dist",
    "build",
    "out",
    ".vite",
    ".turbo",
    "coverage",
];

const RUST_CRATE_TEST_TARGETS: &[RustTestTarget] = &[
    RustTestTarget::lib("cogentlm-core"),
    RustTestTarget::lib("cogentlm-shard"),
    RustTestTarget::lib("cogentlm-sys"),
    RustTestTarget::lib("cogentlm-engine"),
    RustTestTarget::lib("cogentlm-providers"),
    RustTestTarget::lib("cogentlm-client"),
    RustTestTarget::bin("cogentlm-cli", "cogentlm"),
];
const XTASK_TEST_TARGETS: &[RustTestTarget] = &[RustTestTarget::package("xtask")];
const RUST_BINDING_TEST_TARGETS: &[RustTestTarget] = &[
    RustTestTarget::package("cogentlm-napi"),
    RustTestTarget::package("cogentlm-py"),
    RustTestTarget::package("cogentlm-wasm"),
];
const RUST_PUBLIC_API_TEST_TARGETS: &[RustTestTarget] = &[
    RustTestTarget::test("cogentlm-client", "public_api"),
    RustTestTarget::test("cogentlm-providers", "public_api"),
    RustTestTarget::test("cogentlm-shard", "public_api"),
];
const CLI_BLACK_BOX_TEST_TARGETS: &[RustTestTarget] =
    &[RustTestTarget::test("cogentlm-cli", "cli_black_box")];

const CONTRIBUTOR_PROFILE: &[TestProfile] = &[
    TestProfile::Contributor,
    TestProfile::Quick,
    TestProfile::Ci,
    TestProfile::Full,
];
const QUICK_PROFILE: &[TestProfile] = &[TestProfile::Quick, TestProfile::Ci, TestProfile::Full];
const CI_PROFILE: &[TestProfile] = &[TestProfile::Ci, TestProfile::Full];
const FULL_PROFILE: &[TestProfile] = &[TestProfile::Full];
const TEST_PROFILES: &[TestProfile] = &[
    TestProfile::Contributor,
    TestProfile::Quick,
    TestProfile::Ci,
    TestProfile::Full,
];

const TEST_SUITES: &[TestSuite] = &[
    TestSuite {
        id: TestSuiteId::Layout,
        category: TestCategory::Whitebox,
        description: "test layout and hygiene checks",
        requirements: "filesystem",
        profiles: CONTRIBUTOR_PROFILE,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::Layout,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::Xtask,
        category: TestCategory::Whitebox,
        description: "xtask CLI and orchestration unit tests",
        requirements: "cargo",
        profiles: CONTRIBUTOR_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(XTASK_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(XTASK_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::RustCrates,
        category: TestCategory::Whitebox,
        description: "Rust unit tests for core workspace crates",
        requirements: "cargo, native toolchain",
        profiles: QUICK_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_CRATE_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_CRATE_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::RustBindings,
        category: TestCategory::Whitebox,
        description: "Rust unit tests for Node, Python, and WASM binding crates",
        requirements: "cargo, native toolchain",
        profiles: FULL_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_BINDING_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_BINDING_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::PackageTs,
        category: TestCategory::Whitebox,
        description: "browser package TypeScript tests under packages/npm/tests",
        requirements: "bun, wasm build",
        profiles: CI_PROFILE,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::PackageTs,
        discoverer: CaseDiscoverer::PackageTs,
    },
    TestSuite {
        id: TestSuiteId::AppTs,
        category: TestCategory::Whitebox,
        description: "browser app TypeScript tests discovered under apps/",
        requirements: "bun, wasm build",
        profiles: FULL_PROFILE,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::AppTs,
        discoverer: CaseDiscoverer::AppTs,
    },
    TestSuite {
        id: TestSuiteId::RustPublicApi,
        category: TestCategory::Interface,
        description: "crate-level public API integration tests",
        requirements: "cargo, native toolchain",
        profiles: CI_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_PUBLIC_API_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_PUBLIC_API_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::Cli,
        category: TestCategory::Interface,
        description: "CLI black-box integration test",
        requirements: "cargo, native toolchain",
        profiles: FULL_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(CLI_BLACK_BOX_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(CLI_BLACK_BOX_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::NodePackage,
        category: TestCategory::Interface,
        description: "Node binding build/import/router tests",
        requirements: "cargo, node",
        profiles: FULL_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::Any,
        runner: SuiteRunner::NodePackage,
        discoverer: CaseDiscoverer::NodePackage,
    },
    TestSuite {
        id: TestSuiteId::PythonPackage,
        category: TestCategory::Interface,
        description: "Python wheel install/import pytest suite",
        requirements: "cargo, uv, python",
        profiles: FULL_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::PythonPackage,
        discoverer: CaseDiscoverer::PythonPackage,
    },
    TestSuite {
        id: TestSuiteId::BrowserRuntime,
        category: TestCategory::Interface,
        description: "Playwright browser runtime smoke",
        requirements: "bun, wasm build, playwright chromium",
        profiles: FULL_PROFILE,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::BrowserRuntime,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::ModelSmoke,
        category: TestCategory::Interface,
        description: "CLI/Rust/Node/Python local model examples",
        requirements: "sample GGUF model, cargo, node, python",
        profiles: FULL_PROFILE,
        coverage: true,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::ModelSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::LlamaBackendOps,
        category: TestCategory::Interface,
        description: "llama.cpp backend correctness operations",
        requirements: "cmake, ninja, native backend",
        profiles: FULL_PROFILE,
        coverage: false,
        backend_policy: BackendPolicy::Any,
        runner: SuiteRunner::LlamaBackendOps,
        discoverer: CaseDiscoverer::None,
    },
];

/// Runs a workspace test workflow.
pub fn run(sh: &Shell, ctx: &BuildContext, command: TestCommands) -> Result<()> {
    match command {
        TestCommands::List(args) => run_list(ctx, &args),
        TestCommands::Whitebox(args) => run_whitebox(sh, ctx, &args),
        TestCommands::Interface(args) => run_interface(sh, ctx, &args),
        TestCommands::Coverage(args) => run_coverage(sh, ctx, &args),
        TestCommands::All(args) => run_all(sh, ctx, &args),
    }
}

fn run_list(ctx: &BuildContext, args: &TestListArgs) -> Result<()> {
    let suites = TEST_SUITES
        .iter()
        .filter(|suite| list_category_matches(args.category, suite.category))
        .collect::<Vec<_>>();

    match args.format {
        TestListFormat::Text => print_text_list(ctx, &suites, args.cases),
        TestListFormat::Json => print_json_list(ctx, &suites, args.cases),
    }
}

fn run_whitebox(sh: &Shell, ctx: &BuildContext, args: &TestWhiteboxArgs) -> Result<()> {
    let suites = selected_suites(&args.suite, TestCategory::Whitebox)?;
    if args.package.is_some()
        && suites
            .iter()
            .any(|suite| suite.id != TestSuiteId::RustCrates)
    {
        anyhow::bail!("--package is only supported with --suite rust-crates");
    }
    validate_suite_backends(&suites, Backend::Cpu)?;

    for suite in suites {
        run_suite(
            sh,
            ctx,
            suite,
            &SuiteRunOptions {
                backend: Backend::Cpu,
                model: None,
                offline: false,
                package: args.package.as_deref(),
            },
        )?;
    }
    Ok(())
}

fn run_interface(sh: &Shell, ctx: &BuildContext, args: &TestInterfaceArgs) -> Result<()> {
    let suites = selected_suites(&args.suite, TestCategory::Interface)?;
    validate_suite_backends(&suites, args.backend)?;
    for suite in suites {
        run_suite(
            sh,
            ctx,
            suite,
            &SuiteRunOptions {
                backend: args.backend,
                model: args.model.as_deref(),
                offline: args.offline,
                package: None,
            },
        )?;
    }
    Ok(())
}

fn run_all(sh: &Shell, ctx: &BuildContext, args: &TestAllArgs) -> Result<()> {
    let started_at = Instant::now();
    let suites = TEST_SUITES
        .iter()
        .filter(|suite| suite.profiles.contains(&args.profile))
        .collect::<Vec<_>>();
    validate_suite_backends(&suites, args.backend)?;
    output::phase(&format!("Test profile: {}", args.profile.as_str()));
    for suite in suites {
        run_suite(
            sh,
            ctx,
            suite,
            &SuiteRunOptions {
                backend: args.backend,
                model: args.model.as_deref(),
                offline: args.offline,
                package: None,
            },
        )?;
    }
    output::success(format!(
        "Test profile {} complete in {}",
        args.profile.as_str(),
        output::elapsed(started_at.elapsed())
    ));
    Ok(())
}

fn run_suite(
    sh: &Shell,
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    match suite.runner {
        SuiteRunner::Layout => run_layout(ctx),
        SuiteRunner::RustTargets(targets) => {
            let package = if suite.id == TestSuiteId::RustCrates {
                options.package
            } else {
                None
            };
            run_rust_target_tests(sh, ctx, targets, package)
        }
        SuiteRunner::PackageTs => run_package_ts_tests(sh, ctx),
        SuiteRunner::AppTs => run_app_ts_tests(sh, ctx),
        SuiteRunner::NodePackage => run_node_package_tests(sh, ctx, &options.backend),
        SuiteRunner::PythonPackage => run_python_package_tests(sh, ctx, &options.backend),
        SuiteRunner::BrowserRuntime => run_browser_runtime_smoke(sh, ctx),
        SuiteRunner::ModelSmoke => run_model_smoke(sh, ctx, options),
        SuiteRunner::LlamaBackendOps => run_llama_backend_ops_suite(sh, ctx, &options.backend),
    }
}

fn run_layout(ctx: &BuildContext) -> Result<()> {
    output::phase("Test layout");
    let mut violations = Vec::new();
    collect_package_test_layout_violations(ctx, &mut violations)?;
    collect_rust_test_layout_violations(ctx, &mut violations)?;
    collect_catalog_ownership_violations(ctx, &mut violations)?;

    if violations.is_empty() {
        output::success("Test layout check passed");
        return Ok(());
    }

    for violation in &violations {
        output::warning(violation);
    }
    anyhow::bail!(
        "test layout check failed with {} violation(s)",
        violations.len()
    )
}

fn run_rust_target_tests(
    sh: &Shell,
    ctx: &BuildContext,
    targets: &[RustTestTarget],
    package: Option<&str>,
) -> Result<()> {
    output::phase("Rust tests");
    let targets = filtered_rust_targets(targets, package)?;
    run_rust_targets(sh, ctx, &targets, None)
}

fn run_rust_targets(
    sh: &Shell,
    ctx: &BuildContext,
    targets: &[RustTestTarget],
    backend: Option<&Backend>,
) -> Result<()> {
    let _dir = sh.push_dir(ctx.workspace_root());
    for target in targets {
        let package = target.package;
        let mut cargo_test = cmd!(sh, "cargo test -p {package}");
        if let Some(test_kind) = target.kind {
            match test_kind {
                RustTestKind::Lib => {
                    cargo_test = cargo_test.arg("--lib");
                }
                RustTestKind::Bin(binary) => {
                    cargo_test = cargo_test.arg("--bin").arg(binary);
                }
                RustTestKind::Test(test_name) => {
                    cargo_test = cargo_test.arg("--test").arg(test_name);
                }
                RustTestKind::Package => {}
            }
        }
        let cargo_test = apply_toolchains(sh, ctx, cargo_test, backend)?;
        output::run_command(format!("Running {} Rust tests", target.label()), cargo_test)?;
    }
    Ok(())
}

fn run_package_ts_tests(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("White-box browser package TypeScript tests");
    targets::wasm::build(sh, ctx)?;
    let _dir = sh.push_dir(ctx.workspace_root());
    let browser_package_test_dir = BROWSER_PACKAGE_TEST_DIR;
    output::run_command(
        "Running browser package TypeScript tests",
        cmd!(sh, "bun test {browser_package_test_dir}"),
    )
}

fn run_app_ts_tests(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("White-box browser app TypeScript tests");
    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;

    let tests = app_test_files(ctx)?;
    if tests.is_empty() {
        output::warning("No app TypeScript tests were found under apps/");
        return Ok(());
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    output::detail("Test files", tests.len());
    let mut test_cmd = cmd!(sh, "bun test");
    for test in tests {
        test_cmd = test_cmd.arg(test);
    }
    output::run_command("Running app TypeScript tests through Bun", test_cmd)
}

fn run_node_package_tests(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    output::phase("Interface Node package tests");
    targets::node::build(sh, ctx, Some(backend))?;

    let node_dir = ctx.bindings_node_dir();
    let _node = sh.push_dir(&node_dir);
    for backend in node_test_backends(ctx, backend)? {
        output::run_command(
            format!("Running Node.js package tests ({})", backend.as_str()),
            cmd!(sh, "node --test tests/router.test.mjs")
                .env("COGENTLM_NODE_BACKEND", backend.as_str())
                .env("COGENTLM_NODE_TEST_BACKEND", backend.as_str()),
        )?;
    }
    Ok(())
}

fn run_python_package_tests(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    if *backend == Backend::All {
        anyhow::bail!(
            "python-package requires a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    output::phase("Interface Python package tests");
    let wheel = build_python_test_wheel(sh, ctx, backend)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx.tmp_dir().join("python-tests").join(backend.as_str());
    sh.create_dir(&venv_dir)?;
    output::run_command(
        "Creating Python test virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_command(
        "Installing Python test wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel} pytest"
            ),
        ),
    )?;

    let python_tests = ctx.bindings_python_dir().join("tests");
    output::run_command(
        "Running Python package pytest suite",
        cmd!(sh, "{python_exe} -m pytest {python_tests}"),
    )
}

fn run_browser_runtime_smoke(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Interface browser runtime smoke");
    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;
    ensure_playwright_chromium(sh, ctx)?;

    let benchmark_dir = ctx.app_dir("benchmark");
    let _benchmark_dir = sh.push_dir(&benchmark_dir);
    output::run_command(
        "Running benchmark browser:smoke",
        cmd!(sh, "bun run browser:smoke")
            .env("PLAYWRIGHT_BROWSERS_PATH", ctx.playwright_browsers_dir()),
    )
}

fn run_model_smoke(sh: &Shell, ctx: &BuildContext, options: &SuiteRunOptions<'_>) -> Result<()> {
    if options.backend == Backend::All {
        anyhow::bail!(
            "model-smoke requires a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    output::phase("Interface model-backed local smoke tests");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    output::path("Model", &model);
    output::detail("Backend", options.backend.as_str());

    targets::cli::build(sh, ctx, Some(&options.backend))?;
    run_cli_smoke(sh, ctx, &model, &options.backend)?;
    run_rust_generation_smoke(sh, ctx, &model, &options.backend)?;
    run_node_generation_smoke(sh, ctx, &model, &options.backend)?;
    run_python_generation_smoke(sh, ctx, &model, &options.backend)
}

fn run_llama_backend_ops_suite(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    crate::run::run_llama_backend_ops(
        sh,
        ctx,
        &RunLlamaBackendOpsArgs {
            backend: *backend,
            mode: LlamaBackendOpsMode::Test,
            op: None,
            params: None,
            output: LlamaBackendOpsOutput::Console,
        },
    )
}

fn resolve_smoke_model(
    sh: &Shell,
    ctx: &BuildContext,
    model: Option<&Path>,
    offline: bool,
) -> Result<PathBuf> {
    match model {
        Some(model) => Ok(model.to_path_buf()),
        None => sample_model::ensure_sample_model(
            sh,
            ctx,
            SampleModelOptions {
                allow_download: !offline,
            },
        ),
    }
}

fn run_cli_smoke(sh: &Shell, ctx: &BuildContext, model: &Path, backend: &Backend) -> Result<()> {
    output::phase("CLI smoke");
    let cli_dir = ctx.cli_artifacts_dir();
    let cli_exe = cli_dir.join(cli_binary_file_name());
    if !cli_exe.is_file() {
        anyhow::bail!("CLI executable was not staged at {}", cli_exe.display());
    }

    let _dir = sh.push_dir(&cli_dir);
    let mut smoke_cmd = cmd!(
        sh,
        "{cli_exe} {model} {DEFAULT_SMOKE_PROMPT} --max-tokens 1 --temperature 0"
    );
    if *backend != Backend::Cpu {
        smoke_cmd = smoke_cmd.arg("--backend").arg(backend.as_str());
    }
    output::run_command("Running CLI local inference smoke", smoke_cmd)
}

fn run_rust_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    backend: &Backend,
) -> Result<()> {
    output::phase("Rust CogentClient generation smoke");
    output::path("Model", model);
    output::detail("Backend", backend.as_str());

    let _dir = sh.push_dir(ctx.workspace_root());
    for example in RUST_GENERATION_SMOKE_EXAMPLES {
        let mut smoke_cmd = cmd!(sh, "cargo run -p cogentlm-client");
        if *backend != Backend::Cpu {
            smoke_cmd = smoke_cmd.arg("--features").arg(backend.as_str());
        }
        smoke_cmd = smoke_cmd
            .arg("--example")
            .arg(example)
            .arg("--")
            .arg(model)
            .arg(DEFAULT_SMOKE_PROMPT)
            .env("COGENTLM_MAX_TOKENS", "1")
            .env("COGENTLM_TEMPERATURE", "0");
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(backend))?;
        output::run_command(
            format!("Running Rust {} smoke: {example}", backend.as_str()),
            smoke_cmd,
        )
        .with_context(|| format!("Rust {} smoke failed: {example}", backend.as_str()))?;
    }

    Ok(())
}

fn run_node_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    backend: &Backend,
) -> Result<()> {
    output::phase("Node.js generation smoke");
    output::path("Model", model);
    output::detail("Backend", backend.as_str());
    targets::node::build(sh, ctx, Some(backend))?;

    let node_dir = ctx.bindings_node_dir();
    let _dir = sh.push_dir(&node_dir);
    for smoke_script in NODE_GENERATION_SMOKE_SCRIPTS {
        let mut smoke_cmd = cmd!(sh, "node {smoke_script} {model} {DEFAULT_SMOKE_PROMPT}")
            .env("COGENTLM_NODE_BACKEND", backend.as_str())
            .env("COGENTLM_MAX_TOKENS", "1")
            .env("COGENTLM_TEMPERATURE", "0");
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(backend))?;
        output::run_command(
            format!("Running Node {} smoke: {smoke_script}", backend.as_str()),
            smoke_cmd,
        )
        .with_context(|| format!("Node {} smoke failed: {smoke_script}", backend.as_str()))?;
    }

    Ok(())
}

fn run_python_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    backend: &Backend,
) -> Result<()> {
    output::phase("Python generation smoke");
    output::path("Model", model);
    output::detail("Backend", backend.as_str());
    let wheel = build_python_test_wheel(sh, ctx, backend)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx
        .tmp_dir()
        .join("python-model-smoke")
        .join(backend.as_str());
    output::run_command(
        "Creating Python smoke virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_command(
        "Installing Python smoke wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel}"
            ),
        ),
    )?;

    let python_dir = ctx.bindings_python_dir();
    let _dir = sh.push_dir(&python_dir);
    for smoke_script in PYTHON_GENERATION_SMOKE_SCRIPTS {
        let mut smoke_cmd = cmd!(
            sh,
            "{python_exe} {smoke_script} {model} {DEFAULT_SMOKE_PROMPT}"
        )
        .env("COGENTLM_PYTHON_BACKEND", backend.as_str())
        .env("COGENTLM_MAX_TOKENS", "1")
        .env("COGENTLM_TEMPERATURE", "0");
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(backend))?;
        output::run_command(
            format!("Running Python {} smoke: {smoke_script}", backend.as_str()),
            smoke_cmd,
        )
        .with_context(|| format!("Python {} smoke failed: {smoke_script}", backend.as_str()))?;
    }

    Ok(())
}

fn run_coverage(sh: &Shell, ctx: &BuildContext, args: &TestCoverageArgs) -> Result<()> {
    output::phase(&format!("Coverage: {}", args.scope.as_str()));
    validate_coverage_options(args)?;
    ensure_cargo_llvm_cov()?;

    let coverage_root = ctx.build_root().join("coverage");
    let rust_dir = coverage_root.join("rust");
    if coverage_root.exists() {
        sh.remove_path(&coverage_root)?;
    }
    sh.create_dir(&rust_dir)?;
    let rust_lcov = rust_dir.join("lcov.info");
    let rust_html = rust_dir.join("html");

    let _root = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Cleaning Rust coverage counters",
        cmd!(sh, "cargo llvm-cov clean --workspace"),
    )?;
    run_rust_coverage_tests(sh, ctx, args.scope)?;
    output::run_command(
        "Writing Rust LCOV report",
        cmd!(
            sh,
            "cargo llvm-cov report --lcov --output-path {rust_lcov} --ignore-filename-regex third_party|\\.build|target|tests|examples|packages|apps"
        ),
    )?;
    output::run_command(
        "Writing Rust HTML report",
        cmd!(
            sh,
            "cargo llvm-cov report --html --output-dir {rust_html} --ignore-filename-regex third_party|\\.build|target|tests|examples|packages|apps"
        ),
    )?;
    drop(_root);

    if args.scope == TestCoverageScope::All {
        run_node_coverage(sh, ctx, &args.backend)?;
        run_python_coverage(sh, ctx, &args.backend)?;
        run_coverage_interface_smokes(sh, ctx, args)?;
    }

    write_coverage_summary(&coverage_root, args.scope)?;
    output::success(format!(
        "Coverage reports written to {}",
        coverage_root.display()
    ));
    Ok(())
}

fn run_rust_coverage_tests(sh: &Shell, ctx: &BuildContext, scope: TestCoverageScope) -> Result<()> {
    let _dir = sh.push_dir(ctx.workspace_root());
    for target in RUST_CRATE_TEST_TARGETS
        .iter()
        .chain(XTASK_TEST_TARGETS)
        .chain(RUST_BINDING_TEST_TARGETS)
    {
        run_cargo_llvm_cov_target(sh, ctx, target)?;
    }
    if scope == TestCoverageScope::All {
        for target in RUST_PUBLIC_API_TEST_TARGETS
            .iter()
            .chain(CLI_BLACK_BOX_TEST_TARGETS)
        {
            run_cargo_llvm_cov_target(sh, ctx, &target)?;
        }
    }
    Ok(())
}

fn run_cargo_llvm_cov_target(
    sh: &Shell,
    ctx: &BuildContext,
    target: &RustTestTarget,
) -> Result<()> {
    let package = target.package;
    let mut coverage_cmd = cmd!(sh, "cargo llvm-cov --no-report -p {package}");
    if let Some(test_kind) = target.kind {
        match test_kind {
            RustTestKind::Lib => {
                coverage_cmd = coverage_cmd.arg("--lib");
            }
            RustTestKind::Bin(binary) => {
                coverage_cmd = coverage_cmd.arg("--bin").arg(binary);
            }
            RustTestKind::Test(test_name) => {
                coverage_cmd = coverage_cmd.arg("--test").arg(test_name);
            }
            RustTestKind::Package => {}
        }
    }
    let coverage_cmd = apply_toolchains(sh, ctx, coverage_cmd, None)?;
    output::run_command(
        format!("Running coverage for {}", target.label()),
        coverage_cmd,
    )
}

fn run_node_coverage(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    output::phase("Node wrapper coverage");
    targets::node::build(sh, ctx, Some(backend))?;

    let coverage_dir = ctx.build_root().join("coverage").join("node");
    sh.create_dir(&coverage_dir)?;
    let node_dir = ctx.bindings_node_dir();
    let _dir = sh.push_dir(&node_dir);
    output::run_command(
        "Installing Node binding dependencies",
        cmd!(sh, "bun install"),
    )?;
    for backend in node_test_backends(ctx, backend)? {
        output::run_command(
            format!("Running Node c8 coverage ({})", backend.as_str()),
            cmd!(
                sh,
                "bun run c8 --reporter=lcov --reporter=text-summary --reports-dir {coverage_dir} node --test tests/router.test.mjs"
            )
            .env("COGENTLM_NODE_BACKEND", backend.as_str())
            .env("COGENTLM_NODE_TEST_BACKEND", backend.as_str()),
        )?;
    }
    Ok(())
}

fn run_python_coverage(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    if *backend == Backend::All {
        anyhow::bail!("python coverage requires a concrete backend");
    }

    output::phase("Python wrapper coverage");
    let wheel = build_python_test_wheel(sh, ctx, backend)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let coverage_dir = ctx.build_root().join("coverage").join("python");
    sh.create_dir(&coverage_dir)?;
    let lcov = coverage_dir.join("lcov.info");
    let xml = coverage_dir.join("cobertura.xml");
    let html = coverage_dir.join("html");
    let venv_dir = ctx.tmp_dir().join("python-coverage").join(backend.as_str());
    output::run_command(
        "Creating Python coverage virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_command(
        "Installing Python coverage wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel} pytest pytest-cov"
            ),
        ),
    )?;

    let python_tests = ctx.bindings_python_dir().join("tests");
    let python_source = ctx.bindings_python_dir().join("python").join("cogentlm");
    output::run_command(
        "Running Python pytest-cov",
        cmd!(
            sh,
            "{python_exe} -m pytest {python_tests} --cov {python_source} --cov-branch --cov-report lcov:{lcov} --cov-report xml:{xml} --cov-report html:{html}"
        ),
    )
}

fn run_coverage_interface_smokes(
    sh: &Shell,
    ctx: &BuildContext,
    args: &TestCoverageArgs,
) -> Result<()> {
    for suite_id in [TestSuiteId::BrowserRuntime, TestSuiteId::ModelSmoke] {
        let suite = suite_by_id(suite_id)?;
        run_suite(
            sh,
            ctx,
            suite,
            &SuiteRunOptions {
                backend: args.backend,
                model: args.model.as_deref(),
                offline: args.offline,
                package: None,
            },
        )?;
    }
    Ok(())
}

fn write_coverage_summary(coverage_root: &Path, scope: TestCoverageScope) -> Result<()> {
    let rust = parse_lcov_summary(&coverage_root.join("rust").join("lcov.info"))?;
    let node = parse_lcov_summary(&coverage_root.join("node").join("lcov.info"))?;
    let python = parse_lcov_summary(&coverage_root.join("python").join("lcov.info"))?;
    ensure_non_empty_coverage("Rust/native", &rust)?;
    if scope == TestCoverageScope::All {
        ensure_non_empty_coverage("Node wrapper", &node)?;
        ensure_non_empty_coverage("Python wrapper", &python)?;
    }
    let baseline = json!({
        "rust": rust.as_json(),
        "node": node.as_json(),
        "python": python.as_json(),
    });
    let baseline_path = coverage_root.join("baseline.json");
    std::fs::write(&baseline_path, serde_json::to_string_pretty(&baseline)?)
        .with_context(|| format!("failed to write {}", baseline_path.display()))?;

    let summary_path = coverage_root.join("coverage-summary.md");
    let summary = format!(
        "# Coverage baseline\n\n| Area | Covered | Total | Percent |\n| --- | ---: | ---: | ---: |\n| Rust/native | {} | {} | {:.2}% |\n| Node wrapper | {} | {} | {:.2}% |\n| Python wrapper | {} | {} | {:.2}% |\n",
        rust.hit,
        rust.found,
        rust.percent(),
        node.hit,
        node.found,
        node.percent(),
        python.hit,
        python.found,
        python.percent(),
    );
    std::fs::write(&summary_path, summary)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    output::path("Coverage baseline", &baseline_path);
    output::path("Coverage summary", &summary_path);
    Ok(())
}

fn ensure_non_empty_coverage(label: &str, summary: &LcovSummary) -> Result<()> {
    if summary.found == 0 {
        anyhow::bail!("{label} coverage report did not include any first-party lines");
    }
    Ok(())
}

fn print_text_list(ctx: &BuildContext, suites: &[&TestSuite], include_cases: bool) -> Result<()> {
    print_text_profiles();
    println!();
    println!("Test suites:");
    for suite in suites {
        println!(
            "  {:<18} {:<9} {:<18} {}",
            suite.id.as_str(),
            suite.category.as_str(),
            suite.profile_labels().join(","),
            suite.description
        );
        println!("  {:<18} requirements: {}", "", suite.requirements);
    }

    if include_cases {
        println!();
        println!("Test cases:");
        for case in discover_cases(ctx, suites)? {
            println!(
                "  {:<18} {:<42} {}",
                case.suite_id.as_str(),
                case.name,
                case.path
            );
        }
    }
    Ok(())
}

fn print_text_profiles() {
    println!("Test profiles:");
    for profile in TEST_PROFILES {
        println!("  {:<12} {}", profile.as_str(), profile.summary());
        println!(
            "  {:<12} suites: {}",
            "",
            profile_suite_labels(*profile).join(", ")
        );
    }
}

fn print_json_list(ctx: &BuildContext, suites: &[&TestSuite], include_cases: bool) -> Result<()> {
    let profile_values = TEST_PROFILES
        .iter()
        .map(|profile| {
            json!({
                "id": profile.as_str(),
                "description": profile.summary(),
                "suites": profile_suite_labels(*profile),
            })
        })
        .collect::<Vec<_>>();
    let suite_values = suites
        .iter()
        .map(|suite| {
            json!({
                "id": suite.id.as_str(),
                "category": suite.category.as_str(),
                "description": suite.description,
                "requirements": suite.requirements,
                "profiles": suite.profile_labels(),
                "backendPolicy": suite.backend_policy.as_str(),
                "coverage": suite.coverage,
                "caseDiscovery": suite.discoverer.as_str(),
            })
        })
        .collect::<Vec<_>>();
    let cases = if include_cases {
        discover_cases(ctx, suites)?
            .into_iter()
            .map(|case| {
                json!({
                    "suite": case.suite_id.as_str(),
                    "name": case.name,
                    "path": case.path,
                })
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "profiles": profile_values,
            "suites": suite_values,
            "cases": cases,
        }))?
    );
    Ok(())
}

fn profile_suite_labels(profile: TestProfile) -> Vec<&'static str> {
    TEST_SUITES
        .iter()
        .filter(|suite| suite.profiles.contains(&profile))
        .map(|suite| suite.id.as_str())
        .collect()
}

fn discover_cases(ctx: &BuildContext, suites: &[&TestSuite]) -> Result<Vec<TestCase>> {
    let mut cases = Vec::new();
    for suite in suites {
        discover_suite_cases(ctx, suite, &mut cases)?;
    }
    cases.sort_by(|left, right| {
        left.suite_id
            .cmp(&right.suite_id)
            .then(left.name.cmp(&right.name))
            .then(left.path.cmp(&right.path))
    });
    Ok(cases)
}

fn discover_suite_cases(
    ctx: &BuildContext,
    suite: &TestSuite,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    match suite.discoverer {
        CaseDiscoverer::None => {}
        CaseDiscoverer::RustTargets(targets) => {
            discover_rust_cases(ctx, suite.id, rust_target_case_files(ctx, targets)?, cases)?
        }
        CaseDiscoverer::PackageTs => discover_package_ts_cases(ctx, suite.id, cases)?,
        CaseDiscoverer::AppTs => discover_app_ts_cases(ctx, suite.id, cases)?,
        CaseDiscoverer::NodePackage => discover_node_cases(ctx, suite.id, cases)?,
        CaseDiscoverer::PythonPackage => discover_python_cases(ctx, suite.id, cases)?,
    }
    Ok(())
}

fn discover_rust_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    files: Vec<PathBuf>,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    for path in files {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut pending_test = false;
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[test]") || trimmed.starts_with("#[tokio::test]") {
                pending_test = true;
                continue;
            }
            if trimmed.starts_with("#[ignore") {
                continue;
            }
            if pending_test {
                if let Some(name) = parse_rust_fn_name(trimmed) {
                    cases.push(TestCase {
                        suite_id,
                        name,
                        path: display_relative(ctx, &path),
                    });
                }
                pending_test = false;
            }
        }
    }
    Ok(())
}

fn discover_node_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    discover_quoted_test_cases(ctx, suite_id, node_test_files(ctx)?, cases)
}

fn discover_python_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    for path in python_test_files(ctx)? {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if let Some(name) = trimmed.strip_prefix("def test_") {
                if let Some((rest, _)) = name.split_once('(') {
                    cases.push(TestCase {
                        suite_id,
                        name: format!("test_{rest}"),
                        path: display_relative(ctx, &path),
                    });
                }
            }
        }
    }
    Ok(())
}

fn discover_package_ts_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    discover_quoted_test_cases(ctx, suite_id, package_ts_test_files(ctx)?, cases)
}

fn discover_app_ts_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    discover_quoted_test_cases(ctx, suite_id, app_test_files(ctx)?, cases)
}

fn discover_quoted_test_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    files: Vec<PathBuf>,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    for path in files {
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        for line in contents.lines() {
            if let Some(name) = parse_quoted_test_name(line, "test(") {
                cases.push(TestCase {
                    suite_id,
                    name,
                    path: display_relative(ctx, &path),
                });
            }
        }
    }
    Ok(())
}

fn parse_rust_fn_name(line: &str) -> Option<String> {
    let name = line.strip_prefix("fn ")?;
    let (name, _) = name.split_once('(')?;
    Some(name.to_owned())
}

fn parse_quoted_test_name(line: &str, prefix: &str) -> Option<String> {
    let start = line.find(prefix)? + prefix.len();
    let rest = line[start..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let end = rest[1..].find(quote)? + 1;
    Some(rest[1..end].to_owned())
}

fn selected_suites(
    selector: &TestSuiteId,
    category: TestCategory,
) -> Result<Vec<&'static TestSuite>> {
    if *selector == TestSuiteId::All {
        return Ok(TEST_SUITES
            .iter()
            .filter(|suite| suite.category == category)
            .collect());
    }

    let suite = suite_by_id(*selector)?;
    if suite.category != category {
        anyhow::bail!(
            "suite {} is {}, not {}",
            selector.as_str(),
            suite.category.as_str(),
            category.as_str()
        );
    }
    Ok(vec![suite])
}

fn suite_by_id(id: TestSuiteId) -> Result<&'static TestSuite> {
    TEST_SUITES
        .iter()
        .find(|suite| suite.id == id)
        .with_context(|| format!("unknown test suite id: {}", id.as_str()))
}

fn filtered_rust_targets(
    targets: &[RustTestTarget],
    package: Option<&str>,
) -> Result<Vec<RustTestTarget>> {
    let Some(package) = package else {
        return Ok(targets.to_vec());
    };
    let filtered = targets
        .iter()
        .copied()
        .filter(|target| target.package == package)
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        anyhow::bail!("package {package} is not part of the rust-crates suite");
    }
    Ok(filtered)
}

fn list_category_matches(filter: TestListCategory, category: TestCategory) -> bool {
    match filter {
        TestListCategory::All => true,
        TestListCategory::Whitebox => category == TestCategory::Whitebox,
        TestListCategory::Interface => category == TestCategory::Interface,
    }
}

fn ensure_cargo_llvm_cov() -> Result<()> {
    let output = Command::new("cargo")
        .args(["llvm-cov", "--version"])
        .output()
        .context("failed to run cargo llvm-cov --version")?;
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "cargo-llvm-cov is required for `cargo xtask test coverage`; install it with `cargo install cargo-llvm-cov`"
    )
}

fn validate_suite_backends(suites: &[&TestSuite], backend: Backend) -> Result<()> {
    for suite in suites {
        suite.backend_policy.validate(suite.id, backend)?;
    }
    Ok(())
}

fn validate_coverage_options(args: &TestCoverageArgs) -> Result<()> {
    if args.scope == TestCoverageScope::All && args.backend == Backend::All {
        anyhow::bail!(
            "coverage --scope all requires a concrete backend because Python and model smoke coverage cannot run with --backend all"
        );
    }
    Ok(())
}

fn collect_package_test_layout_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    let src = ctx.npm_package_dir().join("src");
    for path in collect_files_with_extension(&src, "ts")? {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.ends_with(".test.ts") {
            violations.push(format!(
                "TypeScript test must live under packages/npm/tests: {}",
                display_relative(ctx, &path)
            ));
        }
    }
    Ok(())
}

fn collect_catalog_ownership_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    let ownership = catalog_file_ownership(ctx)?;
    for path in first_party_test_files(ctx)? {
        let relative = display_relative(ctx, &path);
        match ownership.get(&relative) {
            Some(owners) if owners.len() == 1 => {}
            Some(owners) => violations.push(format!(
                "Test file is owned by multiple catalog suites ({}): {}",
                suite_id_list(owners),
                relative
            )),
            None => violations.push(format!(
                "Test file is not owned by any catalog suite: {relative}"
            )),
        }
    }
    Ok(())
}

fn catalog_file_ownership(ctx: &BuildContext) -> Result<BTreeMap<String, BTreeSet<TestSuiteId>>> {
    let mut ownership = BTreeMap::<String, BTreeSet<TestSuiteId>>::new();
    for suite in TEST_SUITES {
        for path in discoverer_test_files(ctx, suite.discoverer)? {
            ownership
                .entry(display_relative(ctx, &path))
                .or_default()
                .insert(suite.id);
        }
    }
    Ok(ownership)
}

fn discoverer_test_files(ctx: &BuildContext, discoverer: CaseDiscoverer) -> Result<Vec<PathBuf>> {
    match discoverer {
        CaseDiscoverer::None => Ok(Vec::new()),
        CaseDiscoverer::RustTargets(targets) => rust_target_case_files(ctx, targets),
        CaseDiscoverer::PackageTs => package_ts_test_files(ctx),
        CaseDiscoverer::AppTs => app_test_files(ctx),
        CaseDiscoverer::NodePackage => node_test_files(ctx),
        CaseDiscoverer::PythonPackage => python_test_files(ctx),
    }
}

fn first_party_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_first_party_rust_test_files(ctx, &mut files)?;
    files.extend(collect_files_with_suffix(
        &ctx.npm_package_dir(),
        APP_TEST_SUFFIX,
    )?);
    files.extend(app_test_files(ctx)?);
    files.extend(node_test_files(ctx)?);
    files.extend(python_test_files(ctx)?);
    files.extend(first_party_cpp_test_files(ctx)?);
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_first_party_rust_test_files(ctx: &BuildContext, files: &mut Vec<PathBuf>) -> Result<()> {
    for area in [
        ctx.workspace_root().join("crates"),
        ctx.workspace_root().join("bindings"),
    ] {
        for package_root in collect_rust_package_roots(&area)? {
            for path in collect_files_with_extension(&package_root, "rs")? {
                if is_rust_test_file(&package_root, &path)? {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}

fn is_rust_test_file(package_root: &Path, path: &Path) -> Result<bool> {
    if is_allowed_rust_test_file(package_root, path) {
        return Ok(true);
    }
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    if file_name.ends_with("_tests.rs") {
        return Ok(true);
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(contains_test_attribute(&contents))
}

fn first_party_cpp_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for root in [
        ctx.workspace_root().join("crates"),
        ctx.workspace_root().join("bindings"),
    ] {
        for extension in ["c", "cc", "cpp", "h", "hpp"] {
            for path in collect_files_with_extension(&root, extension)? {
                if is_cpp_test_file_name(&path) {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn is_cpp_test_file_name(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = file_name.to_ascii_lowercase();
    lower.starts_with("test")
        || lower.contains("_test")
        || lower.contains("-test")
        || lower.contains(".test.")
}

fn suite_id_list(owners: &BTreeSet<TestSuiteId>) -> String {
    owners
        .iter()
        .map(TestSuiteId::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

fn collect_rust_test_layout_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    for area in [
        ctx.workspace_root().join("crates"),
        ctx.workspace_root().join("bindings"),
    ] {
        for package_root in collect_rust_package_roots(&area)? {
            for path in collect_files_with_extension(&package_root, "rs")? {
                if is_inverted_rust_test_file(&package_root, &path) {
                    violations.push(format!(
                        "Rust unit test file must live under src/tests: {}",
                        display_relative(ctx, &path)
                    ));
                    continue;
                }
                if is_allowed_rust_test_file(&package_root, &path) {
                    continue;
                }

                let contents = std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                if contains_test_attribute(&contents) {
                    violations.push(format!(
                        "Rust test body must live under src/tests or crate-level tests: {}",
                        display_relative(ctx, &path)
                    ));
                }
            }
        }
    }
    Ok(())
}

fn collect_rust_package_roots(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut package_roots = Vec::new();
    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.join("src").is_dir() {
            package_roots.push(path);
        }
    }
    package_roots.sort();
    Ok(package_roots)
}

fn collect_files_with_extension(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                if is_ignored_layout_dir(&path) {
                    continue;
                }
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn collect_files_with_suffix(root: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                if is_ignored_layout_dir(&path) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.ends_with(suffix) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_ignored_layout_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("target" | "node_modules" | ".build" | "third_party" | ".venv")
    )
}

fn is_inverted_rust_test_file(package_root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(package_root) else {
        return false;
    };
    let components = path_components(relative);
    components.first() == Some(&"src")
        && components.get(1) != Some(&"tests")
        && components
            .iter()
            .skip(1)
            .any(|component| *component == "tests")
}

fn is_allowed_rust_test_file(package_root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(package_root) else {
        return false;
    };
    let components = path_components(relative);
    matches!(components.as_slice(), ["tests", ..] | ["src", "tests", ..])
}

fn path_components(path: &Path) -> Vec<&str> {
    path.components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect()
}

fn contains_test_attribute(contents: &str) -> bool {
    contents.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("#[test]") || line.starts_with("#[tokio::test]")
    })
}

fn node_test_backends(ctx: &BuildContext, backend: &Backend) -> Result<Vec<Backend>> {
    let backends = test_backends(backend);
    if *backend != Backend::All {
        return Ok(backends);
    }

    let triplet = node_platform_triplet()?;
    let available = backends
        .into_iter()
        .filter(|backend| node_backend_artifact(ctx, backend, triplet).is_file())
        .collect::<Vec<_>>();
    if available.is_empty() {
        anyhow::bail!("Node --backend all did not produce any testable backend artifacts");
    }
    Ok(available)
}

fn node_backend_artifact(ctx: &BuildContext, backend: &Backend, triplet: &str) -> PathBuf {
    ctx.node_artifacts_dir().join(format!(
        "cogentlm_node_{}.{}.node",
        backend.as_str(),
        triplet
    ))
}

fn node_platform_triplet() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok("win32-x64-msvc"),
        ("macos", "x86_64") => Ok("darwin-x64"),
        ("macos", "aarch64") => Ok("darwin-arm64"),
        ("linux", "x86_64") => Ok("linux-x64-gnu"),
        (os, arch) => anyhow::bail!("unsupported Node test platform: {os} {arch}"),
    }
}

fn test_backends(backend: &Backend) -> Vec<Backend> {
    match backend {
        Backend::All if cfg!(target_os = "macos") => vec![Backend::Cpu, Backend::Metal],
        Backend::All => vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda],
        backend => vec![*backend],
    }
}

fn build_python_test_wheel(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<PathBuf> {
    let uv_exe = setup_uv(sh, ctx)?;
    output::run_command(
        "Ensuring Python 3.12 is available through uv",
        apply_uv_env(ctx, cmd!(sh, "{uv_exe} python install 3.12")),
    )?;

    let python_dir = ctx.bindings_python_dir();
    let dist_dir = ctx.python_artifacts_dir().join("test-wheels");
    prepare_python_test_wheel_dir(sh, &dist_dir)?;
    let target_dir = ctx
        .cargo_python_target_dir(Some(backend))
        .join("pytest-wheel");
    sh.create_dir(&target_dir)?;

    let _dir = sh.push_dir(&python_dir);
    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(
            sh,
            "{uv_exe} tool run maturin build --release --out {dist_dir}"
        ),
    )
    .env("CARGO_TARGET_DIR", &target_dir);
    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(backend))?;
    if *backend != Backend::Cpu {
        maturin_cmd = maturin_cmd.arg("--features").arg(backend.as_str());
    }
    output::run_command("Building Python test wheel", maturin_cmd)?;
    find_wheel_artifact(&dist_dir)?.with_context(|| {
        format!(
            "maturin did not produce a wheel artifact in {}",
            dist_dir.display()
        )
    })
}

fn prepare_python_test_wheel_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;
    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            sh.remove_path(path)?;
        }
    }
    Ok(())
}

fn find_wheel_artifact(dir: &Path) -> Result<Option<PathBuf>> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn ensure_workspace_bun_install(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Installing workspace Bun dependencies",
        cmd!(sh, "bun install"),
    )
}

fn app_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut tests = Vec::new();
    collect_app_test_files(&ctx.apps_root(), &mut tests)?;
    tests.sort();
    Ok(tests)
}

fn package_ts_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    collect_files_with_suffix(&ctx.npm_package_dir().join("tests"), APP_TEST_SUFFIX)
}

fn node_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut files = collect_files_with_suffix(&ctx.bindings_node_dir().join("tests"), ".test.mjs")?;
    files.extend(collect_files_with_suffix(
        &ctx.bindings_node_dir().join("tests"),
        ".test.js",
    )?);
    files.sort();
    files.dedup();
    Ok(files)
}

fn python_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    collect_files_with_extension(&ctx.bindings_python_dir().join("tests"), "py")
}

fn rust_target_case_files(ctx: &BuildContext, targets: &[RustTestTarget]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for target in targets {
        let package_root = rust_package_root(ctx, target.package)?;
        match target.kind {
            Some(RustTestKind::Lib | RustTestKind::Bin(_)) => {
                files.extend(collect_files_with_extension(
                    &package_root.join("src").join("tests"),
                    "rs",
                )?);
            }
            Some(RustTestKind::Package) | None => {
                files.extend(collect_files_with_extension(
                    &package_root.join("src").join("tests"),
                    "rs",
                )?);
                files.extend(collect_files_with_extension(
                    &package_root.join("tests"),
                    "rs",
                )?);
            }
            Some(RustTestKind::Test(test_name)) => {
                let test_file = package_root.join("tests").join(format!("{test_name}.rs"));
                if test_file.exists() {
                    files.push(test_file);
                }
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn rust_package_root(ctx: &BuildContext, package: &str) -> Result<PathBuf> {
    let relative = match package {
        "cogentlm-core" => &["crates", "core"],
        "cogentlm-shard" => &["crates", "shard"],
        "cogentlm-sys" => &["crates", "sys"],
        "cogentlm-engine" => &["crates", "engine"],
        "cogentlm-providers" => &["crates", "providers"],
        "cogentlm-client" => &["crates", "client"],
        "cogentlm-cli" => &["crates", "cli"],
        "xtask" => &["crates", "xtask"],
        "cogentlm-napi" => &["bindings", "node"],
        "cogentlm-py" => &["bindings", "python"],
        "cogentlm-wasm" => &["bindings", "wasm"],
        _ => anyhow::bail!("unknown Rust test package: {package}"),
    };
    Ok(relative
        .iter()
        .fold(ctx.workspace_root().to_path_buf(), |path, component| {
            path.join(component)
        }))
}

fn collect_app_test_files(root: &Path, tests: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }

    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if path.is_dir() {
            if SKIPPED_APP_TEST_DIRS.contains(&file_name) {
                continue;
            }
            collect_app_test_files(&path, tests)?;
            continue;
        }

        if path.is_file() && file_name.ends_with(APP_TEST_SUFFIX) {
            tests.push(path);
        }
    }

    Ok(())
}

fn display_relative(ctx: &BuildContext, path: &Path) -> String {
    path.strip_prefix(ctx.workspace_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

fn python_venv_exe(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn cli_binary_file_name() -> &'static str {
    if cfg!(windows) {
        "cogentlm.exe"
    } else {
        "cogentlm"
    }
}

fn parse_lcov_summary(path: &Path) -> Result<LcovSummary> {
    if !path.exists() {
        return Ok(LcovSummary::default());
    }

    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut summary = LcovSummary::default();
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("LF:") {
            summary.found += value.parse::<usize>().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("LH:") {
            summary.hit += value.parse::<usize>().unwrap_or(0);
        }
    }
    Ok(summary)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestCategory {
    Whitebox,
    Interface,
}

impl TestCategory {
    fn as_str(&self) -> &'static str {
        match self {
            TestCategory::Whitebox => "whitebox",
            TestCategory::Interface => "interface",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TestSuite {
    id: TestSuiteId,
    category: TestCategory,
    description: &'static str,
    requirements: &'static str,
    profiles: &'static [TestProfile],
    coverage: bool,
    backend_policy: BackendPolicy,
    runner: SuiteRunner,
    discoverer: CaseDiscoverer,
}

impl TestSuite {
    fn profile_labels(&self) -> Vec<&'static str> {
        self.profiles.iter().map(TestProfile::as_str).collect()
    }
}

#[derive(Clone, Copy, Debug)]
enum BackendPolicy {
    None,
    Any,
    ConcreteOnly,
}

impl BackendPolicy {
    fn as_str(&self) -> &'static str {
        match self {
            BackendPolicy::None => "none",
            BackendPolicy::Any => "any",
            BackendPolicy::ConcreteOnly => "concrete-only",
        }
    }

    fn validate(&self, suite_id: TestSuiteId, backend: Backend) -> Result<()> {
        if matches!(self, BackendPolicy::ConcreteOnly) && backend == Backend::All {
            anyhow::bail!(
                "{} requires a concrete backend; choose cpu, vulkan, cuda, or metal",
                suite_id.as_str()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum SuiteRunner {
    Layout,
    RustTargets(&'static [RustTestTarget]),
    PackageTs,
    AppTs,
    NodePackage,
    PythonPackage,
    BrowserRuntime,
    ModelSmoke,
    LlamaBackendOps,
}

#[derive(Clone, Copy, Debug)]
enum CaseDiscoverer {
    None,
    RustTargets(&'static [RustTestTarget]),
    PackageTs,
    AppTs,
    NodePackage,
    PythonPackage,
}

impl CaseDiscoverer {
    fn as_str(&self) -> &'static str {
        match self {
            CaseDiscoverer::None => "none",
            CaseDiscoverer::RustTargets(_) => "rust-targets",
            CaseDiscoverer::PackageTs => "package-ts",
            CaseDiscoverer::AppTs => "app-ts",
            CaseDiscoverer::NodePackage => "node-package",
            CaseDiscoverer::PythonPackage => "python-package",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RustTestTarget {
    package: &'static str,
    kind: Option<RustTestKind>,
}

impl RustTestTarget {
    const fn package(package: &'static str) -> Self {
        Self {
            package,
            kind: Some(RustTestKind::Package),
        }
    }

    const fn lib(package: &'static str) -> Self {
        Self {
            package,
            kind: Some(RustTestKind::Lib),
        }
    }

    const fn bin(package: &'static str, binary: &'static str) -> Self {
        Self {
            package,
            kind: Some(RustTestKind::Bin(binary)),
        }
    }

    const fn test(package: &'static str, test_name: &'static str) -> Self {
        Self {
            package,
            kind: Some(RustTestKind::Test(test_name)),
        }
    }

    fn label(&self) -> String {
        match self.kind {
            Some(RustTestKind::Lib) => format!("{} --lib", self.package),
            Some(RustTestKind::Bin(binary)) => format!("{} --bin {binary}", self.package),
            Some(RustTestKind::Test(test_name)) => format!("{} --test {test_name}", self.package),
            Some(RustTestKind::Package) | None => self.package.to_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RustTestKind {
    Package,
    Lib,
    Bin(&'static str),
    Test(&'static str),
}

struct SuiteRunOptions<'a> {
    backend: Backend,
    model: Option<&'a Path>,
    offline: bool,
    package: Option<&'a str>,
}

#[derive(Clone, Debug)]
struct TestCase {
    suite_id: TestSuiteId,
    name: String,
    path: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct LcovSummary {
    found: usize,
    hit: usize,
}

impl LcovSummary {
    fn percent(&self) -> f64 {
        if self.found == 0 {
            0.0
        } else {
            (self.hit as f64 / self.found as f64) * 100.0
        }
    }

    fn as_json(&self) -> serde_json::Value {
        json!({
            "found": self.found,
            "hit": self.hit,
            "percent": self.percent(),
        })
    }
}

#[cfg(test)]
pub(crate) fn catalog_suite_ids() -> BTreeSet<&'static str> {
    TEST_SUITES.iter().map(|suite| suite.id.as_str()).collect()
}

#[cfg(test)]
#[path = "tests/test_tests.rs"]
mod test_tests;
