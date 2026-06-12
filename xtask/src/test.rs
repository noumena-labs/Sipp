//! Cataloged workspace test and coverage orchestration.

use crate::cli::{
    Backend, LlamaBackendOpsMode, LlamaBackendOpsOutput, RunLlamaBackendOpsArgs, TestCommands,
    TestGroupFilter, TestListArgs, TestListFormat, TestSmokeArgs, TestSmokeCase, TestSmokeCaseArgs,
    TestSmokeCommands, TestSmokeExampleBrowserArgs, TestSmokeExamplesGroupArgs,
    TestSmokeFullGroupArgs, TestSmokeGroupTarget, TestSmokeLlamaArgs, TestSmokeModelArgs,
    TestSmokePlaygroundBrowserArgs, TestSmokeSuiteTarget, TestSuiteId, TestUnitArgs,
    TestUnitCommands, TestUnitGroupTarget, TestUnitLayer, TestUnitSuiteTarget, TestVerifyArgs,
    TestVerifyTarget,
};
use crate::javascript;
use crate::output;
use crate::sample_model::{self, SampleModelOptions};
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::{apply_uv_env, setup_uv};
use crate::utils::{ensure_playwright_chromium, BuildContext};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/test_tests.rs"]
mod test_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const DEFAULT_SMOKE_PROMPT: &str = "Describe browser LLM inference.";
const DEFAULT_SMOKE_MAX_TOKENS: u32 = 64;
const DEFAULT_SMOKE_TEMPERATURE: f32 = 0.0;
const GATEWAY_SMOKE_BIND: &str = "127.0.0.1:18787";
const GATEWAY_SMOKE_TOKEN: &str = "example-gateway-smoke-token";
const GATEWAY_SMOKE_START_TIMEOUT: Duration = Duration::from_secs(300);
const RUST_GENERATION_SMOKE_EXAMPLES: &[&str] = &["query", "chat", "embed"];
const RUST_GATEWAY_SMOKE_EXAMPLES: &[&str] = &["gateway_query", "gateway_chat", "gateway_embed"];
const NODE_GENERATION_SMOKE_SCRIPTS: &[&str] = &["query.mjs", "chat.mjs", "embed.mjs"];
const NODE_GATEWAY_SMOKE_SCRIPTS: &[&str] =
    &["gateway_query.mjs", "gateway_chat.mjs", "gateway_embed.mjs"];
const PYTHON_GENERATION_SMOKE_SCRIPTS: &[&str] = &["query.py", "chat.py", "embed.py"];
const PYTHON_GATEWAY_SMOKE_SCRIPTS: &[&str] =
    &["gateway_query.py", "gateway_chat.py", "gateway_embed.py"];
const DEMO_TEST_SUFFIX: &str = ".test.ts";
const SKIPPED_DEMO_TEST_DIRS: &[&str] =
    &["node_modules", "dist", "build", "out", ".vite", "coverage"];

const RUST_CRATE_TEST_TARGETS: &[RustTestTarget] = &[
    RustTestTarget::lib("cogentlm"),
    RustTestTarget::lib("cogentlm-sys"),
    RustTestTarget::lib("cogentlm-gateway"),
    RustTestTarget::package("cogentlm-gateway-server"),
    RustTestTarget::bin("cogentlm-cli", "cogentlm"),
];
const XTASK_TEST_TARGETS: &[RustTestTarget] = &[RustTestTarget::package("xtask")];
const RUST_BINDING_TEST_TARGETS: &[RustTestTarget] = &[
    RustTestTarget::package("cogentlm-napi"),
    RustTestTarget::package("cogentlm-py"),
    RustTestTarget::package("cogentlm-wasm"),
];
const RUST_PUBLIC_API_TEST_TARGETS: &[RustTestTarget] =
    &[RustTestTarget::test("cogentlm", "public_api")];
const CLI_BLACK_BOX_TEST_TARGETS: &[RustTestTarget] =
    &[RustTestTarget::test("cogentlm-cli", "cli_black_box")];

const XTASK_SOURCE_ROOTS: &[&str] = &["xtask/src"];
const RUST_CRATE_SOURCE_ROOTS: &[&str] = &[
    "crates/cogentlm/src",
    "crates/sys/src",
    "lib/gateway/src",
    "apps/gateway-server/src",
    "apps/cli/src",
];
const RUST_BINDING_SOURCE_ROOTS: &[&str] = &[
    "bindings/node/src",
    "bindings/python/src",
    "bindings/wasm/src",
    "bindings/wasm/native",
];
const PACKAGE_TS_SOURCE_ROOTS: &[&str] = &["lib/web/src"];
const DEMO_TS_SOURCE_ROOTS: &[&str] = &["demos"];
const RUST_PUBLIC_API_SOURCE_ROOTS: &[&str] = &["crates/cogentlm/src"];
const CLI_SOURCE_ROOTS: &[&str] = &["apps/cli/src"];
const NODE_PACKAGE_SOURCE_ROOTS: &[&str] = &[
    "bindings/node/src",
    "lib/node/index.d.ts",
    "lib/node/router.js",
    "lib/node/router.d.ts",
];
const PYTHON_PACKAGE_SOURCE_ROOTS: &[&str] = &["lib/python/python/cogentlm"];
const CLI_SMOKE_SOURCE_ROOTS: &[&str] = &["apps/cli/src"];
const RUST_SMOKE_SOURCE_ROOTS: &[&str] = &["examples/rust/src"];
const NODE_SMOKE_SOURCE_ROOTS: &[&str] = &["examples/node", "lib/node"];
const PYTHON_SMOKE_SOURCE_ROOTS: &[&str] = &["examples/python", "lib/python"];
const GATEWAY_EXAMPLE_SMOKE_SOURCE_ROOTS: &[&str] = &[
    "examples/gateway",
    "examples/rust/src",
    "examples/node",
    "examples/python",
    "crates/cogentlm/src/gateway_core",
    "lib/gateway/src",
    "apps/gateway-server/src",
    "crates/cogentlm/src/providers",
];
const BROWSER_EXAMPLE_SMOKE_SOURCE_ROOTS: &[&str] = &["examples/web", "lib/web/src"];
const BROWSER_PLAYGROUND_SMOKE_SOURCE_ROOTS: &[&str] = &["tools/playground"];
const PUBLIC_DOC_RUST_FILES: &[&str] = &[
    "crates/cogentlm/src/lib.rs",
    "bindings/node/src/lib.rs",
    "bindings/python/src/lib.rs",
];
const PUBLIC_DOC_TYPESCRIPT_FILES: &[&str] = &[
    "lib/web/src/index.ts",
    "lib/web/src/character/index.ts",
    "lib/web/src/orchestrator/index.ts",
    "lib/node/index.d.ts",
    "lib/node/router.d.ts",
];
const PUBLIC_DOC_PYTHON_FILES: &[&str] = &["lib/python/python/cogentlm/__init__.py"];

const TEST_SUITES: &[TestSuite] = &[
    TestSuite {
        id: TestSuiteId::Xtask,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Whitebox),
        description: "xtask CLI and orchestration unit tests",
        requirements: "cargo",
        source_roots: XTASK_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(XTASK_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(XTASK_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::RustCrates,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Whitebox),
        description: "Rust unit tests for core workspace crates",
        requirements: "cargo, native toolchain",
        source_roots: RUST_CRATE_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_CRATE_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_CRATE_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::RustBindings,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Whitebox),
        description: "Rust unit tests for Node, Python, and WASM binding crates",
        requirements: "cargo, native toolchain",
        source_roots: RUST_BINDING_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_BINDING_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_BINDING_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::PackageTs,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Whitebox),
        description: "browser package TypeScript tests under lib/web/tests",
        requirements: "bun, wasm build",
        source_roots: PACKAGE_TS_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::PackageTs,
        discoverer: CaseDiscoverer::PackageTs,
    },
    TestSuite {
        id: TestSuiteId::DemoTs,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Whitebox),
        description: "browser demo TypeScript tests discovered under demos/",
        requirements: "bun, wasm build",
        source_roots: DEMO_TS_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::DemoTs,
        discoverer: CaseDiscoverer::DemoTs,
    },
    TestSuite {
        id: TestSuiteId::RustPublicApi,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Interface),
        description: "crate-level public API integration tests",
        requirements: "cargo, native toolchain",
        source_roots: RUST_PUBLIC_API_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(RUST_PUBLIC_API_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(RUST_PUBLIC_API_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::Cli,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Interface),
        description: "CLI black-box integration test",
        requirements: "cargo, native toolchain",
        source_roots: CLI_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::RustTargets(CLI_BLACK_BOX_TEST_TARGETS),
        discoverer: CaseDiscoverer::RustTargets(CLI_BLACK_BOX_TEST_TARGETS),
    },
    TestSuite {
        id: TestSuiteId::NodePackage,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Interface),
        description: "Node binding build/import/router tests",
        requirements: "cargo, node",
        source_roots: NODE_PACKAGE_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::Any,
        runner: SuiteRunner::NodePackage,
        discoverer: CaseDiscoverer::NodePackage,
    },
    TestSuite {
        id: TestSuiteId::PythonPackage,
        group: TestGroup::Unit,
        layer: Some(TestUnitLayer::Interface),
        description: "Python wheel install/import pytest suite",
        requirements: "cargo, uv, python",
        source_roots: PYTHON_PACKAGE_SOURCE_ROOTS,
        coverage: true,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::PythonPackage,
        discoverer: CaseDiscoverer::PythonPackage,
    },
    TestSuite {
        id: TestSuiteId::CliSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "CLI local generation smoke",
        requirements: "sample GGUF model, cargo",
        source_roots: CLI_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::CliSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::RustSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Rust local generation example smoke",
        requirements: "sample GGUF model, cargo",
        source_roots: RUST_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::RustSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::NodeSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Node local generation example smoke",
        requirements: "sample GGUF model, cargo, node",
        source_roots: NODE_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::NodeSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::PythonSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Python local generation example smoke",
        requirements: "sample GGUF model, cargo, uv, python",
        source_roots: PYTHON_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::PythonSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::ExampleGatewaySmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Real local gateway example smoke under examples/gateway",
        requirements: "sample GGUF model, cargo, node, uv, python",
        source_roots: GATEWAY_EXAMPLE_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::ExampleGatewaySmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::ExampleBrowserSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Browser query/chat/embed example smoke under examples/web",
        requirements: "sample GGUF model, bun, wasm build, playwright chromium",
        source_roots: BROWSER_EXAMPLE_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::ConcreteOnly,
        runner: SuiteRunner::ExampleBrowserSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::PlaygroundBrowserSmoke,
        group: TestGroup::Smoke,
        layer: None,
        description: "Playground browser runtime smoke under tools/playground",
        requirements: "bun, wasm build, playwright chromium",
        source_roots: BROWSER_PLAYGROUND_SMOKE_SOURCE_ROOTS,
        coverage: false,
        backend_policy: BackendPolicy::None,
        runner: SuiteRunner::PlaygroundBrowserSmoke,
        discoverer: CaseDiscoverer::None,
    },
    TestSuite {
        id: TestSuiteId::LlamaBackendOps,
        group: TestGroup::Smoke,
        layer: None,
        description: "llama.cpp backend correctness operations",
        requirements: "cmake, ninja, native backend",
        source_roots: &["crates/sys/native"],
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
        TestCommands::Unit(args) => run_unit(sh, ctx, &args),
        TestCommands::Smoke(args) => run_smoke(sh, ctx, &args),
        TestCommands::Verify(args) => run_verify(sh, ctx, &args),
    }
}

fn run_list(ctx: &BuildContext, args: &TestListArgs) -> Result<()> {
    let mut suites = selected_list_suites(args)?;
    let mut cases = if args.cases || args.search.is_some() {
        discover_cases(ctx, &suites)?
    } else {
        Vec::new()
    };
    if let Some(search) = args.search.as_deref() {
        apply_search_filter(&mut suites, &mut cases, search);
    }
    if !args.cases {
        cases.clear();
    }

    match args.format {
        TestListFormat::Text => print_text_list(&suites, &cases),
        TestListFormat::Json => print_json_list(&suites, &cases),
    }
}

fn run_unit(sh: &Shell, ctx: &BuildContext, args: &TestUnitArgs) -> Result<()> {
    let selection = selected_unit_suites(args)?;
    validate_suite_backends(&selection.suites, selection.backend)?;

    let options = SuiteRunOptions {
        backend: selection.backend,
        model: None,
        offline: false,
        package: selection.package.as_deref(),
        prompt: DEFAULT_SMOKE_PROMPT,
        max_tokens: DEFAULT_SMOKE_MAX_TOKENS,
        temperature: DEFAULT_SMOKE_TEMPERATURE,
        cases: &[],
        example_browser: BrowserSmokeRunOptions::default(),
        playground_browser: BrowserSmokeRunOptions::default(),
        llama: LlamaBackendOpsRunOptions::default(),
    };

    run_selected_suites(
        sh,
        ctx,
        &selection.suites,
        &options,
        selection.filters.clone(),
    )
}

fn run_smoke(sh: &Shell, ctx: &BuildContext, args: &TestSmokeArgs) -> Result<()> {
    let selection = selected_smoke_suites(args)?;
    validate_suite_backends(&selection.suites, selection.backend)?;

    let example_browser = BrowserSmokeRunOptions {
        host: selection.browser_host.as_deref(),
        port: selection.browser_port,
        timeout_ms: selection.example_browser_timeout_ms,
        require_webgpu: false,
    };
    let playground_browser = BrowserSmokeRunOptions {
        host: selection.browser_host.as_deref(),
        port: selection.browser_port,
        timeout_ms: selection.playground_browser_timeout_ms,
        require_webgpu: selection.require_webgpu,
    };
    let llama = LlamaBackendOpsRunOptions {
        mode: selection.llama_mode,
        op: selection.llama_op.as_deref(),
        params: selection.llama_params.as_deref(),
        output: selection.llama_output,
    };
    let options = SuiteRunOptions {
        backend: selection.backend,
        model: selection.model.as_deref(),
        offline: selection.offline,
        package: None,
        prompt: &selection.prompt,
        max_tokens: selection.max_tokens,
        temperature: selection.temperature,
        cases: &selection.cases,
        example_browser,
        playground_browser,
        llama,
    };

    run_selected_suites(
        sh,
        ctx,
        &selection.suites,
        &options,
        selection.filters.clone(),
    )
}

fn run_selected_suites(
    sh: &Shell,
    ctx: &BuildContext,
    suites: &[&TestSuite],
    options: &SuiteRunOptions<'_>,
    filters: Value,
) -> Result<()> {
    let mut coverage_state = RunCoverageState::default();
    let mut completed_coverage_suites = Vec::new();
    let mut report = RunReport::new(filters, suites);

    for (index, suite) in suites.iter().enumerate() {
        let started_at = Instant::now();
        if let Err(error) = prepare_suite_case_report_dir(ctx, suite) {
            output::warning(format!(
                "Could not prepare structured test report directory for {}: {error:#}",
                suite.id.as_str()
            ));
        }
        print_suite_start(suite, index + 1, suites.len());
        let result = run_suite(sh, ctx, suite, options, &mut coverage_state);
        let duration_ms = started_at.elapsed().as_millis();
        match result {
            Ok(outcome) => {
                if suite.coverage {
                    completed_coverage_suites.push(*suite);
                    write_coverage_summary(
                        &ctx.build_root().join("coverage"),
                        coverage_report_areas(&completed_coverage_suites),
                    )?;
                }
                let suite_report =
                    SuiteReport::passed(ctx, suite, duration_ms, outcome.counts, outcome.cases);
                print_suite_report(&suite_report);
                report.suites.push(suite_report);
                report.finish("passed");
                write_run_report(ctx, &report)?;
            }
            Err(error) => {
                let message = format!("{error:#}");
                let cases = failed_suite_cases(ctx, suite, options, &error);
                let counts = case_counts(&cases);
                let suite_report =
                    SuiteReport::failed(ctx, suite, duration_ms, message, counts, cases);
                print_suite_report(&suite_report);
                report.suites.push(suite_report);
                for remaining in suites.iter().skip(index + 1) {
                    report.suites.push(SuiteReport::not_run(ctx, remaining));
                }
                report.finish("failed");
                write_run_report(ctx, &report)?;
                print_run_summary(&report);
                output::path("Test run report", &test_run_report_json(ctx));
                return Err(error);
            }
        }
    }

    report.finish("passed");
    write_run_report(ctx, &report)?;
    print_run_summary(&report);
    output::path("Test run report", &test_run_report_json(ctx));
    Ok(())
}

fn run_suite(
    sh: &Shell,
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
    coverage_state: &mut RunCoverageState,
) -> Result<SuiteOutcome> {
    match suite.runner {
        SuiteRunner::RustTargets(targets) => {
            let package = if suite.id == TestSuiteId::RustCrates {
                options.package
            } else {
                None
            };
            run_rust_target_tests(sh, ctx, targets, package, coverage_state)
        }
        SuiteRunner::PackageTs => run_package_ts_tests(sh, ctx),
        SuiteRunner::DemoTs => run_demo_ts_tests(sh, ctx),
        SuiteRunner::NodePackage => run_node_package_tests(sh, ctx, &options.backend),
        SuiteRunner::PythonPackage => run_python_package_tests(sh, ctx, &options.backend),
        SuiteRunner::CliSmoke => run_cli_model_smoke(sh, ctx, options),
        SuiteRunner::RustSmoke => run_rust_model_smoke(sh, ctx, options),
        SuiteRunner::NodeSmoke => run_node_model_smoke(sh, ctx, options),
        SuiteRunner::PythonSmoke => run_python_model_smoke(sh, ctx, options),
        SuiteRunner::ExampleBrowserSmoke => run_browser_example_smoke(sh, ctx, options),
        SuiteRunner::ExampleGatewaySmoke => run_example_gateway_smoke(sh, ctx, options),
        SuiteRunner::PlaygroundBrowserSmoke => {
            run_playground_browser_runtime_smoke(sh, ctx, &options.playground_browser)
        }
        SuiteRunner::LlamaBackendOps => {
            run_llama_backend_ops_suite(sh, ctx, &options.backend, &options.llama)
        }
    }?;

    successful_suite_outcome(ctx, suite, options)
}

fn successful_suite_outcome(
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
) -> Result<SuiteOutcome> {
    let cases = successful_suite_cases(ctx, suite, options)?;
    let counts = if cases.is_empty() {
        known_success_counts(ctx, suite, options)?
    } else {
        case_counts(&cases)
    };
    Ok(SuiteOutcome { counts, cases })
}

fn successful_suite_cases(
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
) -> Result<Vec<TestCaseReport>> {
    let mut cases = Vec::new();
    match suite.discoverer {
        CaseDiscoverer::None => {}
        CaseDiscoverer::RustTargets(targets) if suite.id == TestSuiteId::RustCrates => {
            let targets = filtered_rust_targets(targets, options.package)?;
            discover_rust_cases(
                ctx,
                suite.id,
                rust_target_case_files(ctx, &targets)?,
                &mut cases,
            )?;
        }
        _ => discover_suite_cases(ctx, suite, &mut cases)?,
    }

    if cases.is_empty() {
        return Ok(synthetic_suite_cases(suite, options, CaseStatus::Passed));
    }

    Ok(cases
        .into_iter()
        .map(|case| TestCaseReport::from_case(case, CaseStatus::Passed))
        .collect())
}

fn failed_suite_cases(
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
    error: &anyhow::Error,
) -> Vec<TestCaseReport> {
    match read_structured_case_reports(ctx, suite.id) {
        Ok(cases) if !cases.is_empty() => return cases,
        Ok(_) => {}
        Err(parse_error) => {
            output::warning(format!(
                "Could not parse structured test reports for {}: {parse_error:#}",
                suite.id.as_str()
            ));
        }
    }

    if let Some(log_path) = command_failure_log_path(error) {
        match std::fs::read_to_string(&log_path) {
            Ok(contents) => {
                let cases = parse_libtest_case_reports(&contents, suite.id);
                if !cases.is_empty() {
                    return cases;
                }
            }
            Err(read_error) => {
                output::warning(format!(
                    "Could not read subprocess log {}: {read_error:#}",
                    log_path.display()
                ));
            }
        }
    }

    synthetic_suite_cases(suite, options, CaseStatus::Unknown)
}

fn command_failure_log_path(error: &anyhow::Error) -> Option<PathBuf> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<output::CommandFailure>()
            .map(|failure| failure.log_path().to_path_buf())
    })
}

fn synthetic_suite_cases(
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
    status: CaseStatus,
) -> Vec<TestCaseReport> {
    let names = match suite.runner {
        SuiteRunner::CliSmoke => vec!["cli local generation".to_owned()],
        SuiteRunner::RustSmoke => selected_rust_smoke_examples(options.cases)
            .into_iter()
            .map(str::to_owned)
            .collect(),
        SuiteRunner::NodeSmoke => selected_node_smoke_scripts(options.cases)
            .into_iter()
            .map(str::to_owned)
            .collect(),
        SuiteRunner::PythonSmoke => selected_python_smoke_scripts(options.cases)
            .into_iter()
            .map(str::to_owned)
            .collect(),
        SuiteRunner::ExampleBrowserSmoke => selected_smoke_cases(options.cases)
            .into_iter()
            .map(|case| case.as_str().to_owned())
            .collect(),
        SuiteRunner::ExampleGatewaySmoke => selected_gateway_smoke_labels(options.cases),
        SuiteRunner::PlaygroundBrowserSmoke => vec!["playground browser runtime".to_owned()],
        SuiteRunner::LlamaBackendOps => vec!["llama.cpp backend ops".to_owned()],
        _ => Vec::new(),
    };

    names
        .into_iter()
        .map(|name| TestCaseReport {
            suite_id: suite.id,
            name,
            path: None,
            status,
            error: None,
        })
        .collect()
}

fn case_counts(cases: &[TestCaseReport]) -> Option<TestCounts> {
    if cases.is_empty() || cases.iter().any(|case| case.status == CaseStatus::Unknown) {
        return None;
    }

    let mut counts = TestCounts::default();
    for case in cases {
        match case.status {
            CaseStatus::Passed => counts.passed += 1,
            CaseStatus::Failed => counts.failed += 1,
            CaseStatus::Skipped => counts.skipped += 1,
            CaseStatus::Unknown => {}
        }
    }
    Some(counts)
}

fn known_success_counts(
    ctx: &BuildContext,
    suite: &TestSuite,
    options: &SuiteRunOptions<'_>,
) -> Result<Option<TestCounts>> {
    let total = match suite.runner {
        SuiteRunner::RustTargets(_) => discovered_suite_case_count(ctx, suite, options.package)?,
        SuiteRunner::PackageTs
        | SuiteRunner::DemoTs
        | SuiteRunner::NodePackage
        | SuiteRunner::PythonPackage => discovered_suite_case_count(ctx, suite, None)?,
        SuiteRunner::CliSmoke
        | SuiteRunner::PlaygroundBrowserSmoke
        | SuiteRunner::LlamaBackendOps => 1,
        SuiteRunner::RustSmoke => selected_rust_smoke_examples(options.cases).len(),
        SuiteRunner::NodeSmoke => selected_node_smoke_scripts(options.cases).len(),
        SuiteRunner::PythonSmoke => selected_python_smoke_scripts(options.cases).len(),
        SuiteRunner::ExampleGatewaySmoke => selected_gateway_smoke_labels(options.cases).len(),
        SuiteRunner::ExampleBrowserSmoke => selected_smoke_cases(options.cases).len(),
    };
    Ok(Some(TestCounts::passed(total)))
}

fn discovered_suite_case_count(
    ctx: &BuildContext,
    suite: &TestSuite,
    package: Option<&str>,
) -> Result<usize> {
    let mut cases = Vec::new();
    match suite.discoverer {
        CaseDiscoverer::RustTargets(targets) if suite.id == TestSuiteId::RustCrates => {
            let targets = filtered_rust_targets(targets, package)?;
            discover_rust_cases(
                ctx,
                suite.id,
                rust_target_case_files(ctx, &targets)?,
                &mut cases,
            )?;
        }
        _ => discover_suite_cases(ctx, suite, &mut cases)?,
    }
    Ok(cases.len())
}

fn print_run_summary(report: &RunReport) {
    let summary = report.summary();
    output::phase("Test summary");
    output::detail(
        "Suites",
        format!(
            "{} passed, {} failed, {} not run, {} total",
            summary.suites.passed,
            summary.suites.failed,
            summary.suites.not_run,
            summary.suites.total()
        ),
    );
    if summary.counts.known_suites > 0 {
        output::detail(
            "Tests/checks",
            format!(
                "{} passed, {} failed, {} skipped, {} total",
                summary.counts.counts.passed,
                summary.counts.counts.failed,
                summary.counts.counts.skipped,
                summary.counts.counts.total()
            ),
        );
    }
    if !summary.counts.unknown_suites.is_empty() {
        output::detail("Unknown counts", summary.counts.unknown_suites.join(", "));
    }
    for suite in report
        .suites
        .iter()
        .filter(|suite| suite.status == "failed")
    {
        if let Some(error) = suite.error.as_deref() {
            output::warning(format!("{} failed: {error}", suite.id));
        }
    }
}

fn print_suite_start(suite: &TestSuite, index: usize, total: usize) {
    output::step(format!(
        "Suite {index}/{total}: {} - {}",
        suite.id.as_str(),
        suite.description
    ));
}

fn print_suite_report(suite: &SuiteReport) {
    let counts = suite
        .counts
        .map(|counts| counts.markdown())
        .unwrap_or_else(|| "counts unknown".to_owned());
    let duration = suite
        .duration_ms
        .map(|duration| output::elapsed(std::time::Duration::from_millis(duration)))
        .unwrap_or_else(|| "not run".to_owned());
    let message = format!("Suite {} {} ({duration}; {counts})", suite.id, suite.status);

    match suite.status.as_str() {
        "passed" => output::success(message),
        "failed" => output::warning(message),
        _ => output::detail("Suite", message),
    }

    for case in &suite.cases {
        print_case_report(&suite.id, case);
    }
}

fn print_case_report(suite_id: &str, case: &TestCaseReport) {
    let path = case.path.as_deref().unwrap_or("-");
    let message = format!("{suite_id} :: {} ({path})", case.name);
    match case.status {
        CaseStatus::Passed => output::success(format!("PASS {message}")),
        CaseStatus::Failed => output::warning(format!("FAIL {message}")),
        CaseStatus::Skipped => output::detail("SKIP", message),
        CaseStatus::Unknown => output::detail("UNKNOWN", message),
    }
}

fn verify_test_structure(ctx: &BuildContext) -> Result<()> {
    output::phase("Test structure");
    let violations = collect_test_structure_violations(ctx)?;

    if violations.is_empty() {
        output::success("Test structure check passed");
        return Ok(());
    }

    for violation in &violations {
        output::warning(violation);
    }
    anyhow::bail!(
        "test structure check failed with {} violation(s)",
        violations.len()
    )
}

fn collect_test_structure_violations(ctx: &BuildContext) -> Result<Vec<String>> {
    let mut violations = Vec::new();
    collect_package_test_layout_violations(ctx, &mut violations)?;
    collect_rust_test_layout_violations(ctx, &mut violations)?;
    collect_catalog_ownership_violations(ctx, &mut violations)?;
    Ok(violations)
}

fn run_rust_target_tests(
    sh: &Shell,
    ctx: &BuildContext,
    targets: &[RustTestTarget],
    package: Option<&str>,
    coverage_state: &mut RunCoverageState,
) -> Result<()> {
    output::phase("Rust tests");
    let targets = filtered_rust_targets(targets, package)?;
    run_rust_coverage_targets(sh, ctx, &targets, coverage_state)
}

fn run_rust_coverage_targets(
    sh: &Shell,
    ctx: &BuildContext,
    targets: &[RustTestTarget],
    coverage_state: &mut RunCoverageState,
) -> Result<()> {
    ensure_cargo_llvm_cov()?;
    let coverage_root = ctx.build_root().join("coverage");
    let rust_dir = coverage_root.join("rust");
    sh.create_dir(&rust_dir)?;
    let rust_lcov = rust_dir.join("lcov.info");
    let rust_html = rust_dir.join("html");

    let _dir = sh.push_dir(ctx.workspace_root());
    for target in targets {
        let package = target.package;
        let mut lcov_cmd = cmd!(
            sh,
            "cargo llvm-cov --lcov --output-path {rust_lcov} --ignore-filename-regex third_party|llama\\.cpp|\\.build|target|tests|examples|demos|tools -p {package}"
        );
        if coverage_state.rust_started {
            lcov_cmd = lcov_cmd.arg("--no-clean");
        }
        if let Some(test_kind) = target.kind {
            match test_kind {
                RustTestKind::Lib => {
                    lcov_cmd = lcov_cmd.arg("--lib");
                }
                RustTestKind::Bin(binary) => {
                    lcov_cmd = lcov_cmd.arg("--bin").arg(binary);
                }
                RustTestKind::Test(test_name) => {
                    lcov_cmd = lcov_cmd.arg("--test").arg(test_name);
                }
                RustTestKind::Package => {}
            }
        }
        let lcov_cmd = apply_toolchains(sh, ctx, lcov_cmd, None)?;
        output::run_test_command(
            format!("Running {} Rust coverage tests", target.label()),
            lcov_cmd,
        )?;
        coverage_state.rust_started = true;
    }

    let html_cmd = cmd!(
        sh,
        "cargo llvm-cov --html --no-run --output-dir {rust_html} --ignore-filename-regex third_party|llama\\.cpp|\\.build|target|tests|examples|demos|tools"
    );
    let html_cmd = apply_toolchains(sh, ctx, html_cmd, None)?;
    output::run_build_command("Writing Rust HTML coverage report", html_cmd)?;
    Ok(())
}

fn run_package_ts_tests(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("White-box browser package TypeScript tests");
    targets::wasm::build(sh, ctx)?;
    let browser_package_dir = ctx.browser_package_dir();
    let junit_path = suite_case_report_file(ctx, TestSuiteId::PackageTs, "bun-package.xml");
    let _dir = sh.push_dir(&browser_package_dir);
    output::run_test_command(
        "Running browser package TypeScript tests",
        cmd!(
            sh,
            "bun test tests --reporter=junit --reporter-outfile {junit_path}"
        ),
    )
}

fn run_demo_ts_tests(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("White-box browser demo TypeScript tests");
    let tests = demo_test_files(ctx)?;
    if tests.is_empty() {
        output::warning("No demo TypeScript tests were found under demos/");
        return Ok(());
    }

    let workspaces = demo_test_workspaces(ctx, &tests)?;
    ensure_javascript_workspace_dependencies(sh, ctx, &workspaces)?;
    targets::wasm::build(sh, ctx)?;

    output::detail("Test files", tests.len());
    for (index, test) in tests.into_iter().enumerate() {
        let workspace = demo_test_workspace(ctx, &test)?;
        let relative_test = test.strip_prefix(&workspace).unwrap_or(&test);
        let junit_path =
            suite_case_report_file(ctx, TestSuiteId::DemoTs, &format!("demo-{index}.xml"));
        let _dir = sh.push_dir(&workspace);
        output::run_test_command(
            format!(
                "Running demo TypeScript test {}",
                display_relative(ctx, &test)
            ),
            cmd!(
                sh,
                "bun test {relative_test} --reporter=junit --reporter-outfile {junit_path}"
            ),
        )?;
    }
    Ok(())
}

fn run_node_package_tests(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<()> {
    output::phase("Interface Node package tests");
    targets::node::build(sh, ctx, Some(backend))?;

    let coverage_dir = ctx.build_root().join("coverage").join("node");
    sh.create_dir(&coverage_dir)?;

    let node_dir = ctx.node_package_dir();
    ensure_javascript_workspace_dependencies(sh, ctx, &[node_dir.clone()])?;
    let _node = sh.push_dir(&node_dir);
    for backend in node_test_backends(ctx, backend)? {
        let report_path = suite_case_report_file(
            ctx,
            TestSuiteId::NodePackage,
            &format!("node-{}.tap", backend.as_str()),
        );
        output::run_test_command(
            format!("Running Node.js package tests ({})", backend.as_str()),
            cmd!(
                sh,
                "bunx c8 --reporter=lcov --reports-dir {coverage_dir} node --test --test-reporter=tap --test-reporter-destination {report_path} tests/router.test.mjs"
            )
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
    output::run_build_command(
        "Creating Python test virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_build_command(
        "Installing Python test wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel} pytest pytest-cov"
            ),
        ),
    )?;

    let python_tests = ctx.python_package_project_dir().join("tests");
    let coverage_dir = ctx.build_root().join("coverage").join("python");
    sh.create_dir(&coverage_dir)?;
    let python_lcov = coverage_dir.join("lcov.info");
    let python_cobertura = coverage_dir.join("cobertura.xml");
    let python_html = coverage_dir.join("html");
    let junit_path = suite_case_report_file(ctx, TestSuiteId::PythonPackage, "pytest.xml");
    output::run_test_command(
        "Running Python package pytest suite",
        cmd!(
            sh,
            "{python_exe} -m pytest {python_tests} --junitxml={junit_path} --cov=cogentlm --cov-report=lcov:{python_lcov} --cov-report=xml:{python_cobertura} --cov-report=html:{python_html}"
        ),
    )
}

fn run_cli_model_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("CLI local generation smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    output::path("Model", &model);
    output::detail("Backend", options.backend.as_str());

    targets::cli::build(sh, ctx, Some(&options.backend))?;
    run_cli_smoke(sh, ctx, &model, options)
}

fn run_rust_model_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Rust local generation smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    run_rust_generation_smoke(sh, ctx, &model, options)
}

fn run_node_model_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Node local generation smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    run_node_generation_smoke(sh, ctx, &model, options)
}

fn run_python_model_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Python local generation smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    run_python_generation_smoke(sh, ctx, &model, options)
}

fn run_browser_example_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Browser example smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    output::path("Model", &model);
    let example_dir = ctx.browser_example_dir();
    ensure_javascript_workspace_dependencies(sh, ctx, &[example_dir.clone()])?;
    targets::wasm::build(sh, ctx)?;
    ensure_playwright_chromium(sh, ctx)?;

    output::path("Example workspace", &example_dir);
    output::path("Artifact directory", &ctx.example_artifacts_dir("web"));
    let _example_dir = sh.push_dir(&example_dir);
    let timeout_ms = options.example_browser.timeout_ms.to_string();
    let max_tokens = options.max_tokens.to_string();
    let mut smoke_cmd = cmd!(sh, "node scripts/browser-example-smoke.mjs")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(options.prompt)
        .arg("--max-tokens")
        .arg(max_tokens)
        .arg("--timeout-ms")
        .arg(timeout_ms)
        .env("PLAYWRIGHT_BROWSERS_PATH", ctx.playwright_browsers_dir());
    if let Some(host) = options.example_browser.host {
        smoke_cmd = smoke_cmd.arg("--host").arg(host);
    }
    if let Some(port) = options.example_browser.port {
        smoke_cmd = smoke_cmd.arg("--port").arg(port.to_string());
    }
    for case in selected_smoke_cases(options.cases) {
        smoke_cmd = smoke_cmd.arg("--case").arg(case.as_str());
    }
    output::run_test_command("Running browser example smoke", smoke_cmd)
}

fn run_playground_browser_runtime_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &BrowserSmokeRunOptions<'_>,
) -> Result<()> {
    output::phase("Playground browser runtime smoke");
    let playground_dir = ctx.playground_dir();
    ensure_javascript_workspace_dependencies(sh, ctx, &[playground_dir.clone()])?;
    targets::wasm::build(sh, ctx)?;
    ensure_playwright_chromium(sh, ctx)?;

    output::path("Playground workspace", &playground_dir);
    output::path("Artifact directory", &ctx.tool_artifacts_dir("playground"));
    let _playground_dir = sh.push_dir(&playground_dir);
    let timeout_ms = options.timeout_ms.to_string();
    let mut smoke_cmd = cmd!(sh, "node scripts/playground-runtime-smoke.mjs")
        .arg("--timeout-ms")
        .arg(timeout_ms)
        .env("PLAYWRIGHT_BROWSERS_PATH", ctx.playwright_browsers_dir());
    if let Some(host) = options.host {
        smoke_cmd = smoke_cmd.arg("--host").arg(host);
    }
    if let Some(port) = options.port {
        smoke_cmd = smoke_cmd.arg("--port").arg(port.to_string());
    }
    if options.require_webgpu {
        smoke_cmd = smoke_cmd.arg("--require-webgpu");
    }
    output::run_test_command("Running playground browser runtime smoke", smoke_cmd)
}

fn run_llama_backend_ops_suite(
    sh: &Shell,
    ctx: &BuildContext,
    backend: &Backend,
    options: &LlamaBackendOpsRunOptions<'_>,
) -> Result<()> {
    crate::run::run_llama_backend_ops(
        sh,
        ctx,
        &RunLlamaBackendOpsArgs {
            backend: *backend,
            mode: options.mode,
            op: options.op.map(str::to_owned),
            params: options.params.map(str::to_owned),
            output: options.output,
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

fn run_cli_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("CLI smoke");
    let cli_dir = ctx.cli_artifacts_dir();
    let cli_exe = cli_dir.join(cli_binary_file_name());
    if !cli_exe.is_file() {
        anyhow::bail!("CLI executable was not staged at {}", cli_exe.display());
    }

    let _dir = sh.push_dir(&cli_dir);
    let max_tokens = options.max_tokens.to_string();
    let temperature = format_temperature(options.temperature);
    let mut smoke_cmd = cmd!(sh, "{cli_exe}")
        .arg(model)
        .arg(options.prompt)
        .arg("--max-tokens")
        .arg(max_tokens)
        .arg("--temperature")
        .arg(temperature);
    if options.backend != Backend::Cpu {
        smoke_cmd = smoke_cmd.arg("--backend").arg(options.backend.as_str());
    }
    output::run_test_command("Running CLI local inference smoke", smoke_cmd)
}

fn run_rust_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Rust CogentClient generation smoke");
    output::path("Model", model);
    output::detail("Backend", options.backend.as_str());

    let _dir = sh.push_dir(ctx.workspace_root());
    for example in selected_rust_smoke_examples(options.cases) {
        let mut smoke_cmd = cmd!(sh, "cargo run -p cogentlm-rust-examples");
        if options.backend != Backend::Cpu {
            smoke_cmd = smoke_cmd.arg("--features").arg(options.backend.as_str());
        }
        smoke_cmd = smoke_cmd
            .arg("--bin")
            .arg(example)
            .arg("--")
            .arg(model)
            .arg(options.prompt)
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&options.backend))?;
        output::run_test_command(
            format!("Running Rust {} smoke: {example}", options.backend.as_str()),
            smoke_cmd,
        )
        .with_context(|| format!("Rust {} smoke failed: {example}", options.backend.as_str()))?;
    }

    Ok(())
}

fn run_node_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Node.js generation smoke");
    output::path("Model", model);
    output::detail("Backend", options.backend.as_str());
    targets::node::build(sh, ctx, Some(&options.backend))?;

    let node_dir = ctx.workspace_root().join("examples").join("node");
    let _dir = sh.push_dir(&node_dir);
    for smoke_script in selected_node_smoke_scripts(options.cases) {
        let mut smoke_cmd = cmd!(sh, "node")
            .arg(smoke_script)
            .arg(model)
            .arg(options.prompt)
            .env("COGENTLM_NODE_BACKEND", options.backend.as_str())
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&options.backend))?;
        output::run_test_command(
            format!(
                "Running Node {} smoke: {smoke_script}",
                options.backend.as_str()
            ),
            smoke_cmd,
        )
        .with_context(|| {
            format!(
                "Node {} smoke failed: {smoke_script}",
                options.backend.as_str()
            )
        })?;
    }

    Ok(())
}

fn run_python_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    output::phase("Python generation smoke");
    output::path("Model", model);
    output::detail("Backend", options.backend.as_str());
    let wheel = build_python_test_wheel(sh, ctx, &options.backend)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx
        .tmp_dir()
        .join("python-model-smoke")
        .join(options.backend.as_str());
    output::run_build_command(
        "Creating Python smoke virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_build_command(
        "Installing Python smoke wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel}"
            ),
        ),
    )?;

    let python_dir = ctx.workspace_root().join("examples").join("python");
    let _dir = sh.push_dir(&python_dir);
    for smoke_script in selected_python_smoke_scripts(options.cases) {
        let mut smoke_cmd = cmd!(sh, "{python_exe}")
            .arg(smoke_script)
            .arg(model)
            .arg(options.prompt)
            .env("COGENTLM_PYTHON_BACKEND", options.backend.as_str())
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&options.backend))?;
        output::run_test_command(
            format!(
                "Running Python {} smoke: {smoke_script}",
                options.backend.as_str()
            ),
            smoke_cmd,
        )
        .with_context(|| {
            format!(
                "Python {} smoke failed: {smoke_script}",
                options.backend.as_str()
            )
        })?;
    }

    Ok(())
}

fn run_example_gateway_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
) -> Result<()> {
    if matches!(options.backend, Backend::All) {
        anyhow::bail!(
            "example-gateway requires a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    output::phase("Example local gateway smoke");
    let model = resolve_smoke_model(sh, ctx, options.model, options.offline)?;
    output::path("Model", &model);
    output::detail("Backend", options.backend.as_str());
    run_rust_gateway_smoke(sh, ctx, options, &model)?;
    run_node_gateway_smoke(sh, ctx, options, &model)?;
    run_python_gateway_smoke(sh, ctx, options, &model)?;
    Ok(())
}

fn run_rust_gateway_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
    model: &Path,
) -> Result<()> {
    output::phase("Rust gateway client smoke");
    let _dir = sh.push_dir(ctx.workspace_root());
    for example in selected_rust_gateway_smoke_examples(options.cases) {
        let mut gateway = GatewaySmokeProcess::start(sh, ctx, model, &options.backend)?;
        wait_for_gateway_smoke(gateway.child_mut())?;
        let mut features = vec!["gateway"];
        if options.backend != Backend::Cpu {
            features.push(options.backend.as_str());
        }
        let mut smoke_cmd = cmd!(sh, "cargo run -p cogentlm-rust-examples")
            .arg("--features")
            .arg(features.join(","))
            .arg("--bin")
            .arg(example)
            .arg("--")
            .arg(model)
            .arg(gateway_smoke_target(example))
            .arg(options.prompt)
            .env("COGENTLM_GATEWAY_URL", gateway_smoke_url())
            .env("COGENTLM_GATEWAY_TOKEN", GATEWAY_SMOKE_TOKEN)
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, None)?;
        output::run_test_command(format!("Running Rust gateway smoke: {example}"), smoke_cmd)
            .with_context(|| format!("Rust gateway smoke failed: {example}"))?;
        drop(gateway);
    }
    Ok(())
}

fn run_node_gateway_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
    model: &Path,
) -> Result<()> {
    output::phase("Node.js gateway client smoke");
    targets::node::build(sh, ctx, Some(&Backend::Cpu))?;

    let node_dir = ctx.workspace_root().join("examples").join("node");
    let _dir = sh.push_dir(&node_dir);
    for script in selected_node_gateway_smoke_scripts(options.cases) {
        let mut gateway = GatewaySmokeProcess::start(sh, ctx, model, &options.backend)?;
        wait_for_gateway_smoke(gateway.child_mut())?;
        let mut smoke_cmd = cmd!(sh, "node")
            .arg(script)
            .arg(model)
            .arg(gateway_smoke_target(script))
            .arg(options.prompt)
            .env("COGENTLM_NODE_BACKEND", "cpu")
            .env("COGENTLM_GATEWAY_URL", gateway_smoke_url())
            .env("COGENTLM_GATEWAY_TOKEN", GATEWAY_SMOKE_TOKEN)
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&Backend::Cpu))?;
        output::run_test_command(format!("Running Node gateway smoke: {script}"), smoke_cmd)
            .with_context(|| format!("Node gateway smoke failed: {script}"))?;
        drop(gateway);
    }
    Ok(())
}

fn run_python_gateway_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    options: &SuiteRunOptions<'_>,
    model: &Path,
) -> Result<()> {
    output::phase("Python gateway client smoke");
    let wheel = build_python_test_wheel(sh, ctx, &Backend::Cpu)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx.tmp_dir().join("python-gateway-smoke");
    output::run_build_command(
        "Creating Python gateway smoke virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_build_command(
        "Installing Python gateway smoke wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel}"
            ),
        ),
    )?;

    let python_dir = ctx.workspace_root().join("examples").join("python");
    let _dir = sh.push_dir(&python_dir);
    for script in selected_python_gateway_smoke_scripts(options.cases) {
        let mut gateway = GatewaySmokeProcess::start(sh, ctx, model, &options.backend)?;
        wait_for_gateway_smoke(gateway.child_mut())?;
        let mut smoke_cmd = cmd!(sh, "{python_exe}")
            .arg(script)
            .arg(model)
            .arg(gateway_smoke_target(script))
            .arg(options.prompt)
            .env("COGENTLM_PYTHON_BACKEND", "cpu")
            .env("COGENTLM_GATEWAY_URL", gateway_smoke_url())
            .env("COGENTLM_GATEWAY_TOKEN", GATEWAY_SMOKE_TOKEN)
            .env("COGENTLM_MAX_TOKENS", options.max_tokens.to_string())
            .env(
                "COGENTLM_TEMPERATURE",
                format_temperature(options.temperature),
            );
        smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&Backend::Cpu))?;
        output::run_test_command(format!("Running Python gateway smoke: {script}"), smoke_cmd)
            .with_context(|| format!("Python gateway smoke failed: {script}"))?;
        drop(gateway);
    }
    Ok(())
}

fn wait_for_gateway_smoke(child: &mut Child) -> Result<()> {
    output::phase("Waiting for local gateway");
    let started_at = Instant::now();
    while started_at.elapsed() < GATEWAY_SMOKE_START_TIMEOUT {
        if let Some(status) = child.try_wait().context("failed to poll gateway process")? {
            anyhow::bail!("gateway process exited before readiness: {status}");
        }
        if gateway_smoke_probe() {
            output::success(format!("Gateway is ready at {}", gateway_smoke_url()));
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    anyhow::bail!(
        "gateway did not answer readiness probe at {} within {} seconds",
        gateway_smoke_url(),
        GATEWAY_SMOKE_START_TIMEOUT.as_secs()
    )
}

fn gateway_smoke_probe() -> bool {
    std::net::TcpStream::connect(GATEWAY_SMOKE_BIND).is_ok()
}

fn gateway_smoke_url() -> String {
    format!("http://{GATEWAY_SMOKE_BIND}")
}

fn gateway_smoke_target(case_name: &str) -> &'static str {
    let _ = case_name;
    "local"
}

struct GatewaySmokeProcess {
    child: Option<Child>,
}

impl GatewaySmokeProcess {
    fn start(sh: &Shell, ctx: &BuildContext, model: &Path, backend: &Backend) -> Result<Self> {
        let log_dir = ctx.command_logs_dir();
        std::fs::create_dir_all(&log_dir)
            .with_context(|| format!("failed to create {}", log_dir.display()))?;
        let log_path = log_dir.join("example-gateway-smoke.log");
        let log = std::fs::File::create(&log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        output::path("Gateway log", &log_path);

        let _dir = sh.push_dir(ctx.workspace_root());
        let mut gateway_cmd = cmd!(sh, "cargo run -p cogentlm-gateway-example");
        if *backend != Backend::Cpu {
            gateway_cmd = gateway_cmd.arg("--features").arg(backend.as_str());
        }
        gateway_cmd = gateway_cmd
            .arg("--")
            .arg("--model")
            .arg(model)
            .arg("--bind")
            .arg(GATEWAY_SMOKE_BIND);
        gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(backend))?;

        let mut command: Command = gateway_cmd.quiet().into();
        command
            .stdout(Stdio::from(
                log.try_clone()
                    .context("failed to clone gateway smoke log handle")?,
            ))
            .stderr(Stdio::from(log));

        let child = command.spawn().context("failed to start gateway process")?;
        Ok(Self { child: Some(child) })
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("gateway child is present until drop")
    }
}

impl Drop for GatewaySmokeProcess {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn selected_smoke_cases(cases: &[TestSmokeCase]) -> Vec<TestSmokeCase> {
    if cases.is_empty() {
        return vec![
            TestSmokeCase::Query,
            TestSmokeCase::Chat,
            TestSmokeCase::Embed,
        ];
    }

    let mut selected = Vec::new();
    for case in cases {
        if !selected.contains(case) {
            selected.push(*case);
        }
    }
    selected
}

fn selected_rust_smoke_examples(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => RUST_GENERATION_SMOKE_EXAMPLES[0],
            TestSmokeCase::Chat => RUST_GENERATION_SMOKE_EXAMPLES[1],
            TestSmokeCase::Embed => RUST_GENERATION_SMOKE_EXAMPLES[2],
        })
        .collect()
}

fn selected_rust_gateway_smoke_examples(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => RUST_GATEWAY_SMOKE_EXAMPLES[0],
            TestSmokeCase::Chat => RUST_GATEWAY_SMOKE_EXAMPLES[1],
            TestSmokeCase::Embed => RUST_GATEWAY_SMOKE_EXAMPLES[2],
        })
        .collect()
}

fn selected_node_smoke_scripts(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => NODE_GENERATION_SMOKE_SCRIPTS[0],
            TestSmokeCase::Chat => NODE_GENERATION_SMOKE_SCRIPTS[1],
            TestSmokeCase::Embed => NODE_GENERATION_SMOKE_SCRIPTS[2],
        })
        .collect()
}

fn selected_node_gateway_smoke_scripts(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => NODE_GATEWAY_SMOKE_SCRIPTS[0],
            TestSmokeCase::Chat => NODE_GATEWAY_SMOKE_SCRIPTS[1],
            TestSmokeCase::Embed => NODE_GATEWAY_SMOKE_SCRIPTS[2],
        })
        .collect()
}

fn selected_python_smoke_scripts(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => PYTHON_GENERATION_SMOKE_SCRIPTS[0],
            TestSmokeCase::Chat => PYTHON_GENERATION_SMOKE_SCRIPTS[1],
            TestSmokeCase::Embed => PYTHON_GENERATION_SMOKE_SCRIPTS[2],
        })
        .collect()
}

fn selected_python_gateway_smoke_scripts(cases: &[TestSmokeCase]) -> Vec<&'static str> {
    selected_smoke_cases(cases)
        .into_iter()
        .map(|case| match case {
            TestSmokeCase::Query => PYTHON_GATEWAY_SMOKE_SCRIPTS[0],
            TestSmokeCase::Chat => PYTHON_GATEWAY_SMOKE_SCRIPTS[1],
            TestSmokeCase::Embed => PYTHON_GATEWAY_SMOKE_SCRIPTS[2],
        })
        .collect()
}

fn selected_gateway_smoke_labels(cases: &[TestSmokeCase]) -> Vec<String> {
    let mut labels = Vec::new();
    for example in selected_rust_gateway_smoke_examples(cases) {
        labels.push(format!("rust {example}"));
    }
    for script in selected_node_gateway_smoke_scripts(cases) {
        labels.push(format!("node {script}"));
    }
    for script in selected_python_gateway_smoke_scripts(cases) {
        labels.push(format!("python {script}"));
    }
    labels
}

fn format_temperature(temperature: f32) -> String {
    if temperature.fract() == 0.0 {
        format!("{temperature:.0}")
    } else {
        temperature.to_string()
    }
}

fn run_verify(sh: &Shell, ctx: &BuildContext, args: &TestVerifyArgs) -> Result<()> {
    if args.target == TestVerifyTarget::PublicDocs {
        return run_public_docs_verify(ctx, args);
    }

    let suites = selected_verify_suites(args)?;
    output::phase("Test verification");
    output::detail(
        "Suites",
        suites
            .iter()
            .map(|suite| suite.id.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );

    let mut report = VerifyReport::new(args, &suites);
    let mut failures = Vec::new();

    let structure_result = verify_test_structure(ctx);
    report.checks.push(VerifyCheckReport::from_result(
        "test-structure",
        &structure_result,
    ));
    if let Err(error) = structure_result {
        failures.push(format!("{error:#}"));
    }

    if args.changed {
        let changed_result = validate_changed_coverage(ctx, &suites);
        report.checks.push(VerifyCheckReport::from_result(
            "changed-test-ownership",
            &changed_result,
        ));
        if let Err(error) = changed_result {
            failures.push(format!("{error:#}"));
        }
    } else {
        report
            .checks
            .push(VerifyCheckReport::skipped("changed-test-ownership"));
    }

    let coverage_root = ctx.build_root().join("coverage");
    sh.create_dir(&coverage_root)?;

    let report_areas = coverage_report_areas(&suites);
    let coverage_result = (|| -> Result<CoverageSummaries> {
        if report_areas.rust {
            write_rust_coverage_reports(sh, ctx)?;
        }
        write_coverage_summary(&coverage_root, report_areas)
    })();
    report.checks.push(VerifyCheckReport::from_result(
        "coverage-artifacts",
        &coverage_result,
    ));
    match coverage_result {
        Ok(summaries) => report.coverage = Some(summaries),
        Err(error) => failures.push(format!("{error:#}")),
    }

    let status = if failures.is_empty() {
        "passed"
    } else {
        "failed"
    };
    report.finish(status);
    write_verify_report(ctx, &report)?;
    output::path("Test verification report", &test_verify_report_json(ctx));

    if failures.is_empty() {
        output::success(format!(
            "Coverage reports verified under {}",
            coverage_root.display()
        ));
        return Ok(());
    }

    anyhow::bail!("test verification failed: {}", failures.join("; "))
}

fn run_public_docs_verify(ctx: &BuildContext, args: &TestVerifyArgs) -> Result<()> {
    output::phase("Public API documentation verification");
    output::detail(
        "Files",
        format!(
            "{} Rust, {} TypeScript, {} Python",
            PUBLIC_DOC_RUST_FILES.len(),
            PUBLIC_DOC_TYPESCRIPT_FILES.len(),
            PUBLIC_DOC_PYTHON_FILES.len()
        ),
    );

    let suites: [&TestSuite; 0] = [];
    let mut report = VerifyReport::new(args, &suites);
    let docs_result = verify_public_api_docs(ctx);
    report.checks.push(VerifyCheckReport::from_result(
        "public-api-docs",
        &docs_result,
    ));
    let status = if docs_result.is_ok() {
        "passed"
    } else {
        "failed"
    };
    report.finish(status);
    write_verify_report(ctx, &report)?;
    output::path("Test verification report", &test_verify_report_json(ctx));

    match docs_result {
        Ok(()) => {
            output::success("Public API docs verified");
            Ok(())
        }
        Err(error) => anyhow::bail!("public API documentation verification failed: {error:#}"),
    }
}

fn verify_public_api_docs(ctx: &BuildContext) -> Result<()> {
    let mut violations = Vec::new();
    for relative in PUBLIC_DOC_RUST_FILES {
        let path = ctx.workspace_root().join(relative);
        collect_rust_public_doc_violations(&path, relative, &mut violations)?;
    }
    for relative in PUBLIC_DOC_TYPESCRIPT_FILES {
        let path = ctx.workspace_root().join(relative);
        collect_typescript_public_doc_violations(&path, relative, &mut violations)?;
    }
    for relative in PUBLIC_DOC_PYTHON_FILES {
        let path = ctx.workspace_root().join(relative);
        collect_python_public_doc_violations(&path, relative, &mut violations)?;
    }

    if violations.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "{} public API documentation issue(s):\n{}",
        violations.len(),
        violations.join("\n")
    )
}

fn collect_rust_public_doc_violations(
    path: &Path,
    display: &str,
    violations: &mut Vec<String>,
) -> Result<()> {
    let lines = read_public_doc_lines(path)?;
    if !first_non_blank_starts_with(&lines, "//!") {
        violations.push(format!("{display}:1: missing crate or module rustdoc"));
    }

    let check_facade_items = display == "crates/cogentlm/src/lib.rs";
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let check_top_level_facade_item = check_facade_items && line.starts_with("pub ");
        let Some(target) =
            rust_public_doc_target(&lines, index, trimmed, check_top_level_facade_item)
        else {
            continue;
        };
        if !has_previous_rust_doc_comment(&lines, index) {
            violations.push(format!(
                "{}:{}: missing rustdoc before {}",
                display,
                index + 1,
                target
            ));
        }
    }
    Ok(())
}

fn rust_public_doc_target(
    lines: &[String],
    index: usize,
    trimmed: &str,
    check_top_level_facade_item: bool,
) -> Option<&'static str> {
    if trimmed.starts_with("#[pyclass") {
        return Some("PyO3 class export");
    }
    if trimmed.starts_with("#[pyfunction") {
        return Some("PyO3 function export");
    }
    if trimmed.starts_with("#[pymodule") {
        return Some("PyO3 module export");
    }
    if trimmed.starts_with("#[napi(object)]") {
        return Some("N-API object export");
    }
    if trimmed.starts_with("#[napi(string_enum") {
        return Some("N-API enum export");
    }
    if trimmed.starts_with("#[napi(js_name")
        && next_non_attribute_line(lines, index).is_some_and(|line| line.starts_with("pub struct "))
    {
        return Some("N-API class export");
    }
    if trimmed == "#[napi]"
        && next_non_attribute_line(lines, index).is_some_and(|line| line.starts_with("pub fn "))
    {
        return Some("N-API function export");
    }
    if check_top_level_facade_item && is_rust_facade_public_item(trimmed) {
        return Some("Rust facade item");
    }
    None
}

fn is_rust_facade_public_item(trimmed: &str) -> bool {
    trimmed.starts_with("pub use ")
        || trimmed.starts_with("pub mod ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub trait ")
}

fn next_non_attribute_line(lines: &[String], index: usize) -> Option<&str> {
    lines
        .iter()
        .skip(index + 1)
        .map(|line| line.trim())
        .find(|line| !line.is_empty() && !line.starts_with("#["))
}

fn has_previous_rust_doc_comment(lines: &[String], index: usize) -> bool {
    previous_non_blank_line(lines, index).is_some_and(|line| {
        line.starts_with("///") || line.starts_with("//!") || line.starts_with("#[doc")
    })
}

fn collect_typescript_public_doc_violations(
    path: &Path,
    display: &str,
    violations: &mut Vec<String>,
) -> Result<()> {
    let lines = read_public_doc_lines(path)?;
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !is_typescript_public_export(trimmed) {
            continue;
        }
        if !has_previous_typescript_doc_comment(&lines, index) {
            violations.push(format!(
                "{}:{}: missing JSDoc/TSDoc before export",
                display,
                index + 1
            ));
        }
    }
    Ok(())
}

fn is_typescript_public_export(trimmed: &str) -> bool {
    trimmed.starts_with("export ")
        && !trimmed.starts_with("export *")
        && !trimmed.starts_with("export {};")
}

fn has_previous_typescript_doc_comment(lines: &[String], index: usize) -> bool {
    previous_non_blank_line(lines, index)
        .is_some_and(|line| line.starts_with("/**") || line == "*/" || line.ends_with("*/"))
}

fn collect_python_public_doc_violations(
    path: &Path,
    display: &str,
    violations: &mut Vec<String>,
) -> Result<()> {
    let lines = read_public_doc_lines(path)?;
    if !first_non_blank_starts_with(&lines, "\"\"\"")
        && !first_non_blank_starts_with(&lines, "r\"\"\"")
    {
        violations.push(format!("{display}:1: missing module docstring"));
    }

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if indent != 0 || (!trimmed.starts_with("def ") && !trimmed.starts_with("class ")) {
            continue;
        }
        let Some(name) = python_item_name(trimmed) else {
            continue;
        };
        if name.starts_with('_') {
            continue;
        }
        if !has_following_python_docstring(&lines, index) {
            violations.push(format!(
                "{}:{}: missing Python docstring for {}",
                display,
                index + 1,
                name
            ));
        }
    }
    Ok(())
}

fn python_item_name(trimmed: &str) -> Option<&str> {
    let name_start = trimmed.find(' ')? + 1;
    let rest = &trimmed[name_start..];
    let name_end = rest.find(['(', ':'])?;
    Some(&rest[..name_end])
}

fn has_following_python_docstring(lines: &[String], index: usize) -> bool {
    lines
        .iter()
        .skip(index + 1)
        .map(|line| line.trim_start())
        .find(|line| !line.is_empty())
        .is_some_and(|line| line.starts_with("\"\"\"") || line.starts_with("r\"\"\""))
}

fn first_non_blank_starts_with(lines: &[String], prefix: &str) -> bool {
    lines
        .iter()
        .map(|line| line.trim_start())
        .find(|line| !line.is_empty())
        .is_some_and(|line| line.starts_with(prefix))
}

fn previous_non_blank_line(lines: &[String], index: usize) -> Option<&str> {
    lines[..index]
        .iter()
        .rev()
        .map(|line| line.trim())
        .find(|line| !line.is_empty())
}

fn read_public_doc_lines(path: &Path) -> Result<Vec<String>> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(source.lines().map(str::to_owned).collect())
}

fn write_rust_coverage_reports(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    ensure_cargo_llvm_cov()?;
    let coverage_root = ctx.build_root().join("coverage");
    let rust_dir = coverage_root.join("rust");
    sh.create_dir(&rust_dir)?;
    let rust_lcov = rust_dir.join("lcov.info");
    let rust_html = rust_dir.join("html");

    let _root = sh.push_dir(ctx.workspace_root());
    output::run_build_command(
        "Writing Rust LCOV report",
        cmd!(
            sh,
            "cargo llvm-cov report --lcov --output-path {rust_lcov} --ignore-filename-regex third_party|llama\\.cpp|\\.build|target|tests|examples|demos|tools"
        ),
    )?;
    output::run_build_command(
        "Writing Rust HTML report",
        cmd!(
            sh,
            "cargo llvm-cov report --html --output-dir {rust_html} --ignore-filename-regex third_party|llama\\.cpp|\\.build|target|tests|examples|demos|tools"
        ),
    )?;
    Ok(())
}

fn write_coverage_summary(
    coverage_root: &Path,
    areas: CoverageReportAreas,
) -> Result<CoverageSummaries> {
    let rust = parse_lcov_summary(&coverage_root.join("rust").join("lcov.info"))?;
    let node = parse_lcov_summary(&coverage_root.join("node").join("lcov.info"))?;
    let python = parse_lcov_summary(&coverage_root.join("python").join("lcov.info"))?;
    if areas.rust {
        ensure_non_empty_coverage("Rust/native", &rust)?;
    }
    if areas.node {
        ensure_non_empty_coverage("Node wrapper", &node)?;
    }
    if areas.python {
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
    Ok(CoverageSummaries { rust, node, python })
}

fn ensure_non_empty_coverage(label: &str, summary: &LcovSummary) -> Result<()> {
    if summary.found == 0 {
        anyhow::bail!("{label} coverage report did not include any first-party lines");
    }
    Ok(())
}

fn coverage_report_areas(suites: &[&TestSuite]) -> CoverageReportAreas {
    let mut areas = CoverageReportAreas::default();
    for suite in suites {
        match suite.runner {
            SuiteRunner::RustTargets(_) => areas.rust = true,
            SuiteRunner::NodePackage => areas.node = true,
            SuiteRunner::PythonPackage => areas.python = true,
            SuiteRunner::PackageTs
            | SuiteRunner::DemoTs
            | SuiteRunner::CliSmoke
            | SuiteRunner::RustSmoke
            | SuiteRunner::NodeSmoke
            | SuiteRunner::PythonSmoke
            | SuiteRunner::ExampleGatewaySmoke
            | SuiteRunner::ExampleBrowserSmoke
            | SuiteRunner::PlaygroundBrowserSmoke
            | SuiteRunner::LlamaBackendOps => {}
        }
    }
    areas
}

fn validate_changed_coverage(ctx: &BuildContext, suites: &[&TestSuite]) -> Result<()> {
    output::phase("Coverage change validation");
    let changed_paths = changed_workspace_files(ctx)?;
    if changed_paths.is_empty() {
        output::success("No changed files found for coverage validation");
        return Ok(());
    }

    let test_ownership = catalog_file_ownership(ctx)?;
    let changed_test_suites = changed_paths
        .iter()
        .filter_map(|path| test_ownership.get(path))
        .flat_map(|owners| owners.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut checked_sources = 0;
    let mut violations = Vec::new();

    for path in &changed_paths {
        if test_ownership.contains_key(path) || !is_first_party_source_path(path) {
            continue;
        }
        let owners = source_owner_suites(path, suites);
        if owners.is_empty() {
            continue;
        }
        checked_sources += 1;
        if owners
            .iter()
            .any(|suite| changed_test_suites.contains(&suite.id))
        {
            continue;
        }

        violations.push(format!(
            "{} changed, but no catalog-owned tests changed for suite(s): {}",
            path,
            owners
                .iter()
                .map(|suite| suite.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    if violations.is_empty() {
        output::success(format!(
            "Coverage change validation passed for {checked_sources} source file(s)"
        ));
        return Ok(());
    }

    for violation in &violations {
        output::warning(violation);
    }
    anyhow::bail!(
        "coverage change validation failed with {} source file(s) lacking matching test changes",
        violations.len()
    )
}

fn changed_workspace_files(ctx: &BuildContext) -> Result<BTreeSet<String>> {
    let mut paths = BTreeSet::new();
    collect_git_paths(
        ctx,
        &["diff", "--name-only", "--diff-filter=ACMRT", "HEAD"],
        &mut paths,
    )?;
    collect_git_paths(
        ctx,
        &["ls-files", "--others", "--exclude-standard"],
        &mut paths,
    )?;
    Ok(paths)
}

fn collect_git_paths(
    ctx: &BuildContext,
    args: &[&str],
    paths: &mut BTreeSet<String>,
) -> Result<()> {
    let safe_directory = ctx
        .workspace_root()
        .display()
        .to_string()
        .replace('\\', "/");
    let output = Command::new("git")
        .arg("-c")
        .arg(format!("safe.directory={safe_directory}"))
        .args(args)
        .current_dir(ctx.workspace_root())
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        anyhow::bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let path = normalize_relative_path(line);
        if !path.is_empty() {
            paths.insert(path);
        }
    }
    Ok(())
}

fn source_owner_suites<'a>(path: &str, suites: &'a [&TestSuite]) -> Vec<&'a TestSuite> {
    suites
        .iter()
        .copied()
        .filter(|suite| {
            suite
                .source_roots
                .iter()
                .any(|root| path_matches_root(path, root))
        })
        .collect()
}

fn is_first_party_source_path(path: &str) -> bool {
    let Some(extension) = Path::new(path).extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(
        extension,
        "rs" | "ts" | "tsx" | "js" | "mjs" | "py" | "c" | "cc" | "cpp" | "h" | "hpp"
    ) && !is_probable_test_path(path)
}

fn is_probable_test_path(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("/src/tests/")
        || path.ends_with(".test.ts")
        || path.ends_with(".test.tsx")
        || path.ends_with(".test.js")
        || path.ends_with(".test.mjs")
        || path.ends_with("_tests.rs")
}

fn path_matches_root(path: &str, root: &str) -> bool {
    path == root
        || path
            .strip_prefix(root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn apply_search_filter(
    suites: &mut Vec<&'static TestSuite>,
    cases: &mut Vec<TestCase>,
    search: &str,
) {
    let search = search.trim().to_ascii_lowercase();
    if search.is_empty() {
        return;
    }

    let matching_case_suites = cases
        .iter()
        .filter(|case| case_matches_search(case, &search))
        .map(|case| case.suite_id)
        .collect::<BTreeSet<_>>();
    suites.retain(|suite| {
        suite_matches_search(suite, &search) || matching_case_suites.contains(&suite.id)
    });
    cases.retain(|case| case_matches_search(case, &search));
}

fn suite_matches_search(suite: &TestSuite, search: &str) -> bool {
    contains_search(suite.id.as_str(), search)
        || contains_search(suite.group.as_str(), search)
        || suite
            .layer
            .map_or(false, |layer| contains_search(layer.as_str(), search))
        || contains_search(suite.description, search)
        || contains_search(suite.requirements, search)
        || contains_search(suite.backend_policy.as_str(), search)
        || contains_search(suite.discoverer.as_str(), search)
        || suite
            .source_roots
            .iter()
            .any(|root| contains_search(root, search))
}

fn case_matches_search(case: &TestCase, search: &str) -> bool {
    contains_search(case.suite_id.as_str(), search)
        || contains_search(&case.name, search)
        || contains_search(&case.path, search)
}

fn contains_search(value: &str, search: &str) -> bool {
    value.to_ascii_lowercase().contains(search)
}

fn print_text_list(suites: &[&TestSuite], cases: &[TestCase]) -> Result<()> {
    println!("Test suites:");
    for suite in suites {
        println!(
            "  {:<18} {:<6} {:<9} {:<9} {}",
            suite.id.as_str(),
            suite.group.as_str(),
            suite.layer.map(|layer| layer.as_str()).unwrap_or("-"),
            if suite.coverage { "coverage" } else { "" },
            suite.description
        );
        println!(
            "  {:<18} {:<6} requirements: {}",
            "", "", suite.requirements
        );
    }

    if !cases.is_empty() {
        println!();
        println!("Test cases:");
        for case in cases {
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

fn print_json_list(suites: &[&TestSuite], cases: &[TestCase]) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&list_json_value(suites, cases))?
    );
    Ok(())
}

fn list_json_value(suites: &[&TestSuite], cases: &[TestCase]) -> Value {
    let suite_values = suites
        .iter()
        .map(|suite| {
            json!({
                "id": suite.id.as_str(),
                "group": suite.group.as_str(),
                "layer": suite.layer.map(|layer| layer.as_str()),
                "description": suite.description,
                "requirements": suite.requirements,
                "sourceRoots": suite.source_roots,
                "backendPolicy": suite.backend_policy.as_str(),
                "coverage": suite.coverage,
                "caseDiscovery": suite.discoverer.as_str(),
            })
        })
        .collect::<Vec<_>>();
    let case_values = cases
        .iter()
        .map(|case| {
            json!({
                "suite": case.suite_id.as_str(),
                "name": case.name,
                "path": case.path,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "suites": suite_values,
        "cases": case_values,
    })
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
        CaseDiscoverer::DemoTs => discover_demo_ts_cases(ctx, suite.id, cases)?,
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

fn discover_demo_ts_cases(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
    cases: &mut Vec<TestCase>,
) -> Result<()> {
    discover_quoted_test_cases(ctx, suite_id, demo_test_files(ctx)?, cases)
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
    let line = line.strip_prefix("async ").unwrap_or(line);
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

fn selected_list_suites(args: &TestListArgs) -> Result<Vec<&'static TestSuite>> {
    if args.layer.is_some() && args.group == TestGroupFilter::Smoke {
        anyhow::bail!("--layer only applies to unit suites");
    }

    Ok(TEST_SUITES
        .iter()
        .filter(|suite| group_filter_matches(args.group, suite.group))
        .filter(|suite| args.layer.map_or(true, |layer| suite.layer == Some(layer)))
        .collect())
}

fn selected_unit_suites(args: &TestUnitArgs) -> Result<UnitSelection> {
    let mut selection = UnitSelection::default();
    match &args.command {
        TestUnitCommands::Suite(args) => apply_unit_suite_selection(&mut selection, &args.target)?,
        TestUnitCommands::Group(args) => apply_unit_group_selection(&mut selection, &args.target),
    }
    Ok(selection)
}

fn apply_unit_suite_selection(
    selection: &mut UnitSelection,
    target: &TestUnitSuiteTarget,
) -> Result<()> {
    match target {
        TestUnitSuiteTarget::Xtask => {
            selection.suites = vec![suite_by_id(TestSuiteId::Xtask)?];
            selection.filters = unit_filters("suite", "xtask");
        }
        TestUnitSuiteTarget::RustCrates(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::RustCrates)?];
            selection.package = args.package.clone();
            selection.filters = json!({
                "command": "unit",
                "namespace": "suite",
                "target": "rust-crates",
                "package": args.package.as_deref(),
            });
        }
        TestUnitSuiteTarget::RustBindings => {
            selection.suites = vec![suite_by_id(TestSuiteId::RustBindings)?];
            selection.filters = unit_filters("suite", "rust-bindings");
        }
        TestUnitSuiteTarget::BrowserPackage => {
            selection.suites = vec![suite_by_id(TestSuiteId::PackageTs)?];
            selection.filters = unit_filters("suite", "browser-package");
        }
        TestUnitSuiteTarget::Demos => {
            selection.suites = vec![suite_by_id(TestSuiteId::DemoTs)?];
            selection.filters = unit_filters("suite", "demos");
        }
        TestUnitSuiteTarget::Api => {
            selection.suites = vec![suite_by_id(TestSuiteId::RustPublicApi)?];
            selection.filters = unit_filters("suite", "api");
        }
        TestUnitSuiteTarget::Cli => {
            selection.suites = vec![suite_by_id(TestSuiteId::Cli)?];
            selection.filters = unit_filters("suite", "cli");
        }
        TestUnitSuiteTarget::NodePackage(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::NodePackage)?];
            selection.backend = args.backend;
            selection.filters = json!({
                "command": "unit",
                "namespace": "suite",
                "target": "node-package",
                "backend": args.backend.as_str(),
            });
        }
        TestUnitSuiteTarget::PythonPackage(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::PythonPackage)?];
            selection.backend = args.backend;
            selection.filters = json!({
                "command": "unit",
                "namespace": "suite",
                "target": "python-package",
                "backend": args.backend.as_str(),
            });
        }
    }
    Ok(())
}

fn apply_unit_group_selection(selection: &mut UnitSelection, target: &TestUnitGroupTarget) {
    match target {
        TestUnitGroupTarget::Whitebox => {
            selection.suites = unit_suites_by_layer(TestUnitLayer::Whitebox);
            selection.filters = unit_filters("group", "whitebox");
        }
        TestUnitGroupTarget::Interface => {
            selection.suites = unit_suites_by_layer(TestUnitLayer::Interface);
            selection.filters = unit_filters("group", "interface");
        }
        TestUnitGroupTarget::Full => {
            selection.suites = suites_by_group(TestGroup::Unit);
            selection.filters = unit_filters("group", "full");
        }
    }
}

fn selected_smoke_suites(args: &TestSmokeArgs) -> Result<SmokeSelection> {
    let mut selection = SmokeSelection::default();
    match &args.command {
        TestSmokeCommands::Suite(args) => {
            apply_smoke_suite_selection(&mut selection, &args.target)?
        }
        TestSmokeCommands::Group(args) => {
            apply_smoke_group_selection(&mut selection, &args.target)?
        }
    }
    Ok(selection)
}

fn apply_smoke_suite_selection(
    selection: &mut SmokeSelection,
    target: &TestSmokeSuiteTarget,
) -> Result<()> {
    match target {
        TestSmokeSuiteTarget::Cli(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::CliSmoke)?];
            selection.apply_model_args(args);
            selection.filters = smoke_filters("suite", "cli", args, &[], json!({}));
        }
        TestSmokeSuiteTarget::ExampleRust(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::RustSmoke)?];
            selection.apply_case_args(args);
            selection.filters =
                smoke_filters("suite", "example-rust", &args.model, &args.cases, json!({}));
        }
        TestSmokeSuiteTarget::ExampleNode(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::NodeSmoke)?];
            selection.apply_case_args(args);
            selection.filters =
                smoke_filters("suite", "example-node", &args.model, &args.cases, json!({}));
        }
        TestSmokeSuiteTarget::ExamplePython(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::PythonSmoke)?];
            selection.apply_case_args(args);
            selection.filters = smoke_filters(
                "suite",
                "example-python",
                &args.model,
                &args.cases,
                json!({}),
            );
        }
        TestSmokeSuiteTarget::ExampleGateway(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::ExampleGatewaySmoke)?];
            selection.apply_case_args(args);
            selection.filters = smoke_filters(
                "suite",
                "example-gateway",
                &args.model,
                &args.cases,
                json!({}),
            );
        }
        TestSmokeSuiteTarget::ExampleBrowser(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::ExampleBrowserSmoke)?];
            selection.apply_example_browser_args(args);
            selection.filters = browser_example_filters("suite", "example-browser", args);
        }
        TestSmokeSuiteTarget::PlaygroundBrowser(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::PlaygroundBrowserSmoke)?];
            selection.apply_playground_browser_args(args);
            selection.filters = playground_browser_filters("suite", "playground-browser", args);
        }
        TestSmokeSuiteTarget::LlamaBackendOps(args) => {
            selection.suites = vec![suite_by_id(TestSuiteId::LlamaBackendOps)?];
            selection.apply_llama_args(args);
            selection.filters = json!({
                "command": "smoke",
                "namespace": "suite",
                "target": "llama-backend-ops",
                "backend": args.backend.as_str(),
                "mode": args.mode.as_str(),
                "op": args.op.as_deref(),
                "params": args.params.as_deref(),
                "output": args.output.as_str(),
            });
        }
    }
    Ok(())
}

fn apply_smoke_group_selection(
    selection: &mut SmokeSelection,
    target: &TestSmokeGroupTarget,
) -> Result<()> {
    match target {
        TestSmokeGroupTarget::Examples(args) => apply_examples_group_selection(selection, args),
        TestSmokeGroupTarget::LocalModel(args) => {
            selection.suites = local_model_smoke_suites()?;
            selection.apply_model_args(args);
            selection.filters = smoke_filters("group", "local-model", args, &[], json!({}));
            Ok(())
        }
        TestSmokeGroupTarget::Full(args) => apply_full_group_selection(selection, args),
    }
}

fn apply_examples_group_selection(
    selection: &mut SmokeSelection,
    args: &TestSmokeExamplesGroupArgs,
) -> Result<()> {
    selection.suites = example_smoke_suites()?;
    selection.apply_case_args(&args.cases);
    selection.browser_host = args.browser_host.clone();
    selection.browser_port = args.browser_port;
    selection.example_browser_timeout_ms = args.browser_timeout_ms;
    selection.filters = smoke_filters(
        "group",
        "examples",
        &args.cases.model,
        &args.cases.cases,
        json!({
            "browserHost": args.browser_host.as_deref(),
            "browserPort": args.browser_port,
            "browserTimeoutMs": args.browser_timeout_ms,
        }),
    );
    Ok(())
}

fn apply_full_group_selection(
    selection: &mut SmokeSelection,
    args: &TestSmokeFullGroupArgs,
) -> Result<()> {
    selection.suites = vec![
        suite_by_id(TestSuiteId::CliSmoke)?,
        suite_by_id(TestSuiteId::RustSmoke)?,
        suite_by_id(TestSuiteId::NodeSmoke)?,
        suite_by_id(TestSuiteId::PythonSmoke)?,
        suite_by_id(TestSuiteId::ExampleGatewaySmoke)?,
        suite_by_id(TestSuiteId::ExampleBrowserSmoke)?,
        suite_by_id(TestSuiteId::PlaygroundBrowserSmoke)?,
        suite_by_id(TestSuiteId::LlamaBackendOps)?,
    ];
    selection.apply_model_args(&args.model);
    selection.example_browser_timeout_ms = args.example_browser_timeout_ms;
    selection.playground_browser_timeout_ms = args.playground_browser_timeout_ms;
    selection.filters = smoke_filters(
        "group",
        "full",
        &args.model,
        &[],
        json!({
            "exampleBrowserTimeoutMs": args.example_browser_timeout_ms,
            "playgroundBrowserTimeoutMs": args.playground_browser_timeout_ms,
        }),
    );
    Ok(())
}

fn local_model_smoke_suites() -> Result<Vec<&'static TestSuite>> {
    Ok(vec![
        suite_by_id(TestSuiteId::CliSmoke)?,
        suite_by_id(TestSuiteId::RustSmoke)?,
        suite_by_id(TestSuiteId::NodeSmoke)?,
        suite_by_id(TestSuiteId::PythonSmoke)?,
    ])
}

fn example_smoke_suites() -> Result<Vec<&'static TestSuite>> {
    Ok(vec![
        suite_by_id(TestSuiteId::RustSmoke)?,
        suite_by_id(TestSuiteId::NodeSmoke)?,
        suite_by_id(TestSuiteId::PythonSmoke)?,
        suite_by_id(TestSuiteId::ExampleGatewaySmoke)?,
        suite_by_id(TestSuiteId::ExampleBrowserSmoke)?,
    ])
}

fn selected_verify_suites(args: &TestVerifyArgs) -> Result<Vec<&'static TestSuite>> {
    if args.target == TestVerifyTarget::PublicDocs {
        return Ok(Vec::new());
    }

    let explicit = !matches!(
        args.target,
        TestVerifyTarget::All | TestVerifyTarget::Whitebox | TestVerifyTarget::Interface
    );
    let suites = verify_target_suites(args.target)?;
    if explicit {
        let uncovered = suites
            .iter()
            .filter(|suite| !suite.coverage)
            .map(|suite| suite.id.as_str())
            .collect::<Vec<_>>();
        if !uncovered.is_empty() {
            anyhow::bail!(
                "suite(s) are not coverage-capable: {}",
                uncovered.join(", ")
            );
        }
    }

    let suites = suites
        .into_iter()
        .filter(|suite| suite.coverage)
        .collect::<Vec<_>>();
    if suites.is_empty() {
        anyhow::bail!("no coverage-capable suites matched the selected filters");
    }
    Ok(suites)
}

fn verify_target_suites(target: TestVerifyTarget) -> Result<Vec<&'static TestSuite>> {
    match target {
        TestVerifyTarget::All => Ok(suites_by_group(TestGroup::Unit)),
        TestVerifyTarget::Whitebox => Ok(unit_suites_by_layer(TestUnitLayer::Whitebox)),
        TestVerifyTarget::Interface => Ok(unit_suites_by_layer(TestUnitLayer::Interface)),
        TestVerifyTarget::Xtask => Ok(vec![suite_by_id(TestSuiteId::Xtask)?]),
        TestVerifyTarget::Rust => Ok(vec![suite_by_id(TestSuiteId::RustCrates)?]),
        TestVerifyTarget::Bindings => Ok(vec![suite_by_id(TestSuiteId::RustBindings)?]),
        TestVerifyTarget::BrowserPackage => Ok(vec![suite_by_id(TestSuiteId::PackageTs)?]),
        TestVerifyTarget::Demos => Ok(vec![suite_by_id(TestSuiteId::DemoTs)?]),
        TestVerifyTarget::Api => Ok(vec![suite_by_id(TestSuiteId::RustPublicApi)?]),
        TestVerifyTarget::Cli => Ok(vec![suite_by_id(TestSuiteId::Cli)?]),
        TestVerifyTarget::Node => Ok(vec![suite_by_id(TestSuiteId::NodePackage)?]),
        TestVerifyTarget::Python => Ok(vec![suite_by_id(TestSuiteId::PythonPackage)?]),
        TestVerifyTarget::PublicDocs => Ok(Vec::new()),
    }
}

fn suites_by_group(group: TestGroup) -> Vec<&'static TestSuite> {
    TEST_SUITES
        .iter()
        .filter(|suite| suite.group == group)
        .collect()
}

fn unit_suites_by_layer(layer: TestUnitLayer) -> Vec<&'static TestSuite> {
    TEST_SUITES
        .iter()
        .filter(|suite| suite.group == TestGroup::Unit && suite.layer == Some(layer))
        .collect()
}

fn group_filter_matches(filter: TestGroupFilter, group: TestGroup) -> bool {
    match filter {
        TestGroupFilter::All => true,
        TestGroupFilter::Unit => group == TestGroup::Unit,
        TestGroupFilter::Smoke => group == TestGroup::Smoke,
    }
}

fn unit_filters(namespace: &str, target: &str) -> Value {
    json!({
        "command": "unit",
        "namespace": namespace,
        "target": target,
    })
}

fn smoke_filters(
    namespace: &str,
    target: &str,
    args: &TestSmokeModelArgs,
    cases: &[TestSmokeCase],
    extra: Value,
) -> Value {
    json!({
        "command": "smoke",
        "namespace": namespace,
        "target": target,
        "backend": args.backend.as_str(),
        "model": args.model.as_ref().map(|model| model.display().to_string()),
        "offline": args.offline,
        "prompt": args.prompt.as_str(),
        "maxTokens": args.max_tokens,
        "temperature": args.temperature,
        "cases": cases.iter().map(TestSmokeCase::as_str).collect::<Vec<_>>(),
        "extra": extra,
    })
}

fn playground_browser_filters(
    namespace: &str,
    target: &str,
    args: &TestSmokePlaygroundBrowserArgs,
) -> Value {
    json!({
        "command": "smoke",
        "namespace": namespace,
        "target": target,
        "host": args.host.as_deref(),
        "port": args.port,
        "timeoutMs": args.timeout_ms,
        "requireWebgpu": args.require_webgpu,
    })
}

fn browser_example_filters(
    namespace: &str,
    target: &str,
    args: &TestSmokeExampleBrowserArgs,
) -> Value {
    json!({
        "command": "smoke",
        "namespace": namespace,
        "target": target,
        "model": args.model.as_ref().map(|model| model.display().to_string()),
        "offline": args.offline,
        "prompt": args.prompt.as_str(),
        "maxTokens": args.max_tokens,
        "cases": args.cases.iter().map(TestSmokeCase::as_str).collect::<Vec<_>>(),
        "host": args.host.as_deref(),
        "port": args.port,
        "timeoutMs": args.timeout_ms,
    })
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

fn ensure_cargo_llvm_cov() -> Result<()> {
    let output = Command::new("cargo")
        .args(["llvm-cov", "--version"])
        .output()
        .context("failed to run cargo llvm-cov --version")?;
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "cargo-llvm-cov is required for Rust coverage; install it with `cargo install cargo-llvm-cov`"
    )
}

fn validate_suite_backends(suites: &[&TestSuite], backend: Backend) -> Result<()> {
    for suite in suites {
        suite.backend_policy.validate(suite.id, backend)?;
    }
    Ok(())
}

#[cfg(test)]
fn validate_package_filter(suites: &[&TestSuite], package: Option<&str>) -> Result<()> {
    if package.is_some()
        && (suites.len() != 1
            || suites.first().map(|suite| suite.id) != Some(TestSuiteId::RustCrates))
    {
        anyhow::bail!("--package is only supported with `test unit suite rust-crates`");
    }
    Ok(())
}

fn collect_package_test_layout_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    let src = ctx.browser_package_dir().join("src");
    for path in collect_files_with_extension(&src, "ts")? {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.ends_with(".test.ts") {
            violations.push(format!(
                "TypeScript test must live under lib/web/tests: {}",
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
        CaseDiscoverer::DemoTs => demo_test_files(ctx),
        CaseDiscoverer::NodePackage => node_test_files(ctx),
        CaseDiscoverer::PythonPackage => python_test_files(ctx),
    }
}

fn first_party_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_first_party_rust_test_files(ctx, &mut files)?;
    files.extend(collect_files_with_suffix(
        &ctx.browser_package_dir(),
        DEMO_TEST_SUFFIX,
    )?);
    files.extend(demo_test_files(ctx)?);
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
        ctx.workspace_root().join("lib"),
        ctx.workspace_root().join("apps"),
        ctx.workspace_root().join("xtask"),
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
        ctx.workspace_root().join("lib"),
        ctx.workspace_root().join("apps"),
        ctx.workspace_root().join("xtask"),
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
        ctx.workspace_root().join("lib"),
        ctx.workspace_root().join("apps"),
        ctx.workspace_root().join("xtask"),
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
    if root.join("src").is_dir() {
        package_roots.push(root.to_path_buf());
    }
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
        Some(
            "target"
                | "node_modules"
                | ".build"
                | "third_party"
                | "llama.cpp"
                | ".venv"
                | ".pytest_cache"
        )
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
    output::run_build_command(
        "Ensuring Python 3.12 is available through uv",
        apply_uv_env(ctx, cmd!(sh, "{uv_exe} python install 3.12")),
    )?;

    let python_dir = ctx.python_package_project_dir();
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
    output::run_build_command("Building Python test wheel", maturin_cmd)?;
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

fn ensure_javascript_workspace_dependencies(
    sh: &Shell,
    ctx: &BuildContext,
    package_dirs: &[PathBuf],
) -> Result<()> {
    javascript::install_root_workspace_dependencies(
        sh,
        ctx,
        "Installing JavaScript workspace dependencies",
        package_dirs,
    )
}

fn demo_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut tests = Vec::new();
    collect_demo_test_files(&ctx.demos_root(), &mut tests)?;
    tests.sort();
    Ok(tests)
}

fn demo_test_workspaces(ctx: &BuildContext, tests: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut workspaces = Vec::new();
    for test in tests {
        let workspace = demo_test_workspace(ctx, test)?;
        if !workspaces.contains(&workspace) {
            workspaces.push(workspace);
        }
    }
    Ok(workspaces)
}

fn demo_test_workspace(ctx: &BuildContext, test: &Path) -> Result<PathBuf> {
    let relative = test
        .strip_prefix(ctx.demos_root())
        .with_context(|| format!("demo test is outside demos/: {}", test.display()))?;
    let demo_name = relative
        .components()
        .next()
        .with_context(|| format!("demo test has no demo workspace: {}", test.display()))?;
    Ok(ctx.demos_root().join(demo_name.as_os_str()))
}

fn package_ts_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    collect_files_with_suffix(&ctx.browser_package_dir().join("tests"), DEMO_TEST_SUFFIX)
}

fn node_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut files = collect_files_with_suffix(&ctx.node_package_dir().join("tests"), ".test.mjs")?;
    files.extend(collect_files_with_suffix(
        &ctx.node_package_dir().join("tests"),
        ".test.js",
    )?);
    files.sort();
    files.dedup();
    Ok(files)
}

fn python_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    collect_files_with_extension(&ctx.python_package_project_dir().join("tests"), "py")
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
    let relative: &[&str] = match package {
        "cogentlm" => &["crates", "cogentlm"],
        "cogentlm-sys" => &["crates", "sys"],
        "cogentlm-gateway" => &["lib", "gateway"],
        "cogentlm-gateway-server" => &["apps", "gateway-server"],
        "cogentlm-cli" => &["apps", "cli"],
        "xtask" => &["xtask"],
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

fn collect_demo_test_files(root: &Path, tests: &mut Vec<PathBuf>) -> Result<()> {
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
            if SKIPPED_DEMO_TEST_DIRS.contains(&file_name) {
                continue;
            }
            collect_demo_test_files(&path, tests)?;
            continue;
        }

        if path.is_file() && file_name.ends_with(DEMO_TEST_SUFFIX) {
            tests.push(path);
        }
    }

    Ok(())
}

fn display_relative(ctx: &BuildContext, path: &Path) -> String {
    let path = path
        .strip_prefix(ctx.workspace_root())
        .unwrap_or(path)
        .display()
        .to_string();
    normalize_relative_path(&path)
}

fn normalize_relative_path(path: &str) -> String {
    path.trim().trim_start_matches("./").replace('\\', "/")
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

fn write_run_report(ctx: &BuildContext, report: &RunReport) -> Result<()> {
    let report_dir = test_report_dir(ctx);
    std::fs::create_dir_all(&report_dir)
        .with_context(|| format!("failed to create {}", report_dir.display()))?;
    let json_path = test_run_report_json(ctx);
    let markdown_path = test_run_report_markdown(ctx);
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&report.as_json(ctx))?,
    )
    .with_context(|| format!("failed to write {}", json_path.display()))?;
    std::fs::write(&markdown_path, report.as_markdown(ctx))
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;
    Ok(())
}

fn write_verify_report(ctx: &BuildContext, report: &VerifyReport) -> Result<()> {
    let report_dir = test_report_dir(ctx);
    std::fs::create_dir_all(&report_dir)
        .with_context(|| format!("failed to create {}", report_dir.display()))?;
    let json_path = test_verify_report_json(ctx);
    let markdown_path = test_verify_report_markdown(ctx);
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&report.as_json(ctx))?,
    )
    .with_context(|| format!("failed to write {}", json_path.display()))?;
    std::fs::write(&markdown_path, report.as_markdown(ctx))
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;
    Ok(())
}

fn test_report_dir(ctx: &BuildContext) -> PathBuf {
    ctx.build_root().join("test")
}

fn test_run_report_json(ctx: &BuildContext) -> PathBuf {
    test_report_dir(ctx).join("run-report.json")
}

fn test_run_report_markdown(ctx: &BuildContext) -> PathBuf {
    test_report_dir(ctx).join("run-report.md")
}

fn test_verify_report_json(ctx: &BuildContext) -> PathBuf {
    test_report_dir(ctx).join("verify-report.json")
}

fn test_verify_report_markdown(ctx: &BuildContext) -> PathBuf {
    test_report_dir(ctx).join("verify-report.md")
}

fn prepare_suite_case_report_dir(ctx: &BuildContext, suite: &TestSuite) -> Result<()> {
    let dir = suite_case_report_dir(ctx, suite.id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to remove {}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))
}

fn suite_case_report_dir(ctx: &BuildContext, suite_id: TestSuiteId) -> PathBuf {
    test_report_dir(ctx)
        .join("case-reports")
        .join(suite_id.as_str())
}

fn suite_case_report_file(ctx: &BuildContext, suite_id: TestSuiteId, name: &str) -> PathBuf {
    suite_case_report_dir(ctx, suite_id).join(name)
}

fn read_structured_case_reports(
    ctx: &BuildContext,
    suite_id: TestSuiteId,
) -> Result<Vec<TestCaseReport>> {
    let dir = suite_case_report_dir(ctx, suite_id);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut cases = Vec::new();
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("xml") => cases.extend(parse_junit_case_reports(&contents, suite_id)),
            Some("tap") => cases.extend(parse_tap_case_reports(&contents, suite_id)),
            _ => {}
        }
    }
    cases.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.path.cmp(&right.path))
            .then(left.status.as_str().cmp(right.status.as_str()))
    });
    Ok(cases)
}

fn parse_junit_case_reports(contents: &str, suite_id: TestSuiteId) -> Vec<TestCaseReport> {
    let mut reports = Vec::new();
    let mut remaining = contents;

    while let Some(start) = remaining.find("<testcase") {
        remaining = &remaining[start..];
        let Some(tag_end) = remaining.find('>') else {
            break;
        };
        let tag = &remaining[..=tag_end];
        let self_closing = tag.trim_end().ends_with("/>");
        let (body, next) = if self_closing {
            ("", &remaining[tag_end + 1..])
        } else if let Some(end) = remaining[tag_end + 1..].find("</testcase>") {
            let body_start = tag_end + 1;
            let body_end = body_start + end;
            (
                &remaining[body_start..body_end],
                &remaining[body_end + "</testcase>".len()..],
            )
        } else {
            break;
        };
        remaining = next;

        let Some(name) = xml_attr(tag, "name") else {
            continue;
        };
        let path = xml_attr(tag, "file").or_else(|| xml_attr(tag, "classname"));
        let failure = body.contains("<failure");
        let error = body.contains("<error");
        let skipped = body.contains("<skipped");
        let status = if failure || error {
            CaseStatus::Failed
        } else if skipped {
            CaseStatus::Skipped
        } else {
            CaseStatus::Passed
        };
        let message = if failure {
            xml_element_attr(body, "failure", "message")
        } else if error {
            xml_element_attr(body, "error", "message")
        } else {
            None
        };
        reports.push(TestCaseReport {
            suite_id,
            name,
            path,
            status,
            error: message,
        });
    }

    reports
}

fn parse_tap_case_reports(contents: &str, suite_id: TestSuiteId) -> Vec<TestCaseReport> {
    let mut reports = Vec::new();
    for line in contents.lines().map(clean_log_line) {
        let (status, rest) = if let Some(rest) = line.strip_prefix("ok ") {
            (CaseStatus::Passed, rest)
        } else if let Some(rest) = line.strip_prefix("not ok ") {
            (CaseStatus::Failed, rest)
        } else {
            continue;
        };
        let Some((_, name)) = rest.split_once(" - ") else {
            continue;
        };
        let status = if name.contains("# SKIP") {
            CaseStatus::Skipped
        } else {
            status
        };
        let name = name
            .split_once('#')
            .map(|(name, _)| name)
            .unwrap_or(name)
            .trim();
        if name.is_empty() {
            continue;
        }
        reports.push(TestCaseReport {
            suite_id,
            name: name.to_owned(),
            path: None,
            status,
            error: None,
        });
    }
    reports
}

fn parse_libtest_case_reports(contents: &str, suite_id: TestSuiteId) -> Vec<TestCaseReport> {
    let mut reports = Vec::new();
    for line in contents.lines().map(clean_log_line) {
        let Some(rest) = line.strip_prefix("test ") else {
            continue;
        };
        let Some((name, status_text)) = rest.rsplit_once(" ... ") else {
            continue;
        };
        let status = if status_text.starts_with("ok") {
            CaseStatus::Passed
        } else if status_text.starts_with("FAILED") {
            CaseStatus::Failed
        } else if status_text.starts_with("ignored") {
            CaseStatus::Skipped
        } else {
            continue;
        };
        reports.push(TestCaseReport {
            suite_id,
            name: name.trim().to_owned(),
            path: None,
            status,
            error: None,
        });
    }
    reports
}

fn clean_log_line(line: &str) -> &str {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("[stdout] ") {
        rest.trim_start()
    } else if let Some(rest) = trimmed.strip_prefix("[stderr] ") {
        rest.trim_start()
    } else if let Some(rest) = trimmed.strip_prefix("[pty] ") {
        rest.trim_start()
    } else {
        trimmed
    }
}

fn xml_element_attr(contents: &str, element: &str, attr: &str) -> Option<String> {
    let start = contents.find(&format!("<{element}"))?;
    let tag = &contents[start..];
    let end = tag.find('>')?;
    xml_attr(&tag[..=end], attr)
}

fn xml_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)? + needle.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(xml_unescape(&rest[..end]))
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn coverage_artifacts_for_suite(ctx: &BuildContext, suite: &TestSuite) -> Vec<String> {
    let coverage_root = ctx.build_root().join("coverage");
    let artifacts = match suite.runner {
        SuiteRunner::RustTargets(_) => vec![
            coverage_root.join("rust").join("lcov.info"),
            coverage_root.join("rust").join("html"),
        ],
        SuiteRunner::NodePackage => vec![coverage_root.join("node").join("lcov.info")],
        SuiteRunner::PythonPackage => vec![
            coverage_root.join("python").join("lcov.info"),
            coverage_root.join("python").join("cobertura.xml"),
            coverage_root.join("python").join("html"),
        ],
        SuiteRunner::PackageTs
        | SuiteRunner::DemoTs
        | SuiteRunner::CliSmoke
        | SuiteRunner::RustSmoke
        | SuiteRunner::NodeSmoke
        | SuiteRunner::PythonSmoke
        | SuiteRunner::ExampleGatewaySmoke
        | SuiteRunner::ExampleBrowserSmoke
        | SuiteRunner::PlaygroundBrowserSmoke
        | SuiteRunner::LlamaBackendOps => Vec::new(),
    };
    artifacts
        .iter()
        .map(|artifact| display_relative(ctx, artifact))
        .collect()
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn duration_millis(duration: u128) -> u64 {
    u64::try_from(duration).unwrap_or(u64::MAX)
}

fn markdown_cell(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('|', "\\|")
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
enum TestGroup {
    Unit,
    Smoke,
}

impl TestGroup {
    fn as_str(&self) -> &'static str {
        match self {
            TestGroup::Unit => "unit",
            TestGroup::Smoke => "smoke",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TestSuite {
    id: TestSuiteId,
    group: TestGroup,
    layer: Option<TestUnitLayer>,
    description: &'static str,
    requirements: &'static str,
    source_roots: &'static [&'static str],
    coverage: bool,
    backend_policy: BackendPolicy,
    runner: SuiteRunner,
    discoverer: CaseDiscoverer,
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
    RustTargets(&'static [RustTestTarget]),
    PackageTs,
    DemoTs,
    NodePackage,
    PythonPackage,
    CliSmoke,
    RustSmoke,
    NodeSmoke,
    PythonSmoke,
    ExampleGatewaySmoke,
    ExampleBrowserSmoke,
    PlaygroundBrowserSmoke,
    LlamaBackendOps,
}

#[derive(Clone, Copy, Debug)]
enum CaseDiscoverer {
    None,
    RustTargets(&'static [RustTestTarget]),
    PackageTs,
    DemoTs,
    NodePackage,
    PythonPackage,
}

impl CaseDiscoverer {
    fn as_str(&self) -> &'static str {
        match self {
            CaseDiscoverer::None => "none",
            CaseDiscoverer::RustTargets(_) => "rust-targets",
            CaseDiscoverer::PackageTs => "package-ts",
            CaseDiscoverer::DemoTs => "demo-ts",
            CaseDiscoverer::NodePackage => "node-package",
            CaseDiscoverer::PythonPackage => "python-package",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RustTestKind {
    Package,
    Lib,
    Bin(&'static str),
    Test(&'static str),
}

struct UnitSelection {
    suites: Vec<&'static TestSuite>,
    package: Option<String>,
    backend: Backend,
    filters: Value,
}

impl Default for UnitSelection {
    fn default() -> Self {
        Self {
            suites: Vec::new(),
            package: None,
            backend: Backend::Cpu,
            filters: json!({}),
        }
    }
}

struct SmokeSelection {
    suites: Vec<&'static TestSuite>,
    backend: Backend,
    model: Option<PathBuf>,
    offline: bool,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    cases: Vec<TestSmokeCase>,
    browser_host: Option<String>,
    browser_port: Option<u16>,
    example_browser_timeout_ms: u64,
    playground_browser_timeout_ms: u64,
    require_webgpu: bool,
    llama_mode: LlamaBackendOpsMode,
    llama_op: Option<String>,
    llama_params: Option<String>,
    llama_output: LlamaBackendOpsOutput,
    filters: Value,
}

impl Default for SmokeSelection {
    fn default() -> Self {
        Self {
            suites: Vec::new(),
            backend: Backend::Cpu,
            model: None,
            offline: false,
            prompt: DEFAULT_SMOKE_PROMPT.to_owned(),
            max_tokens: DEFAULT_SMOKE_MAX_TOKENS,
            temperature: DEFAULT_SMOKE_TEMPERATURE,
            cases: Vec::new(),
            browser_host: None,
            browser_port: None,
            example_browser_timeout_ms: 30_000,
            playground_browser_timeout_ms: 30_000,
            require_webgpu: false,
            llama_mode: LlamaBackendOpsMode::Test,
            llama_op: None,
            llama_params: None,
            llama_output: LlamaBackendOpsOutput::Console,
            filters: json!({}),
        }
    }
}

impl SmokeSelection {
    fn apply_model_args(&mut self, args: &TestSmokeModelArgs) {
        self.backend = args.backend;
        self.model = args.model.clone();
        self.offline = args.offline;
        self.prompt = args.prompt.clone();
        self.max_tokens = args.max_tokens;
        self.temperature = args.temperature;
    }

    fn apply_case_args(&mut self, args: &TestSmokeCaseArgs) {
        self.apply_model_args(&args.model);
        self.cases = args.cases.clone();
    }

    fn apply_example_browser_args(&mut self, args: &TestSmokeExampleBrowserArgs) {
        self.model = args.model.clone();
        self.offline = args.offline;
        self.prompt = args.prompt.clone();
        self.max_tokens = args.max_tokens;
        self.cases = args.cases.clone();
        self.browser_host = args.host.clone();
        self.browser_port = args.port;
        self.example_browser_timeout_ms = args.timeout_ms;
    }

    fn apply_playground_browser_args(&mut self, args: &TestSmokePlaygroundBrowserArgs) {
        self.browser_host = args.host.clone();
        self.browser_port = args.port;
        self.playground_browser_timeout_ms = args.timeout_ms;
        self.require_webgpu = args.require_webgpu;
    }

    fn apply_llama_args(&mut self, args: &TestSmokeLlamaArgs) {
        self.backend = args.backend;
        self.llama_mode = args.mode;
        self.llama_op = args.op.clone();
        self.llama_params = args.params.clone();
        self.llama_output = args.output;
    }
}

struct SuiteRunOptions<'a> {
    backend: Backend,
    model: Option<&'a Path>,
    offline: bool,
    package: Option<&'a str>,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
    cases: &'a [TestSmokeCase],
    example_browser: BrowserSmokeRunOptions<'a>,
    playground_browser: BrowserSmokeRunOptions<'a>,
    llama: LlamaBackendOpsRunOptions<'a>,
}

#[derive(Default)]
struct BrowserSmokeRunOptions<'a> {
    host: Option<&'a str>,
    port: Option<u16>,
    timeout_ms: u64,
    require_webgpu: bool,
}

struct LlamaBackendOpsRunOptions<'a> {
    mode: LlamaBackendOpsMode,
    op: Option<&'a str>,
    params: Option<&'a str>,
    output: LlamaBackendOpsOutput,
}

impl Default for LlamaBackendOpsRunOptions<'_> {
    fn default() -> Self {
        Self {
            mode: LlamaBackendOpsMode::Test,
            op: None,
            params: None,
            output: LlamaBackendOpsOutput::Console,
        }
    }
}

#[derive(Default)]
struct RunCoverageState {
    rust_started: bool,
}

struct SuiteOutcome {
    counts: Option<TestCounts>,
    cases: Vec<TestCaseReport>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaseStatus {
    Passed,
    Failed,
    Skipped,
    Unknown,
}

impl CaseStatus {
    fn as_str(&self) -> &'static str {
        match self {
            CaseStatus::Passed => "passed",
            CaseStatus::Failed => "failed",
            CaseStatus::Skipped => "skipped",
            CaseStatus::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TestCaseReport {
    suite_id: TestSuiteId,
    name: String,
    path: Option<String>,
    status: CaseStatus,
    error: Option<String>,
}

impl TestCaseReport {
    fn from_case(case: TestCase, status: CaseStatus) -> Self {
        Self {
            suite_id: case.suite_id,
            name: case.name,
            path: Some(case.path),
            status,
            error: None,
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "suite": self.suite_id.as_str(),
            "name": self.name,
            "path": self.path,
            "status": self.status.as_str(),
            "error": self.error,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct TestCounts {
    passed: usize,
    failed: usize,
    skipped: usize,
}

impl TestCounts {
    fn passed(passed: usize) -> Self {
        Self {
            passed,
            failed: 0,
            skipped: 0,
        }
    }

    fn total(&self) -> usize {
        self.passed + self.failed + self.skipped
    }

    fn add(&mut self, other: TestCounts) {
        self.passed += other.passed;
        self.failed += other.failed;
        self.skipped += other.skipped;
    }

    fn as_json(&self) -> Value {
        json!({
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total(),
        })
    }

    fn markdown(&self) -> String {
        format!(
            "{} passed, {} failed, {} skipped, {} total",
            self.passed,
            self.failed,
            self.skipped,
            self.total()
        )
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
struct RunSummary {
    suites: SuiteStatusSummary,
    counts: CountSummary,
}

impl RunSummary {
    fn as_json(&self) -> Value {
        json!({
            "suites": self.suites.as_json(),
            "counts": self.counts.as_json(),
        })
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
struct SuiteStatusSummary {
    passed: usize,
    failed: usize,
    not_run: usize,
}

impl SuiteStatusSummary {
    fn total(&self) -> usize {
        self.passed + self.failed + self.not_run
    }

    fn as_json(&self) -> Value {
        json!({
            "passed": self.passed,
            "failed": self.failed,
            "notRun": self.not_run,
            "total": self.total(),
        })
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
struct CountSummary {
    counts: TestCounts,
    known_suites: usize,
    unknown_suites: Vec<String>,
}

impl CountSummary {
    fn as_json(&self) -> Value {
        json!({
            "passed": self.counts.passed,
            "failed": self.counts.failed,
            "skipped": self.counts.skipped,
            "total": self.counts.total(),
            "knownSuites": self.known_suites,
            "unknownSuites": self.unknown_suites,
        })
    }
}

struct RunReport {
    generated_at_unix_seconds: u64,
    finished_at_unix_seconds: Option<u64>,
    started_at: Instant,
    duration_ms: Option<u64>,
    status: String,
    filters: Value,
    selected_suites: Vec<String>,
    suites: Vec<SuiteReport>,
}

impl RunReport {
    fn new(filters: Value, suites: &[&TestSuite]) -> Self {
        Self {
            generated_at_unix_seconds: now_unix_seconds(),
            finished_at_unix_seconds: None,
            started_at: Instant::now(),
            duration_ms: None,
            status: "running".to_owned(),
            filters,
            selected_suites: suites
                .iter()
                .map(|suite| suite.id.as_str().to_owned())
                .collect(),
            suites: Vec::new(),
        }
    }

    fn finish(&mut self, status: &str) {
        self.status = status.to_owned();
        self.finished_at_unix_seconds = Some(now_unix_seconds());
        self.duration_ms = Some(duration_millis(self.started_at.elapsed().as_millis()));
    }

    fn summary(&self) -> RunSummary {
        let mut summary = RunSummary::default();
        for suite in &self.suites {
            match suite.status.as_str() {
                "passed" => summary.suites.passed += 1,
                "failed" => summary.suites.failed += 1,
                "not_run" => summary.suites.not_run += 1,
                _ => {}
            }

            if let Some(counts) = suite.counts {
                summary.counts.known_suites += 1;
                summary.counts.counts.add(counts);
            } else {
                summary.counts.unknown_suites.push(suite.id.clone());
            }
        }
        summary
    }

    fn as_json(&self, ctx: &BuildContext) -> Value {
        json!({
            "kind": "test-run",
            "status": self.status.clone(),
            "generatedAtUnixSeconds": self.generated_at_unix_seconds,
            "finishedAtUnixSeconds": self.finished_at_unix_seconds,
            "durationMs": self.duration_ms,
            "filters": self.filters.clone(),
            "selectedSuites": self.selected_suites.clone(),
            "summary": self.summary().as_json(),
            "logDir": display_relative(ctx, &ctx.command_logs_dir()),
            "suites": self.suites.iter().map(|suite| suite.as_json()).collect::<Vec<_>>(),
        })
    }

    fn as_markdown(&self, ctx: &BuildContext) -> String {
        let summary = self.summary();
        let mut markdown = format!(
            "# Test run report\n\nStatus: `{}`\n\nSuites: {} passed, {} failed, {} not run, {} total\n\nTests/checks: {}\n\nUnknown counts: {}\n\nLog directory: `{}`\n\n| Suite | Status | Duration | Counts | Coverage | Error |\n| --- | --- | ---: | --- | --- | --- |\n",
            self.status,
            summary.suites.passed,
            summary.suites.failed,
            summary.suites.not_run,
            summary.suites.total(),
            summary.counts.counts.markdown(),
            if summary.counts.unknown_suites.is_empty() {
                "-".to_owned()
            } else {
                summary.counts.unknown_suites.join(", ")
            },
            display_relative(ctx, &ctx.command_logs_dir()),
        );
        for suite in &self.suites {
            markdown.push_str(&format!(
                "| `{}` | `{}` | {} | `{}` | `{}` | {} |\n",
                suite.id,
                suite.status,
                suite
                    .duration_ms
                    .map(|duration| duration.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                suite
                    .counts
                    .map(|counts| counts.markdown())
                    .unwrap_or_else(|| "unknown".to_owned()),
                suite.coverage_status,
                suite
                    .error
                    .as_deref()
                    .map(markdown_cell)
                    .unwrap_or_else(|| "-".to_owned()),
            ));
        }
        if self.suites.iter().any(|suite| !suite.cases.is_empty()) {
            markdown.push_str(
                "\n## Cases\n\n| Suite | Case | Status | Path | Error |\n| --- | --- | --- | --- | --- |\n",
            );
            for suite in &self.suites {
                for case in &suite.cases {
                    markdown.push_str(&format!(
                        "| `{}` | {} | `{}` | `{}` | {} |\n",
                        suite.id,
                        markdown_cell(&case.name),
                        case.status.as_str(),
                        case.path
                            .as_deref()
                            .map(markdown_cell)
                            .unwrap_or_else(|| "-".to_owned()),
                        case.error
                            .as_deref()
                            .map(markdown_cell)
                            .unwrap_or_else(|| "-".to_owned()),
                    ));
                }
            }
        }
        markdown
    }
}

struct SuiteReport {
    id: String,
    group: String,
    layer: Option<String>,
    status: String,
    duration_ms: Option<u64>,
    coverage_status: String,
    coverage_artifacts: Vec<String>,
    counts: Option<TestCounts>,
    cases: Vec<TestCaseReport>,
    error: Option<String>,
}

impl SuiteReport {
    fn passed(
        ctx: &BuildContext,
        suite: &TestSuite,
        duration_ms: u128,
        counts: Option<TestCounts>,
        cases: Vec<TestCaseReport>,
    ) -> Self {
        Self {
            id: suite.id.as_str().to_owned(),
            group: suite.group.as_str().to_owned(),
            layer: suite.layer.map(|layer| layer.as_str()).map(str::to_owned),
            status: "passed".to_owned(),
            duration_ms: Some(duration_millis(duration_ms)),
            coverage_status: if suite.coverage {
                "written".to_owned()
            } else {
                "not_applicable".to_owned()
            },
            coverage_artifacts: coverage_artifacts_for_suite(ctx, suite),
            counts,
            cases,
            error: None,
        }
    }

    fn failed(
        ctx: &BuildContext,
        suite: &TestSuite,
        duration_ms: u128,
        error: String,
        counts: Option<TestCounts>,
        cases: Vec<TestCaseReport>,
    ) -> Self {
        Self {
            id: suite.id.as_str().to_owned(),
            group: suite.group.as_str().to_owned(),
            layer: suite.layer.map(|layer| layer.as_str()).map(str::to_owned),
            status: "failed".to_owned(),
            duration_ms: Some(duration_millis(duration_ms)),
            coverage_status: if suite.coverage {
                "failed".to_owned()
            } else {
                "not_applicable".to_owned()
            },
            coverage_artifacts: coverage_artifacts_for_suite(ctx, suite),
            counts,
            cases,
            error: Some(error),
        }
    }

    fn not_run(ctx: &BuildContext, suite: &TestSuite) -> Self {
        Self {
            id: suite.id.as_str().to_owned(),
            group: suite.group.as_str().to_owned(),
            layer: suite.layer.map(|layer| layer.as_str()).map(str::to_owned),
            status: "not_run".to_owned(),
            duration_ms: None,
            coverage_status: if suite.coverage {
                "not_run".to_owned()
            } else {
                "not_applicable".to_owned()
            },
            coverage_artifacts: coverage_artifacts_for_suite(ctx, suite),
            counts: None,
            cases: Vec::new(),
            error: None,
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "id": self.id.clone(),
            "group": self.group.clone(),
            "layer": self.layer.clone(),
            "status": self.status.clone(),
            "durationMs": self.duration_ms,
            "coverage": {
                "status": self.coverage_status.clone(),
                "artifacts": self.coverage_artifacts.clone(),
            },
            "counts": self.counts.map(|counts| {
                let mut value = counts.as_json();
                if let Value::Object(object) = &mut value {
                    object.insert("status".to_owned(), Value::String("known".to_owned()));
                }
                value
            }).unwrap_or_else(|| json!({
                "status": "unknown",
            })),
            "cases": self.cases.iter().map(TestCaseReport::as_json).collect::<Vec<_>>(),
            "error": self.error.clone(),
        })
    }
}

struct VerifyReport {
    generated_at_unix_seconds: u64,
    finished_at_unix_seconds: Option<u64>,
    started_at: Instant,
    duration_ms: Option<u64>,
    status: String,
    filters: Value,
    selected_suites: Vec<String>,
    checks: Vec<VerifyCheckReport>,
    coverage: Option<CoverageSummaries>,
}

impl VerifyReport {
    fn new(args: &TestVerifyArgs, suites: &[&TestSuite]) -> Self {
        Self {
            generated_at_unix_seconds: now_unix_seconds(),
            finished_at_unix_seconds: None,
            started_at: Instant::now(),
            duration_ms: None,
            status: "running".to_owned(),
            filters: json!({
                "target": args.target.as_str(),
                "changed": args.changed,
            }),
            selected_suites: suites
                .iter()
                .map(|suite| suite.id.as_str().to_owned())
                .collect(),
            checks: Vec::new(),
            coverage: None,
        }
    }

    fn finish(&mut self, status: &str) {
        self.status = status.to_owned();
        self.finished_at_unix_seconds = Some(now_unix_seconds());
        self.duration_ms = Some(duration_millis(self.started_at.elapsed().as_millis()));
    }

    fn as_json(&self, ctx: &BuildContext) -> Value {
        json!({
            "kind": "test-verify",
            "status": self.status.clone(),
            "generatedAtUnixSeconds": self.generated_at_unix_seconds,
            "finishedAtUnixSeconds": self.finished_at_unix_seconds,
            "durationMs": self.duration_ms,
            "filters": self.filters.clone(),
            "selectedSuites": self.selected_suites.clone(),
            "coverageDir": display_relative(ctx, &ctx.build_root().join("coverage")),
            "checks": self.checks.iter().map(|check| check.as_json()).collect::<Vec<_>>(),
            "coverage": self.coverage.as_ref().map(CoverageSummaries::as_json),
        })
    }

    fn as_markdown(&self, ctx: &BuildContext) -> String {
        let mut markdown = format!(
            "# Test verification report\n\nStatus: `{}`\n\nCoverage directory: `{}`\n\n| Check | Status | Error |\n| --- | --- | --- |\n",
            self.status,
            display_relative(ctx, &ctx.build_root().join("coverage")),
        );
        for check in &self.checks {
            markdown.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                check.id,
                check.status,
                check
                    .error
                    .as_deref()
                    .map(markdown_cell)
                    .unwrap_or_else(|| "-".to_owned()),
            ));
        }
        markdown
    }
}

struct VerifyCheckReport {
    id: String,
    status: String,
    error: Option<String>,
}

impl VerifyCheckReport {
    fn from_result<T>(id: &str, result: &Result<T>) -> Self {
        match result {
            Ok(_) => Self {
                id: id.to_owned(),
                status: "passed".to_owned(),
                error: None,
            },
            Err(error) => Self {
                id: id.to_owned(),
                status: "failed".to_owned(),
                error: Some(format!("{error:#}")),
            },
        }
    }

    fn skipped(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            status: "skipped".to_owned(),
            error: None,
        }
    }

    fn as_json(&self) -> Value {
        json!({
            "id": self.id.clone(),
            "status": self.status.clone(),
            "error": self.error.clone(),
        })
    }
}

#[derive(Clone, Debug)]
struct TestCase {
    suite_id: TestSuiteId,
    name: String,
    path: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct CoverageReportAreas {
    rust: bool,
    node: bool,
    python: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct CoverageSummaries {
    rust: LcovSummary,
    node: LcovSummary,
    python: LcovSummary,
}

impl CoverageSummaries {
    fn as_json(&self) -> Value {
        json!({
            "rust": self.rust.as_json(),
            "node": self.node.as_json(),
            "python": self.python.as_json(),
        })
    }
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
