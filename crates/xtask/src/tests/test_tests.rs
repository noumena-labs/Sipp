//! Tests the `test` module in `xtask`.
//!
//! Covers developer automation helpers, catalog logic, and terminal formatting with deterministic fixtures instead of invoking external toolchains.

use clap::error::ErrorKind;
use clap::Parser;

use crate::cli::{
    Backend, Cli, Commands, LlamaBackendOpsMode, RunCommands, RunLlamaCommands, TestCategoryFilter,
    TestCommands, TestSuiteId, TestVerifyArgs,
};
use crate::utils::BuildContext;

use super::{
    catalog_suite_ids, collect_catalog_ownership_violations, discover_cases, list_json_value,
    selected_suites, selected_verify_suites, suite_by_id, validate_suite_backends, RunReport,
    SuiteReport, TestCategory, NODE_GENERATION_SMOKE_SCRIPTS, PYTHON_GENERATION_SMOKE_SCRIPTS,
    RUST_GENERATION_SMOKE_EXAMPLES, TEST_SUITES,
};

#[test]
fn catalog_suite_ids_are_unique() {
    let unique = catalog_suite_ids();

    assert_eq!(unique.len(), TEST_SUITES.len());
}

#[test]
fn catalog_has_whitebox_and_interface_suites() {
    assert!(TEST_SUITES
        .iter()
        .any(|suite| suite.category == TestCategory::Whitebox));
    assert!(TEST_SUITES
        .iter()
        .any(|suite| suite.category == TestCategory::Interface));
}

#[test]
fn new_test_commands_parse() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "run",
        "--suite",
        "rust-crates",
        "--package",
        "cogentlm-core",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Run(args) = command else {
        panic!("expected run command");
    };
    assert_eq!(args.suite, vec![TestSuiteId::RustCrates]);
    assert_eq!(args.package.as_deref(), Some("cogentlm-core"));
}

#[test]
fn test_run_accepts_category_and_repeated_suites() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "run",
        "--category",
        "interface",
        "--suite",
        "node-package",
        "--suite",
        "python-package",
        "--backend",
        "cpu",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Run(args) = command else {
        panic!("expected run command");
    };
    assert_eq!(args.category, TestCategoryFilter::Interface);
    assert_eq!(
        args.suite,
        vec![TestSuiteId::NodePackage, TestSuiteId::PythonPackage]
    );
    assert_eq!(args.backend, Backend::Cpu);
}

#[test]
fn test_verify_accepts_category_and_repeated_suites() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "verify",
        "--category",
        "whitebox",
        "--suite",
        "xtask",
        "--suite",
        "rust-crates",
        "--changed",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Verify(args) = command else {
        panic!("expected verify command");
    };
    assert_eq!(args.category, TestCategoryFilter::Whitebox);
    assert_eq!(
        args.suite,
        vec![TestSuiteId::Xtask, TestSuiteId::RustCrates]
    );
    assert!(args.changed);
}

#[test]
fn invalid_suite_names_are_rejected_by_clap() {
    assert!(Cli::try_parse_from(["xtask", "test", "run", "--suite", "nope"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "run", "--suite", "layout"]).is_err());
}

#[test]
fn suite_category_mismatches_are_rejected() {
    assert!(selected_suites(&[TestSuiteId::NodePackage], TestCategoryFilter::Whitebox).is_err());
    assert!(selected_suites(&[TestSuiteId::PackageTs], TestCategoryFilter::Interface).is_err());
}

#[test]
fn empty_run_selection_selects_every_suite() {
    assert_eq!(
        selected_suites(&[], TestCategoryFilter::All).unwrap().len(),
        TEST_SUITES.len()
    );
}

#[test]
fn coverage_rejects_explicit_non_coverage_suites() {
    let args = TestVerifyArgs {
        category: TestCategoryFilter::All,
        suite: vec![TestSuiteId::PackageTs],
        changed: false,
    };

    assert!(selected_verify_suites(&args).is_err());
}

#[test]
fn backend_preflight_rejects_all_for_concrete_only_suites() {
    let python = suite_by_id(TestSuiteId::PythonPackage).unwrap();
    assert!(validate_suite_backends(&[python], Backend::All).is_err());

    let node = suite_by_id(TestSuiteId::NodePackage).unwrap();
    assert!(validate_suite_backends(&[node], Backend::All).is_ok());
}

#[test]
fn coverage_rejects_model_smoke_because_it_runs_no_report() {
    let args = TestVerifyArgs {
        category: TestCategoryFilter::All,
        suite: vec![TestSuiteId::ModelSmoke],
        changed: false,
    };

    assert!(selected_verify_suites(&args).is_err());
}

#[test]
fn run_report_serializes_suite_status_and_coverage_artifacts() {
    let ctx = BuildContext::new().unwrap();
    let args = crate::cli::TestRunArgs {
        category: TestCategoryFilter::All,
        suite: vec![TestSuiteId::Xtask],
        package: None,
        backend: Backend::Cpu,
        model: None,
        offline: false,
    };
    let suite = suite_by_id(TestSuiteId::Xtask).unwrap();
    let mut report = RunReport::new(&args, &[suite]);

    report.suites.push(SuiteReport::passed(&ctx, suite, 42));
    report.finish("passed");

    let value = report.as_json(&ctx);
    assert_eq!(value["status"], "passed");
    assert_eq!(value["suites"][0]["status"], "passed");
    assert_eq!(value["suites"][0]["coverage"]["status"], "written");
    assert!(value["suites"][0]["coverage"]["artifacts"][0]
        .as_str()
        .unwrap()
        .ends_with(".build/coverage/rust/lcov.info"));
}

#[test]
fn model_smoke_uses_generation_only_examples() {
    assert_eq!(RUST_GENERATION_SMOKE_EXAMPLES, ["query", "chat"]);
    assert_eq!(
        NODE_GENERATION_SMOKE_SCRIPTS,
        ["examples/query.mjs", "examples/chat.mjs"]
    );
    assert_eq!(
        PYTHON_GENERATION_SMOKE_SCRIPTS,
        ["examples/query.py", "examples/chat.py"]
    );
}

#[test]
fn list_json_does_not_include_profiles() {
    let suites = [suite_by_id(TestSuiteId::Xtask).unwrap()];
    let value = list_json_value(&suites, &[]);

    assert!(value.get("profiles").is_none());
    assert!(value["suites"][0].get("profiles").is_none());
    assert_eq!(value["suites"][0]["id"], "xtask");
}

#[test]
fn case_discovery_includes_each_discoverable_language_surface() {
    let ctx = BuildContext::new().unwrap();
    let suites = [
        suite_by_id(TestSuiteId::PackageTs).unwrap(),
        suite_by_id(TestSuiteId::AppTs).unwrap(),
        suite_by_id(TestSuiteId::NodePackage).unwrap(),
        suite_by_id(TestSuiteId::PythonPackage).unwrap(),
    ];
    let cases = discover_cases(&ctx, &suites).unwrap();

    assert!(cases
        .iter()
        .any(|case| case.suite_id == TestSuiteId::PackageTs));
    assert!(cases.iter().any(|case| case.suite_id == TestSuiteId::AppTs));
    assert!(cases
        .iter()
        .any(|case| case.suite_id == TestSuiteId::NodePackage));
    assert!(cases
        .iter()
        .any(|case| case.suite_id == TestSuiteId::PythonPackage));
}

#[test]
fn rust_crate_discovery_matches_unit_runner_scope() {
    let ctx = BuildContext::new().unwrap();
    let rust_crates = [suite_by_id(TestSuiteId::RustCrates).unwrap()];
    let cases = discover_cases(&ctx, &rust_crates).unwrap();

    assert!(cases
        .iter()
        .all(|case| !case.path.contains("tests/public_api.rs")));
}

#[test]
fn catalog_owns_every_first_party_test_file_once() {
    let ctx = BuildContext::new().unwrap();
    let mut violations = Vec::new();

    collect_catalog_ownership_violations(&ctx, &mut violations).unwrap();

    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn old_test_commands_are_rejected() {
    assert!(Cli::try_parse_from(["xtask", "test", "core"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "rust-api"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "model-smoke"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "all"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "whitebox"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "interface"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "coverage"]).is_err());
}

#[test]
fn test_help_uses_clap_help_subcommand() {
    assert_eq!(
        Cli::try_parse_from(["xtask", "test", "help"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        Cli::try_parse_from(["xtask", "test", "help", "run"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        Cli::try_parse_from(["xtask", "test", "help", "verify"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DisplayHelp
    );
}

#[test]
fn old_run_test_commands_are_rejected() {
    assert!(Cli::try_parse_from(["xtask", "run", "all"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "run", "apps", "test"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "run", "bindings", "node"]).is_err());
}

#[test]
fn run_keeps_app_and_llama_groups() {
    let cli = Cli::parse_from(["xtask", "run", "apps", "build", "examples"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    assert!(matches!(command, RunCommands::Apps { .. }));

    let cli = Cli::parse_from(["xtask", "run", "llama", "backend-ops"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    let RunCommands::Llama { command } = command else {
        panic!("expected llama run command");
    };
    let RunLlamaCommands::BackendOps(args) = command;
    assert_eq!(args.mode, LlamaBackendOpsMode::Support);
}
