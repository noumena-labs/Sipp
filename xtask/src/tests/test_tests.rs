//! Tests the `test` module in `xtask`.
//!
//! Covers catalog selection, reporting, layout validation, test-case discovery,
//! and coverage-summary helpers with deterministic fixtures instead of invoking
//! external toolchains or model-backed tests.

use clap::error::ErrorKind;
use clap::Parser;

use crate::cli::{
    Backend, Cli, Commands, LlamaBackendOpsMode, RunCommands, RunLlamaCommands, TestCommands,
    TestSmokeArgs, TestSmokeCase, TestSmokeCaseArgs, TestSmokeCommands, TestSmokeExamplesGroupArgs,
    TestSmokeFullGroupArgs, TestSmokeGroupArgs, TestSmokeGroupTarget, TestSmokeModelArgs,
    TestSmokeSuiteArgs, TestSmokeSuiteTarget, TestSuiteId, TestUnitArgs, TestUnitCommands,
    TestUnitGroupArgs, TestUnitGroupTarget, TestUnitLayer, TestUnitSuiteArgs, TestUnitSuiteTarget,
    TestVerifyArgs, TestVerifyTarget,
};
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    apply_search_filter, case_counts, catalog_suite_ids, collect_catalog_ownership_violations,
    collect_files_with_extension, collect_files_with_suffix, collect_python_public_doc_violations,
    collect_rust_public_doc_violations, collect_typescript_public_doc_violations,
    contains_test_attribute, coverage_report_areas, discover_cases, display_relative,
    duration_millis, filtered_rust_targets, is_allowed_rust_test_file, is_cpp_test_file_name,
    is_first_party_source_path, is_inverted_rust_test_file, is_probable_test_path, list_json_value,
    markdown_cell, normalize_relative_path, parse_junit_case_reports, parse_lcov_summary,
    parse_libtest_case_reports, parse_quoted_test_name, parse_rust_fn_name, parse_tap_case_reports,
    path_components, path_matches_root, python_venv_exe, selected_smoke_suites,
    selected_unit_suites, selected_verify_suites, source_owner_suites, suite_by_id, test_backends,
    validate_package_filter, validate_suite_backends, CaseDiscoverer, CaseStatus,
    CoverageSummaries, LcovSummary, RunReport, RustTestTarget, SuiteReport, TestCase,
    TestCaseReport, TestCounts, TestGroup, VerifyCheckReport, VerifyReport,
    NODE_GENERATION_SMOKE_SCRIPTS, PYTHON_GENERATION_SMOKE_SCRIPTS, RUST_CRATE_TEST_TARGETS,
    RUST_GENERATION_SMOKE_EXAMPLES, TEST_SUITES,
};

#[test]
fn catalog_suite_ids_are_unique() {
    let unique = catalog_suite_ids();

    assert_eq!(unique.len(), TEST_SUITES.len());
}

#[test]
fn catalog_has_whitebox_and_interface_suites() {
    assert!(TEST_SUITES.iter().any(
        |suite| suite.group == TestGroup::Unit && suite.layer == Some(TestUnitLayer::Whitebox)
    ));
    assert!(TEST_SUITES.iter().any(
        |suite| suite.group == TestGroup::Unit && suite.layer == Some(TestUnitLayer::Interface)
    ));
    assert!(TEST_SUITES
        .iter()
        .any(|suite| suite.group == TestGroup::Smoke));
}

#[test]
fn new_test_commands_parse() {
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
        panic!("expected rust target");
    };
    assert_eq!(args.package.as_deref(), Some("cogentlm-core"));
}

#[test]
fn test_unit_accepts_interface_binding_targets() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "unit",
        "suite",
        "node-package",
        "--backend",
        "cpu",
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
    let TestUnitSuiteTarget::NodePackage(args) = args.target else {
        panic!("expected node target");
    };
    assert_eq!(args.backend, Backend::Cpu);
}

#[test]
fn test_smoke_accepts_node_model_options_and_cases() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "smoke",
        "suite",
        "example-node",
        "--backend",
        "cpu",
        "--model",
        "sample.gguf",
        "--prompt",
        "hello",
        "--max-tokens",
        "2",
        "--temperature",
        "0.5",
        "--case",
        "query",
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
    let TestSmokeSuiteTarget::ExampleNode(args) = args.target else {
        panic!("expected node smoke target");
    };
    assert_eq!(args.model.backend, Backend::Cpu);
    assert_eq!(
        args.model.model.as_deref(),
        Some(std::path::Path::new("sample.gguf"))
    );
    assert_eq!(args.model.prompt, "hello");
    assert_eq!(args.model.max_tokens, 2);
    assert_eq!(args.model.temperature, 0.5);
    assert_eq!(args.cases, vec![TestSmokeCase::Query]);
}

#[test]
fn test_smoke_accepts_provider_gateway_target() {
    let cli = Cli::parse_from(["xtask", "test", "smoke", "suite", "provider-gateway"]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Smoke(args) = command else {
        panic!("expected smoke command");
    };
    assert!(matches!(
        args.command,
        TestSmokeCommands::Suite(TestSmokeSuiteArgs {
            target: TestSmokeSuiteTarget::ProviderGateway
        })
    ));
}

#[test]
fn test_verify_accepts_target() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "verify",
        "--target",
        "whitebox",
        "--changed",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Verify(args) = command else {
        panic!("expected verify command");
    };
    assert_eq!(args.target, TestVerifyTarget::Whitebox);
    assert!(args.changed);
}

#[test]
fn test_verify_accepts_public_docs_target() {
    let cli = Cli::parse_from(["xtask", "test", "verify", "--target", "public-docs"]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Verify(args) = command else {
        panic!("expected verify command");
    };
    assert_eq!(args.target, TestVerifyTarget::PublicDocs);
    assert!(!args.changed);
}

#[test]
fn public_docs_target_selects_no_coverage_suites() {
    let args = TestVerifyArgs {
        target: TestVerifyTarget::PublicDocs,
        changed: false,
    };

    assert!(selected_verify_suites(&args).unwrap().is_empty());
}

#[test]
fn public_doc_checker_accepts_documented_exports() {
    let temp = TempDir::new("public-docs-accepts");
    let rust = temp.write(
        "lib.rs",
        "//! Module docs.\n\n/// Exported type.\n#[napi(object)]\npub struct Exported {}\n",
    );
    let ts = temp.write(
        "index.ts",
        "/** Exported type. */\nexport interface Exported {}\n",
    );
    let python = temp.write(
        "__init__.py",
        "\"\"\"Module docs.\"\"\"\n\n\ndef exported():\n    \"\"\"Return an exported value.\"\"\"\n    return 1\n",
    );
    let mut violations = Vec::new();

    collect_rust_public_doc_violations(&rust, "lib.rs", &mut violations).unwrap();
    collect_typescript_public_doc_violations(&ts, "index.ts", &mut violations).unwrap();
    collect_python_public_doc_violations(&python, "__init__.py", &mut violations).unwrap();

    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn public_doc_checker_reports_missing_exports() {
    let temp = TempDir::new("public-docs-reports");
    let rust = temp.write("lib.rs", "#[napi(object)]\npub struct Exported {}\n");
    let ts = temp.write("index.ts", "export interface Exported {}\n");
    let python = temp.write("__init__.py", "def exported():\n    return 1\n");
    let mut violations = Vec::new();

    collect_rust_public_doc_violations(&rust, "lib.rs", &mut violations).unwrap();
    collect_typescript_public_doc_violations(&ts, "index.ts", &mut violations).unwrap();
    collect_python_public_doc_violations(&python, "__init__.py", &mut violations).unwrap();

    assert_eq!(violations.len(), 5);
    assert!(violations
        .iter()
        .any(|violation| violation.contains("missing crate or module rustdoc")));
    assert!(violations
        .iter()
        .any(|violation| violation.contains("missing rustdoc")));
    assert!(violations
        .iter()
        .any(|violation| violation.contains("missing JSDoc/TSDoc")));
    assert!(violations
        .iter()
        .any(|violation| violation.contains("missing module docstring")));
    assert!(violations
        .iter()
        .any(|violation| violation.contains("missing Python docstring")));
}

#[test]
fn old_flag_based_test_commands_are_rejected_by_clap() {
    assert!(Cli::try_parse_from(["xtask", "test", "run", "--suite", "nope"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "list", "--category", "unit"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "list", "--suite", "xtask"]).is_err());
}

#[test]
fn unit_group_selection_expands_expected_suites() {
    let selection = selected_unit_suites(&TestUnitArgs {
        command: TestUnitCommands::Group(TestUnitGroupArgs {
            target: TestUnitGroupTarget::Interface,
        }),
    })
    .unwrap();

    assert_eq!(
        selection
            .suites
            .iter()
            .map(|suite| suite.id)
            .collect::<Vec<_>>(),
        vec![
            TestSuiteId::RustPublicApi,
            TestSuiteId::Cli,
            TestSuiteId::NodePackage,
            TestSuiteId::PythonPackage
        ]
    );
}

#[test]
fn full_unit_group_selects_every_unit_suite() {
    let selection = selected_unit_suites(&TestUnitArgs {
        command: TestUnitCommands::Group(TestUnitGroupArgs {
            target: TestUnitGroupTarget::Full,
        }),
    })
    .unwrap();

    assert_eq!(
        selection.suites.len(),
        TEST_SUITES
            .iter()
            .filter(|suite| suite.group == TestGroup::Unit)
            .count()
    );
}

#[test]
fn unit_suite_selection_selects_one_suite() {
    let selection = selected_unit_suites(&TestUnitArgs {
        command: TestUnitCommands::Suite(TestUnitSuiteArgs {
            target: TestUnitSuiteTarget::RustCrates(crate::cli::TestUnitRustArgs {
                package: Some("cogentlm-core".to_owned()),
            }),
        }),
    })
    .unwrap();

    assert_eq!(
        selection
            .suites
            .iter()
            .map(|suite| suite.id)
            .collect::<Vec<_>>(),
        vec![TestSuiteId::RustCrates]
    );
    assert_eq!(selection.package.as_deref(), Some("cogentlm-core"));
}

#[test]
fn smoke_groups_expand_to_expected_suites() {
    let model_args = TestSmokeModelArgs {
        backend: Backend::Cpu,
        model: None,
        offline: false,
        prompt: "hello".to_owned(),
        max_tokens: 1,
        temperature: 0.0,
    };
    let local_model = selected_smoke_suites(&TestSmokeArgs {
        command: TestSmokeCommands::Group(TestSmokeGroupArgs {
            target: TestSmokeGroupTarget::LocalModel(model_args.clone()),
        }),
    })
    .unwrap();
    assert_eq!(
        local_model
            .suites
            .iter()
            .map(|suite| suite.id)
            .collect::<Vec<_>>(),
        vec![
            TestSuiteId::CliSmoke,
            TestSuiteId::RustSmoke,
            TestSuiteId::NodeSmoke,
            TestSuiteId::PythonSmoke
        ]
    );

    let examples = selected_smoke_suites(&TestSmokeArgs {
        command: TestSmokeCommands::Group(TestSmokeGroupArgs {
            target: TestSmokeGroupTarget::Examples(TestSmokeExamplesGroupArgs {
                cases: TestSmokeCaseArgs {
                    model: model_args.clone(),
                    cases: vec![TestSmokeCase::Query],
                },
                browser_host: None,
                browser_port: None,
                browser_timeout_ms: 30_000,
            }),
        }),
    })
    .unwrap();
    assert_eq!(
        examples
            .suites
            .iter()
            .map(|suite| suite.id)
            .collect::<Vec<_>>(),
        vec![
            TestSuiteId::RustSmoke,
            TestSuiteId::NodeSmoke,
            TestSuiteId::PythonSmoke,
            TestSuiteId::ExampleBrowserSmoke
        ]
    );
    assert_eq!(examples.cases, vec![TestSmokeCase::Query]);

    let all = selected_smoke_suites(&TestSmokeArgs {
        command: TestSmokeCommands::Group(TestSmokeGroupArgs {
            target: TestSmokeGroupTarget::Full(TestSmokeFullGroupArgs {
                model: model_args,
                example_browser_timeout_ms: 30_000,
                playground_browser_timeout_ms: 30_000,
            }),
        }),
    })
    .unwrap();
    assert_eq!(
        all.suites.iter().map(|suite| suite.id).collect::<Vec<_>>(),
        vec![
            TestSuiteId::CliSmoke,
            TestSuiteId::RustSmoke,
            TestSuiteId::NodeSmoke,
            TestSuiteId::PythonSmoke,
            TestSuiteId::ExampleBrowserSmoke,
            TestSuiteId::PlaygroundBrowserSmoke,
            TestSuiteId::ProviderGatewaySmoke,
            TestSuiteId::LlamaBackendOps
        ]
    );
}

#[test]
fn coverage_rejects_explicit_non_coverage_suites() {
    let args = TestVerifyArgs {
        target: TestVerifyTarget::BrowserPackage,
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
        target: TestVerifyTarget::Demos,
        changed: false,
    };

    assert!(selected_verify_suites(&args).is_err());
}

#[test]
fn run_report_serializes_suite_status_and_coverage_artifacts() {
    let ctx = BuildContext::new().unwrap();
    let suite = suite_by_id(TestSuiteId::Xtask).unwrap();
    let mut report = RunReport::new(
        serde_json::json!({
            "command": "unit",
            "target": "xtask",
        }),
        &[suite],
    );

    report.suites.push(SuiteReport::passed(
        &ctx,
        suite,
        42,
        Some(TestCounts::passed(7)),
        vec![TestCaseReport {
            suite_id: TestSuiteId::Xtask,
            name: "case | one".to_owned(),
            path: Some("xtask/src/tests/test_tests.rs".to_owned()),
            status: CaseStatus::Passed,
            error: None,
        }],
    ));
    report.finish("passed");

    let value = report.as_json(&ctx);
    assert_eq!(value["status"], "passed");
    assert_eq!(value["summary"]["suites"]["passed"], 1);
    assert_eq!(value["summary"]["suites"]["total"], 1);
    assert_eq!(value["summary"]["counts"]["passed"], 7);
    assert_eq!(
        value["summary"]["counts"]["unknownSuites"],
        serde_json::json!([])
    );
    assert_eq!(value["suites"][0]["status"], "passed");
    assert_eq!(value["suites"][0]["group"], "unit");
    assert_eq!(value["suites"][0]["layer"], "whitebox");
    assert_eq!(value["suites"][0]["counts"]["status"], "known");
    assert_eq!(value["suites"][0]["counts"]["total"], 7);
    assert_eq!(value["suites"][0]["cases"][0]["status"], "passed");
    assert_eq!(value["suites"][0]["cases"][0]["name"], "case | one");
    assert_eq!(value["suites"][0]["coverage"]["status"], "written");
    assert!(value["suites"][0]["coverage"]["artifacts"][0]
        .as_str()
        .unwrap()
        .ends_with(".build/coverage/rust/lcov.info"));
}

#[test]
fn run_report_summarizes_failed_and_unknown_suite_counts() {
    let ctx = BuildContext::new().unwrap();
    let passed = suite_by_id(TestSuiteId::Xtask).unwrap();
    let failed = suite_by_id(TestSuiteId::RustCrates).unwrap();
    let not_run = suite_by_id(TestSuiteId::NodePackage).unwrap();
    let mut report = RunReport::new(serde_json::json!({}), &[passed, failed, not_run]);

    report.suites.push(SuiteReport::passed(
        &ctx,
        passed,
        10,
        Some(TestCounts::passed(3)),
        Vec::new(),
    ));
    report.suites.push(SuiteReport::failed(
        &ctx,
        failed,
        20,
        "bad | value\nnext".to_owned(),
        None,
        vec![TestCaseReport {
            suite_id: TestSuiteId::RustCrates,
            name: "bad | case".to_owned(),
            path: Some("crates/core/src/tests/error_tests.rs".to_owned()),
            status: CaseStatus::Failed,
            error: Some("bad | value\nnext".to_owned()),
        }],
    ));
    report.suites.push(SuiteReport::not_run(&ctx, not_run));
    report.finish("failed");

    let value = report.as_json(&ctx);
    assert_eq!(value["summary"]["suites"]["passed"], 1);
    assert_eq!(value["summary"]["suites"]["failed"], 1);
    assert_eq!(value["summary"]["suites"]["notRun"], 1);
    assert_eq!(value["summary"]["suites"]["total"], 3);
    assert_eq!(value["summary"]["counts"]["passed"], 3);
    assert_eq!(
        value["summary"]["counts"]["unknownSuites"],
        serde_json::json!(["rust-crates", "node-package"])
    );
    assert_eq!(value["suites"][1]["counts"]["status"], "unknown");

    let markdown = report.as_markdown(&ctx);
    assert!(markdown.contains("Suites: 1 passed, 1 failed, 1 not run, 3 total"));
    assert!(markdown.contains("Tests/checks: 3 passed, 0 failed, 0 skipped, 3 total"));
    assert!(markdown.contains("Unknown counts: rust-crates, node-package"));
    assert!(markdown.contains("bad \\| case"));
    assert!(markdown.contains("bad \\| value next"));
}

#[test]
fn case_report_parsers_handle_junit_tap_and_libtest() {
    let junit = r#"
        <testsuite>
          <testcase classname="pkg.tests" name="passes" file="tests/pkg.test.ts" />
          <testcase classname="pkg.tests" name="fails">
            <failure message="bad &amp; broken">trace</failure>
          </testcase>
          <testcase classname="pkg.tests" name="skips"><skipped /></testcase>
        </testsuite>
    "#;
    let junit_cases = parse_junit_case_reports(junit, TestSuiteId::PackageTs);
    assert_eq!(junit_cases.len(), 3);
    assert_eq!(junit_cases[0].status, CaseStatus::Passed);
    assert_eq!(junit_cases[1].status, CaseStatus::Failed);
    assert_eq!(junit_cases[1].error.as_deref(), Some("bad & broken"));
    assert_eq!(junit_cases[2].status, CaseStatus::Skipped);

    let tap = "ok 1 - router imports binding\nnot ok 2 - router rejects invalid backend\nok 3 - optional # SKIP unavailable\n";
    let tap_cases = parse_tap_case_reports(tap, TestSuiteId::NodePackage);
    assert_eq!(tap_cases[0].status, CaseStatus::Passed);
    assert_eq!(tap_cases[1].status, CaseStatus::Failed);
    assert_eq!(tap_cases[2].status, CaseStatus::Skipped);

    let libtest = "[stdout] test tests::passes ... ok\n[stdout] test tests::fails ... FAILED\n[stdout] test tests::skips ... ignored\n";
    let libtest_cases = parse_libtest_case_reports(libtest, TestSuiteId::Xtask);
    assert_eq!(libtest_cases[0].name, "tests::passes");
    assert_eq!(libtest_cases[0].status, CaseStatus::Passed);
    assert_eq!(libtest_cases[1].status, CaseStatus::Failed);
    assert_eq!(libtest_cases[2].status, CaseStatus::Skipped);
}

#[test]
fn case_counts_ignore_unknown_outcomes() {
    let cases = vec![
        TestCaseReport {
            suite_id: TestSuiteId::Xtask,
            name: "passes".to_owned(),
            path: None,
            status: CaseStatus::Passed,
            error: None,
        },
        TestCaseReport {
            suite_id: TestSuiteId::Xtask,
            name: "skips".to_owned(),
            path: None,
            status: CaseStatus::Skipped,
            error: None,
        },
    ];
    assert_eq!(
        case_counts(&cases),
        Some(TestCounts {
            passed: 1,
            failed: 0,
            skipped: 1,
        })
    );

    let mut unknown = cases;
    unknown.push(TestCaseReport {
        suite_id: TestSuiteId::Xtask,
        name: "opaque".to_owned(),
        path: None,
        status: CaseStatus::Unknown,
        error: None,
    });
    assert_eq!(case_counts(&unknown), None);
}

#[test]
fn model_smoke_uses_generation_only_examples() {
    assert_eq!(RUST_GENERATION_SMOKE_EXAMPLES, ["query", "chat"]);
    assert_eq!(NODE_GENERATION_SMOKE_SCRIPTS, ["query.mjs", "chat.mjs"]);
    assert_eq!(PYTHON_GENERATION_SMOKE_SCRIPTS, ["query.py", "chat.py"]);
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
        suite_by_id(TestSuiteId::DemoTs).unwrap(),
        suite_by_id(TestSuiteId::NodePackage).unwrap(),
        suite_by_id(TestSuiteId::PythonPackage).unwrap(),
    ];
    let cases = discover_cases(&ctx, &suites).unwrap();

    assert!(cases
        .iter()
        .any(|case| case.suite_id == TestSuiteId::PackageTs));
    assert!(cases
        .iter()
        .any(|case| case.suite_id == TestSuiteId::DemoTs));
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
    assert!(Cli::try_parse_from(["xtask", "test", "unit"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "whitebox"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "interface"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "xtask"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "rust"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "bindings"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "browser-package"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "demos"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "api"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "cli"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "node"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "unit", "python"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "all"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "model"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "browser"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "rust"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "node"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "python"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "cli"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "provider-gateway"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "test", "smoke", "llama"]).is_err());
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
        Cli::try_parse_from(["xtask", "test", "help", "unit"])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        Cli::try_parse_from(["xtask", "test", "help", "smoke"])
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
    assert!(Cli::try_parse_from(["xtask", "run", "apps", "build", "chat"]).is_err());
    assert!(Cli::try_parse_from(["xtask", "run", "bindings", "node"]).is_err());
}

#[test]
fn run_keeps_demo_example_tool_and_llama_groups() {
    let cli = Cli::parse_from(["xtask", "run", "demos", "build", "chat"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    assert!(matches!(command, RunCommands::Demos { .. }));

    let cli = Cli::parse_from(["xtask", "run", "examples", "serve", "browser"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    assert!(matches!(command, RunCommands::Examples { .. }));

    let cli = Cli::parse_from(["xtask", "run", "tools", "build", "playground"]);
    let Commands::Run { command } = cli.command else {
        panic!("expected run command");
    };
    assert!(matches!(command, RunCommands::Tools { .. }));

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

#[test]
fn package_filter_only_applies_to_rust_crates_suite() {
    let rust_crates = suite_by_id(TestSuiteId::RustCrates).unwrap();
    assert!(validate_package_filter(&[rust_crates], Some("cogentlm-core")).is_ok());

    let xtask = suite_by_id(TestSuiteId::Xtask).unwrap();
    assert!(validate_package_filter(&[xtask], Some("xtask")).is_err());
    assert!(validate_package_filter(&[rust_crates, xtask], Some("cogentlm-core")).is_err());
}

#[test]
fn filtered_rust_targets_reject_unknown_packages() {
    let targets = filtered_rust_targets(RUST_CRATE_TEST_TARGETS, Some("cogentlm-core")).unwrap();
    assert_eq!(targets, vec![RustTestTarget::lib("cogentlm-core")]);
    assert!(filtered_rust_targets(RUST_CRATE_TEST_TARGETS, Some("xtask")).is_err());
}

#[test]
fn search_filter_matches_suite_metadata_and_cases() {
    let mut suites = vec![
        suite_by_id(TestSuiteId::Xtask).unwrap(),
        suite_by_id(TestSuiteId::NodePackage).unwrap(),
    ];
    let mut cases = vec![
        TestCase {
            suite_id: TestSuiteId::NodePackage,
            name: "router routes aliases".to_owned(),
            path: "lib/node/tests/router.test.mjs".to_owned(),
        },
        TestCase {
            suite_id: TestSuiteId::Xtask,
            name: "catalog ids".to_owned(),
            path: "xtask/src/tests/test_tests.rs".to_owned(),
        },
    ];

    apply_search_filter(&mut suites, &mut cases, "router");

    assert_eq!(suites.len(), 1);
    assert_eq!(suites[0].id, TestSuiteId::NodePackage);
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].suite_id, TestSuiteId::NodePackage);
}

#[test]
fn source_path_helpers_classify_roots_and_tests() {
    let suites = [
        suite_by_id(TestSuiteId::Xtask).unwrap(),
        suite_by_id(TestSuiteId::RustCrates).unwrap(),
    ];

    assert!(path_matches_root("xtask/src/test.rs", "xtask/src"));
    assert!(!path_matches_root("xtask-extra/src/lib.rs", "xtask"));
    assert_eq!(
        source_owner_suites("xtask/src/test.rs", &suites)
            .iter()
            .map(|suite| suite.id)
            .collect::<Vec<_>>(),
        vec![TestSuiteId::Xtask]
    );
    assert!(is_first_party_source_path("xtask/src/test.rs"));
    assert!(!is_first_party_source_path("xtask/src/tests/test_tests.rs"));
    assert!(is_probable_test_path("lib/web/tests/router.test.ts"));
    assert!(is_probable_test_path("xtask/src/tests/test_tests.rs"));
}

#[test]
fn rust_test_layout_helpers_detect_allowed_and_inverted_paths() {
    let package_root = std::path::Path::new("crate");
    assert!(is_allowed_rust_test_file(
        package_root,
        std::path::Path::new("crate/src/tests/foo_tests.rs")
    ));
    assert!(is_allowed_rust_test_file(
        package_root,
        std::path::Path::new("crate/tests/public_api.rs")
    ));
    assert!(is_inverted_rust_test_file(
        package_root,
        std::path::Path::new("crate/src/foo/tests/bar_tests.rs")
    ));
    assert_eq!(
        path_components(std::path::Path::new("src/tests/foo_tests.rs")),
        vec!["src", "tests", "foo_tests.rs"]
    );
    assert!(contains_test_attribute("    #[test]\nfn case() {}"));
    assert!(contains_test_attribute(
        "    #[tokio::test]\nasync fn case() {}"
    ));
}

#[test]
fn filesystem_collectors_are_sorted_and_skip_ignored_dirs() {
    let temp = TempDir::new("test-collectors");
    temp.write("root/b.rs", "");
    temp.write("root/a.rs", "");
    temp.write("root/target/ignored.rs", "");
    temp.write("root/c.test.ts", "");
    temp.write("root/nested/d.test.ts", "");

    let rust_files = collect_files_with_extension(&temp.join("root"), "rs").unwrap();
    assert_eq!(
        rust_files,
        vec![temp.join("root/a.rs"), temp.join("root/b.rs")]
    );

    let ts_tests = collect_files_with_suffix(&temp.join("root"), ".test.ts").unwrap();
    assert_eq!(
        ts_tests,
        vec![
            temp.join("root/c.test.ts"),
            temp.join("root/nested/d.test.ts")
        ]
    );
}

#[test]
fn cpp_and_rust_case_name_parsers_handle_supported_shapes() {
    assert!(is_cpp_test_file_name(std::path::Path::new(
        "test_router.cpp"
    )));
    assert!(is_cpp_test_file_name(std::path::Path::new(
        "router-test.cc"
    )));
    assert!(!is_cpp_test_file_name(std::path::Path::new("router.cpp")));
    assert_eq!(
        parse_rust_fn_name("fn parses_case() {"),
        Some("parses_case".to_owned())
    );
    assert_eq!(
        parse_rust_fn_name("async fn parses_async_case() {"),
        Some("parses_async_case".to_owned())
    );
    assert_eq!(
        parse_quoted_test_name("test(\"routes aliases\", () => {})", "test("),
        Some("routes aliases".to_owned())
    );
}

#[test]
fn coverage_helpers_summarize_selected_report_areas_and_lcov() {
    let xtask = suite_by_id(TestSuiteId::Xtask).unwrap();
    let node = suite_by_id(TestSuiteId::NodePackage).unwrap();
    let python = suite_by_id(TestSuiteId::PythonPackage).unwrap();
    let areas = coverage_report_areas(&[xtask, node, python]);
    assert!(areas.rust);
    assert!(areas.node);
    assert!(areas.python);

    let temp = TempDir::new("test-lcov");
    let lcov = temp.write(
        "lcov.info",
        "TN:\nSF:file.rs\nLF:10\nLH:7\nend_of_record\nSF:file2.rs\nLF:5\nLH:5\nend_of_record\n",
    );
    let summary = parse_lcov_summary(&lcov).unwrap();
    assert_eq!(summary.found, 15);
    assert_eq!(summary.hit, 12);
    assert_eq!(LcovSummary::default().percent(), 0.0);
    assert_eq!(
        CoverageSummaries {
            rust: summary,
            node: LcovSummary::default(),
            python: LcovSummary::default(),
        }
        .as_json()["rust"]["hit"],
        12
    );
}

#[test]
fn report_markdown_and_json_escape_dynamic_values() {
    let ctx = BuildContext::new().unwrap();
    let args = TestVerifyArgs {
        target: TestVerifyTarget::Xtask,
        changed: true,
    };
    let suite = suite_by_id(TestSuiteId::Xtask).unwrap();
    let mut report = VerifyReport::new(&args, &[suite]);
    let error = anyhow::anyhow!("bad | value\nnext");
    let result: anyhow::Result<()> = Err(error);
    report
        .checks
        .push(VerifyCheckReport::from_result("coverage", &result));
    report.checks.push(VerifyCheckReport::skipped("changed"));
    report.finish("failed");

    let json = report.as_json(&ctx);
    assert_eq!(json["kind"], "test-verify");
    assert_eq!(json["checks"][0]["status"], "failed");
    assert_eq!(json["checks"][1]["status"], "skipped");
    assert!(report.as_markdown(&ctx).contains("bad \\| value next"));
}

#[test]
fn formatting_helpers_are_platform_and_overflow_safe() {
    let ctx = BuildContext::new().unwrap();
    assert_eq!(normalize_relative_path(" ./a\\b "), "a/b");
    assert!(display_relative(&ctx, &ctx.build_root()).contains(".build"));
    assert_eq!(duration_millis(u128::from(u64::MAX) + 1), u64::MAX);
    assert_eq!(markdown_cell("a|b\r\nc"), "a\\|b  c");
    assert!(python_venv_exe(std::path::Path::new(".venv"))
        .display()
        .to_string()
        .contains("python"));
}

#[test]
fn backend_and_discoverer_labels_are_stable() {
    assert_eq!(test_backends(&Backend::Cpu), vec![Backend::Cpu]);
    assert_eq!(CaseDiscoverer::None.as_str(), "none");
    assert_eq!(CaseDiscoverer::PackageTs.as_str(), "package-ts");
}
