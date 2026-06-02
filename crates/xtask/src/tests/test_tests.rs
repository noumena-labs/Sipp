//! Unit tests for cataloged test orchestration.

use clap::Parser;

use crate::cli::{
    Backend, Cli, Commands, LlamaBackendOpsMode, RunCommands, RunLlamaCommands, TestCommands,
    TestProfile, TestSuiteId,
};
use crate::utils::BuildContext;

use super::{
    catalog_suite_ids, collect_catalog_ownership_violations, discover_cases, profile_suite_labels,
    selected_suites, suite_by_id, validate_suite_backends, TestCategory, TEST_SUITES,
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
fn ci_profile_keeps_expected_gate_suites() {
    let ci_suites = TEST_SUITES
        .iter()
        .filter(|suite| suite.profiles.contains(&TestProfile::Ci))
        .map(|suite| suite.id)
        .collect::<Vec<_>>();

    assert!(ci_suites.contains(&TestSuiteId::Layout));
    assert!(ci_suites.contains(&TestSuiteId::Xtask));
    assert!(ci_suites.contains(&TestSuiteId::RustCrates));
    assert!(ci_suites.contains(&TestSuiteId::RustPublicApi));
    assert!(ci_suites.contains(&TestSuiteId::PackageTs));
    assert!(!ci_suites.contains(&TestSuiteId::ModelSmoke));
}

#[test]
fn contributor_profile_stays_public_safe() {
    let contributor_suites = TEST_SUITES
        .iter()
        .filter(|suite| suite.profiles.contains(&TestProfile::Contributor))
        .map(|suite| suite.id)
        .collect::<Vec<_>>();

    assert_eq!(
        contributor_suites,
        vec![TestSuiteId::Layout, TestSuiteId::Xtask]
    );
}

#[test]
fn profile_suite_labels_explain_cumulative_profiles() {
    assert_eq!(
        profile_suite_labels(TestProfile::Contributor),
        vec!["layout", "xtask"]
    );
    assert_eq!(
        profile_suite_labels(TestProfile::Quick),
        vec!["layout", "xtask", "rust-crates"]
    );
    assert_eq!(
        profile_suite_labels(TestProfile::Ci),
        vec![
            "layout",
            "xtask",
            "rust-crates",
            "package-ts",
            "rust-public-api"
        ]
    );
    assert!(profile_suite_labels(TestProfile::Full).contains(&"model-smoke"));
}

#[test]
fn new_test_commands_parse() {
    let cli = Cli::parse_from([
        "xtask",
        "test",
        "whitebox",
        "--suite",
        "rust-crates",
        "--package",
        "cogentlm-core",
    ]);

    let Commands::Test { command } = cli.command else {
        panic!("expected test command");
    };
    let TestCommands::Whitebox(args) = command else {
        panic!("expected whitebox command");
    };
    assert_eq!(args.suite, TestSuiteId::RustCrates);
    assert_eq!(args.package.as_deref(), Some("cogentlm-core"));
}

#[test]
fn invalid_suite_names_are_rejected_by_clap() {
    assert!(Cli::try_parse_from(["xtask", "test", "whitebox", "--suite", "nope"]).is_err());
}

#[test]
fn suite_category_mismatches_are_rejected() {
    assert!(selected_suites(&TestSuiteId::NodePackage, TestCategory::Whitebox).is_err());
    assert!(selected_suites(&TestSuiteId::PackageTs, TestCategory::Interface).is_err());
}

#[test]
fn backend_preflight_rejects_all_for_concrete_only_suites() {
    let python = suite_by_id(TestSuiteId::PythonPackage).unwrap();
    assert!(validate_suite_backends(&[python], Backend::All).is_err());

    let node = suite_by_id(TestSuiteId::NodePackage).unwrap();
    assert!(validate_suite_backends(&[node], Backend::All).is_ok());
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
        .all(|case| !case.path.contains("tests\\public_api.rs")
            && !case.path.contains("tests/public_api.rs")));
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
