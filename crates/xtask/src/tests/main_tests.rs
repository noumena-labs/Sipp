//! Tests the `main` binary module in `xtask`.
//!
//! Covers command-summary helpers and backend defaults without invoking the
//! binary dispatcher, subprocesses, or build orchestration.

use xtask::cli::{
    Backend, BackendArgs, BuildCommands, CleanArgs, Commands, TestCategoryFilter, TestCommands,
    TestListArgs, TestListFormat, TestRunArgs, TestVerifyArgs,
};

use super::{backend_summary, build_summary, command_summary, test_summary};

#[test]
fn command_summary_labels_top_level_workflows() {
    assert_eq!(
        command_summary(&Commands::Clean(CleanArgs {
            purge: false,
            toolchains: false,
            dry_run: true,
        })),
        "Clean workspace"
    );
    assert_eq!(
        command_summary(&Commands::Build {
            target: BuildCommands::Core,
        }),
        "Build native Rust workspace"
    );
}

#[test]
fn test_summary_labels_test_subcommands() {
    assert_eq!(
        test_summary(&TestCommands::List(TestListArgs {
            category: TestCategoryFilter::All,
            suite: Vec::new(),
            cases: false,
            search: None,
            format: TestListFormat::Text,
        })),
        "List test suites"
    );
    assert_eq!(
        test_summary(&TestCommands::Run(TestRunArgs {
            category: TestCategoryFilter::Whitebox,
            suite: Vec::new(),
            package: None,
            backend: Backend::Vulkan,
            model: None,
            offline: false,
        })),
        "Run tests (vulkan)"
    );
    assert_eq!(
        test_summary(&TestCommands::Verify(TestVerifyArgs {
            category: TestCategoryFilter::All,
            suite: Vec::new(),
            changed: true,
        })),
        "Verify tests"
    );
}

#[test]
fn build_summary_labels_backend_defaults_and_explicit_backends() {
    assert_eq!(
        build_summary(&BuildCommands::All),
        "Build all default targets"
    );
    assert_eq!(
        build_summary(&BuildCommands::Node(BackendArgs { backend: None })),
        "Build Node.js bindings (cpu)"
    );
    assert_eq!(
        build_summary(&BuildCommands::Python(BackendArgs {
            backend: Some(Backend::Cuda),
        })),
        "Build Python bindings (cuda)"
    );
    assert_eq!(
        build_summary(&BuildCommands::Cli(BackendArgs {
            backend: Some(Backend::All),
        })),
        "Build Rust CLI distribution (all)"
    );
}

#[test]
fn backend_summary_uses_cpu_for_missing_backend() {
    assert_eq!(backend_summary(None), "cpu");
    assert_eq!(backend_summary(Some(Backend::Metal)), "metal");
}
