//! Tests the `main` binary module in `xtask`.
//!
//! Covers command-summary helpers and backend defaults without invoking the
//! binary dispatcher, subprocesses, or build orchestration.

use xtask::cli::{
    Backend, BackendArgs, BuildCommands, CleanArgs, Commands, TestCommands, TestGroupFilter,
    TestListArgs, TestListFormat, TestSmokeArgs, TestSmokeBrowserArgs, TestSmokeTarget,
    TestUnitArgs, TestVerifyArgs, TestVerifyTarget,
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
            group: TestGroupFilter::All,
            layer: None,
            cases: false,
            search: None,
            format: TestListFormat::Text,
        })),
        "List test suites"
    );
    assert_eq!(
        test_summary(&TestCommands::Unit(TestUnitArgs { target: None })),
        "Run unit tests"
    );
    assert_eq!(
        test_summary(&TestCommands::Smoke(TestSmokeArgs {
            target: TestSmokeTarget::Browser(TestSmokeBrowserArgs {
                host: None,
                port: None,
                timeout_ms: 30_000,
                require_rust_engine: false,
                require_gguf_ingest: false,
                require_webgpu: false,
            }),
        })),
        "Run smoke tests"
    );
    assert_eq!(
        test_summary(&TestCommands::Verify(TestVerifyArgs {
            target: TestVerifyTarget::All,
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
