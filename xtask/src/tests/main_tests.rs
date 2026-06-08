//! Tests the `main` binary module in `xtask`.
//!
//! Covers command-summary helpers and backend defaults without invoking the
//! binary dispatcher, subprocesses, or build orchestration.

use xtask::cli::{
    Backend, BackendArgs, BuildCommands, CleanArgs, Commands, RunCommands, RunGatewayServerArgs,
    RunGatewayServerCommand, RunGatewayServerSourceArgs, TestCommands, TestGroupFilter,
    TestListArgs, TestListFormat, TestSmokeArgs, TestSmokeCaseArgs, TestSmokeCommands,
    TestSmokeModelArgs, TestSmokePlaygroundBrowserArgs, TestSmokeSuiteArgs, TestSmokeSuiteTarget,
    TestUnitArgs, TestUnitCommands, TestUnitGroupArgs, TestUnitGroupTarget, TestVerifyArgs,
    TestVerifyTarget,
};

use super::{
    backend_summary, build_summary, command_summary, effective_output_options, test_summary,
    OutputOptions,
};

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
        test_summary(&TestCommands::Unit(TestUnitArgs {
            command: TestUnitCommands::Group(TestUnitGroupArgs {
                target: TestUnitGroupTarget::Full,
            }),
        })),
        "Run unit tests"
    );
    assert_eq!(
        test_summary(&TestCommands::Smoke(TestSmokeArgs {
            command: TestSmokeCommands::Suite(TestSmokeSuiteArgs {
                target: TestSmokeSuiteTarget::PlaygroundBrowser(TestSmokePlaygroundBrowserArgs {
                    host: None,
                    port: None,
                    timeout_ms: 30_000,
                    require_rust_engine: false,
                    require_gguf_ingest: false,
                    require_webgpu: false,
                }),
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
    assert_eq!(
        build_summary(&BuildCommands::GatewayServer(BackendArgs {
            backend: Some(Backend::Vulkan),
        })),
        "Build gateway-server (vulkan)"
    );
}

#[test]
fn backend_summary_uses_cpu_for_missing_backend() {
    assert_eq!(backend_summary(None), "cpu");
    assert_eq!(backend_summary(Some(Backend::Metal)), "metal");
}

#[test]
fn effective_output_options_keep_build_and_run_compact_by_default() {
    let build = Commands::Build {
        target: BuildCommands::Core,
    };
    let run = Commands::Run {
        command: RunCommands::GatewayServer(RunGatewayServerArgs {
            command: RunGatewayServerCommand::Check(RunGatewayServerSourceArgs {
                config: "apps/gateway-server/config/development.toml".into(),
                backend: Backend::Cpu,
            }),
        }),
    };

    assert_eq!(
        effective_output_options(&build, false, false),
        OutputOptions {
            stream_subprocess: false,
            plain: false,
            final_status: true,
        }
    );
    assert_eq!(
        effective_output_options(&run, false, false),
        OutputOptions {
            stream_subprocess: false,
            plain: false,
            final_status: true,
        }
    );
    assert_eq!(
        effective_output_options(&build, true, false),
        OutputOptions {
            stream_subprocess: true,
            plain: false,
            final_status: true,
        }
    );
    assert_eq!(
        effective_output_options(&build, false, true),
        OutputOptions {
            stream_subprocess: false,
            plain: true,
            final_status: true,
        }
    );
}

#[test]
fn effective_output_options_keep_test_execution_compact_by_default() {
    let unit = Commands::Test {
        command: TestCommands::Unit(TestUnitArgs {
            command: TestUnitCommands::Group(TestUnitGroupArgs {
                target: TestUnitGroupTarget::Whitebox,
            }),
        }),
    };
    let smoke = Commands::Test {
        command: TestCommands::Smoke(TestSmokeArgs {
            command: TestSmokeCommands::Suite(TestSmokeSuiteArgs {
                target: TestSmokeSuiteTarget::ExampleGateway(TestSmokeCaseArgs {
                    model: TestSmokeModelArgs {
                        backend: Backend::Cpu,
                        model: None,
                        offline: false,
                        prompt: "hello".to_owned(),
                        max_tokens: 1,
                        temperature: 0.0,
                    },
                    cases: Vec::new(),
                }),
            }),
        }),
    };

    assert_eq!(
        effective_output_options(&unit, false, false),
        OutputOptions {
            stream_subprocess: false,
            plain: false,
            final_status: true,
        }
    );
    assert_eq!(
        effective_output_options(&smoke, false, true),
        OutputOptions {
            stream_subprocess: false,
            plain: true,
            final_status: true,
        }
    );
}

#[test]
fn effective_output_options_keep_information_commands_plain() {
    let test_list = Commands::Test {
        command: TestCommands::List(TestListArgs {
            group: TestGroupFilter::All,
            layer: None,
            cases: false,
            search: None,
            format: TestListFormat::Text,
        }),
    };

    assert_eq!(
        effective_output_options(&test_list, false, false),
        OutputOptions {
            stream_subprocess: false,
            plain: true,
            final_status: true,
        }
    );
}

#[test]
fn effective_output_options_keep_json_test_list_machine_readable() {
    let test_list = Commands::Test {
        command: TestCommands::List(TestListArgs {
            group: TestGroupFilter::All,
            layer: None,
            cases: true,
            search: None,
            format: TestListFormat::Json,
        }),
    };

    assert_eq!(
        effective_output_options(&test_list, false, false),
        OutputOptions {
            stream_subprocess: false,
            plain: true,
            final_status: false,
        }
    );
}
