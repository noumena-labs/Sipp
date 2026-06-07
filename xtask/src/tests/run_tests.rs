//! Tests the `run` module in `xtask`.
//!
//! Covers backend selection, executable discovery, recursive file search, and
//! pre-execution error paths with fake build directories instead of invoking
//! CMake, Vite, or llama.cpp binaries.

use crate::cli::{
    Backend, LlamaBackendOpsMode, LlamaBackendOpsOutput, RunCommands, RunGatewayExampleCase,
    RunLlamaBackendOpsArgs, RunLlamaCommands,
};
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    find_file_recursive, find_llama_backend_ops_exe, gateway_example_alias, host_binding_backends,
    node_gateway_example_script, python_gateway_example_script, run, rust_gateway_example_bin,
    validate_gateway_example_backend,
};

#[test]
fn host_backend_expansion_matches_supported_platform_family() {
    let backends = host_binding_backends();
    assert_eq!(backends[0], Backend::Cpu);
    if cfg!(target_os = "macos") {
        assert_eq!(backends, [Backend::Cpu, Backend::Metal]);
    } else {
        assert_eq!(backends, [Backend::Cpu, Backend::Vulkan, Backend::Cuda]);
    }
}

#[test]
fn recursive_file_search_finds_nested_files_and_ignores_missing_roots() {
    let temp = TempDir::new("run-find-file");
    let expected = temp.write("a/b/target.txt", "fixture");

    assert_eq!(
        find_file_recursive(temp.path(), "target.txt").unwrap(),
        Some(expected)
    );
    assert_eq!(
        find_file_recursive(temp.path(), "missing.txt").unwrap(),
        None
    );
    assert_eq!(
        find_file_recursive(&temp.join("missing-root"), "target.txt").unwrap(),
        None
    );
}

#[test]
fn backend_ops_executable_discovery_checks_known_locations_then_recurses() {
    let temp = TempDir::new("run-backend-ops");
    let exe_name = if cfg!(windows) {
        "test-backend-ops.exe"
    } else {
        "test-backend-ops"
    };
    let direct = temp.write(format!("build/bin/{exe_name}"), "");
    assert_eq!(
        find_llama_backend_ops_exe(&temp.join("build")).unwrap(),
        direct
    );

    let temp = TempDir::new("run-backend-ops-recursive");
    let nested = temp.write(format!("build/deep/path/{exe_name}"), "");
    assert_eq!(
        find_llama_backend_ops_exe(&temp.join("build")).unwrap(),
        nested
    );
    assert!(find_llama_backend_ops_exe(&temp.join("missing")).is_err());
}

#[test]
fn llama_correctness_mode_is_rejected_before_external_commands() {
    let temp = TempDir::new("run-llama-reject");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let sh = xshell::Shell::new().unwrap();

    let error = run(
        &sh,
        &ctx,
        RunCommands::Llama {
            command: RunLlamaCommands::BackendOps(RunLlamaBackendOpsArgs {
                backend: Backend::Cpu,
                mode: LlamaBackendOpsMode::Test,
                op: None,
                params: None,
                output: LlamaBackendOpsOutput::Console,
            }),
        },
    )
    .unwrap_err();

    assert!(format!("{error:#}").contains("correctness checks moved"));
}

#[test]
fn gateway_example_case_mappings_match_client_files_and_aliases() {
    assert_eq!(gateway_example_alias(RunGatewayExampleCase::Query), "local");
    assert_eq!(gateway_example_alias(RunGatewayExampleCase::Chat), "local");
    assert_eq!(gateway_example_alias(RunGatewayExampleCase::Embed), "local");

    assert_eq!(
        rust_gateway_example_bin(RunGatewayExampleCase::Query),
        "gateway_query"
    );
    assert_eq!(
        rust_gateway_example_bin(RunGatewayExampleCase::Chat),
        "gateway_chat"
    );
    assert_eq!(
        rust_gateway_example_bin(RunGatewayExampleCase::Embed),
        "gateway_embed"
    );
    assert_eq!(
        node_gateway_example_script(RunGatewayExampleCase::Query),
        "gateway_query.mjs"
    );
    assert_eq!(
        node_gateway_example_script(RunGatewayExampleCase::Chat),
        "gateway_chat.mjs"
    );
    assert_eq!(
        node_gateway_example_script(RunGatewayExampleCase::Embed),
        "gateway_embed.mjs"
    );
    assert_eq!(
        python_gateway_example_script(RunGatewayExampleCase::Query),
        "gateway_query.py"
    );
    assert_eq!(
        python_gateway_example_script(RunGatewayExampleCase::Chat),
        "gateway_chat.py"
    );
    assert_eq!(
        python_gateway_example_script(RunGatewayExampleCase::Embed),
        "gateway_embed.py"
    );
}

#[test]
fn gateway_example_backend_all_is_rejected() {
    let error = validate_gateway_example_backend(&Backend::All).unwrap_err();

    assert!(format!("{error:#}").contains("concrete backend"));
    assert!(validate_gateway_example_backend(&Backend::Cpu).is_ok());
}
