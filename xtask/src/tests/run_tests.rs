//! Tests the `run` module in `xtask`.
//!
//! Covers backend selection, executable discovery, recursive file search, and
//! pre-execution error paths with fake build directories instead of invoking
//! CMake, Vite, or llama.cpp binaries.

use crate::cli::{
    Backend, LlamaBackendOpsMode, LlamaBackendOpsOutput, RunCommands, RunGatewayExampleCase,
    RunGatewayLocalServeArgs, RunLlamaBackendOpsArgs, RunLlamaCommands,
};
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    find_file_recursive, find_llama_backend_ops_exe, gateway_example_alias,
    gateway_web_allowed_origins, host_binding_backends, node_gateway_example_script,
    python_gateway_example_script, run, rust_gateway_example_bin, validate_gateway_example_backend,
    write_local_gateway_config, write_local_gateway_example_config, LocalGatewayConfigOptions,
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
    assert_eq!(
        gateway_example_alias(RunGatewayExampleCase::Embed),
        "local-embed"
    );

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
fn gateway_web_origins_include_selected_host_port_and_loopback_peer() {
    assert_eq!(
        gateway_web_allowed_origins("localhost", 4173),
        vec![
            "http://localhost:4173".to_owned(),
            "http://127.0.0.1:4173".to_owned()
        ]
    );
    assert_eq!(
        gateway_web_allowed_origins("0.0.0.0", 5174),
        vec![
            "http://127.0.0.1:5174".to_owned(),
            "http://localhost:5174".to_owned()
        ]
    );
}

#[test]
fn local_gateway_config_uses_dynamic_browser_origins() {
    let temp = TempDir::new("run-gateway-config");
    let model = temp.write("models/model.gguf", "fixture");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let args = RunGatewayLocalServeArgs {
        model,
        bind: "127.0.0.1:18888".to_owned(),
        token_env: "TEST_GATEWAY_TOKEN".to_owned(),
        backend: Backend::Cpu,
    };

    let config = write_local_gateway_example_config(&ctx, &args).unwrap();
    let contents = std::fs::read_to_string(config).unwrap();

    assert!(contents.contains("bind = \"127.0.0.1:18888\""));
    assert!(contents.contains("token_env = \"TEST_GATEWAY_TOKEN\""));
    assert!(contents
        .contains("allowed_origins = [\"http://127.0.0.1:5173\", \"http://localhost:5173\"]"));
    assert!(contents.contains("name = \"local\""));
    assert!(contents.contains("name = \"local-embed\""));

    let custom_origins = gateway_web_allowed_origins("localhost", 4173);
    let config = write_local_gateway_config(
        &ctx,
        LocalGatewayConfigOptions {
            model: &args.model,
            bind: &args.bind,
            token_env: &args.token_env,
            allowed_origins: &custom_origins,
        },
    )
    .unwrap();
    let contents = std::fs::read_to_string(config).unwrap();
    assert!(contents
        .contains("allowed_origins = [\"http://localhost:4173\", \"http://127.0.0.1:4173\"]"));
}

#[test]
fn gateway_example_backend_all_is_rejected() {
    let error = validate_gateway_example_backend(&Backend::All).unwrap_err();

    assert!(format!("{error:#}").contains("concrete backend"));
    assert!(validate_gateway_example_backend(&Backend::Cpu).is_ok());
}
