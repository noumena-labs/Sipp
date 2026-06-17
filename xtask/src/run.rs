//! Developer run workflows for long-lived demos and non-test diagnostics.

use crate::cli::{
    Backend, DemoName, DemoServeMode, ExampleName, LlamaBackendOpsMode, RunBrowserExampleServeArgs,
    RunCommands, RunDemoServeArgs, RunDemosCommands, RunExampleServeArgs, RunExampleServeTarget,
    RunExamplesCommands, RunGatewayExampleArgs, RunGatewayExampleCase, RunGatewayExampleClientArgs,
    RunGatewayExampleCommonArgs, RunGatewayExampleTarget, RunGatewayExampleWebArgs,
    RunGatewayLocalServeArgs, RunGatewayOpenAiServeArgs, RunGatewayServerArgs,
    RunGatewayServerCommand, RunGatewayServerSourceArgs, RunLlamaBackendOpsArgs, RunLlamaCommands,
    RunToolServeArgs, RunToolsCommands, ToolName, WasmThreading,
};
use crate::javascript;
use crate::output;
use crate::sample_model::{self, SampleModelOptions};
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::{apply_uv_env, setup_uv};
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use xshell::{cmd, Cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/run_tests.rs"]
mod run_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const LLAMA_BACKEND_OPS_TARGET: &str = "test-backend-ops";
const GATEWAY_RUN_TOKEN_ENV: &str = "SIPP_GATEWAY_TOKEN";
const GATEWAY_RUN_START_TIMEOUT: Duration = Duration::from_secs(300);

/// Runs a developer workflow.
pub fn run(sh: &Shell, ctx: &BuildContext, command: RunCommands) -> Result<()> {
    match command {
        RunCommands::GatewayServer(args) => run_gateway_server(sh, ctx, &args),
        RunCommands::Demos { command } => run_demos(sh, ctx, command),
        RunCommands::Examples { command } => run_examples(sh, ctx, command),
        RunCommands::Tools { command } => run_tools(sh, ctx, command),
        RunCommands::Llama { command } => run_llama(sh, ctx, command),
    }
}

fn run_demos(sh: &Shell, ctx: &BuildContext, command: RunDemosCommands) -> Result<()> {
    match command {
        RunDemosCommands::Build(args) => build_one_demo(sh, ctx, args.demo),
        RunDemosCommands::Serve(args) => serve_demo(sh, ctx, &args),
    }
}

fn run_examples(sh: &Shell, ctx: &BuildContext, command: RunExamplesCommands) -> Result<()> {
    match command {
        RunExamplesCommands::Serve(args) => serve_example(sh, ctx, &args),
        RunExamplesCommands::Gateway(args) => run_gateway_example(sh, ctx, &args),
    }
}

fn run_tools(sh: &Shell, ctx: &BuildContext, command: RunToolsCommands) -> Result<()> {
    match command {
        RunToolsCommands::Build(args) => build_one_tool(sh, ctx, args.tool),
        RunToolsCommands::Serve(args) => serve_tool(sh, ctx, &args),
    }
}

fn run_llama(sh: &Shell, ctx: &BuildContext, command: RunLlamaCommands) -> Result<()> {
    match command {
        RunLlamaCommands::BackendOps(args) => {
            if matches!(args.mode, LlamaBackendOpsMode::Test) {
                anyhow::bail!(
                    "llama.cpp correctness checks moved to `cargo xtask test smoke suite llama-backend-ops`"
                );
            }
            run_llama_backend_ops(sh, ctx, &args)
        }
    }
}

fn run_gateway_server(sh: &Shell, ctx: &BuildContext, args: &RunGatewayServerArgs) -> Result<()> {
    match &args.command {
        RunGatewayServerCommand::Check(args) => {
            run_gateway_server_source(sh, ctx, args, "check", false)
        }
        RunGatewayServerCommand::Serve(args) => {
            run_gateway_server_source(sh, ctx, args, "serve", true)
        }
    }
}

fn run_gateway_server_source(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayServerSourceArgs,
    command: &'static str,
    long_running: bool,
) -> Result<()> {
    let phase = format!("Gateway-server {command}");
    output::phase(&phase);
    let config = resolve_gateway_server_config_path(ctx, &args.config)?;
    output::path("Config", &config);
    output::detail("Backend", args.backend.as_str());

    targets::gateway_server::build(sh, ctx, Some(&args.backend))?;
    let dist_dir = ctx.gateway_server_artifacts_dir();
    let gateway = dist_dir.join(targets::gateway_server::gateway_binary_file_name());
    if !gateway.is_file() {
        anyhow::bail!("gateway-server binary is missing: {}", gateway.display());
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    let mut gateway_cmd = cmd!(sh, "{gateway} {command} --config {config}");
    gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(&args.backend))?;
    gateway_cmd = apply_gateway_server_runtime_env(gateway_cmd, &dist_dir);

    if long_running {
        output::run_long_command("Starting gateway-server", gateway_cmd)
            .context("gateway-server failed")
    } else {
        output::run_command("Checking gateway-server config", gateway_cmd)
            .context("gateway-server config check failed")
    }
}

fn resolve_gateway_server_config_path(ctx: &BuildContext, config: &Path) -> Result<PathBuf> {
    let path = if config.is_absolute() {
        config.to_path_buf()
    } else {
        ctx.workspace_root().join(config)
    };
    if !path.is_file() {
        anyhow::bail!("gateway config file does not exist: {}", path.display());
    }
    path.canonicalize()
        .with_context(|| format!("failed to canonicalize {}", path.display()))
}

fn apply_gateway_server_runtime_env<'a>(command: Cmd<'a>, dist_dir: &Path) -> Cmd<'a> {
    let path = dist_dir.display().to_string();
    if cfg!(windows) {
        let current = env::var("PATH").unwrap_or_default();
        return command.env("PATH", prepend_path(&path, &current));
    }
    if cfg!(target_os = "macos") {
        let current = env::var("DYLD_LIBRARY_PATH").unwrap_or_default();
        return command.env("DYLD_LIBRARY_PATH", prepend_path(&path, &current));
    }
    let current = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    command.env("LD_LIBRARY_PATH", prepend_path(&path, &current))
}

fn prepend_path(path: &str, current: &str) -> String {
    if current.is_empty() {
        path.to_string()
    } else if cfg!(windows) {
        format!("{path};{current}")
    } else {
        format!("{path}:{current}")
    }
}

fn build_one_demo(sh: &Shell, ctx: &BuildContext, demo: DemoName) -> Result<()> {
    output::phase(&format!("Build browser demo: {}", demo.slug()));
    ensure_javascript_workspace_dependencies(sh, ctx, &ctx.demo_dir(demo.slug()))?;
    targets::wasm::build(sh, ctx, WasmThreading::All)?;
    build_demo_only(sh, ctx, demo)
}

fn build_demo_only(sh: &Shell, ctx: &BuildContext, demo: DemoName) -> Result<()> {
    let demo_dir = ctx.demo_dir(demo.slug());
    output::phase(&format!("Demo build: {}", demo.slug()));
    output::path("Demo workspace", &demo_dir);
    output::path("Artifact directory", &ctx.demo_artifacts_dir(demo.slug()));

    let _dir = sh.push_dir(&demo_dir);
    output::run_build_command(
        format!("Building {} demo", demo.slug()),
        cmd!(sh, "bun run build"),
    )
    .with_context(|| format!("failed to build {} demo", demo.slug()))
}

fn serve_demo(sh: &Shell, ctx: &BuildContext, args: &RunDemoServeArgs) -> Result<()> {
    output::phase(&format!("Serve browser demo: {}", args.demo.slug()));
    output::detail("Mode", args.mode.as_str());
    output::path("Demo workspace", &ctx.demo_dir(args.demo.slug()));

    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &ctx.demo_dir(args.demo.slug()))?;
        targets::wasm::build(sh, ctx, WasmThreading::All)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_demo_only(sh, ctx, args.demo)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    let demo_dir = ctx.demo_dir(args.demo.slug());
    let _dir = sh.push_dir(&demo_dir);
    let mut serve_cmd = match args.mode {
        DemoServeMode::Dev => cmd!(sh, "bunx --bun vite"),
        DemoServeMode::Preview => cmd!(sh, "bunx --bun vite preview"),
    };

    if let Some(host) = &args.host {
        serve_cmd = serve_cmd.arg("--host").arg(host);
    }
    if let Some(port) = args.port {
        serve_cmd = serve_cmd.arg("--port").arg(port.to_string());
    }

    output::run_long_command(
        format!(
            "Starting {} Vite server for {}",
            args.mode.as_str(),
            args.demo.slug()
        ),
        serve_cmd,
    )
    .with_context(|| format!("{} demo server failed", args.demo.slug()))
}

fn build_one_tool(sh: &Shell, ctx: &BuildContext, tool: ToolName) -> Result<()> {
    output::phase(&format!("Build tool: {}", tool.slug()));
    ensure_javascript_workspace_dependencies(sh, ctx, &tool_dir(ctx, tool))?;
    targets::wasm::build(sh, ctx, WasmThreading::All)?;
    build_tool_only(sh, ctx, tool)
}

fn build_tool_only(sh: &Shell, ctx: &BuildContext, tool: ToolName) -> Result<()> {
    let tool_dir = tool_dir(ctx, tool);
    output::phase(&format!("Tool build: {}", tool.slug()));
    output::path("Tool workspace", &tool_dir);
    output::path("Artifact directory", &ctx.tool_artifacts_dir(tool.slug()));

    let _dir = sh.push_dir(&tool_dir);
    output::run_build_command(
        format!("Building {} tool", tool.slug()),
        cmd!(sh, "bun run build"),
    )
    .with_context(|| format!("failed to build {} tool", tool.slug()))
}

fn serve_tool(sh: &Shell, ctx: &BuildContext, args: &RunToolServeArgs) -> Result<()> {
    output::phase(&format!("Serve tool: {}", args.tool.slug()));
    output::detail("Mode", args.mode.as_str());
    output::path("Tool workspace", &tool_dir(ctx, args.tool));

    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &tool_dir(ctx, args.tool))?;
        targets::wasm::build(sh, ctx, WasmThreading::All)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_tool_only(sh, ctx, args.tool)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    serve_vite_workspace(
        sh,
        &tool_dir(ctx, args.tool),
        args.mode,
        args.host.as_deref(),
        args.port,
        format!(
            "Starting {} Vite server for {} tool",
            args.mode.as_str(),
            args.tool.slug()
        ),
        format!("{} tool server failed", args.tool.slug()),
    )
}

fn serve_example(sh: &Shell, ctx: &BuildContext, args: &RunExampleServeArgs) -> Result<()> {
    match &args.target {
        RunExampleServeTarget::Browser(args) => serve_browser_example(sh, ctx, args),
        RunExampleServeTarget::GatewayLocal(args) => serve_local_gateway_example(sh, ctx, args),
        RunExampleServeTarget::GatewayOpenAi(args) => serve_openai_gateway_example(sh, ctx, args),
    }
}

fn serve_browser_example(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunBrowserExampleServeArgs,
) -> Result<()> {
    let example = ExampleName::Browser;
    output::phase(&format!("Serve example: {}", example.label()));
    output::detail("Mode", args.mode.as_str());
    output::path("Example workspace", &example_dir(ctx, example));

    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &example_dir(ctx, example))?;
        targets::wasm::build(sh, ctx, WasmThreading::All)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_example_only(sh, ctx, example)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    serve_vite_workspace(
        sh,
        &example_dir(ctx, example),
        args.mode,
        args.host.as_deref(),
        args.port,
        format!(
            "Starting {} Vite server for {} example",
            args.mode.as_str(),
            example.label()
        ),
        format!("{} example server failed", example.label()),
    )
}

fn serve_local_gateway_example(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayLocalServeArgs,
) -> Result<()> {
    output::phase("Serve local gateway example");
    output::path("Model", &args.model);
    output::detail("Bind", &args.bind);
    output::detail("Backend", args.backend.as_str());
    if !args.model.is_file() {
        anyhow::bail!(
            "gateway model file does not exist: {}",
            args.model.display()
        );
    }

    run_local_gateway_server(
        sh,
        ctx,
        &args.model,
        &args.bind,
        &args.backend,
        "local gateway example",
    )
}

fn serve_openai_gateway_example(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayOpenAiServeArgs,
) -> Result<()> {
    output::phase("Serve OpenAI gateway example");
    output::detail("Bind", &args.bind);
    output::detail("Gateway token env", &args.token_env);
    output::detail("OpenAI key env", &args.api_key_env);
    validate_secret_env(&args.token_env)?;
    validate_secret_env(&args.api_key_env)?;

    let config_path = write_openai_gateway_example_config(ctx, args)?;
    run_production_gateway_server(
        sh,
        ctx,
        &config_path,
        &Backend::Cpu,
        "OpenAI gateway example",
    )
}

fn run_gateway_example(sh: &Shell, ctx: &BuildContext, args: &RunGatewayExampleArgs) -> Result<()> {
    match &args.target {
        RunGatewayExampleTarget::Rust(args) => run_rust_gateway_example_client(sh, ctx, args),
        RunGatewayExampleTarget::Node(args) => run_node_gateway_example_client(sh, ctx, args),
        RunGatewayExampleTarget::Python(args) => run_python_gateway_example_client(sh, ctx, args),
        RunGatewayExampleTarget::Web(args) => run_web_gateway_example(sh, ctx, args),
    }
}

fn run_rust_gateway_example_client(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayExampleClientArgs,
) -> Result<()> {
    output::phase("Run Rust gateway example");
    output::detail("Case", args.common.case.as_str());
    output::detail("Gateway bind", &args.common.bind);
    output::detail("Gateway backend", args.common.backend.as_str());
    validate_gateway_example_common(&args.common)?;
    let model = resolve_gateway_example_model(sh, ctx, &args.common)?;
    output::path("Model", &model);

    let mut gateway = start_local_gateway_example(sh, ctx, &args.common, &model)?;
    wait_for_gateway_example(&args.common.bind, gateway.child_mut()?)?;

    let target = gateway_example_target(args.common.case);
    let bin = rust_gateway_example_bin(args.common.case);
    let gateway_url = gateway_url(&args.common.bind);
    let _dir = sh.push_dir(ctx.workspace_root());
    let mut client_cmd = cmd!(sh, "cargo run -p sipp-rust-examples")
        .arg("--features")
        .arg("gateway")
        .arg("--bin")
        .arg(bin)
        .arg("--")
        .arg(&model)
        .arg(target)
        .arg(&args.prompt)
        .env("SIPP_GATEWAY_URL", gateway_url)
        .env(GATEWAY_RUN_TOKEN_ENV, &args.common.token)
        .env("SIPP_MAX_TOKENS", args.max_tokens.to_string())
        .env("SIPP_TEMPERATURE", format_temperature(args.temperature));
    client_cmd = apply_toolchains(sh, ctx, client_cmd, None)?;
    output::run_long_command(format!("Running Rust gateway example: {bin}"), client_cmd)
        .with_context(|| format!("Rust gateway example failed: {bin}"))
}

fn run_node_gateway_example_client(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayExampleClientArgs,
) -> Result<()> {
    output::phase("Run Node gateway example");
    output::detail("Case", args.common.case.as_str());
    output::detail("Gateway bind", &args.common.bind);
    output::detail("Gateway backend", args.common.backend.as_str());
    validate_gateway_example_common(&args.common)?;
    let model = resolve_gateway_example_model(sh, ctx, &args.common)?;
    output::path("Model", &model);
    targets::node::build(sh, ctx, Some(&Backend::Cpu))?;

    let mut gateway = start_local_gateway_example(sh, ctx, &args.common, &model)?;
    wait_for_gateway_example(&args.common.bind, gateway.child_mut()?)?;

    let target = gateway_example_target(args.common.case);
    let script = node_gateway_example_script(args.common.case);
    let gateway_url = gateway_url(&args.common.bind);
    let node_dir = ctx.examples_root().join("node");
    let _dir = sh.push_dir(&node_dir);
    let mut client_cmd = cmd!(sh, "node")
        .arg(script)
        .arg(&model)
        .arg(target)
        .arg(&args.prompt)
        .env("SIPP_NODE_BACKEND", "cpu")
        .env("SIPP_GATEWAY_URL", gateway_url)
        .env(GATEWAY_RUN_TOKEN_ENV, &args.common.token)
        .env("SIPP_MAX_TOKENS", args.max_tokens.to_string())
        .env("SIPP_TEMPERATURE", format_temperature(args.temperature));
    client_cmd = apply_toolchains(sh, ctx, client_cmd, Some(&Backend::Cpu))?;
    output::run_long_command(
        format!("Running Node gateway example: {script}"),
        client_cmd,
    )
    .with_context(|| format!("Node gateway example failed: {script}"))
}

fn run_python_gateway_example_client(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayExampleClientArgs,
) -> Result<()> {
    output::phase("Run Python gateway example");
    output::detail("Case", args.common.case.as_str());
    output::detail("Gateway bind", &args.common.bind);
    output::detail("Gateway backend", args.common.backend.as_str());
    validate_gateway_example_common(&args.common)?;
    let model = resolve_gateway_example_model(sh, ctx, &args.common)?;
    output::path("Model", &model);
    let wheel = build_python_gateway_run_wheel(sh, ctx)?;
    let python_exe = install_python_gateway_run_venv(sh, ctx, &wheel)?;

    let mut gateway = start_local_gateway_example(sh, ctx, &args.common, &model)?;
    wait_for_gateway_example(&args.common.bind, gateway.child_mut()?)?;

    let target = gateway_example_target(args.common.case);
    let script = python_gateway_example_script(args.common.case);
    let gateway_url = gateway_url(&args.common.bind);
    let python_dir = ctx.examples_root().join("python");
    let _dir = sh.push_dir(&python_dir);
    let mut client_cmd = cmd!(sh, "{python_exe}")
        .arg(script)
        .arg(&model)
        .arg(target)
        .arg(&args.prompt)
        .env("SIPP_PYTHON_BACKEND", "cpu")
        .env("SIPP_GATEWAY_URL", gateway_url)
        .env(GATEWAY_RUN_TOKEN_ENV, &args.common.token)
        .env("SIPP_MAX_TOKENS", args.max_tokens.to_string())
        .env("SIPP_TEMPERATURE", format_temperature(args.temperature));
    client_cmd = apply_toolchains(sh, ctx, client_cmd, Some(&Backend::Cpu))?;
    output::run_long_command(
        format!("Running Python gateway example: {script}"),
        client_cmd,
    )
    .with_context(|| format!("Python gateway example failed: {script}"))
}

fn run_web_gateway_example(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunGatewayExampleWebArgs,
) -> Result<()> {
    output::phase("Run web gateway example");
    output::detail("Case", args.common.case.as_str());
    output::detail("Gateway bind", &args.common.bind);
    output::detail("Gateway backend", args.common.backend.as_str());
    output::detail("Vite host", &args.host);
    output::detail("Vite port", args.port);
    validate_gateway_example_common(&args.common)?;
    let model = resolve_gateway_example_model(sh, ctx, &args.common)?;
    output::path("Model", &model);

    let example = ExampleName::Browser;
    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &example_dir(ctx, example))?;
        targets::wasm::build(sh, ctx, WasmThreading::All)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_example_only(sh, ctx, example)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    let mut gateway = start_local_gateway_example(sh, ctx, &args.common, &model)?;
    wait_for_gateway_example(&args.common.bind, gateway.child_mut()?)?;

    output::detail("Gateway URL", gateway_url(&args.common.bind));
    output::detail("Gateway target", gateway_example_target(args.common.case));
    output::detail("Gateway token", gateway_token_display(&args.common.token));
    output::detail("Open page", gateway_example_page(args.common.case));

    serve_vite_workspace(
        sh,
        &example_dir(ctx, example),
        args.mode,
        Some(&args.host),
        Some(args.port),
        format!(
            "Starting {} Vite server for gateway web example",
            args.mode.as_str()
        ),
        "gateway web example server failed".to_owned(),
    )
}

fn build_example_only(sh: &Shell, ctx: &BuildContext, example: ExampleName) -> Result<()> {
    let example_dir = example_dir(ctx, example);
    output::phase(&format!("Example build: {}", example.label()));
    output::path("Example workspace", &example_dir);
    output::path(
        "Artifact directory",
        &ctx.example_artifacts_dir(example.dir_name()),
    );

    let _dir = sh.push_dir(&example_dir);
    output::run_build_command(
        format!("Building {} example", example.label()),
        cmd!(sh, "bun run build"),
    )
    .with_context(|| format!("failed to build {} example", example.label()))
}

fn write_openai_gateway_example_config(
    ctx: &BuildContext,
    args: &RunGatewayOpenAiServeArgs,
) -> Result<PathBuf> {
    let config_dir = ctx.tmp_dir().join("examples").join("gateway");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("failed to create {}", config_dir.display()))?;
    let config_path = config_dir.join("openai-gateway.toml");
    let contents = format!(
        r#"public_bind = {bind}
management_bind = "127.0.0.1:9090"
max_request_bytes = 1048576
drain_timeout_seconds = 120
force_close_timeout_seconds = 5
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

[[tokens]]
env = {token_env}
caller = "xtask-openai"

[[aliases]]
name = "openai-chat"
type = "openai"
model = {chat_model}
api_key_env = {api_key_env}

[aliases.limits]
[aliases.limits.global]
max_concurrent_requests = 8
max_requests_per_minute = 120

[aliases.operations]
query = true
chat = true
embed = false

[[aliases]]
name = "openai-embed"
type = "openai"
model = {embed_model}
api_key_env = {api_key_env}

[aliases.limits]
[aliases.limits.global]
max_concurrent_requests = 8
max_requests_per_minute = 120

[aliases.operations]
query = false
chat = false
embed = true
"#,
        bind = toml_string(&args.bind),
        token_env = toml_string(&args.token_env),
        chat_model = toml_string(&args.chat_model),
        embed_model = toml_string(&args.embed_model),
        api_key_env = toml_string(&args.api_key_env),
    );
    fs::write(&config_path, contents)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    output::path("Generated gateway config", &config_path);
    Ok(config_path)
}

fn run_production_gateway_server(
    sh: &Shell,
    ctx: &BuildContext,
    config_path: &Path,
    backend: &Backend,
    label: &'static str,
) -> Result<()> {
    if matches!(backend, Backend::All) {
        anyhow::bail!(
            "gateway examples require a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    let mut gateway_cmd = cmd!(sh, "cargo run -p sipp-gateway-server -- serve");
    if *backend != Backend::Cpu {
        gateway_cmd = gateway_cmd.arg("--features").arg(backend.as_str());
    }
    gateway_cmd = gateway_cmd.arg("--config").arg(config_path);
    gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(backend))?;
    output::run_long_command(format!("Starting {label}"), gateway_cmd)
        .with_context(|| format!("{label} failed"))
}

fn run_local_gateway_server(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    bind: &str,
    backend: &Backend,
    label: &'static str,
) -> Result<()> {
    validate_gateway_example_backend(backend)?;
    let _dir = sh.push_dir(ctx.workspace_root());
    let mut gateway_cmd = cmd!(sh, "cargo run -p sipp-gateway-example");
    if *backend != Backend::Cpu {
        gateway_cmd = gateway_cmd.arg("--features").arg(backend.as_str());
    }
    gateway_cmd = gateway_cmd
        .arg("--")
        .arg("--model")
        .arg(model)
        .arg("--bind")
        .arg(bind);
    gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(backend))?;
    output::run_long_command(format!("Starting {label}"), gateway_cmd)
        .with_context(|| format!("{label} failed"))
}

fn start_local_gateway_example(
    sh: &Shell,
    ctx: &BuildContext,
    common: &RunGatewayExampleCommonArgs,
    model: &Path,
) -> Result<ManagedGatewayProcess> {
    validate_gateway_example_common(common)?;
    ManagedGatewayProcess::start(sh, ctx, model, &common.bind, &common.backend)
}

pub(crate) fn validate_gateway_example_backend(backend: &Backend) -> Result<()> {
    if matches!(backend, Backend::All) {
        anyhow::bail!(
            "gateway examples require a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }
    Ok(())
}

fn validate_gateway_example_common(common: &RunGatewayExampleCommonArgs) -> Result<()> {
    validate_gateway_example_backend(&common.backend)?;
    if common.token.trim().is_empty() {
        anyhow::bail!("gateway token must not be empty");
    }
    Ok(())
}

fn resolve_gateway_example_model(
    sh: &Shell,
    ctx: &BuildContext,
    common: &RunGatewayExampleCommonArgs,
) -> Result<PathBuf> {
    if let Some(model) = &common.model {
        if !model.is_file() {
            anyhow::bail!("gateway model file does not exist: {}", model.display());
        }
        return Ok(model.to_path_buf());
    }

    sample_model::ensure_sample_model(
        sh,
        ctx,
        SampleModelOptions {
            allow_download: false,
        },
    )
    .with_context(|| {
        format!(
            "failed to use cached sample model at {}; pass --model <model.gguf> or run setup to install it",
            sample_model::sample_model_path(ctx).display()
        )
    })
}

fn wait_for_gateway_example(bind: &str, child: &mut Child) -> Result<()> {
    output::phase("Waiting for local gateway");
    let probe_addr = gateway_client_addr(bind);
    let started_at = Instant::now();
    while started_at.elapsed() < GATEWAY_RUN_START_TIMEOUT {
        if let Some(status) = child.try_wait().context("failed to poll gateway process")? {
            anyhow::bail!("gateway process exited before readiness: {status}");
        }
        if TcpStream::connect(&probe_addr).is_ok() {
            output::success(format!("Gateway is ready at {}", gateway_url(bind)));
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    anyhow::bail!(
        "gateway did not answer readiness probe at {} within {} seconds",
        gateway_url(bind),
        GATEWAY_RUN_START_TIMEOUT.as_secs()
    )
}

struct ManagedGatewayProcess {
    child: Option<Child>,
}

impl ManagedGatewayProcess {
    fn start(
        sh: &Shell,
        ctx: &BuildContext,
        model: &Path,
        bind: &str,
        backend: &Backend,
    ) -> Result<Self> {
        let log_dir = ctx.command_logs_dir();
        fs::create_dir_all(&log_dir)
            .with_context(|| format!("failed to create {}", log_dir.display()))?;
        let log_path = log_dir.join("example-gateway-run.log");
        let log = fs::File::create(&log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        output::path("Gateway log", &log_path);

        let _dir = sh.push_dir(ctx.workspace_root());
        let mut gateway_cmd = cmd!(sh, "cargo run -p sipp-gateway-example");
        if *backend != Backend::Cpu {
            gateway_cmd = gateway_cmd.arg("--features").arg(backend.as_str());
        }
        gateway_cmd = gateway_cmd
            .arg("--")
            .arg("--model")
            .arg(model)
            .arg("--bind")
            .arg(bind);
        gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(backend))?;

        let mut command: Command = gateway_cmd.quiet().into();
        command
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                log.try_clone()
                    .context("failed to clone gateway run log handle")?,
            ))
            .stderr(Stdio::from(log));

        let child = command.spawn().context("failed to start gateway process")?;
        Ok(Self { child: Some(child) })
    }

    fn child_mut(&mut self) -> Result<&mut Child> {
        self.child
            .as_mut()
            .context("gateway child is not available")
    }
}

impl Drop for ManagedGatewayProcess {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn validate_secret_env(name: &str) -> Result<()> {
    read_secret_env(name).map(|_| ())
}

fn read_secret_env(name: &str) -> Result<String> {
    let value = env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(value)
}

fn toml_string(value: &str) -> String {
    serde_json::to_string(value).expect("string serialization cannot fail")
}

fn url_host(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_owned()
    }
}

pub(crate) fn gateway_example_target(case: RunGatewayExampleCase) -> &'static str {
    let _ = case;
    "local"
}

pub(crate) fn rust_gateway_example_bin(case: RunGatewayExampleCase) -> &'static str {
    match case {
        RunGatewayExampleCase::Query => "gateway_query",
        RunGatewayExampleCase::Chat => "gateway_chat",
        RunGatewayExampleCase::Embed => "gateway_embed",
    }
}

pub(crate) fn node_gateway_example_script(case: RunGatewayExampleCase) -> &'static str {
    match case {
        RunGatewayExampleCase::Query => "gateway_query.mjs",
        RunGatewayExampleCase::Chat => "gateway_chat.mjs",
        RunGatewayExampleCase::Embed => "gateway_embed.mjs",
    }
}

pub(crate) fn python_gateway_example_script(case: RunGatewayExampleCase) -> &'static str {
    match case {
        RunGatewayExampleCase::Query => "gateway_query.py",
        RunGatewayExampleCase::Chat => "gateway_chat.py",
        RunGatewayExampleCase::Embed => "gateway_embed.py",
    }
}

fn gateway_example_page(case: RunGatewayExampleCase) -> &'static str {
    match case {
        RunGatewayExampleCase::Query => "/gateway_query.html",
        RunGatewayExampleCase::Chat => "/gateway_chat.html",
        RunGatewayExampleCase::Embed => "/gateway_embed.html",
    }
}

fn gateway_token_display(token: &str) -> &'static str {
    if token == "dev-token" {
        "dev-token"
    } else {
        "<redacted; use the value passed to --token>"
    }
}

fn gateway_url(bind: &str) -> String {
    format!("http://{}", gateway_client_addr(bind))
}

fn gateway_client_addr(bind: &str) -> String {
    match bind.parse::<SocketAddr>() {
        Ok(addr) => {
            let host = if addr.ip().is_unspecified() {
                "127.0.0.1".to_owned()
            } else {
                addr.ip().to_string()
            };
            format!("{}:{}", url_host(&host), addr.port())
        }
        Err(_) => bind.to_owned(),
    }
}

fn build_python_gateway_run_wheel(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let backend = Backend::Cpu;
    let uv_exe = setup_uv(sh, ctx)?;
    output::run_build_command(
        "Ensuring Python 3.12 is available through uv",
        apply_uv_env(ctx, cmd!(sh, "{uv_exe} python install 3.12")),
    )?;

    let python_dir = ctx.python_package_project_dir();
    let dist_dir = ctx.python_artifacts_dir().join("gateway-run-wheels");
    prepare_python_gateway_run_wheel_dir(sh, &dist_dir)?;
    let target_dir = ctx
        .cargo_python_target_dir(Some(&backend))
        .join("gateway-run-wheel");
    sh.create_dir(&target_dir)?;

    let _dir = sh.push_dir(&python_dir);
    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(
            sh,
            "{uv_exe} tool run --python 3.12 maturin build --release --out {dist_dir}"
        ),
    )
    .env("CARGO_TARGET_DIR", &target_dir);
    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(&backend))?;
    output::run_build_command("Building Python gateway run wheel", maturin_cmd)?;
    find_python_wheel_artifact(&dist_dir)?.with_context(|| {
        format!(
            "maturin did not produce a wheel artifact in {}",
            dist_dir.display()
        )
    })
}

fn install_python_gateway_run_venv(
    sh: &Shell,
    ctx: &BuildContext,
    wheel: &Path,
) -> Result<PathBuf> {
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx.tmp_dir().join("python-gateway-run");
    output::run_build_command(
        "Creating Python gateway run virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_build_command(
        "Installing Python gateway run wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel}"
            ),
        ),
    )?;
    Ok(python_exe)
}

fn prepare_python_gateway_run_wheel_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;
    for entry in
        fs::read_dir(dist_dir).with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            sh.remove_path(path)?;
        }
    }
    Ok(())
}

fn find_python_wheel_artifact(dir: &Path) -> Result<Option<PathBuf>> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn python_venv_exe(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn format_temperature(temperature: f32) -> String {
    if temperature.fract() == 0.0 {
        format!("{temperature:.0}")
    } else {
        temperature.to_string()
    }
}

fn serve_vite_workspace(
    sh: &Shell,
    workspace: &Path,
    mode: DemoServeMode,
    host: Option<&str>,
    port: Option<u16>,
    label: String,
    error_context: String,
) -> Result<()> {
    let _dir = sh.push_dir(workspace);
    let mut serve_cmd = match mode {
        DemoServeMode::Dev => cmd!(sh, "bunx --bun vite"),
        DemoServeMode::Preview => cmd!(sh, "bunx --bun vite preview"),
    };

    if let Some(host) = host {
        serve_cmd = serve_cmd.arg("--host").arg(host);
    }
    if let Some(port) = port {
        serve_cmd = serve_cmd.arg("--port").arg(port.to_string());
    }

    output::run_long_command(label, serve_cmd).with_context(|| error_context)
}

pub(crate) fn run_llama_backend_ops(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunLlamaBackendOpsArgs,
) -> Result<()> {
    output::phase("llama.cpp backend operations");
    output::detail("Backend", args.backend.as_str());
    output::detail("Mode", args.mode.as_str());
    output::path("Source", &ctx.llama_cpp_dir());

    if matches!(args.backend, Backend::All) {
        let mut ran = Vec::new();
        for backend in host_binding_backends() {
            let optional = *backend != Backend::Cpu;
            match run_llama_backend_ops_for_backend(sh, ctx, backend, args) {
                Ok(()) => ran.push(*backend),
                Err(error) if optional => {
                    output::warning(format!(
                        "Skipped optional llama.cpp {} backend ops: {error:#}",
                        backend.as_str()
                    ));
                }
                Err(error) => return Err(error),
            }
        }
        output::detail("llama.cpp backend ops", output::backend_list(&ran));
        return Ok(());
    }

    run_llama_backend_ops_for_backend(sh, ctx, &args.backend, args)
}

fn run_llama_backend_ops_for_backend(
    sh: &Shell,
    ctx: &BuildContext,
    backend: &Backend,
    args: &RunLlamaBackendOpsArgs,
) -> Result<()> {
    let source_dir = ctx.llama_cpp_dir();
    let build_dir = ctx.cmake_llama_build_dir(backend);
    output::phase(&format!("llama.cpp backend: {}", backend.as_str()));
    output::path("CMake build directory", &build_dir);

    let mut configure_cmd = cmd!(
        sh,
        "cmake -S {source_dir} -B {build_dir} -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF"
    );
    configure_cmd = configure_cmd.arg("-DGGML_CUDA=OFF");
    configure_cmd = configure_cmd.arg("-DGGML_METAL=OFF");
    configure_cmd = configure_cmd.arg("-DGGML_VULKAN=OFF");
    configure_cmd = match *backend {
        Backend::Cpu => configure_cmd,
        Backend::Cuda => configure_cmd.arg("-DGGML_CUDA=ON"),
        Backend::Metal => configure_cmd.arg("-DGGML_METAL=ON"),
        Backend::Vulkan => configure_cmd.arg("-DGGML_VULKAN=ON"),
        Backend::All => {
            anyhow::bail!("Backend::All cannot be configured as a single llama.cpp build")
        }
    };
    configure_cmd = apply_toolchains(sh, ctx, configure_cmd, Some(backend))?;
    output::run_build_command(
        format!("Configuring llama.cpp {}", backend.as_str()),
        configure_cmd,
    )?;

    let target = LLAMA_BACKEND_OPS_TARGET;
    let mut build_cmd = cmd!(
        sh,
        "cmake --build {build_dir} --target {target} --parallel --config Release"
    );
    build_cmd = apply_toolchains(sh, ctx, build_cmd, Some(backend))?;
    output::run_build_command(
        format!("Building llama.cpp {}", LLAMA_BACKEND_OPS_TARGET),
        build_cmd,
    )?;

    let test_exe = find_llama_backend_ops_exe(&build_dir)?;
    output::path("Backend ops executable", &test_exe);

    let mut test_cmd = cmd!(sh, "{test_exe}");
    test_cmd = test_cmd.arg(args.mode.as_str());
    if *backend == Backend::Cpu {
        test_cmd = test_cmd.arg("-b").arg("CPU");
    }
    if let Some(op) = &args.op {
        test_cmd = test_cmd.arg("-o").arg(op);
    }
    if let Some(params) = &args.params {
        test_cmd = test_cmd.arg("-p").arg(params);
    }
    test_cmd = test_cmd.arg("--output").arg(args.output.as_str());
    test_cmd = apply_toolchains(sh, ctx, test_cmd, Some(backend))?;

    output::run_test_command(
        format!("Running llama.cpp {} backend ops", backend.as_str()),
        test_cmd,
    )
}

fn ensure_javascript_workspace_dependencies(
    sh: &Shell,
    ctx: &BuildContext,
    package_dir: &Path,
) -> Result<()> {
    javascript::install_root_workspace_dependencies(
        sh,
        ctx,
        "Installing JavaScript workspace dependencies",
        &[package_dir.to_path_buf()],
    )
}

fn example_dir(ctx: &BuildContext, example: ExampleName) -> PathBuf {
    match example {
        ExampleName::Browser => ctx.browser_example_dir(),
    }
}

fn tool_dir(ctx: &BuildContext, tool: ToolName) -> PathBuf {
    match tool {
        ToolName::Playground => ctx.playground_dir(),
    }
}

fn host_binding_backends() -> &'static [Backend] {
    if cfg!(target_os = "macos") {
        &[Backend::Cpu, Backend::Metal]
    } else {
        &[Backend::Cpu, Backend::Vulkan, Backend::Cuda]
    }
}

fn find_llama_backend_ops_exe(build_dir: &Path) -> Result<PathBuf> {
    let exe_name = if cfg!(windows) {
        format!("{LLAMA_BACKEND_OPS_TARGET}.exe")
    } else {
        LLAMA_BACKEND_OPS_TARGET.to_owned()
    };
    let candidates = [
        build_dir.join("bin").join(&exe_name),
        build_dir.join("bin").join("Release").join(&exe_name),
        build_dir.join("tests").join(&exe_name),
    ];

    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    find_file_recursive(build_dir, &exe_name)?.with_context(|| {
        format!(
            "failed to find {exe_name} under {} after build",
            build_dir.display()
        )
    })
}

fn find_file_recursive(root: &Path, file_name: &str) -> Result<Option<PathBuf>> {
    if !root.exists() {
        return Ok(None);
    }

    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.is_file() && path.file_name().and_then(|name| name.to_str()) == Some(file_name) {
            return Ok(Some(path));
        }
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, file_name)? {
                return Ok(Some(found));
            }
        }
    }

    Ok(None)
}
