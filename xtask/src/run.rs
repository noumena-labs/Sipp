//! Developer run workflows for long-lived demos and non-test diagnostics.

use crate::cli::{
    Backend, DemoName, DemoServeMode, ExampleName, LlamaBackendOpsMode, RunBrowserExampleServeArgs,
    RunCommands, RunDemoServeArgs, RunDemosCommands, RunExampleServeArgs, RunExampleServeTarget,
    RunExamplesCommands, RunGatewayLocalServeArgs, RunGatewayOpenAiServeArgs,
    RunLlamaBackendOpsArgs, RunLlamaCommands, RunToolServeArgs, RunToolsCommands, ToolName,
};
use crate::javascript;
use crate::output;
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

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

/// Runs a developer workflow.
pub fn run(sh: &Shell, ctx: &BuildContext, command: RunCommands) -> Result<()> {
    match command {
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

fn build_one_demo(sh: &Shell, ctx: &BuildContext, demo: DemoName) -> Result<()> {
    output::phase(&format!("Build browser demo: {}", demo.slug()));
    ensure_javascript_workspace_dependencies(sh, ctx, &ctx.demo_dir(demo.slug()))?;
    targets::wasm::build(sh, ctx)?;
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
        targets::wasm::build(sh, ctx)?;
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
    targets::wasm::build(sh, ctx)?;
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
        targets::wasm::build(sh, ctx)?;
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
        targets::wasm::build(sh, ctx)?;
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
    output::detail("Gateway token env", &args.token_env);
    output::detail("Backend", args.backend.as_str());
    validate_secret_env(&args.token_env)?;
    if !args.model.is_file() {
        anyhow::bail!(
            "gateway model file does not exist: {}",
            args.model.display()
        );
    }

    let config_path = write_local_gateway_example_config(ctx, args)?;
    run_gateway_server(
        sh,
        ctx,
        &config_path,
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
    run_gateway_server(
        sh,
        ctx,
        &config_path,
        &Backend::Cpu,
        "OpenAI gateway example",
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

fn write_local_gateway_example_config(
    ctx: &BuildContext,
    args: &RunGatewayLocalServeArgs,
) -> Result<PathBuf> {
    let config_dir = ctx.tmp_dir().join("examples").join("gateway");
    fs::create_dir_all(&config_dir)
        .with_context(|| format!("failed to create {}", config_dir.display()))?;
    let config_path = config_dir.join("local-gateway.toml");
    let contents = format!(
        r#"[server]
bind = {bind}

[auth]
token_env = {token_env}

[limits]
max_request_bytes = 1048576

[cors]
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

[[aliases]]
name = "local"
operations = ["query", "chat", "embed"]

[aliases.limits]
max_concurrent_requests = 4
max_requests_per_minute = 60

[aliases.backend]
kind = "local_cogent_engine"
model_path = {model_path}

[aliases.backend.runtime.context]
n_ctx = 2048
embeddings = true

[aliases.backend.runtime.scheduler]
continuous_batching = true
prefill_chunk_size = 0

[aliases.backend.runtime.cache]
mode = "live_slot_prefix"

[aliases.backend.runtime.observability]
runtime_metrics = true
backend_profiling = false
"#,
        bind = toml_string(&args.bind),
        token_env = toml_string(&args.token_env),
        model_path = toml_string(&args.model.display().to_string()),
    );
    fs::write(&config_path, contents)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    output::path("Generated gateway config", &config_path);
    Ok(config_path)
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
        r#"[server]
bind = {bind}

[auth]
token_env = {token_env}

[limits]
max_request_bytes = 1048576

[cors]
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

[[aliases]]
name = "openai-chat"
operations = ["query", "chat"]

[aliases.limits]
max_concurrent_requests = 8
max_requests_per_minute = 120

[aliases.backend]
kind = "open_ai"
model = {chat_model}
api_key_env = {api_key_env}

[[aliases]]
name = "openai-embed"
operations = ["embed"]

[aliases.limits]
max_concurrent_requests = 8
max_requests_per_minute = 120

[aliases.backend]
kind = "open_ai"
model = {embed_model}
api_key_env = {api_key_env}
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

fn run_gateway_server(
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
    let mut gateway_cmd = cmd!(sh, "cargo run -p cogentlm-gateway");
    if *backend != Backend::Cpu {
        gateway_cmd = gateway_cmd.arg("--features").arg(backend.as_str());
    }
    gateway_cmd = gateway_cmd
        .arg("--")
        .arg("serve")
        .arg("--config")
        .arg(config_path);
    gateway_cmd = apply_toolchains(sh, ctx, gateway_cmd, Some(backend))?;
    output::run_long_command(format!("Starting {label}"), gateway_cmd)
        .with_context(|| format!("{label} failed"))
}

fn validate_secret_env(name: &str) -> Result<()> {
    let value = env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(())
}

fn toml_string(value: &str) -> String {
    serde_json::to_string(value).expect("string serialization cannot fail")
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
