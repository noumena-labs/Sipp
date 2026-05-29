//! Developer run workflows for apps, bindings, and standalone llama.cpp checks.

use crate::cli::{
    AppName, AppServeMode, Backend, LlamaBackendOpsMode, LlamaBackendOpsOutput, RunAllArgs,
    RunAppServeArgs, RunAppsCommands, RunBindingSmokeArgs, RunBindingsCommands, RunBrowserArgs,
    RunCommands, RunLlamaBackendOpsArgs, RunLlamaCommands,
};
use crate::output;
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::setup_uv;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

const NODE_SMOKE_SCRIPT: &str = "examples/node_smoke.mjs";
const PYTHON_SMOKE_SCRIPT: &str = "examples/python_smoke.py";
const APP_TEST_SUFFIX: &str = ".test.ts";
const SKIPPED_APP_TEST_DIRS: &[&str] = &[
    "node_modules",
    "dist",
    "build",
    "out",
    ".vite",
    ".turbo",
    "coverage",
];
const BROWSER_PACKAGE_TEST_DIR: &str = "packages/npm/src";
const LLAMA_BACKEND_OPS_TARGET: &str = "test-backend-ops";

/// Runs a developer workflow.
pub fn run(sh: &Shell, ctx: &BuildContext, command: RunCommands) -> Result<()> {
    match command {
        RunCommands::All(args) => run_all(sh, ctx, &args),
        RunCommands::Apps { command } => run_apps(sh, ctx, command),
        RunCommands::Bindings { command } => run_bindings(sh, ctx, command),
        RunCommands::Llama { command } => run_llama(sh, ctx, command),
    }
}

fn run_all(sh: &Shell, ctx: &BuildContext, args: &RunAllArgs) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Run all finite developer workflows");
    output::path("Workspace", ctx.workspace_root());
    output::path("Model", &args.model);
    output::detail("Backend", args.backend.as_str());

    build_all_apps(sh, ctx)?;
    output::phase("App TypeScript tests");
    run_app_tests_only(sh, ctx)?;
    run_binding_browser_inner(sh, ctx, &RunBrowserArgs { ingest: false }, false)?;
    run_node_smokes(sh, ctx, &binding_options_from_all(args))?;
    run_python_smokes(sh, ctx, &binding_options_from_all(args))?;
    run_llama_backend_ops(
        sh,
        ctx,
        &RunLlamaBackendOpsArgs {
            backend: args.backend,
            mode: LlamaBackendOpsMode::Test,
            op: None,
            params: None,
            output: LlamaBackendOpsOutput::Console,
        },
    )?;

    output::success(format!(
        "All run workflows complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    Ok(())
}

fn run_apps(sh: &Shell, ctx: &BuildContext, command: RunAppsCommands) -> Result<()> {
    match command {
        RunAppsCommands::Build(args) => build_one_app(sh, ctx, args.app),
        RunAppsCommands::Test => run_app_tests(sh, ctx),
        RunAppsCommands::Serve(args) => serve_app(sh, ctx, &args),
    }
}

fn run_bindings(sh: &Shell, ctx: &BuildContext, command: RunBindingsCommands) -> Result<()> {
    match command {
        RunBindingsCommands::All(args) => {
            run_binding_browser(sh, ctx, &RunBrowserArgs { ingest: false })?;
            run_node_smokes(sh, ctx, &binding_options_from_smoke(&args))?;
            run_python_smokes(sh, ctx, &binding_options_from_smoke(&args))
        }
        RunBindingsCommands::Browser(args) => run_binding_browser(sh, ctx, &args),
        RunBindingsCommands::Node(args) => {
            run_node_smokes(sh, ctx, &binding_options_from_smoke(&args))
        }
        RunBindingsCommands::Python(args) => {
            run_python_smokes(sh, ctx, &binding_options_from_smoke(&args))
        }
    }
}

fn run_llama(sh: &Shell, ctx: &BuildContext, command: RunLlamaCommands) -> Result<()> {
    match command {
        RunLlamaCommands::BackendOps(args) => run_llama_backend_ops(sh, ctx, &args),
    }
}

fn build_one_app(sh: &Shell, ctx: &BuildContext, app: AppName) -> Result<()> {
    output::phase(&format!("Build browser app: {}", app.slug()));
    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;
    build_app_only(sh, ctx, app)
}

fn build_all_apps(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Build browser apps");
    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;

    for app in AppName::all() {
        build_app_only(sh, ctx, *app)?;
    }

    Ok(())
}

fn build_app_only(sh: &Shell, ctx: &BuildContext, app: AppName) -> Result<()> {
    let app_dir = ctx.app_dir(app.slug());
    output::phase(&format!("App build: {}", app.slug()));
    output::path("App workspace", &app_dir);
    output::path("Artifact directory", &ctx.app_artifacts_dir(app.slug()));

    let _dir = sh.push_dir(&app_dir);
    output::run_command(
        format!("Building {} app", app.slug()),
        cmd!(sh, "bun run build"),
    )
    .with_context(|| format!("failed to build {} app", app.slug()))
}

fn run_app_tests(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("App TypeScript tests");
    output::path("Workspace", ctx.workspace_root());

    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;
    run_app_tests_only(sh, ctx)
}

fn run_app_tests_only(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let tests = app_test_files(ctx)?;
    if tests.is_empty() {
        output::warning("No app TypeScript tests were found under apps/");
        return Ok(());
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    output::detail("Test files", tests.len());
    let mut test_cmd = cmd!(sh, "bun test");
    for test in tests {
        test_cmd = test_cmd.arg(test);
    }

    output::run_command(
        "Running app TypeScript tests through Bun",
        test_cmd,
    )
}

fn serve_app(sh: &Shell, ctx: &BuildContext, args: &RunAppServeArgs) -> Result<()> {
    output::phase(&format!("Serve browser app: {}", args.app.slug()));
    output::detail("Mode", args.mode.as_str());
    output::path("App workspace", &ctx.app_dir(args.app.slug()));

    if !args.no_build {
        ensure_workspace_bun_install(sh, ctx)?;
        targets::wasm::build(sh, ctx)?;
        if matches!(args.mode, AppServeMode::Preview) {
            build_app_only(sh, ctx, args.app)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    let app_dir = ctx.app_dir(args.app.slug());
    let _dir = sh.push_dir(&app_dir);
    let mut serve_cmd = match args.mode {
        AppServeMode::Dev => cmd!(sh, "bunx --bun vite"),
        AppServeMode::Preview => cmd!(sh, "bunx --bun vite preview"),
    };

    if let Some(host) = &args.host {
        serve_cmd = serve_cmd.arg("--host").arg(host);
    }
    if let Some(port) = args.port {
        serve_cmd = serve_cmd.arg("--port").arg(port.to_string());
    }

    output::step(format!(
        "Starting {} Vite server for {}",
        args.mode.as_str(),
        args.app.slug()
    ));
    serve_cmd
        .run()
        .with_context(|| format!("{} app server failed", args.app.slug()))
}

fn run_binding_browser(sh: &Shell, ctx: &BuildContext, args: &RunBrowserArgs) -> Result<()> {
    run_binding_browser_inner(sh, ctx, args, true)
}

fn run_binding_browser_inner(
    sh: &Shell,
    ctx: &BuildContext,
    args: &RunBrowserArgs,
    build_browser_package: bool,
) -> Result<()> {
    output::phase("Browser/WASM binding checks");
    if build_browser_package {
        targets::wasm::build(sh, ctx)?;
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    let browser_package_test_dir = BROWSER_PACKAGE_TEST_DIR;
    output::run_command(
        "Running browser package TypeScript tests",
        cmd!(sh, "bun test {browser_package_test_dir}"),
    )?;

    run_benchmark_browser_smoke(sh, ctx, args.ingest)
}

fn run_benchmark_browser_smoke(sh: &Shell, ctx: &BuildContext, require_ingest: bool) -> Result<()> {
    output::phase("Browser runtime smoke");
    let benchmark_dir = ctx.app_dir("benchmark");
    output::path("Benchmark app", &benchmark_dir);

    let _dir = sh.push_dir(&benchmark_dir);
    let script = if require_ingest {
        "browser:smoke:ingest"
    } else {
        "browser:smoke"
    };

    output::run_command(
        format!("Running benchmark {script}"),
        cmd!(sh, "bun run {script}"),
    )
}

fn run_node_smokes(sh: &Shell, ctx: &BuildContext, options: &BindingSmokeOptions<'_>) -> Result<()> {
    output::phase("Node.js binding smoke");
    output::path("Model", options.model);
    output::detail("Backend", options.backend.as_str());

    if *options.backend == Backend::All {
        targets::node::build(sh, ctx, Some(&Backend::All))?;
        run_node_smokes_for_built_backends(sh, ctx, options)
    } else {
        targets::node::build(sh, ctx, Some(options.backend))?;
        run_node_smoke(sh, ctx, options.backend, options)
    }
}

fn run_node_smokes_for_built_backends(
    sh: &Shell,
    ctx: &BuildContext,
    options: &BindingSmokeOptions<'_>,
) -> Result<()> {
    let mut ran = Vec::new();
    for backend in host_binding_backends() {
        if node_backend_artifact_exists(ctx, backend)? {
            run_node_smoke(sh, ctx, backend, options)?;
            ran.push(*backend);
        } else {
            output::warning(format!(
                "Skipping Node {} smoke; no built artifact was found",
                backend.as_str()
            ));
        }
    }

    if ran.is_empty() {
        anyhow::bail!("no Node backend artifacts were available for smoke tests");
    }

    output::detail("Node smoke backends", output::backend_list(&ran));
    Ok(())
}

fn run_node_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    backend: &Backend,
    options: &BindingSmokeOptions<'_>,
) -> Result<()> {
    let backend_value = backend.as_str();
    let model = options.model;
    let prompt = options.prompt;
    let smoke_script = NODE_SMOKE_SCRIPT;
    let node_dir = ctx.bindings_node_dir();
    let _dir = sh.push_dir(&node_dir);

    let mut smoke_cmd = cmd!(
        sh,
        "node {smoke_script} {model} {prompt} --backend {backend_value}"
    )
    .env("COGENTLM_NODE_BACKEND", backend_value);

    if let Some(gpu_layers) = options.gpu_layers {
        smoke_cmd = smoke_cmd.arg("--gpu-layers").arg(gpu_layers.to_string());
    }

    output::run_command(
        format!("Running Node {backend_value} smoke"),
        smoke_cmd,
    )
    .with_context(|| format!("Node {backend_value} smoke failed"))
}

fn run_python_smokes(
    sh: &Shell,
    ctx: &BuildContext,
    options: &BindingSmokeOptions<'_>,
) -> Result<()> {
    output::phase("Python binding smoke");
    output::path("Model", options.model);
    output::detail("Backend", options.backend.as_str());

    if *options.backend == Backend::All {
        targets::python::build(sh, ctx, Some(&Backend::All))?;
        run_python_smokes_for_built_backends(sh, ctx, options)
    } else {
        targets::python::build(sh, ctx, Some(options.backend))?;
        run_python_smoke(sh, ctx, options.backend, options)
    }
}

fn run_python_smokes_for_built_backends(
    sh: &Shell,
    ctx: &BuildContext,
    options: &BindingSmokeOptions<'_>,
) -> Result<()> {
    let mut ran = Vec::new();
    for backend in host_binding_backends() {
        if python_backend_artifact_exists(ctx, backend)? {
            run_python_smoke(sh, ctx, backend, options)?;
            ran.push(*backend);
        } else {
            output::warning(format!(
                "Skipping Python {} smoke; no built artifact was found",
                backend.as_str()
            ));
        }
    }

    if ran.is_empty() {
        anyhow::bail!("no Python backend artifacts were available for smoke tests");
    }

    output::detail("Python smoke backends", output::backend_list(&ran));
    Ok(())
}

fn run_python_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    backend: &Backend,
    options: &BindingSmokeOptions<'_>,
) -> Result<()> {
    let backend_value = backend.as_str();
    let model = options.model;
    let prompt = options.prompt;
    let smoke_script = PYTHON_SMOKE_SCRIPT;
    let python_dir = ctx.bindings_python_dir();
    let python_source_dir = python_dir.join("python");
    let uv_exe = setup_uv(sh, ctx)?;
    let _dir = sh.push_dir(&python_dir);

    let mut smoke_cmd = cmd!(
        sh,
        "{uv_exe} run python {smoke_script} {model} {prompt} --backend {backend_value}"
    )
    .env("COGENTLM_PYTHON_BACKEND", backend_value)
    .env("PYTHONPATH", python_source_dir);

    if let Some(gpu_layers) = options.gpu_layers {
        smoke_cmd = smoke_cmd.arg("--gpu-layers").arg(gpu_layers.to_string());
    }

    output::run_command(
        format!("Running Python {backend_value} smoke"),
        smoke_cmd,
    )
    .with_context(|| format!("Python {backend_value} smoke failed"))
}

fn run_llama_backend_ops(
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
    output::run_command(
        format!("Configuring llama.cpp {}", backend.as_str()),
        configure_cmd,
    )?;

    let target = LLAMA_BACKEND_OPS_TARGET;
    let mut build_cmd = cmd!(
        sh,
        "cmake --build {build_dir} --target {target} --parallel --config Release"
    );
    build_cmd = apply_toolchains(sh, ctx, build_cmd, Some(backend))?;
    output::run_command(
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

    output::run_command(
        format!("Running llama.cpp {} backend ops", backend.as_str()),
        test_cmd,
    )
}

fn ensure_workspace_bun_install(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Installing workspace Bun dependencies",
        cmd!(sh, "bun install"),
    )
}

fn host_binding_backends() -> &'static [Backend] {
    if cfg!(target_os = "macos") {
        &[Backend::Cpu, Backend::Metal]
    } else {
        &[Backend::Cpu, Backend::Vulkan, Backend::Cuda]
    }
}

fn app_test_files(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut tests = Vec::new();
    collect_app_test_files(&ctx.apps_root(), &mut tests)?;
    tests.sort();
    Ok(tests)
}

fn collect_app_test_files(root: &Path, tests: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }

    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if path.is_dir() {
            if SKIPPED_APP_TEST_DIRS.contains(&file_name) {
                continue;
            }
            collect_app_test_files(&path, tests)?;
            continue;
        }

        if path.is_file() && file_name.ends_with(APP_TEST_SUFFIX) {
            tests.push(path);
        }
    }

    Ok(())
}

fn node_backend_artifact_exists(ctx: &BuildContext, backend: &Backend) -> Result<bool> {
    let artifact_dir = ctx.node_artifacts_dir();
    if !artifact_dir.exists() {
        return Ok(false);
    }

    let prefix = format!("cogentlm_node_{}.", backend.as_str());
    for entry in std::fs::read_dir(&artifact_dir)
        .with_context(|| format!("failed to read {}", artifact_dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&prefix)
            && path.extension().and_then(|ext| ext.to_str()) == Some("node")
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn python_backend_artifact_exists(ctx: &BuildContext, backend: &Backend) -> Result<bool> {
    let binary_dir = ctx.bindings_python_binary_dir();
    if binary_dir.exists() && contains_python_backend_binary(&binary_dir, backend)? {
        return Ok(true);
    }

    if *backend == Backend::Cpu {
        let package_dir = ctx.bindings_python_package_dir();
        return Ok(contains_direct_python_extension(&package_dir)?);
    }

    Ok(false)
}

fn contains_python_backend_binary(dir: &Path, backend: &Backend) -> Result<bool> {
    let prefix = format!("_native_{}", backend.as_str());
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&prefix) && is_python_extension(&path) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn contains_direct_python_extension(dir: &Path) -> Result<bool> {
    if !dir.exists() {
        return Ok(false);
    }

    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with("_native") && is_python_extension(&path) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_python_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("pyd" | "so")
    )
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

struct BindingSmokeOptions<'a> {
    model: &'a Path,
    backend: &'a Backend,
    prompt: &'a str,
    gpu_layers: Option<u32>,
}

fn binding_options_from_smoke(args: &RunBindingSmokeArgs) -> BindingSmokeOptions<'_> {
    BindingSmokeOptions {
        model: &args.model,
        backend: &args.backend,
        prompt: &args.prompt,
        gpu_layers: args.gpu_layers,
    }
}

fn binding_options_from_all(args: &RunAllArgs) -> BindingSmokeOptions<'_> {
    BindingSmokeOptions {
        model: &args.model,
        backend: &args.backend,
        prompt: &args.prompt,
        gpu_layers: args.gpu_layers,
    }
}
