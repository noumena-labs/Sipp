//! Developer run workflows for long-lived apps and non-test diagnostics.

use crate::cli::{
    AppName, AppServeMode, Backend, LlamaBackendOpsMode, RunAppServeArgs, RunAppsCommands,
    RunCommands, RunLlamaBackendOpsArgs, RunLlamaCommands,
};
use crate::output;
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
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
        RunCommands::Apps { command } => run_apps(sh, ctx, command),
        RunCommands::Llama { command } => run_llama(sh, ctx, command),
    }
}

fn run_apps(sh: &Shell, ctx: &BuildContext, command: RunAppsCommands) -> Result<()> {
    match command {
        RunAppsCommands::Build(args) => build_one_app(sh, ctx, args.app),
        RunAppsCommands::Serve(args) => serve_app(sh, ctx, &args),
    }
}

fn run_llama(sh: &Shell, ctx: &BuildContext, command: RunLlamaCommands) -> Result<()> {
    match command {
        RunLlamaCommands::BackendOps(args) => {
            if matches!(args.mode, LlamaBackendOpsMode::Test) {
                anyhow::bail!(
                    "llama.cpp correctness checks moved to `cargo xtask test run --suite llama-backend-ops`"
                );
            }
            run_llama_backend_ops(sh, ctx, &args)
        }
    }
}

fn build_one_app(sh: &Shell, ctx: &BuildContext, app: AppName) -> Result<()> {
    output::phase(&format!("Build browser app: {}", app.slug()));
    ensure_workspace_bun_install(sh, ctx)?;
    targets::wasm::build(sh, ctx)?;
    build_app_only(sh, ctx, app)
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

    output::run_long_command(
        format!(
            "Starting {} Vite server for {}",
            args.mode.as_str(),
            args.app.slug()
        ),
        serve_cmd,
    )
    .with_context(|| format!("{} app server failed", args.app.slug()))
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
