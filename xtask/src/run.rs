//! Developer run workflows for long-lived demos and non-test diagnostics.

use crate::cli::{
    Backend, BenchmarkName, DemoName, DemoServeMode, ExampleName, LlamaBackendOpsMode,
    RunBenchmarkServeArgs, RunBenchmarksCommands, RunCommands, RunDemoServeArgs, RunDemosCommands,
    RunExampleServeArgs, RunExamplesCommands, RunLlamaBackendOpsArgs, RunLlamaCommands,
};
use crate::javascript;
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
        RunCommands::Demos { command } => run_demos(sh, ctx, command),
        RunCommands::Examples { command } => run_examples(sh, ctx, command),
        RunCommands::Benchmarks { command } => run_benchmarks(sh, ctx, command),
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

fn run_benchmarks(sh: &Shell, ctx: &BuildContext, command: RunBenchmarksCommands) -> Result<()> {
    match command {
        RunBenchmarksCommands::Build(args) => build_one_benchmark(sh, ctx, args.benchmark),
        RunBenchmarksCommands::Serve(args) => serve_benchmark(sh, ctx, &args),
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

fn build_one_benchmark(sh: &Shell, ctx: &BuildContext, benchmark: BenchmarkName) -> Result<()> {
    output::phase(&format!("Build benchmark: {}", benchmark.slug()));
    ensure_javascript_workspace_dependencies(sh, ctx, &benchmark_dir(ctx, benchmark))?;
    targets::wasm::build(sh, ctx)?;
    build_benchmark_only(sh, ctx, benchmark)
}

fn build_benchmark_only(sh: &Shell, ctx: &BuildContext, benchmark: BenchmarkName) -> Result<()> {
    let benchmark_dir = benchmark_dir(ctx, benchmark);
    output::phase(&format!("Benchmark build: {}", benchmark.slug()));
    output::path("Benchmark workspace", &benchmark_dir);
    output::path(
        "Artifact directory",
        &ctx.benchmark_artifacts_dir(benchmark.slug()),
    );

    let _dir = sh.push_dir(&benchmark_dir);
    output::run_build_command(
        format!("Building {} benchmark", benchmark.slug()),
        cmd!(sh, "bun run build"),
    )
    .with_context(|| format!("failed to build {} benchmark", benchmark.slug()))
}

fn serve_benchmark(sh: &Shell, ctx: &BuildContext, args: &RunBenchmarkServeArgs) -> Result<()> {
    output::phase(&format!("Serve benchmark: {}", args.benchmark.slug()));
    output::detail("Mode", args.mode.as_str());
    output::path("Benchmark workspace", &benchmark_dir(ctx, args.benchmark));

    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &benchmark_dir(ctx, args.benchmark))?;
        targets::wasm::build(sh, ctx)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_benchmark_only(sh, ctx, args.benchmark)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    serve_vite_workspace(
        sh,
        &benchmark_dir(ctx, args.benchmark),
        args.mode,
        args.host.as_deref(),
        args.port,
        format!(
            "Starting {} Vite server for {} benchmark",
            args.mode.as_str(),
            args.benchmark.slug()
        ),
        format!("{} benchmark server failed", args.benchmark.slug()),
    )
}

fn serve_example(sh: &Shell, ctx: &BuildContext, args: &RunExampleServeArgs) -> Result<()> {
    output::phase(&format!("Serve example: {}", args.example.label()));
    output::detail("Mode", args.mode.as_str());
    output::path("Example workspace", &example_dir(ctx, args.example));

    if !args.no_build {
        ensure_javascript_workspace_dependencies(sh, ctx, &example_dir(ctx, args.example))?;
        targets::wasm::build(sh, ctx)?;
        if matches!(args.mode, DemoServeMode::Preview) {
            build_example_only(sh, ctx, args.example)?;
        }
    } else {
        output::warning("Skipping browser package build before serving");
    }

    serve_vite_workspace(
        sh,
        &example_dir(ctx, args.example),
        args.mode,
        args.host.as_deref(),
        args.port,
        format!(
            "Starting {} Vite server for {} example",
            args.mode.as_str(),
            args.example.label()
        ),
        format!("{} example server failed", args.example.label()),
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

fn benchmark_dir(ctx: &BuildContext, benchmark: BenchmarkName) -> PathBuf {
    match benchmark {
        BenchmarkName::Browser => ctx.benchmark_browser_dir(),
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
