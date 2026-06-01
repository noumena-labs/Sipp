//! Workspace test orchestration.

use crate::cli::{
    Backend, TestAllArgs, TestBrowserArgs, TestCommands, TestModelSmokeArgs, TestNativeBindingArgs,
};
use crate::output;
use crate::sample_model::{self, SampleModelOptions};
use crate::targets;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::{apply_uv_env, setup_uv};
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

const RUST_GENERATION_SMOKE_EXAMPLES: &[&str] = &["query", "chat"];
const NODE_GENERATION_SMOKE_SCRIPTS: &[&str] = &["examples/query.mjs", "examples/chat.mjs"];
const PYTHON_GENERATION_SMOKE_SCRIPTS: &[&str] = &["examples/query.py", "examples/chat.py"];
const CORE_UNIT_TEST_PACKAGES: &[&str] = &[
    "cogentlm-core",
    "cogentlm-shard",
    "cogentlm-sys",
    "cogentlm-engine",
    "cogentlm-providers",
    "cogentlm-client",
];

/// Runs a workspace test workflow.
pub fn run(sh: &Shell, ctx: &BuildContext, command: TestCommands) -> Result<()> {
    match command {
        TestCommands::All(args) => run_all(sh, ctx, &args),
        TestCommands::Layout => run_layout(ctx),
        TestCommands::Core => run_core(sh, ctx),
        TestCommands::RustApi => run_rust_api(sh, ctx),
        TestCommands::Browser(args) => run_browser(sh, ctx, &args),
        TestCommands::Node(args) => run_node(sh, ctx, &args),
        TestCommands::Python(args) => run_python(sh, ctx, &args),
        TestCommands::ModelSmoke(args) => run_model_smoke(sh, ctx, &args),
    }
}

fn run_all(sh: &Shell, ctx: &BuildContext, args: &TestAllArgs) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Full test workflow");
    run_layout(ctx)?;
    run_core(sh, ctx)?;
    run_rust_api(sh, ctx)?;
    run_browser(sh, ctx, &TestBrowserArgs { no_model: false })?;
    run_node(
        sh,
        ctx,
        &TestNativeBindingArgs {
            backend: args.backend,
        },
    )?;
    run_python(
        sh,
        ctx,
        &TestNativeBindingArgs {
            backend: args.backend,
        },
    )?;
    run_model_smoke(
        sh,
        ctx,
        &TestModelSmokeArgs {
            model: args.model.clone(),
            backend: args.backend,
            prompt: args.prompt.clone(),
            gpu_layers: args.gpu_layers,
            offline: args.offline,
        },
    )?;
    output::success(format!(
        "Full test workflow complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    Ok(())
}

fn run_layout(ctx: &BuildContext) -> Result<()> {
    output::phase("Test layout");
    let mut violations = Vec::new();
    collect_package_test_layout_violations(ctx, &mut violations)?;
    collect_rust_test_layout_violations(ctx, &mut violations)?;

    if violations.is_empty() {
        output::success("Test layout check passed");
        return Ok(());
    }

    for violation in &violations {
        output::warning(violation);
    }
    anyhow::bail!(
        "test layout check failed with {} violation(s)",
        violations.len()
    )
}

fn run_core(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Rust core unit tests");
    let _dir = sh.push_dir(ctx.workspace_root());
    for package in CORE_UNIT_TEST_PACKAGES {
        let cargo_test =
            apply_toolchains(sh, ctx, cmd!(sh, "cargo test -p {package} --lib"), None)?;
        output::run_command(format!("Running {package} unit tests"), cargo_test)?;
    }
    let cli_test = apply_toolchains(
        sh,
        ctx,
        cmd!(sh, "cargo test -p cogentlm-cli --bin cogentlm"),
        None,
    )?;
    output::run_command("Running cogentlm-cli unit tests", cli_test)?;
    let xtask_test = apply_toolchains(sh, ctx, cmd!(sh, "cargo test -p xtask"), None)?;
    output::run_command("Running xtask Rust tests", xtask_test)
}

fn run_rust_api(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Rust public API integration tests");
    let _dir = sh.push_dir(ctx.workspace_root());
    for (package, test_name) in [
        ("cogentlm-client", "public_api"),
        ("cogentlm-providers", "public_api"),
        ("cogentlm-shard", "public_api"),
        ("cogentlm-cli", "cli_black_box"),
    ] {
        let cargo_test = apply_toolchains(
            sh,
            ctx,
            cmd!(sh, "cargo test -p {package} --test {test_name}"),
            None,
        )?;
        output::run_command(
            format!("Running {package} integration test {test_name}"),
            cargo_test,
        )?;
    }
    Ok(())
}

fn run_browser(sh: &Shell, ctx: &BuildContext, args: &TestBrowserArgs) -> Result<()> {
    output::phase("Browser package tests");
    targets::wasm::build(sh, ctx)?;
    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Running browser package TypeScript tests",
        cmd!(sh, "bun test packages/npm/tests"),
    )?;

    if args.no_model {
        output::detail("Browser runtime smoke", "skipped by --no-model");
        return Ok(());
    }

    ensure_workspace_bun_install(sh, ctx)?;
    let benchmark_dir = ctx.app_dir("benchmark");
    let _benchmark_dir = sh.push_dir(&benchmark_dir);
    output::run_command(
        "Running benchmark Playwright browser runtime smoke",
        cmd!(sh, "bun run browser:smoke"),
    )
}

fn run_node(sh: &Shell, ctx: &BuildContext, args: &TestNativeBindingArgs) -> Result<()> {
    output::phase("Node binding tests");
    targets::node::build(sh, ctx, Some(&args.backend))?;

    let _root = sh.push_dir(ctx.workspace_root());
    let cargo_test = apply_toolchains(
        sh,
        ctx,
        cmd!(sh, "cargo test -p cogentlm-napi"),
        Some(&args.backend),
    )?;
    output::run_command("Running Node binding Rust tests", cargo_test)?;
    drop(_root);

    let node_dir = ctx.bindings_node_dir();
    let _node = sh.push_dir(&node_dir);
    for backend in node_test_backends(ctx, &args.backend)? {
        output::run_command(
            format!("Running Node.js black-box tests ({})", backend.as_str()),
            cmd!(sh, "node --test tests/router.test.mjs")
                .env("COGENTLM_NODE_BACKEND", backend.as_str())
                .env("COGENTLM_NODE_TEST_BACKEND", backend.as_str()),
        )?;
    }
    Ok(())
}

fn run_python(sh: &Shell, ctx: &BuildContext, args: &TestNativeBindingArgs) -> Result<()> {
    if args.backend == Backend::All {
        anyhow::bail!(
            "python tests require a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    output::phase("Python binding tests");
    let wheel = build_python_test_wheel(sh, ctx, &args.backend)?;
    let _root = sh.push_dir(ctx.workspace_root());
    let cargo_test = apply_toolchains(
        sh,
        ctx,
        cmd!(sh, "cargo test -p cogentlm-py"),
        Some(&args.backend),
    )?;
    output::run_command("Running Python binding Rust tests", cargo_test)?;
    drop(_root);

    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx
        .tmp_dir()
        .join("python-tests")
        .join(args.backend.as_str());
    sh.create_dir(&venv_dir)?;
    output::run_command(
        "Creating Python test virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_command(
        "Installing Python test wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel} pytest"
            ),
        ),
    )?;

    let python_tests = ctx.bindings_python_dir().join("tests");
    output::run_command(
        "Running Python black-box pytest suite",
        cmd!(sh, "{python_exe} -m pytest {python_tests}"),
    )
}

fn run_model_smoke(sh: &Shell, ctx: &BuildContext, args: &TestModelSmokeArgs) -> Result<()> {
    if args.backend == Backend::All {
        anyhow::bail!(
            "model-smoke requires a concrete backend; choose cpu, vulkan, cuda, or metal"
        );
    }

    output::phase("Model-backed local smoke tests");
    let model = resolve_smoke_model(sh, ctx, args)?;
    output::path("Model", &model);
    output::detail("Backend", args.backend.as_str());

    targets::cli::build(sh, ctx, Some(&args.backend))?;
    run_cli_smoke(sh, ctx, &model, args)?;
    run_rust_generation_smoke(sh, ctx, &model, args)?;
    run_node_generation_smoke(sh, ctx, &model, args)?;
    run_python_generation_smoke(sh, ctx, &model, args)
}

fn resolve_smoke_model(
    sh: &Shell,
    ctx: &BuildContext,
    args: &TestModelSmokeArgs,
) -> Result<PathBuf> {
    match &args.model {
        Some(model) => Ok(model.clone()),
        None => sample_model::ensure_sample_model(
            sh,
            ctx,
            SampleModelOptions {
                allow_download: !args.offline,
            },
        ),
    }
}

fn run_cli_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    args: &TestModelSmokeArgs,
) -> Result<()> {
    output::phase("CLI smoke");
    let cli_dir = ctx.cli_artifacts_dir();
    let cli_exe = cli_dir.join(cli_binary_file_name());
    if !cli_exe.is_file() {
        anyhow::bail!("CLI executable was not staged at {}", cli_exe.display());
    }

    let _dir = sh.push_dir(&cli_dir);
    let prompt = &args.prompt;
    let mut smoke_cmd = cmd!(
        sh,
        "{cli_exe} {model} {prompt} --max-tokens 1 --temperature 0"
    );
    if args.backend != Backend::Cpu {
        smoke_cmd = smoke_cmd.arg("--backend").arg(args.backend.as_str());
    }
    if let Some(gpu_layers) = args.gpu_layers {
        smoke_cmd = smoke_cmd.arg("--gpu-layers").arg(gpu_layers.to_string());
    }
    output::run_command("Running CLI local inference smoke", smoke_cmd)
}

fn run_rust_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    args: &TestModelSmokeArgs,
) -> Result<()> {
    output::phase("Rust CogentClient generation smoke");
    output::path("Model", model);
    output::detail("Backend", args.backend.as_str());

    let _dir = sh.push_dir(ctx.workspace_root());
    for backend in test_backends(&args.backend) {
        for example in RUST_GENERATION_SMOKE_EXAMPLES {
            let mut smoke_cmd = cmd!(sh, "cargo run -p cogentlm-client");
            if backend != Backend::Cpu {
                smoke_cmd = smoke_cmd.arg("--features").arg(backend.as_str());
            }
            smoke_cmd = smoke_cmd
                .arg("--example")
                .arg(example)
                .arg("--")
                .arg(model)
                .arg(&args.prompt);
            if let Some(gpu_layers) = args.gpu_layers {
                smoke_cmd = smoke_cmd.env("COGENTLM_GPU_LAYERS", gpu_layers.to_string());
            }
            smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&backend))?;
            output::run_command(
                format!("Running Rust {} smoke: {example}", backend.as_str()),
                smoke_cmd,
            )
            .with_context(|| format!("Rust {} smoke failed: {example}", backend.as_str()))?;
        }
    }

    Ok(())
}

fn run_node_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    args: &TestModelSmokeArgs,
) -> Result<()> {
    output::phase("Node.js generation smoke");
    output::path("Model", model);
    output::detail("Backend", args.backend.as_str());
    targets::node::build(sh, ctx, Some(&args.backend))?;

    let node_dir = ctx.bindings_node_dir();
    let _dir = sh.push_dir(&node_dir);
    for backend in test_backends(&args.backend) {
        for smoke_script in NODE_GENERATION_SMOKE_SCRIPTS {
            let prompt = &args.prompt;
            let mut smoke_cmd = cmd!(sh, "node {smoke_script} {model} {prompt}")
                .env("COGENTLM_NODE_BACKEND", backend.as_str());
            if let Some(gpu_layers) = args.gpu_layers {
                smoke_cmd = smoke_cmd.env("COGENTLM_GPU_LAYERS", gpu_layers.to_string());
            }
            smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&backend))?;
            output::run_command(
                format!("Running Node {} smoke: {smoke_script}", backend.as_str()),
                smoke_cmd,
            )
            .with_context(|| format!("Node {} smoke failed: {smoke_script}", backend.as_str()))?;
        }
    }

    Ok(())
}

fn run_python_generation_smoke(
    sh: &Shell,
    ctx: &BuildContext,
    model: &Path,
    args: &TestModelSmokeArgs,
) -> Result<()> {
    output::phase("Python generation smoke");
    output::path("Model", model);
    output::detail("Backend", args.backend.as_str());
    let wheel = build_python_test_wheel(sh, ctx, &args.backend)?;
    let uv_exe = setup_uv(sh, ctx)?;
    let venv_dir = ctx
        .tmp_dir()
        .join("python-model-smoke")
        .join(args.backend.as_str());
    output::run_command(
        "Creating Python smoke virtual environment",
        apply_uv_env(
            ctx,
            cmd!(sh, "{uv_exe} venv --clear --python 3.12 {venv_dir}"),
        ),
    )?;
    let python_exe = python_venv_exe(&venv_dir);
    output::run_command(
        "Installing Python smoke wheel",
        apply_uv_env(
            ctx,
            cmd!(
                sh,
                "{uv_exe} pip install --python {python_exe} --force-reinstall {wheel}"
            ),
        ),
    )?;

    let python_dir = ctx.bindings_python_dir();
    let _dir = sh.push_dir(&python_dir);
    for backend in test_backends(&args.backend) {
        for smoke_script in PYTHON_GENERATION_SMOKE_SCRIPTS {
            let prompt = &args.prompt;
            let mut smoke_cmd = cmd!(sh, "{python_exe} {smoke_script} {model} {prompt}")
                .env("COGENTLM_PYTHON_BACKEND", backend.as_str());
            if let Some(gpu_layers) = args.gpu_layers {
                smoke_cmd = smoke_cmd.env("COGENTLM_GPU_LAYERS", gpu_layers.to_string());
            }
            smoke_cmd = apply_toolchains(sh, ctx, smoke_cmd, Some(&backend))?;
            output::run_command(
                format!("Running Python {} smoke: {smoke_script}", backend.as_str()),
                smoke_cmd,
            )
            .with_context(|| format!("Python {} smoke failed: {smoke_script}", backend.as_str()))?;
        }
    }

    Ok(())
}

fn test_backends(backend: &Backend) -> Vec<Backend> {
    match backend {
        Backend::All if cfg!(target_os = "macos") => vec![Backend::Cpu, Backend::Metal],
        Backend::All => vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda],
        backend => vec![*backend],
    }
}

fn collect_package_test_layout_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    let src = ctx.npm_package_dir().join("src");
    for path in collect_files_with_extension(&src, "ts")? {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.ends_with(".test.ts") {
            violations.push(format!(
                "TypeScript test must live under packages/npm/tests: {}",
                display_relative(ctx, &path)
            ));
        }
    }
    Ok(())
}

fn collect_rust_test_layout_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    for area in [
        ctx.workspace_root().join("crates"),
        ctx.workspace_root().join("bindings"),
    ] {
        for package_root in collect_rust_package_roots(&area)? {
            for path in collect_files_with_extension(&package_root, "rs")? {
                if is_inverted_rust_test_file(&package_root, &path) {
                    violations.push(format!(
                        "Rust unit test file must live under src/tests: {}",
                        display_relative(ctx, &path)
                    ));
                    continue;
                }
                if is_allowed_rust_test_file(&package_root, &path) {
                    continue;
                }

                let contents = std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                if contains_test_attribute(&contents) {
                    violations.push(format!(
                        "Rust test body must live under src/tests or crate-level tests: {}",
                        display_relative(ctx, &path)
                    ));
                }
            }
        }
    }
    Ok(())
}

fn collect_rust_package_roots(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut package_roots = Vec::new();
    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.join("src").is_dir() {
            package_roots.push(path);
        }
    }
    package_roots.sort();
    Ok(package_roots)
}

fn collect_files_with_extension(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                if is_ignored_layout_dir(&path) {
                    continue;
                }
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_ignored_layout_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("target" | "node_modules" | ".build" | "third_party")
    )
}

fn is_inverted_rust_test_file(package_root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(package_root) else {
        return false;
    };
    let components = path_components(relative);
    components.first() == Some(&"src")
        && components.get(1) != Some(&"tests")
        && components
            .iter()
            .skip(1)
            .any(|component| *component == "tests")
}

fn is_allowed_rust_test_file(package_root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(package_root) else {
        return false;
    };
    let components = path_components(relative);
    matches!(components.as_slice(), ["tests", ..] | ["src", "tests", ..])
}

fn path_components(path: &Path) -> Vec<&str> {
    path.components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect()
}

fn contains_test_attribute(contents: &str) -> bool {
    contents.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("#[test]") || line.starts_with("#[tokio::test]")
    })
}

fn display_relative(ctx: &BuildContext, path: &Path) -> String {
    path.strip_prefix(ctx.workspace_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

fn node_test_backends(ctx: &BuildContext, backend: &Backend) -> Result<Vec<Backend>> {
    let backends = test_backends(backend);
    if *backend != Backend::All {
        return Ok(backends);
    }

    let triplet = node_platform_triplet()?;
    let available = backends
        .into_iter()
        .filter(|backend| node_backend_artifact(ctx, backend, triplet).is_file())
        .collect::<Vec<_>>();
    if available.is_empty() {
        anyhow::bail!("Node --backend all did not produce any testable backend artifacts");
    }
    Ok(available)
}

fn node_backend_artifact(ctx: &BuildContext, backend: &Backend, triplet: &str) -> PathBuf {
    ctx.node_artifacts_dir().join(format!(
        "cogentlm_node_{}.{}.node",
        backend.as_str(),
        triplet
    ))
}

fn node_platform_triplet() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok("win32-x64-msvc"),
        ("macos", "x86_64") => Ok("darwin-x64"),
        ("macos", "aarch64") => Ok("darwin-arm64"),
        ("linux", "x86_64") => Ok("linux-x64-gnu"),
        (os, arch) => anyhow::bail!("unsupported Node test platform: {os} {arch}"),
    }
}

fn build_python_test_wheel(sh: &Shell, ctx: &BuildContext, backend: &Backend) -> Result<PathBuf> {
    let uv_exe = setup_uv(sh, ctx)?;
    output::run_command(
        "Ensuring Python 3.12 is available through uv",
        apply_uv_env(ctx, cmd!(sh, "{uv_exe} python install 3.12")),
    )?;

    let python_dir = ctx.bindings_python_dir();
    let dist_dir = ctx.python_artifacts_dir().join("test-wheels");
    prepare_python_test_wheel_dir(sh, &dist_dir)?;
    let target_dir = ctx
        .cargo_python_target_dir(Some(backend))
        .join("pytest-wheel");
    sh.create_dir(&target_dir)?;

    let _dir = sh.push_dir(&python_dir);
    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(
            sh,
            "{uv_exe} tool run maturin build --release --out {dist_dir}"
        ),
    )
    .env("CARGO_TARGET_DIR", &target_dir);
    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(backend))?;
    if *backend != Backend::Cpu {
        maturin_cmd = maturin_cmd.arg("--features").arg(backend.as_str());
    }
    output::run_command("Building Python test wheel", maturin_cmd)?;
    find_wheel_artifact(&dist_dir)?.with_context(|| {
        format!(
            "maturin did not produce a wheel artifact in {}",
            dist_dir.display()
        )
    })
}

fn prepare_python_test_wheel_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;
    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            sh.remove_path(path)?;
        }
    }
    Ok(())
}

fn find_wheel_artifact(dir: &Path) -> Result<Option<PathBuf>> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn ensure_workspace_bun_install(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Installing workspace Bun dependencies",
        cmd!(sh, "bun install"),
    )
}

fn python_venv_exe(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn cli_binary_file_name() -> &'static str {
    if cfg!(windows) {
        "cogentlm.exe"
    } else {
        "cogentlm"
    }
}
