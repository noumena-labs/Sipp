//! Python binding build target.
//!
use crate::cli::Backend;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

const PYTHON_PACKAGE_NAME: &str = "cogentlm";
const PYTHON_NATIVE_MODULE_NAME: &str = "_native";
const PYTHON_BACKEND_BINARY_DIR: &str = "binaries";

/// Builds the Python bindings for the selected backend.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    println!("=> Building Python Bindings...");

    // Get our hermetic `uv` executable
    let uv_exe = crate::toolchains::python::setup_uv(sh, ctx)?;

    println!("   Ensuring Python environment is bootstrapped...");
    cmd!(sh, "{uv_exe} python install 3.12").run()?;

    let _dir = sh.push_dir(python_project_dir(ctx));

    // Pass the hermetic uv_exe down into our specialized build paths
    if matches!(backend, Some(Backend::All)) {
        return build_fat_wheel(sh, ctx, &uv_exe);
    }

    build_develop(sh, ctx, backend, &uv_exe)
}

fn build_develop(
    sh: &Shell,
    ctx: &BuildContext,
    backend: Option<&Backend>,
    uv_exe: &Path,
) -> Result<()> {
    println!("=> Building Python Bindings (Develop)...");
    let python_dir = python_project_dir(ctx);
    let _dir = sh.push_dir(&python_dir);

    // If one doesn't exist, we use uv to create a blazing-fast hermetic .venv
    let venv_dir = python_dir.join(".venv");
    if !venv_dir.exists() {
        println!("   Creating local Python virtual environment (.venv)...");
        cmd!(sh, "{uv_exe} venv --python 3.12").run()?;
    }
    let binary_dir = backend_binary_dir(ctx);
    prepare_backend_binary_dir(sh, &binary_dir)?;

    let target_dir = ctx.cargo_python_target_dir(backend);
    sh.create_dir(&target_dir)?;

    // FIX: Use {uv_exe} tool run instead of global uv run
    let mut maturin_cmd = cmd!(sh, "{uv_exe} tool run maturin develop --release --uv")
        .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, backend)?;

    match backend {
        Some(Backend::Cpu) | None => {
            println!("   Hardware Backend: CPU (Default)");
        }
        Some(b) => {
            let feature = b.as_str();
            println!("   Hardware Backend: {}", feature.to_uppercase());
            maturin_cmd = maturin_cmd.arg("--features").arg(feature);
        }
    }

    maturin_cmd.run()?;
    Ok(())
}

fn build_fat_wheel(sh: &Shell, ctx: &BuildContext, uv_exe: &Path) -> Result<()> {
    println!("=> Building Python Backend-Fat Wheel...");
    let binary_dir = backend_binary_dir(ctx);
    prepare_backend_binary_dir(sh, &binary_dir)?;

    let dist_dir = ctx.python_artifacts_dir();
    prepare_dist_dir(sh, &dist_dir)?;

    let backends_to_build = backends_to_build();
    let mut built = Vec::new();
    let mut skipped = Vec::new();

    for backend in backends_to_build {
        let optional = backend != Backend::Cpu;
        match build_backend_variant(sh, ctx, &binary_dir, &backend, uv_exe) {
            Ok(path) => {
                println!("   Staged {}", path.display());
                built.push(backend);
            }
            Err(error) if optional => {
                eprintln!(
                    "   Warning: skipped optional {} backend: {error:#}",
                    backend.as_str()
                );
                skipped.push(backend);
            }
            Err(error) => return Err(error),
        }
    }

    let target_dir = ctx.cargo_python_target_dir(None);
    sh.create_dir(&target_dir)?;

    // FIX: Use {uv_exe} tool run
    let mut maturin_cmd = cmd!(
        sh,
        "{uv_exe} tool run maturin build --release --out {dist_dir}"
    )
    .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, None)?;
    maturin_cmd
        .run()
        .context("failed to build Python backend-fat wheel")?;

    // ... (rest of the logging remains the same)
    Ok(())
}

fn build_backend_variant(
    sh: &Shell,
    ctx: &BuildContext,
    binary_dir: &Path,
    backend: &Backend,
    uv_exe: &Path,
) -> Result<PathBuf> {
    if matches!(backend, Backend::All) {
        anyhow::bail!("Backend::All cannot be built as a single Python variant");
    }

    let feature = backend.as_str();
    println!("--------------------------------------------------");
    println!("Compiling Python Variant: {}", feature.to_uppercase());
    println!("--------------------------------------------------");

    let staging_dir = ctx.tmp_dir().join("python").join("wheels").join(feature);
    if staging_dir.exists() {
        sh.remove_path(&staging_dir)?;
    }
    sh.create_dir(&staging_dir)?;

    let target_dir = ctx.cargo_python_target_dir(Some(backend));
    sh.create_dir(&target_dir)?;

    // FIX: Use {uv_exe} tool run
    let mut maturin_cmd = cmd!(
        sh,
        "{uv_exe} tool run maturin build --release --out {staging_dir}"
    )
    .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(backend))?;

    if *backend != Backend::Cpu {
        maturin_cmd = maturin_cmd.arg("--features").arg(feature);
    }

    maturin_cmd
        .run()
        .with_context(|| format!("failed to build Python {feature} backend"))?;

    let wheel =
        find_wheel_artifact(&staging_dir)?.context("maturin did not produce a wheel artifact")?;

    let extracted_dir = ctx.tmp_dir().join("python").join("extracted").join(feature);
    if extracted_dir.exists() {
        sh.remove_path(&extracted_dir)?;
    }
    sh.create_dir(&extracted_dir)?;

    // FIX: Use {uv_exe} run python
    cmd!(
        sh,
        "{uv_exe} run python -m zipfile -e {wheel} {extracted_dir}"
    )
    .run()
    .with_context(|| format!("failed to extract {}", wheel.display()))?;

    let native = find_native_extension(&extracted_dir)?.with_context(|| {
        format!("wheel did not contain {PYTHON_PACKAGE_NAME}/{PYTHON_NATIVE_MODULE_NAME} extension")
    })?;
    let native_file_name = native
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("invalid native extension path {}", native.display()))?;
    let native_suffix = native_file_name
        .strip_prefix(PYTHON_NATIVE_MODULE_NAME)
        .with_context(|| format!("unexpected native extension name: {native_file_name}"))?;
    let dest = binary_dir.join(format!(
        "{PYTHON_NATIVE_MODULE_NAME}_{feature}{native_suffix}"
    ));

    sh.copy_file(&native, &dest)?;
    sh.remove_path(&staging_dir)?;
    sh.remove_path(&extracted_dir)?;

    Ok(dest)
}

// --------------------------------------------------------------------------------------------
// HELPERS
// --------------------------------------------------------------------------------------------

fn python_project_dir(ctx: &BuildContext) -> PathBuf {
    ctx.workspace_root().join("bindings").join("python")
}

fn python_package_dir(ctx: &BuildContext) -> PathBuf {
    python_project_dir(ctx)
        .join("python")
        .join(PYTHON_PACKAGE_NAME)
}

fn backend_binary_dir(ctx: &BuildContext) -> PathBuf {
    python_package_dir(ctx).join(PYTHON_BACKEND_BINARY_DIR)
}

fn backends_to_build() -> Vec<Backend> {
    if cfg!(target_os = "macos") {
        vec![Backend::Cpu, Backend::Metal]
    } else {
        vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
    }
}

fn prepare_backend_binary_dir(sh: &Shell, dir: &Path) -> Result<()> {
    sh.create_dir(dir)?;

    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.file_name().and_then(|name| name.to_str()) == Some(".gitkeep") {
            continue;
        }
        sh.remove_path(path)?;
    }

    Ok(())
}

fn prepare_dist_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;

    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with("cogentlm-")
            && path.extension().and_then(|ext| ext.to_str()) == Some("whl")
        {
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

fn find_native_extension(dir: &Path) -> Result<Option<PathBuf>> {
    let package_dir = dir.join(PYTHON_PACKAGE_NAME);
    for entry in std::fs::read_dir(&package_dir)
        .with_context(|| format!("failed to read {}", package_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() || !is_python_extension(&path) {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(PYTHON_NATIVE_MODULE_NAME) {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn is_python_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("pyd" | "so")
    )
}
