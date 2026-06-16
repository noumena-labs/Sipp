//! Python binding build target.

use crate::cli::Backend;
use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::toolchains::python::apply_uv_env;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/targets/python_tests.rs"]
mod python_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const PYTHON_PACKAGE_NAME: &str = "sipp";
const PYTHON_BACKEND_PACKAGE_PREFIX: &str = "sipp-backend";

/// Builds the Python bindings for the selected backend.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    output::phase("Python bindings");
    output::detail("Backend request", backend_label(backend));
    output::path("Python package workspace", &python_project_dir(ctx));
    output::path("PyO3 binding crate", &ctx.bindings_python_dir());

    let uv_exe = crate::toolchains::python::setup_uv(sh, ctx)?;

    output::run_build_command(
        "Ensuring Python 3.12 is available through uv",
        apply_uv_env(ctx, cmd!(sh, "{uv_exe} python install 3.12")),
    )?;

    if matches!(backend, Some(Backend::All)) {
        return build_package_wheels(sh, ctx, &uv_exe);
    }

    build_develop(sh, ctx, backend, &uv_exe)
}

fn build_develop(
    sh: &Shell,
    ctx: &BuildContext,
    backend: Option<&Backend>,
    uv_exe: &Path,
) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Python develop build");

    let python_dir = python_project_dir(ctx);
    let _dir = sh.push_dir(&python_dir);

    let venv_dir = python_dir.join(".venv");
    if !venv_dir.exists() {
        output::run_build_command(
            "Creating local Python virtual environment",
            apply_uv_env(ctx, cmd!(sh, "{uv_exe} venv --python 3.12")),
        )?;
    } else {
        output::success("Using existing Python virtual environment");
    }

    let target_dir = ctx.cargo_python_target_dir(backend);
    sh.create_dir(&target_dir)?;
    output::path("Cargo target dir", &target_dir);

    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(sh, "{uv_exe} tool run maturin develop --release --uv"),
    )
    .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, backend)?;

    match backend {
        Some(Backend::Cpu) | None => {
            output::detail("Hardware backend", "CPU (default)");
        }
        Some(b) => {
            let feature = b.as_str();
            output::detail("Hardware backend", feature.to_uppercase());
            maturin_cmd = maturin_cmd.arg("--features").arg(feature);
        }
    }

    output::run_build_command("Running maturin develop", maturin_cmd)?;
    output::success(format!(
        "Python develop build complete in {}",
        output::elapsed(started_at.elapsed())
    ));

    Ok(())
}

fn build_package_wheels(sh: &Shell, ctx: &BuildContext, uv_exe: &Path) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Python package wheel set");

    let dist_dir = ctx.python_artifacts_dir();
    prepare_dist_dir(sh, &dist_dir)?;
    output::path("Wheel output directory", &dist_dir);

    let backends_to_build = backends_to_build();
    output::detail(
        "Expanded backends",
        output::backend_list(&backends_to_build),
    );

    let mut built = Vec::new();
    let mut skipped = Vec::new();

    for backend in backends_to_build {
        let optional = backend != Backend::Cpu && !require_all_backends();
        match build_package_wheel(sh, ctx, &dist_dir, &backend, uv_exe) {
            Ok(path) => {
                output::artifact(&path);
                built.push(backend);
            }
            Err(error) if optional => {
                output::warning(format!(
                    "Skipped optional {} backend: {error:#}",
                    backend.as_str()
                ));
                skipped.push(backend);
            }
            Err(error) => return Err(error),
        }
    }

    output::success(format!(
        "Python package wheel set complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    output::detail("Built wheels", output::backend_list(&built));

    if !skipped.is_empty() {
        output::detail("Skipped optional variants", output::backend_list(&skipped));
    }

    Ok(())
}

fn build_package_wheel(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    backend: &Backend,
    uv_exe: &Path,
) -> Result<PathBuf> {
    if matches!(backend, Backend::All) {
        anyhow::bail!("Backend::All cannot be built as a single Python variant");
    }

    if *backend == Backend::Cpu {
        return build_sipp_wheel(sh, ctx, dist_dir, uv_exe);
    }

    build_backend_package_wheel(sh, ctx, dist_dir, backend, uv_exe)
}

fn build_sipp_wheel(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    uv_exe: &Path,
) -> Result<PathBuf> {
    output::phase("Python package: SIPP");

    let python_dir = python_project_dir(ctx);
    let _dir = sh.push_dir(&python_dir);
    let target_dir = ctx.cargo_python_target_dir(Some(&Backend::Cpu));
    sh.create_dir(&target_dir)?;
    output::path("Cargo target dir", &target_dir);

    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(
            sh,
            "{uv_exe} tool run maturin build --release --out {dist_dir}"
        ),
    )
    .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(&Backend::Cpu))?;
    output::run_build_command("Building Python sipp wheel", maturin_cmd)
        .context("failed to build Python sipp wheel")?;

    find_wheel_artifact_for_distribution(dist_dir, PYTHON_PACKAGE_NAME)?.with_context(|| {
        format!(
            "maturin did not produce a {PYTHON_PACKAGE_NAME} wheel artifact in {}",
            dist_dir.display()
        )
    })
}

fn build_backend_package_wheel(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    backend: &Backend,
    uv_exe: &Path,
) -> Result<PathBuf> {
    let feature = backend.as_str();
    let distribution = backend_distribution_name(backend);
    output::phase(&format!(
        "Python backend package: {}",
        distribution.to_uppercase()
    ));

    let project_dir = backend_package_project_dir(ctx, backend);
    if !project_dir.exists() {
        anyhow::bail!(
            "Python backend package project is missing: {}",
            project_dir.display()
        );
    }
    let wheel_dir = ctx
        .tmp_dir()
        .join("python")
        .join("backend-wheels")
        .join(feature);
    if wheel_dir.exists() {
        sh.remove_path(&wheel_dir)?;
    }
    sh.create_dir(&wheel_dir)?;

    let target_dir = ctx.cargo_python_target_dir(Some(backend));
    sh.create_dir(&target_dir)?;
    output::path("Cargo target dir", &target_dir);
    output::path("Backend package workspace", &project_dir);

    let _dir = sh.push_dir(&project_dir);
    let mut maturin_cmd = apply_uv_env(
        ctx,
        cmd!(
            sh,
            "{uv_exe} tool run maturin build --release --out {wheel_dir}"
        ),
    )
    .env("CARGO_TARGET_DIR", &target_dir);

    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, Some(backend))?;
    maturin_cmd = maturin_cmd.arg("--features").arg(feature);
    if let Some(auditwheel) = backend_auditwheel_mode(backend) {
        output::detail("Auditwheel policy", auditwheel);
        maturin_cmd = maturin_cmd.arg("--auditwheel").arg(auditwheel);
    }

    output::run_build_command(
        format!("Building Python {feature} backend wheel"),
        maturin_cmd,
    )
    .with_context(|| format!("failed to build Python {feature} backend"))?;

    let wheel =
        find_wheel_artifact_for_distribution(&wheel_dir, &distribution)?.with_context(|| {
            format!(
                "maturin did not produce a {distribution} wheel artifact in {}",
                wheel_dir.display()
            )
        })?;
    let dest = dist_dir.join(
        wheel
            .file_name()
            .with_context(|| format!("invalid wheel path {}", wheel.display()))?,
    );
    sh.copy_file(&wheel, &dest)?;

    Ok(dest)
}

fn python_project_dir(ctx: &BuildContext) -> PathBuf {
    ctx.python_package_project_dir()
}

fn backend_package_project_dir(ctx: &BuildContext, backend: &Backend) -> PathBuf {
    python_project_dir(ctx)
        .join("backends")
        .join(backend.as_str())
}

fn backends_to_build() -> Vec<Backend> {
    if cfg!(target_os = "macos") {
        vec![Backend::Cpu, Backend::Metal]
    } else {
        vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
    }
}

fn backend_distribution_name(backend: &Backend) -> String {
    format!("{PYTHON_BACKEND_PACKAGE_PREFIX}-{}", backend.as_str())
}

// Dev Linux GPU backend wheels may depend on host driver/runtime stacks.
// Maturin's repair mode tries to copy every DT_NEEDED library and fails on
// driver-only libraries such as libcuda.so.1, so dev builds can keep the audit
// visible without repairing. Release builds leave this unset and stay strict.
fn backend_auditwheel_mode(backend: &Backend) -> Option<&'static str> {
    if cfg!(target_os = "linux")
        && matches!(backend, Backend::Cuda | Backend::Vulkan)
        && gpu_auditwheel_warn_enabled()
    {
        Some("warn")
    } else {
        None
    }
}

fn gpu_auditwheel_warn_enabled() -> bool {
    matches!(
        env::var("SIPP_PYTHON_GPU_AUDITWHEEL").as_deref(),
        Ok("warn")
    )
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
        if (file_name.starts_with("sipp-") || file_name.starts_with("sipp_backend_"))
            && path.extension().and_then(|ext| ext.to_str()) == Some("whl")
        {
            sh.remove_path(path)?;
        }
    }

    Ok(())
}

fn find_wheel_artifact_for_distribution(dir: &Path, distribution: &str) -> Result<Option<PathBuf>> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("whl")
            && wheel_matches_distribution(&path, distribution)
        {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn wheel_matches_distribution(path: &Path, distribution: &str) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let normalized = distribution.replace('-', "_");
    file_name.starts_with(&format!("{normalized}-"))
}

fn backend_label(backend: Option<&Backend>) -> &'static str {
    backend.map(Backend::as_str).unwrap_or("cpu (default)")
}

fn require_all_backends() -> bool {
    matches!(env::var("SIPP_REQUIRE_ALL_BACKENDS").as_deref(), Ok("1"))
}
