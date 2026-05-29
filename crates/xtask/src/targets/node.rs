//! Node.js binding build target.

use crate::cli::Backend;
use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

const NODE_BINARY_NAME: &str = "cogentlm_node";

/// Builds Node.js bindings for the selected backend or backend set.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Node.js bindings");
    output::detail("Backend request", backend_label(backend));

    let node_dir = ctx.workspace_root().join("bindings").join("node");
    output::path("Binding workspace", &node_dir);
    output::path("Artifact directory", &ctx.node_artifacts_dir());

    let _dir = sh.push_dir(&node_dir);

    output::run_command("Installing Node dependencies", cmd!(sh, "bun install"))?;

    let dist_dir = ctx.node_artifacts_dir();
    prepare_dist_dir(sh, ctx, &dist_dir)?;

    let best_effort = matches!(backend, Some(Backend::All));
    let backends_to_build = backends_to_build(backend);
    let mut built = Vec::new();
    let mut skipped = Vec::new();

    for backend in backends_to_build {
        let optional = best_effort && backend != Backend::Cpu;
        match build_backend_variant(sh, ctx, &dist_dir, &backend) {
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
        "Node build complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    output::detail("Built variants", output::backend_list(&built));

    if !skipped.is_empty() {
        output::detail("Skipped optional variants", output::backend_list(&skipped));
    }

    Ok(())
}

fn backends_to_build(backend: Option<&Backend>) -> Vec<Backend> {
    match backend {
        Some(Backend::All) => {
            if cfg!(target_os = "macos") {
                vec![Backend::Cpu, Backend::Metal]
            } else {
                vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
            }
        }
        Some(backend) => vec![backend.clone()],
        None => vec![Backend::Cpu],
    }
}

fn prepare_dist_dir(sh: &Shell, ctx: &BuildContext, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;

    let staging_dir = ctx.tmp_dir().join("node");
    if staging_dir.exists() {
        output::step(format!(
            "Removing stale Node staging directory {}",
            staging_dir.display()
        ));
        sh.remove_path(&staging_dir)?;
    }

    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&format!("{NODE_BINARY_NAME}_"))
            && path.extension().and_then(|ext| ext.to_str()) == Some("node")
        {
            output::step(format!("Removing stale Node artifact {}", path.display()));
            sh.remove_path(path)?;
        }
    }

    Ok(())
}

fn build_backend_variant(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    backend: &Backend,
) -> Result<PathBuf> {
    if matches!(backend, Backend::All) {
        anyhow::bail!("Backend::All cannot be built as a single Node variant");
    }

    let feature = backend.as_str();
    output::phase(&format!("Node backend: {}", feature.to_uppercase()));

    let staging_dir = ctx.tmp_dir().join("node").join(feature);
    if staging_dir.exists() {
        sh.remove_path(&staging_dir)?;
    }
    sh.create_dir(&staging_dir)?;

    let target_dir = ctx.cargo_node_target_dir(backend);
    output::path("Cargo target dir", &target_dir);
    output::path("Staging directory", &staging_dir);

    let mut napi_cmd = cmd!(
        sh,
        "bunx napi build --platform --release --no-js --output-dir {staging_dir} --target-dir {target_dir}"
    );
    napi_cmd = apply_toolchains(sh, ctx, napi_cmd, Some(backend))?;

    if *backend != Backend::Cpu {
        napi_cmd = napi_cmd.arg("--features").arg(feature);
    }

    output::run_command(format!("Compiling Node {feature} backend"), napi_cmd)
        .with_context(|| format!("failed to build Node {feature} backend"))?;

    let artifact = find_artifact(&staging_dir)?.with_context(|| {
        format!(
            "napi did not produce a .node artifact in {}",
            staging_dir.display()
        )
    })?;
    let file_name = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("invalid Node artifact path {}", artifact.display()))?;
    let renamed = file_name.replacen(
        NODE_BINARY_NAME,
        &format!("{NODE_BINARY_NAME}_{feature}"),
        1,
    );
    if renamed == file_name {
        anyhow::bail!("unexpected Node artifact name: {file_name}");
    }

    let dest = dist_dir.join(renamed);
    sh.copy_file(&artifact, &dest)?;
    sh.remove_path(&staging_dir)?;

    Ok(dest)
}

fn backend_label(backend: Option<&Backend>) -> &'static str {
    backend.map(Backend::as_str).unwrap_or("cpu (default)")
}

fn find_artifact(dir: &Path) -> Result<Option<PathBuf>> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("node") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}
