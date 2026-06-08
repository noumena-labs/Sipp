//! Standalone gateway-server distribution build target.

use crate::cli::Backend;
use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const GATEWAY_BINARY_NAME: &str = "cogentlm-gateway";
const BACKEND_DL_FEATURE: &str = "backend-dl";

/// Builds a staged gateway-server distribution for the selected backend set.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Gateway-server distribution");
    output::detail("Backend request", backend_label(backend));

    let dist_dir = ctx.gateway_server_artifacts_dir();
    output::path("Artifact directory", &dist_dir);
    prepare_dist_dir(sh, &dist_dir)?;

    let best_effort = matches!(backend, Some(Backend::All));
    let backends_to_build = backends_to_build(backend);
    output::detail(
        "Expanded backends",
        output::backend_list(&backends_to_build),
    );

    let mut built = Vec::new();
    let mut skipped = Vec::new();

    for backend in backends_to_build {
        let copy_executable = built.is_empty();
        let optional = best_effort && backend != Backend::Cpu;
        match build_backend_variant(sh, ctx, &dist_dir, &backend, copy_executable) {
            Ok(()) => built.push(backend),
            Err(error) if optional => {
                output::warning(format!(
                    "Skipped optional gateway-server {} backend: {error:#}",
                    backend.as_str()
                ));
                skipped.push(backend);
            }
            Err(error) => return Err(error),
        }
    }

    output::success(format!(
        "Gateway-server distribution build complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    output::detail("Built variants", output::backend_list(&built));

    if !skipped.is_empty() {
        output::detail("Skipped optional variants", output::backend_list(&skipped));
    }

    Ok(())
}

fn build_backend_variant(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    backend: &Backend,
    copy_executable: bool,
) -> Result<()> {
    if matches!(backend, Backend::All) {
        anyhow::bail!("Backend::All cannot be built as a single gateway-server variant");
    }

    let feature = backend.as_str();
    output::phase(&format!(
        "Gateway-server backend: {}",
        feature.to_uppercase()
    ));

    let target_dir = ctx.cargo_gateway_server_target_dir(backend);
    let cmake_dir = ctx.cmake_gateway_server_sys_dir(backend);
    sh.create_dir(&target_dir)?;
    sh.create_dir(&cmake_dir)?;
    let cmake_dir = cmake_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", cmake_dir.display()))?;
    output::path("Cargo target dir", &target_dir);
    output::path("CMake sys dir", &cmake_dir);

    let mut cargo_cmd = cmd!(
        sh,
        "cargo build --release --package cogentlm-gateway-server --target-dir {target_dir}"
    )
    .env("COGENTLM_SYS_CMAKE_OUT_DIR", &cmake_dir);
    cargo_cmd = apply_toolchains(sh, ctx, cargo_cmd, Some(backend))?;
    cargo_cmd = cargo_cmd.arg("--features").arg(cargo_features(backend));

    output::run_build_command(
        format!("Compiling gateway-server {feature} backend"),
        cargo_cmd,
    )
    .with_context(|| format!("failed to build gateway-server {feature} backend"))?;

    if copy_executable {
        let executable = copy_gateway_executable(sh, &target_dir, dist_dir)?;
        output::artifact(&executable);
    }

    let summary = copy_runtime_artifacts(
        sh,
        &cmake_dir,
        dist_dir,
        copy_executable,
        if copy_executable { None } else { Some(backend) },
    )?;
    if copy_executable && summary.base_files == 0 {
        anyhow::bail!(
            "gateway-server {feature} backend did not produce base runtime libraries in {}",
            cmake_dir.display()
        );
    }
    if summary.plugin_files == 0 {
        anyhow::bail!(
            "gateway-server {feature} backend did not produce ggml backend plugins in {}",
            cmake_dir.display()
        );
    }
    output::detail("Runtime files copied", summary.total_files);

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
        Some(backend) => vec![*backend],
        None => vec![Backend::Cpu],
    }
}

fn prepare_dist_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    if dist_dir.exists() {
        output::step(format!(
            "Removing stale gateway-server artifact directory {}",
            dist_dir.display()
        ));
        sh.remove_path(dist_dir)?;
    }
    sh.create_dir(dist_dir)?;
    Ok(())
}

fn copy_gateway_executable(sh: &Shell, target_dir: &Path, dist_dir: &Path) -> Result<PathBuf> {
    let source = target_dir.join("release").join(gateway_binary_file_name());
    if !source.exists() {
        anyhow::bail!("cargo did not produce {}", source.display());
    }

    let dest = dist_dir.join(gateway_binary_file_name());
    replace_file(sh, &source, &dest)?;
    Ok(dest)
}

fn copy_runtime_artifacts(
    sh: &Shell,
    cmake_dir: &Path,
    dist_dir: &Path,
    copy_base: bool,
    plugin_backend: Option<&Backend>,
) -> Result<RuntimeCopySummary> {
    let mut copied_names = HashSet::new();
    let mut summary = RuntimeCopySummary::default();

    for file in collect_runtime_candidates(cmake_dir)? {
        let Some(file_name) = file.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(kind) = runtime_file_kind(file_name) else {
            continue;
        };

        let copy = match kind {
            RuntimeFileKind::Base => copy_base,
            RuntimeFileKind::Plugin => plugin_backend
                .map(|backend| plugin_matches_backend(file_name, backend))
                .unwrap_or(true),
            RuntimeFileKind::MetalSidecar => {
                copy_base || matches!(plugin_backend, Some(Backend::Metal))
            }
        };
        if !copy || !copied_names.insert(file_name.to_owned()) {
            continue;
        }

        let dest = dist_dir.join(file_name);
        replace_file(sh, &file, &dest)?;
        output::artifact(&dest);
        summary.total_files += 1;
        match kind {
            RuntimeFileKind::Base => summary.base_files += 1,
            RuntimeFileKind::Plugin => summary.plugin_files += 1,
            RuntimeFileKind::MetalSidecar => {}
        }
    }

    Ok(summary)
}

fn replace_file(sh: &Shell, source: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        sh.remove_path(dest)?;
    }
    sh.copy_file(source, dest)?;
    Ok(())
}

fn collect_runtime_candidates(cmake_dir: &Path) -> Result<Vec<PathBuf>> {
    let roots = [
        cmake_dir.join("bin"),
        cmake_dir.join("lib"),
        cmake_dir.join("lib64"),
        cmake_dir.join("build"),
    ];
    let mut files = Vec::new();

    for root in roots {
        collect_files_recursive(&root, &mut files)?;
    }

    files.sort();
    Ok(files)
}

fn collect_files_recursive(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }

    Ok(())
}

fn runtime_file_kind(file_name: &str) -> Option<RuntimeFileKind> {
    let lower = file_name.to_ascii_lowercase();
    if is_base_runtime_file(&lower) {
        Some(RuntimeFileKind::Base)
    } else if is_backend_plugin_file(&lower) {
        Some(RuntimeFileKind::Plugin)
    } else if is_metal_sidecar_file(&lower) {
        Some(RuntimeFileKind::MetalSidecar)
    } else {
        None
    }
}

fn is_base_runtime_file(file_name: &str) -> bool {
    if cfg!(windows) {
        return matches!(
            file_name,
            "llama.dll"
                | "llama-common.dll"
                | "llama-common-base.dll"
                | "ggml.dll"
                | "ggml-base.dll"
                | "cogent_shim.dll"
                | "mtmd.dll"
                | "cpp-httplib.dll"
        );
    }

    let is_shared = file_name.contains(".so") || file_name.ends_with(".dylib");
    is_shared
        && [
            "libllama.",
            "libllama-common.",
            "libllama-common-base.",
            "libggml.",
            "libggml-base.",
            "libcogent_shim.",
            "libmtmd.",
            "libcpp-httplib.",
        ]
        .iter()
        .any(|prefix| file_name.starts_with(prefix))
}

fn is_backend_plugin_file(file_name: &str) -> bool {
    if cfg!(windows) {
        file_name.starts_with("ggml-")
            && file_name.ends_with(".dll")
            && file_name != "ggml-base.dll"
    } else {
        file_name.starts_with("libggml-")
            && file_name.ends_with(".so")
            && !file_name.starts_with("libggml-base.")
    }
}

fn is_metal_sidecar_file(file_name: &str) -> bool {
    matches!(
        file_name,
        "default.metallib" | "ggml-common.h" | "ggml-metal.metal" | "ggml-metal-impl.h"
    )
}

fn plugin_matches_backend(file_name: &str, backend: &Backend) -> bool {
    let lower = file_name.to_ascii_lowercase();
    match backend {
        Backend::Cpu => lower.contains("ggml-cpu"),
        Backend::Cuda => lower.contains("ggml-cuda"),
        Backend::Metal => lower.contains("ggml-metal"),
        Backend::Vulkan => lower.contains("ggml-vulkan"),
        Backend::All => false,
    }
}

fn cargo_features(backend: &Backend) -> String {
    if *backend == Backend::Cpu {
        BACKEND_DL_FEATURE.to_owned()
    } else {
        format!("{BACKEND_DL_FEATURE},{}", backend.as_str())
    }
}

pub(crate) fn gateway_binary_file_name() -> &'static str {
    if cfg!(windows) {
        "cogentlm-gateway.exe"
    } else {
        GATEWAY_BINARY_NAME
    }
}

fn backend_label(backend: Option<&Backend>) -> &'static str {
    backend.map(Backend::as_str).unwrap_or("cpu (default)")
}

#[derive(Default)]
struct RuntimeCopySummary {
    total_files: usize,
    base_files: usize,
    plugin_files: usize,
}

#[derive(Clone, Copy)]
enum RuntimeFileKind {
    Base,
    Plugin,
    MetalSidecar,
}
