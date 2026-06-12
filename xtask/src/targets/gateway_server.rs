//! Standalone gateway-server distribution build target.

use crate::cli::Backend;
use crate::javascript;
use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/targets/gateway_server_tests.rs"]
mod gateway_server_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const GATEWAY_BINARY_NAME: &str = "sipp-gateway";
const BACKEND_DL_FEATURE: &str = "backend-dl";
const ADMIN_UI_FINGERPRINT_FILE: &str = "admin-ui.sha256";

/// Builds a staged gateway-server distribution for the selected backend set.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    let started_at = Instant::now();
    output::phase("Gateway-server distribution");
    output::detail("Backend request", backend_label(backend));

    let dist_dir = ctx.gateway_server_artifacts_dir();
    output::path("Artifact directory", &dist_dir);
    let admin_ui_dist = build_admin_ui(sh, ctx)?;
    prepare_dist_dir(sh, &dist_dir)?;
    copy_admin_ui(sh, &admin_ui_dist, &dist_dir.join("admin-ui"))?;

    let best_effort = matches!(backend, Some(Backend::All));
    let backends_to_build = backends_to_build(backend);
    output::detail(
        "Expanded backends",
        output::backend_list(&backends_to_build),
    );

    let mut built = Vec::new();
    let mut skipped = Vec::new();
    let mut expected_runtime_files = HashSet::new();

    for backend in backends_to_build {
        let copy_executable = built.is_empty();
        let optional = best_effort && backend != Backend::Cpu;
        match build_backend_variant(sh, ctx, &dist_dir, &backend, copy_executable) {
            Ok(files) => {
                expected_runtime_files.extend(files);
                built.push(backend);
            }
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
    clean_stale_runtime_artifacts(sh, &dist_dir, &expected_runtime_files)?;

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

fn build_admin_ui(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let admin_ui_dir = ctx
        .workspace_root()
        .join("apps")
        .join("gateway-server")
        .join("admin-ui");
    let dist = admin_ui_dir.join("dist");
    let fingerprint = admin_ui_fingerprint(ctx, &admin_ui_dir)?;
    let fingerprint_path = admin_ui_fingerprint_path(ctx);
    if dist.join("index.html").is_file()
        && std::fs::read_to_string(&fingerprint_path)
            .map(|current| current.trim() == fingerprint)
            .unwrap_or(false)
    {
        output::step("Gateway Admin Dashboard inputs unchanged; skipping build");
        return Ok(dist);
    }

    javascript::install_root_workspace_dependencies(
        sh,
        ctx,
        "Installing gateway Admin Dashboard dependencies",
        std::slice::from_ref(&admin_ui_dir),
    )?;
    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_build_command(
        "Building gateway Admin Dashboard",
        cmd!(sh, "bun run --filter sipp-gateway-admin-ui build"),
    )?;
    if !dist.join("index.html").is_file() {
        anyhow::bail!(
            "gateway Admin Dashboard build did not produce {}",
            dist.join("index.html").display()
        );
    }
    if let Some(parent) = fingerprint_path.parent() {
        sh.create_dir(parent)?;
    }
    std::fs::write(&fingerprint_path, fingerprint)
        .with_context(|| format!("failed to write {}", fingerprint_path.display()))?;
    Ok(dist)
}

fn copy_admin_ui(sh: &Shell, source: &Path, dest: &Path) -> Result<()> {
    sh.create_dir(dest)?;
    let mut expected_files = HashSet::new();
    for file in collect_files_recursive_flat(source)? {
        let relative = file
            .strip_prefix(source)
            .with_context(|| format!("dashboard asset is outside {}", source.display()))?;
        expected_files.insert(relative.to_path_buf());
        let target = dest.join(relative);
        if let Some(parent) = target.parent() {
            sh.create_dir(parent)?;
        }
        replace_file(sh, &file, &target)?;
    }
    remove_stale_files(sh, dest, &expected_files)?;
    output::artifact(dest);
    Ok(())
}

fn admin_ui_fingerprint(ctx: &BuildContext, admin_ui_dir: &Path) -> Result<String> {
    let mut files = Vec::new();
    collect_admin_ui_fingerprint_files(admin_ui_dir, &mut files)?;
    for path in [
        ctx.workspace_root().join("package.json"),
        ctx.workspace_root().join("bun.lock"),
        ctx.workspace_root().join("bun.lockb"),
    ] {
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    files.dedup();

    let mut hasher = Sha256::new();
    for file in files {
        let relative = file
            .strip_prefix(ctx.workspace_root())
            .unwrap_or(file.as_path())
            .to_string_lossy()
            .replace('\\', "/");
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        let bytes =
            std::fs::read(&file).with_context(|| format!("failed to read {}", file.display()))?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn admin_ui_fingerprint_path(ctx: &BuildContext) -> PathBuf {
    ctx.tmp_dir()
        .join("gateway-server")
        .join(ADMIN_UI_FINGERPRINT_FILE)
}

fn collect_admin_ui_fingerprint_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries = std::fs::read_dir(&dir)
            .with_context(|| format!("failed to read {}", dir.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                if is_ignored_admin_ui_dir(&path) {
                    continue;
                }
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }

    Ok(())
}

fn is_ignored_admin_ui_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("dist" | "node_modules")
    )
}

fn remove_stale_files(sh: &Shell, dest: &Path, expected_files: &HashSet<PathBuf>) -> Result<()> {
    if !dest.exists() {
        return Ok(());
    }

    for file in collect_files_recursive_flat(dest)? {
        let relative = file
            .strip_prefix(dest)
            .with_context(|| format!("dashboard artifact is outside {}", dest.display()))?;
        if !expected_files.contains(relative) {
            sh.remove_path(file)?;
        }
    }

    Ok(())
}

fn build_backend_variant(
    sh: &Shell,
    ctx: &BuildContext,
    dist_dir: &Path,
    backend: &Backend,
    copy_executable: bool,
) -> Result<HashSet<String>> {
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
        "cargo build --release --package sipp-gateway-server --target-dir {target_dir}"
    )
    .env("SIPP_SYS_CMAKE_OUT_DIR", &cmake_dir);
    cargo_cmd = apply_toolchains(sh, ctx, cargo_cmd, Some(backend))?;
    cargo_cmd = cargo_cmd.arg("--features").arg(cargo_features(backend));

    output::run_build_command(
        format!("Compiling gateway-server {feature} backend"),
        cargo_cmd,
    )
    .with_context(|| format!("failed to build gateway-server {feature} backend"))?;

    let mut copied_files = HashSet::new();
    if copy_executable {
        let executable = copy_gateway_executable(sh, &target_dir, dist_dir)?;
        if let Some(file_name) = executable.file_name().and_then(|name| name.to_str()) {
            copied_files.insert(file_name.to_owned());
        }
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
    copied_files.extend(summary.copied_names);

    Ok(copied_files)
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
    sh.create_dir(dist_dir)?;
    Ok(())
}

fn clean_stale_runtime_artifacts(
    sh: &Shell,
    dist_dir: &Path,
    expected_files: &HashSet<String>,
) -> Result<()> {
    if !dist_dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if is_gateway_runtime_artifact(file_name) && !expected_files.contains(file_name) {
            output::step(format!(
                "Removing stale gateway-server runtime artifact {}",
                path.display()
            ));
            sh.remove_path(path)?;
        }
    }

    Ok(())
}

fn is_gateway_runtime_artifact(file_name: &str) -> bool {
    file_name == gateway_binary_file_name() || runtime_file_kind(file_name).is_some()
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
        if !copy || !summary.copied_names.insert(file_name.to_owned()) {
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
    if dest.exists() && files_equal(source, dest)? {
        return Ok(());
    }
    if dest.exists() {
        sh.remove_path(dest)?;
    }
    sh.copy_file(source, dest)?;
    Ok(())
}

fn files_equal(left: &Path, right: &Path) -> Result<bool> {
    let left_meta =
        std::fs::metadata(left).with_context(|| format!("failed to stat {}", left.display()))?;
    let right_meta =
        std::fs::metadata(right).with_context(|| format!("failed to stat {}", right.display()))?;
    if left_meta.len() != right_meta.len() {
        return Ok(false);
    }
    let left_bytes =
        std::fs::read(left).with_context(|| format!("failed to read {}", left.display()))?;
    let right_bytes =
        std::fs::read(right).with_context(|| format!("failed to read {}", right.display()))?;
    Ok(left_bytes == right_bytes)
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

fn collect_files_recursive_flat(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_files_recursive(root, &mut files)?;
    Ok(files)
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
                | "sipp_shim.dll"
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
            "libsipp_shim.",
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
        "sipp-gateway.exe"
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
    copied_names: HashSet<String>,
}

#[derive(Clone, Copy)]
enum RuntimeFileKind {
    Base,
    Plugin,
    MetalSidecar,
}
