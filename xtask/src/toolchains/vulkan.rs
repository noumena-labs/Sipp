//! Vulkan SDK bootstrapping.

use crate::output;
use crate::utils::BuildContext;
#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::Result;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

pub(crate) const VULKAN_VERSION: &str = "1.4.350.0";

/// Ensures a hermetic Vulkan SDK is available under the build directory.
pub(crate) fn setup_vulkan(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let toolchain_root = ctx.toolchain_dir();
    let vulkan_dir = toolchain_root.join("vulkan");

    let (os_path, filename, bin_path) = if cfg!(windows) {
        (
            "windows",
            format!("vulkansdk-windows-X64-{VULKAN_VERSION}.exe"),
            vulkan_dir.join("Bin").join("glslc.exe"),
        )
    } else if cfg!(target_os = "macos") {
        (
            "mac",
            format!("vulkansdk-macos-{VULKAN_VERSION}.zip"),
            vulkan_dir.join("macOS").join("bin").join("glslc"),
        )
    } else {
        (
            "linux",
            format!("vulkansdk-linux-x86_64-{VULKAN_VERSION}.tar.xz"),
            vulkan_dir
                .join(VULKAN_VERSION)
                .join("x86_64")
                .join("bin")
                .join("glslc"),
        )
    };

    if !bin_path.exists() {
        output::phase("Vulkan SDK");
        output::detail("Version", VULKAN_VERSION);
        output::path("Install directory", &vulkan_dir);
        sh.create_dir(&vulkan_dir)?;

        let url =
            format!("https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/{os_path}/{filename}");
        let archive_path = toolchain_root.join(&filename);

        output::detail("Download", &url);

        output::run_command(
            "Downloading Vulkan SDK (~400MB)",
            cmd!(sh, "curl -f -L -o {archive_path} {url}"),
        )?;

        if cfg!(windows) {
            output::run_command(
                "Installing Vulkan SDK",
                cmd!(sh, "{archive_path} --root {vulkan_dir} --accept-licenses --default-answer --confirm-command install copy_only=1"),
            )?;
        } else if cfg!(target_os = "macos") {
            output::run_command(
                "Extracting Vulkan SDK",
                cmd!(sh, "unzip -oq {archive_path} -d {vulkan_dir}"),
            )?;
        } else {
            output::run_command(
                "Extracting Vulkan SDK",
                cmd!(sh, "tar -xf {archive_path} -C {vulkan_dir}"),
            )?;
        }

        sh.remove_path(&archive_path)?;
    } else {
        output::success(format!("Using Vulkan SDK at {}", vulkan_dir.display()));
    }

    ensure_linux_loader_symlink(&vulkan_dir)?;

    Ok(vulkan_dir)
}

#[cfg(target_os = "linux")]
fn ensure_linux_loader_symlink(vulkan_dir: &Path) -> Result<()> {
    let lib_dir = vulkan_dir.join(VULKAN_VERSION).join("x86_64").join("lib");
    let link_path = lib_dir.join("libvulkan.so");
    if link_path.exists() {
        return Ok(());
    }

    let Some(target_path) = linux_loader_target(&lib_dir)? else {
        return Ok(());
    };

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        target_path
            .file_name()
            .with_context(|| format!("invalid Vulkan loader path {}", target_path.display()))?,
        &link_path,
    )
    .with_context(|| {
        format!(
            "failed to create Vulkan loader symlink {} -> {}",
            link_path.display(),
            target_path.display()
        )
    })?;

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn ensure_linux_loader_symlink(_vulkan_dir: &Path) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn linux_loader_target(lib_dir: &Path) -> Result<Option<PathBuf>> {
    let preferred = lib_dir.join("libvulkan.so.1");
    if preferred.exists() {
        return Ok(Some(preferred));
    }

    let mut candidates = Vec::new();
    if lib_dir.exists() {
        for entry in std::fs::read_dir(lib_dir)
            .with_context(|| format!("failed to read Vulkan SDK lib dir {}", lib_dir.display()))?
        {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("libvulkan.so.") {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    Ok(candidates.into_iter().next())
}
