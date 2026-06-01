//! Vulkan SDK bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::PathBuf;
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
                cmd!(sh, "unzip -q {archive_path} -d {vulkan_dir}"),
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

    Ok(vulkan_dir)
}
