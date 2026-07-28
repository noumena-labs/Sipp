//! Vulkan SDK bootstrapping.

use crate::output;
use crate::utils::BuildContext;
#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::Result;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
compile_error!("the managed Vulkan SDK supports Windows, macOS, and Linux hosts");

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/toolchains/vulkan_tests.rs"]
mod vulkan_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const VULKAN_VERSION: &str = "1.4.350.0";

#[derive(Clone, Copy)]
enum VulkanHost {
    #[cfg(any(test, target_os = "windows"))]
    Windows,
    #[cfg(any(test, target_os = "macos"))]
    Macos,
    #[cfg(any(test, target_os = "linux"))]
    Linux,
}

impl VulkanHost {
    #[cfg(target_os = "windows")]
    fn current() -> Self {
        Self::Windows
    }

    #[cfg(target_os = "macos")]
    fn current() -> Self {
        Self::Macos
    }

    #[cfg(target_os = "linux")]
    fn current() -> Self {
        Self::Linux
    }

    fn download_path(self) -> &'static str {
        match self {
            #[cfg(any(test, target_os = "windows"))]
            Self::Windows => "windows",
            #[cfg(any(test, target_os = "macos"))]
            Self::Macos => "mac",
            #[cfg(any(test, target_os = "linux"))]
            Self::Linux => "linux",
        }
    }

    fn archive_name(self) -> String {
        match self {
            #[cfg(any(test, target_os = "windows"))]
            Self::Windows => format!("vulkansdk-windows-X64-{VULKAN_VERSION}.exe"),
            #[cfg(any(test, target_os = "macos"))]
            Self::Macos => format!("vulkansdk-macos-{VULKAN_VERSION}.zip"),
            #[cfg(any(test, target_os = "linux"))]
            Self::Linux => format!("vulkansdk-linux-x86_64-{VULKAN_VERSION}.tar.xz"),
        }
    }
}

/// Platform-specific paths for one managed Vulkan SDK version.
pub(crate) struct VulkanLayout {
    install_dir: PathBuf,
    /// Directory assigned to `VULKAN_SDK`.
    pub(crate) sdk_dir: PathBuf,
    /// Directory containing Vulkan SDK command-line tools.
    pub(crate) bin_dir: PathBuf,
    /// Managed GLSL compiler used as the installation marker.
    pub(crate) glslc: PathBuf,
}

impl VulkanLayout {
    fn new(cache_dir: &Path, host: VulkanHost) -> Self {
        let install_dir = cache_dir.join(VULKAN_VERSION);
        let (sdk_dir, bin_name, compiler_name) = match host {
            #[cfg(any(test, target_os = "windows"))]
            VulkanHost::Windows => (install_dir.clone(), "Bin", "glslc.exe"),
            #[cfg(any(test, target_os = "macos"))]
            VulkanHost::Macos => (install_dir.join("macOS"), "bin", "glslc"),
            #[cfg(any(test, target_os = "linux"))]
            VulkanHost::Linux => (install_dir.join("x86_64"), "bin", "glslc"),
        };
        let bin_dir = sdk_dir.join(bin_name);
        let glslc = bin_dir.join(compiler_name);

        Self {
            install_dir,
            sdk_dir,
            bin_dir,
            glslc,
        }
    }

    /// Returns the managed Vulkan SDK layout for the current host.
    pub(crate) fn current(ctx: &BuildContext) -> Self {
        Self::new(&ctx.vulkan_dir(), VulkanHost::current())
    }

    /// Returns whether the pinned SDK compiler is installed.
    pub(crate) fn is_installed(&self) -> bool {
        self.glslc.is_file()
    }
}

/// Ensures a hermetic Vulkan SDK is available under the build directory.
pub(crate) fn setup_vulkan(sh: &Shell, ctx: &BuildContext) -> Result<VulkanLayout> {
    let host = VulkanHost::current();
    let layout = VulkanLayout::current(ctx);

    if !layout.is_installed() {
        output::phase("Vulkan SDK");
        output::detail("Version", VULKAN_VERSION);
        output::path("Install directory", &layout.install_dir);
        sh.create_dir(ctx.vulkan_dir())?;
        let tmp_dir = ctx.tmp_dir();
        sh.create_dir(&tmp_dir)?;

        let filename = host.archive_name();
        let url = format!(
            "https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/{}/{filename}",
            host.download_path()
        );
        let archive_path = tmp_dir.join(filename);

        output::detail("Download", &url);
        output::run_command(
            "Downloading Vulkan SDK (~400MB)",
            cmd!(sh, "curl -f -L -o {archive_path} {url}"),
        )?;

        sh.remove_path(&layout.install_dir)?;

        install_archive(sh, ctx, &layout, &archive_path)?;
        if !layout.is_installed() {
            anyhow::bail!(
                "Vulkan SDK {VULKAN_VERSION} did not install {}",
                layout.glslc.display()
            );
        }
        sh.remove_path(&archive_path)?;
        output::success(format!(
            "Installed Vulkan SDK at {}",
            layout.sdk_dir.display()
        ));
    } else {
        output::success(format!("Using Vulkan SDK at {}", layout.sdk_dir.display()));
    }

    #[cfg(target_os = "linux")]
    ensure_linux_loader_symlink(&layout)?;

    Ok(layout)
}

fn install_archive(
    sh: &Shell,
    _ctx: &BuildContext,
    _layout: &VulkanLayout,
    archive_path: &Path,
) -> Result<()> {
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    {
        #[cfg(target_os = "windows")]
        let installer = archive_path.to_path_buf();
        #[cfg(target_os = "macos")]
        let (installer, staging_dir) = {
            let staging_dir = _ctx
                .tmp_dir()
                .join(format!("vulkan-macos-{VULKAN_VERSION}"));
            sh.remove_path(&staging_dir)?;
            sh.create_dir(&staging_dir)?;
            output::run_command(
                "Extracting Vulkan SDK installer",
                cmd!(sh, "unzip -oq {archive_path} -d {staging_dir}"),
            )?;
            (macos_installer_path(&staging_dir), staging_dir)
        };

        let install_dir = &_layout.install_dir;
        output::run_command(
            "Installing Vulkan SDK",
            cmd!(sh, "{installer} --root {install_dir} --accept-licenses --default-answer --confirm-command install copy_only=1"),
        )?;
        #[cfg(target_os = "macos")]
        sh.remove_path(staging_dir)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    {
        let vulkan_dir = _ctx.vulkan_dir();
        output::run_command(
            "Extracting Vulkan SDK",
            cmd!(sh, "tar -xf {archive_path} -C {vulkan_dir}"),
        )
    }
}

#[cfg(any(test, target_os = "macos"))]
fn macos_installer_path(staging_dir: &Path) -> PathBuf {
    let name = format!("vulkansdk-macOS-{VULKAN_VERSION}");
    staging_dir
        .join(format!("{name}.app"))
        .join("Contents")
        .join("MacOS")
        .join(name)
}

#[cfg(target_os = "linux")]
fn ensure_linux_loader_symlink(layout: &VulkanLayout) -> Result<()> {
    let lib_dir = layout.sdk_dir.join("lib");
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
