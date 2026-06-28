//! CMake bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

pub(crate) const CMAKE_VERSION: &str = "3.30.5";

/// Ensures a hermetic CMake install is available under `.build/toolchain`.
pub(crate) fn setup_cmake(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let cmake_dir = ctx.cmake_toolchain_dir();
    let cmake_bin_dir = ctx.cmake_bin_dir()?;
    let cmake_exe = ctx.cmake_exe()?;

    if !cmake_exe.exists() {
        output::phase("CMake build tool");
        output::detail("Version", CMAKE_VERSION);
        output::path("Install directory", &cmake_dir);
        sh.create_dir(&cmake_dir)?;

        let filename = cmake_filename()?;
        let url = format!(
            "https://github.com/Kitware/CMake/releases/download/v{CMAKE_VERSION}/{filename}"
        );
        let archive_path = cmake_dir.join(filename);

        output::detail("Download", &url);
        output::run_command(
            "Downloading CMake",
            cmd!(sh, "curl -f -L -o {archive_path} {url}"),
        )?;

        if cfg!(windows) {
            output::run_command(
                "Extracting CMake",
                cmd!(sh, "tar -xf {archive_path} -C {cmake_dir}"),
            )?;
        } else {
            output::run_command(
                "Extracting CMake",
                cmd!(sh, "tar -xzf {archive_path} -C {cmake_dir}"),
            )?;
        }
        sh.remove_path(archive_path)?;
    } else {
        output::success(format!("Using CMake at {}", cmake_exe.display()));
    }

    Ok(cmake_bin_dir)
}

fn cmake_filename() -> Result<String> {
    let platform = cmake_archive_platform()?;
    let extension = if cfg!(windows) { "zip" } else { "tar.gz" };
    Ok(format!("cmake-{CMAKE_VERSION}-{platform}.{extension}"))
}

pub(crate) fn cmake_bin_dir(cmake_dir: &Path) -> Result<PathBuf> {
    let install_dir = cmake_install_dir(cmake_dir)?;
    if cfg!(target_os = "macos") {
        Ok(install_dir.join("CMake.app").join("Contents").join("bin"))
    } else {
        Ok(install_dir.join("bin"))
    }
}

pub(crate) fn cmake_install_dir(cmake_dir: &Path) -> Result<PathBuf> {
    Ok(cmake_dir.join(format!(
        "cmake-{CMAKE_VERSION}-{}",
        cmake_archive_platform()?
    )))
}

pub(crate) fn cmake_archive_platform() -> Result<&'static str> {
    if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        Ok("windows-x86_64")
    } else if cfg!(target_os = "macos") {
        Ok("macos-universal")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        Ok("linux-x86_64")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        Ok("linux-aarch64")
    } else {
        anyhow::bail!("managed CMake is not available for this host platform")
    }
}
