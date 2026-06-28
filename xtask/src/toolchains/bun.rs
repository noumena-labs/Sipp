//! Bun JavaScript runtime bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/toolchains/bun_tests.rs"]
mod bun_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) const BUN_VERSION: &str = "1.3.14";

/// Ensures the hermetic Bun executable is available under `.build/toolchain`.
pub(crate) fn setup_bun(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let bun_dir = ctx.bun_toolchain_dir();
    let bun_exe = ctx.bun_exe();

    if !bun_exe.exists() || !bun_version_matches(&bun_exe) {
        output::phase("Bun JavaScript toolchain");
        output::detail("Version", BUN_VERSION);
        output::path("Install directory", &bun_dir);
        sh.create_dir(&bun_dir)?;

        let target = bun_target()?;
        let filename = format!("{target}.zip");
        let url = format!(
            "https://github.com/oven-sh/bun/releases/download/bun-v{BUN_VERSION}/{filename}"
        );
        let archive_path = bun_dir.join(&filename);
        if bun_exe.exists() {
            sh.remove_path(&bun_exe)?;
        }

        output::detail("Download", &url);
        output::run_command(
            "Downloading Bun",
            cmd!(sh, "curl -f -L -o {archive_path} {url}"),
        )?;

        if cfg!(windows) {
            output::run_command(
                "Extracting Bun",
                cmd!(sh, "tar -xf {archive_path} -C {bun_dir}"),
            )?;
        } else {
            output::run_command(
                "Extracting Bun",
                cmd!(sh, "unzip -oq {archive_path} -d {bun_dir}"),
            )?;
        }

        let nested_bun_exe = bun_dir.join(target).join(bun_executable_name());
        if nested_bun_exe.exists() {
            sh.copy_file(&nested_bun_exe, &bun_exe)
                .with_context(|| format!("failed to stage Bun at {}", bun_exe.display()))?;
            make_unix_executable(&bun_exe)?;
        } else if !bun_exe.exists() {
            anyhow::bail!(
                "Bun archive contained neither {} nor {}",
                nested_bun_exe.display(),
                bun_exe.display()
            );
        }

        sh.remove_path(&archive_path)?;
        let extracted_dir = bun_dir.join(target);
        if extracted_dir.exists() {
            sh.remove_path(extracted_dir)?;
        }
        output::success(format!("Installed Bun at {}", bun_exe.display()));
    } else {
        output::success(format!("Using Bun at {}", bun_exe.display()));
    }

    prepend_to_path(&bun_dir)?;

    Ok(bun_exe)
}

pub(crate) fn bun_version_matches(path: &Path) -> bool {
    bun_version(path).as_deref() == Some(BUN_VERSION)
}

pub(crate) fn bun_version(path: &Path) -> Option<String> {
    let output = Command::new(path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn bun_target() -> Result<&'static str> {
    if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
        Ok("bun-windows-x64")
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        Ok("bun-darwin-aarch64")
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        Ok("bun-darwin-x64")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        Ok("bun-linux-aarch64")
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        Ok("bun-linux-x64")
    } else {
        anyhow::bail!("managed Bun is not available for this host platform")
    }
}

fn bun_executable_name() -> &'static str {
    if cfg!(windows) {
        "bun.exe"
    } else {
        "bun"
    }
}

fn prepend_to_path(dir: &Path) -> Result<()> {
    let current = env::var_os("PATH").unwrap_or_default();
    let mut paths = env::split_paths(&current).collect::<Vec<_>>();
    if !paths.iter().any(|path| path == dir) {
        paths.insert(0, dir.to_path_buf());
        let joined = env::join_paths(paths).context("failed to update PATH for managed Bun")?;
        env::set_var("PATH", joined);
    }
    Ok(())
}

#[cfg(unix)]
fn make_unix_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)
        .with_context(|| format!("failed to mark {} executable", path.display()))
}

#[cfg(not(unix))]
fn make_unix_executable(_path: &Path) -> Result<()> {
    Ok(())
}
