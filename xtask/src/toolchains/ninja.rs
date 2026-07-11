//! Ninja build tool bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

const NINJA_VERSION: &str = "1.13.2";

/// Ensures Ninja is available when the host platform needs a hermetic copy.
pub(crate) fn setup_ninja(sh: &Shell, ctx: &BuildContext) -> Result<Option<PathBuf>> {
    let ninja_dir = ctx.ninja_toolchain_dir();
    let ninja_exe = ctx.ninja_exe();

    if !ninja_exe.exists() {
        output::phase("Ninja build tool");
        output::path("Install directory", &ninja_dir);
        sh.create_dir(&ninja_dir)?;

        let filename = ninja_filename()?;
        let url = format!(
            "https://github.com/ninja-build/ninja/releases/download/v{NINJA_VERSION}/{filename}"
        );
        let zip_path = ninja_dir.join(filename);

        output::run_command(
            "Downloading Ninja",
            cmd!(sh, "curl -f -L -o {zip_path} {url}"),
        )?;
        if cfg!(windows) {
            output::run_command(
                "Extracting Ninja",
                cmd!(sh, "tar -xf {zip_path} -C {ninja_dir}"),
            )?;
        } else {
            output::run_command(
                "Extracting Ninja",
                cmd!(sh, "unzip -oq {zip_path} -d {ninja_dir}"),
            )?;
        }
        if !ninja_exe.exists() {
            anyhow::bail!(
                "Ninja archive did not contain expected executable {}",
                ninja_exe.display()
            );
        }
        make_unix_executable(&ninja_exe)?;
        sh.remove_path(zip_path)?;
    } else {
        output::success(format!("Using Ninja at {}", ninja_exe.display()));
    }

    Ok(Some(ninja_dir))
}

fn ninja_filename() -> Result<&'static str> {
    if cfg!(windows) {
        Ok("ninja-win.zip")
    } else if cfg!(target_os = "macos") {
        Ok("ninja-mac.zip")
    } else if cfg!(target_os = "linux") {
        Ok("ninja-linux.zip")
    } else {
        anyhow::bail!("managed Ninja is not available for this host platform")
    }
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
