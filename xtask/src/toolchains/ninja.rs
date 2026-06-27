//! Ninja build tool bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

const NINJA_VERSION: &str = "1.13.2";

/// Ensures Ninja is available when the host platform needs a hermetic copy.
pub(crate) fn setup_ninja(sh: &Shell, ctx: &BuildContext) -> Result<Option<PathBuf>> {
    let ninja_dir = ctx.toolchain_dir().join("ninja");
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

        output::run_command("Downloading Ninja", cmd!(sh, "curl -L -o {zip_path} {url}"))?;
        output::run_command(
            "Extracting Ninja",
            cmd!(sh, "tar -xf {zip_path} -C {ninja_dir}"),
        )?;
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

    let mut permissions = std::fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_unix_executable(_path: &Path) -> Result<()> {
    Ok(())
}
