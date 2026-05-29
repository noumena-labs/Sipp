//! Ninja build tool bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::PathBuf;
use xshell::{cmd, Shell};

const NINJA_VERSION: &str = "1.13.2";

/// Ensures Ninja is available when the host platform needs a hermetic copy.
pub(crate) fn setup_ninja(sh: &Shell, ctx: &BuildContext) -> Result<Option<PathBuf>> {
    if cfg!(windows) {
        let ninja_dir = ctx.toolchain_dir().join("ninja");
        let ninja_exe = ninja_dir.join("ninja.exe");

        if !ninja_exe.exists() {
            output::phase("Ninja build tool");
            output::path("Install directory", &ninja_dir);
            sh.create_dir(&ninja_dir)?;

            let url = format!(
                "https://github.com/ninja-build/ninja/releases/download/v{}/ninja-win.zip",
                NINJA_VERSION
            );
            let zip_path = ninja_dir.join("ninja-win.zip");

            output::run_command("Downloading Ninja", cmd!(sh, "curl -L -o {zip_path} {url}"))?;
            output::run_command(
                "Extracting Ninja",
                cmd!(sh, "tar -xf {zip_path} -C {ninja_dir}"),
            )?;
            sh.remove_path(zip_path)?;
        } else {
            output::success(format!("Using Ninja at {}", ninja_exe.display()));
        }
        Ok(Some(ninja_dir))
    } else {
        output::detail("Ninja", "using host build tools");
        Ok(None)
    }
}
