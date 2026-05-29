use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::PathBuf;
use xshell::{cmd, Shell};

/// Ensures the hermetic uv executable is available under `.build/toolchain`.
pub(crate) fn setup_uv(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let root = ctx.workspace_root();
    let uv_dir = root.join(".build").join("toolchain").join("uv");
    let uv_exe = if cfg!(windows) {
        uv_dir.join("uv.exe")
    } else {
        uv_dir.join("uv")
    };

    if !uv_exe.exists() {
        output::phase("Python uv toolchain");
        output::path("Install directory", &uv_dir);
        std::fs::create_dir_all(&uv_dir)?;

        let (target, ext) = if cfg!(target_os = "windows") {
            ("x86_64-pc-windows-msvc", "zip")
        } else if cfg!(target_os = "macos") {
            if cfg!(target_arch = "aarch64") {
                ("aarch64-apple-darwin", "tar.gz")
            } else {
                ("x86_64-apple-darwin", "tar.gz")
            }
        } else {
            ("x86_64-unknown-linux-gnu", "tar.gz")
        };

        let url =
            format!("https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.{ext}");
        let archive_path = uv_dir.join(format!("uv.{ext}"));

        output::detail("Download", &url);
        output::run_command(
            "Downloading uv",
            cmd!(sh, "curl -f -L -o {archive_path} {url}"),
        )?;

        if ext == "zip" {
            output::run_command(
                "Extracting uv",
                cmd!(sh, "tar -xf {archive_path} -C {uv_dir}"),
            )?;
        } else {
            output::run_command(
                "Extracting uv",
                cmd!(sh, "tar -xzf {archive_path} -C {uv_dir}"),
            )?;
        }

        let subfolder = uv_dir.join(format!("uv-{target}"));
        let extracted_bin_sub = subfolder.join(if cfg!(windows) { "uv.exe" } else { "uv" });
        if !extracted_bin_sub.exists() {
            anyhow::bail!("uv archive did not contain {}", extracted_bin_sub.display());
        }

        sh.copy_file(&extracted_bin_sub, &uv_exe)
            .with_context(|| format!("failed to stage uv at {}", uv_exe.display()))?;

        sh.remove_path(&archive_path)?;
        sh.remove_path(subfolder)?;
        output::success(format!("Installed uv at {}", uv_exe.display()));
    } else {
        output::success(format!("Using uv at {}", uv_exe.display()));
    }

    Ok(uv_exe)
}
