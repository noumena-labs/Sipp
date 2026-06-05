use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::path::PathBuf;
use xshell::{cmd, Cmd, Shell};

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
        let nested_uv_exe = subfolder.join(if cfg!(windows) { "uv.exe" } else { "uv" });
        if nested_uv_exe.exists() {
            sh.copy_file(&nested_uv_exe, &uv_exe)
                .with_context(|| format!("failed to stage uv at {}", uv_exe.display()))?;
        } else if !uv_exe.exists() {
            anyhow::bail!(
                "uv archive contained neither {} nor {}",
                nested_uv_exe.display(),
                uv_exe.display()
            );
        }

        sh.remove_path(&archive_path)?;
        if subfolder.exists() {
            sh.remove_path(subfolder)?;
        }
        output::success(format!("Installed uv at {}", uv_exe.display()));
    } else {
        output::success(format!("Using uv at {}", uv_exe.display()));
    }

    Ok(uv_exe)
}

/// Applies workspace-local uv cache paths so commands do not depend on user cache state.
pub(crate) fn apply_uv_env<'a>(ctx: &BuildContext, command: Cmd<'a>) -> Cmd<'a> {
    command
        .env("UV_CACHE_DIR", ctx.build_root().join("uv-cache"))
        .env("UV_PYTHON_INSTALL_DIR", ctx.toolchain_dir().join("python"))
        .env("UV_TOOL_DIR", ctx.toolchain_dir().join("uv-tools"))
        .env("UV_TOOL_BIN_DIR", ctx.toolchain_dir().join("uv-tool-bin"))
}
