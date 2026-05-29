use crate::utils::BuildContext;
use anyhow::Result;
use std::path::PathBuf;
use xshell::{cmd, Shell};

pub fn setup_uv(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let root = ctx.workspace_root();
    let uv_dir = root.join(".build").join("toolchain").join("uv");
    let uv_exe = if cfg!(windows) {
        uv_dir.join("uv.exe")
    } else {
        uv_dir.join("uv")
    };

    if !uv_exe.exists() {
        println!("=> Bootstrapping hermetic `uv` (Python toolchain)...");
        std::fs::create_dir_all(&uv_dir)?;

        // Map OS and Architecture to Astral's release targets
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

        println!("   Downloading uv from: {url}");
        cmd!(sh, "curl -f -L -o {archive_path} {url}").run()?;

        // Extract the archive
        if ext == "zip" {
            cmd!(sh, "tar -xf {archive_path} -C {uv_dir}").run()?;
        } else {
            cmd!(sh, "tar -xzf {archive_path} -C {uv_dir}").run()?;
        }

        // The archive extracts into a subfolder (e.g., `uv-x86_64-pc-windows-msvc/uv.exe`).
        // We move the binary up to the root of our `uv` folder for easy access.
        let subfolder = uv_dir.join(format!("uv-{target}"));
        let extracted_bin_sub = subfolder.join(if cfg!(windows) { "uv.exe" } else { "uv" });

        if extracted_bin_sub.exists() {
            sh.copy_file(&extracted_bin_sub, &uv_exe)?;
            let _ = sh.remove_path(&subfolder);
        }

        // Clean up the archive and the extracted subfolder
        sh.remove_path(&archive_path)?;
        sh.remove_path(uv_dir.join(format!("uv-{target}")))?;
    }

    Ok(uv_exe)
}
