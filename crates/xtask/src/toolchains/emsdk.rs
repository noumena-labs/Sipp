//! Emscripten SDK bootstrapping and command wrapping.

use crate::utils::BuildContext;
use anyhow::Result;
use std::path::Path;
use std::thread;
use std::time::Duration;
use xshell::{cmd, Shell};

const EMSDK_VERSION: &str = "4.0.23";

/// Ensures the configured Emscripten SDK is cloned, installed, and active.
pub(crate) fn setup_emsdk(sh: &Shell, ctx: &BuildContext) -> Result<std::path::PathBuf> {
    let toolchain_root = ctx.toolchain_dir();
    let emsdk_dir = toolchain_root.join("emsdk");

    if !emsdk_dir.exists() {
        println!("=> Bootstrapping hermetic Emscripten toolchain...");
        std::fs::create_dir_all(&toolchain_root)?;
        let _toolchain_dir = sh.push_dir(&toolchain_root);
        cmd!(
            sh,
            "git clone https://github.com/emscripten-core/emsdk.git emsdk"
        )
        .run()?;
    }

    let _dir = sh.push_dir(&emsdk_dir);
    println!("=> Activating emsdk v{EMSDK_VERSION}...");

    if cfg!(windows) {
        install_emsdk_windows(sh)?;
        cmd!(sh, "cmd.exe /c emsdk.bat activate {EMSDK_VERSION}")
            .env_remove("SHELL")
            .env_remove("MSYSTEM")
            .run()?;
    } else {
        cmd!(sh, "bash -c")
            .arg(format!("./emsdk install {EMSDK_VERSION}"))
            .run()?;
        cmd!(sh, "bash -c")
            .arg(format!("./emsdk activate {EMSDK_VERSION}"))
            .run()?;
    }

    Ok(emsdk_dir)
}

/// Runs a command after loading the Emscripten environment.
pub(crate) fn run_with_emsdk(
    sh: &Shell,
    emsdk_dir: &Path,
    ninja_dir: Option<&Path>,
    command: &str,
) -> Result<()> {
    if cfg!(windows) {
        let bat = emsdk_dir.join("emsdk_env.bat");
        let temp_script = sh.current_dir().join(".run_emsdk_wrapper.bat");

        let path_injection = if let Some(ninja_dir) = ninja_dir {
            format!("set PATH={};%PATH%\r\n", ninja_dir.display())
        } else {
            String::new()
        };

        let emcmake = emsdk_dir
            .join("upstream")
            .join("emscripten")
            .join("emcmake.bat");
        let emmake = emsdk_dir
            .join("upstream")
            .join("emscripten")
            .join("emmake.bat");

        let script_content = format!(
            "@echo off\r\n\
            call \"{}\"\r\n\
            {}\
            set EMCMAKE={}\r\n\
            set EMMAKE={}\r\n\
            {}\r\n",
            bat.display(),
            path_injection,
            emcmake.display(),
            emmake.display(),
            command
        );

        sh.write_file(&temp_script, &script_content)?;
        let result = cmd!(sh, "cmd.exe /c {temp_script}")
            .env_remove("SHELL")
            .env_remove("MSYSTEM")
            .run();

        let _ = sh.remove_path(&temp_script);
        result?;
    } else {
        let script = emsdk_dir.join("emsdk_env.sh").display().to_string();
        let emcmake = emsdk_dir
            .join("upstream")
            .join("emscripten")
            .join("emcmake");
        let emmake = emsdk_dir.join("upstream").join("emscripten").join("emmake");

        let full_cmd = format!(
            "source \"{}\" && export EMCMAKE=\"{}\" && export EMMAKE=\"{}\" && {}",
            script,
            emcmake.display(),
            emmake.display(),
            command
        );
        cmd!(sh, "bash -c").arg(full_cmd).run()?;
    }
    Ok(())
}

fn install_emsdk_windows(sh: &Shell) -> Result<()> {
    let mut attempts = 0;
    let max_attempts = 5;

    loop {
        attempts += 1;
        let result = cmd!(sh, "cmd.exe /c emsdk.bat install {EMSDK_VERSION}")
            .env("EMSDK_USE_CURL", "1")
            .env_remove("SHELL")
            .env_remove("MSYSTEM")
            .run();

        if result.is_ok() {
            return Ok(());
        }

        if attempts >= max_attempts {
            anyhow::bail!(
                "emsdk install failed after {max_attempts} attempts. Please check your network connection."
            );
        }

        println!(
            "   Download truncated or locked by Windows Defender. Retrying ({attempts}/{max_attempts})..."
        );
        thread::sleep(Duration::from_secs(2));
    }
}
