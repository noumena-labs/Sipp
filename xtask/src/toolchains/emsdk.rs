//! Emscripten SDK bootstrapping and command wrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::thread;
use std::time::Duration;
use xshell::{cmd, Cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/toolchains/emsdk_tests.rs"]
mod emsdk_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const EMSDK_VERSION: &str = "4.0.23";

/// Ensures the configured Emscripten SDK is cloned, installed, and active.
pub(crate) fn setup_emsdk(sh: &Shell, ctx: &BuildContext) -> Result<std::path::PathBuf> {
    let toolchain_root = ctx.toolchain_dir();
    let emsdk_dir = toolchain_root.join("emsdk");

    if !emsdk_dir.exists() {
        output::phase("Emscripten SDK");
        output::path("Install directory", &emsdk_dir);
        std::fs::create_dir_all(&toolchain_root)?;
        let _toolchain_dir = sh.push_dir(&toolchain_root);
        output::run_command(
            "Cloning emsdk",
            cmd!(
                sh,
                "git clone https://github.com/emscripten-core/emsdk.git emsdk"
            ),
        )?;
    } else {
        output::success(format!("Using emsdk at {}", emsdk_dir.display()));
    }

    let _dir = sh.push_dir(&emsdk_dir);
    output::detail("Emscripten version", EMSDK_VERSION);

    if cfg!(windows) {
        patch_emsdk_windows(&emsdk_dir)?;
        install_emsdk_windows(sh, &emsdk_dir)?;
        activate_emsdk_windows(sh, &emsdk_dir)?;
    } else {
        output::run_command(
            format!("Installing emsdk {EMSDK_VERSION}"),
            cmd!(sh, "bash -c").arg(format!("./emsdk install {EMSDK_VERSION}")),
        )?;
        output::run_command(
            format!("Activating emsdk {EMSDK_VERSION}"),
            cmd!(sh, "bash -c").arg(format!("./emsdk activate {EMSDK_VERSION}")),
        )?;
    }

    Ok(emsdk_dir)
}

/// Runs a command after loading the Emscripten environment.
pub(crate) fn run_with_emsdk(
    sh: &Shell,
    emsdk_dir: &Path,
    ninja_dir: Option<&Path>,
    label: &str,
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
        let result = output::run_build_command(
            label,
            clean_windows_emsdk_env(cmd!(sh, "cmd.exe /c {temp_script}")),
        );

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
        output::run_build_command(label, cmd!(sh, "bash -c").arg(full_cmd))?;
    }
    Ok(())
}

fn install_emsdk_windows(sh: &Shell, emsdk_dir: &Path) -> Result<()> {
    if emsdk_is_installed(emsdk_dir)? {
        output::success(format!("Using installed emsdk {EMSDK_VERSION}"));
        return Ok(());
    }

    let mut attempts = 0;
    let max_attempts = 5;

    loop {
        attempts += 1;
        let result = output::run_command(
            format!("Installing emsdk {EMSDK_VERSION}"),
            clean_windows_emsdk_env(
                cmd!(sh, "cmd.exe /c emsdk.bat install {EMSDK_VERSION}").env("EMSDK_USE_CURL", "1"),
            ),
        );

        if result.is_ok() {
            return Ok(());
        }

        if attempts >= max_attempts {
            anyhow::bail!(
                "emsdk install failed after {max_attempts} attempts. Please check your network connection."
            );
        }

        output::warning(format!(
            "Download truncated or locked by Windows Defender; retrying ({attempts}/{max_attempts})"
        ));
        thread::sleep(Duration::from_secs(2));
    }
}

fn activate_emsdk_windows(sh: &Shell, emsdk_dir: &Path) -> Result<()> {
    if emsdk_is_active(emsdk_dir)? {
        output::success(format!("Using active emsdk {EMSDK_VERSION}"));
        return Ok(());
    }

    output::run_command(
        format!("Activating emsdk {EMSDK_VERSION}"),
        clean_windows_emsdk_env(cmd!(sh, "cmd.exe /c emsdk.bat activate {EMSDK_VERSION}")),
    )?;
    Ok(())
}

fn emsdk_is_active(emsdk_dir: &Path) -> Result<bool> {
    Ok(emsdk_is_installed(emsdk_dir)? && emsdk_dir.join(".emscripten").exists())
}

fn emsdk_is_installed(emsdk_dir: &Path) -> Result<bool> {
    let version_file = emsdk_dir.join("upstream").join(".emsdk_version");
    let Ok(actual_tool_id) = std::fs::read_to_string(&version_file) else {
        return Ok(false);
    };

    if actual_tool_id.trim() != expected_emsdk_tool_id(emsdk_dir)? {
        return Ok(false);
    }

    let emscripten_dir = emsdk_dir.join("upstream").join("emscripten");
    let required_paths = [
        emscripten_dir.join("emcc.bat"),
        emscripten_dir.join("emcmake.bat"),
        emscripten_dir.join("emmake.bat"),
        emsdk_dir.join("node"),
        emsdk_dir.join("python"),
    ];

    Ok(required_paths.iter().all(|path| path.exists()))
}

fn expected_emsdk_tool_id(emsdk_dir: &Path) -> Result<String> {
    let tags_path = emsdk_dir.join("emscripten-releases-tags.json");
    let tags = std::fs::read_to_string(&tags_path)
        .with_context(|| format!("failed to read {}", tags_path.display()))?;
    let tags: serde_json::Value = serde_json::from_str(&tags)
        .with_context(|| format!("failed to parse {}", tags_path.display()))?;
    let release_hash = tags
        .get("releases")
        .and_then(|releases| releases.get(EMSDK_VERSION))
        .and_then(serde_json::Value::as_str)
        .with_context(|| format!("emsdk release {EMSDK_VERSION} is not listed"))?;

    Ok(format!("releases-{release_hash}-64bit"))
}

fn patch_emsdk_windows(emsdk_dir: &Path) -> Result<()> {
    const OLD: &str = "\
# platform.machine() may return AMD64 on windows, so standardize the case.
machine = os.getenv('EMSDK_ARCH', platform.machine().lower())
";
    const NEW: &str = "\
# platform.machine() may return AMD64 on windows, so standardize the case.
machine = os.getenv('EMSDK_ARCH')
if not machine:
  machine = platform.machine().lower()
";

    let path = emsdk_dir.join("emsdk.py");
    let original = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let source = original.replace("\r\n", "\n").replace('\r', "\n");

    let patched = if source.contains(NEW) {
        source
    } else if source.contains(OLD) {
        source.replace(OLD, NEW)
    } else {
        bail!("emsdk.py Windows platform detection patch target was not found");
    };

    if patched == original {
        return Ok(());
    }

    std::fs::write(&path, patched)
        .with_context(|| format!("failed to patch {}", path.display()))?;
    output::success("Patched emsdk Windows platform detection");
    Ok(())
}

fn clean_windows_emsdk_env<'a>(cmd: Cmd<'a>) -> Cmd<'a> {
    // Avoid host shell state and WMI-backed OS detection inside emsdk.py.
    cmd.env("EMSDK_OS", "windows")
        .env("EMSDK_ARCH", "x86_64")
        .env_remove("SHELL")
        .env_remove("MSYSTEM")
        .env_remove("EMSDK")
        .env_remove("EMSDK_PYTHON")
        .env_remove("EM_CONFIG")
        .env_remove("EMSCRIPTEN")
        .env_remove("EMCC")
        .env_remove("EMXX")
        .env_remove("EMAR")
        .env_remove("EMRANLIB")
        .env_remove("EMCMAKE")
        .env_remove("EMMAKE")
        .env_remove("EMSDK_NODE")
        .env_remove("NODE_JS")
        .env_remove("BINARYEN_ROOT")
        .env_remove("LLVM_ROOT")
        .env_remove("EM_CACHE")
}
