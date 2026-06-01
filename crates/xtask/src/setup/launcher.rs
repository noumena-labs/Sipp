//! Cached launcher installer for the short `clm` developer command.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::path::Path;
use xshell::Shell;

const UNIX_LAUNCHER: &str = r#"#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
TARGET="$ROOT/.build/xtask/debug/xtask"
STAMP="$ROOT/.build/xtask/clm.stamp"

needs_build=0
if [ ! -x "$TARGET" ] || [ ! -f "$STAMP" ]; then
  needs_build=1
elif find "$ROOT/crates/xtask/src" "$ROOT/crates/xtask/Cargo.toml" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" "$ROOT/.cargo/config.toml" -newer "$STAMP" -print -quit 2>/dev/null | grep -q .; then
  needs_build=1
fi

if [ "$needs_build" = "1" ]; then
  (cd "$ROOT" && cargo build --target-dir .build/xtask --package xtask --quiet) || exit $?
  mkdir -p "$(dirname "$STAMP")"
  : > "$STAMP"
fi

exec "$TARGET" "$@"
"#;

const WINDOWS_CMD_LAUNCHER: &str = r#"@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0clm.ps1" %*
exit /b %ERRORLEVEL%
"#;

const WINDOWS_PS_LAUNCHER: &str = r#"$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptRoot "..\..")
$Target = Join-Path $Root ".build\xtask\debug\xtask.exe"
$Stamp = Join-Path $Root ".build\xtask\clm.stamp"

$SourceRoots = @(
  (Join-Path $Root "crates\xtask\src"),
  (Join-Path $Root "crates\xtask\Cargo.toml"),
  (Join-Path $Root "Cargo.toml"),
  (Join-Path $Root "Cargo.lock"),
  (Join-Path $Root ".cargo\config.toml")
)

$SourceFiles = @()
foreach ($SourceRoot in $SourceRoots) {
  if (Test-Path $SourceRoot) {
    $Item = Get-Item $SourceRoot
    if ($Item.PSIsContainer) {
      $SourceFiles += Get-ChildItem $SourceRoot -Recurse -File -Include *.rs
    } else {
      $SourceFiles += $Item
    }
  }
}

$NeedsBuild = !(Test-Path $Target) -or !(Test-Path $Stamp)
if (!$NeedsBuild) {
  $StampTime = (Get-Item $Stamp).LastWriteTimeUtc
  foreach ($SourceFile in $SourceFiles) {
    if ($SourceFile.LastWriteTimeUtc -gt $StampTime) {
      $NeedsBuild = $true
      break
    }
  }
}

if ($NeedsBuild) {
  Push-Location $Root
  try {
    cargo build --target-dir .build/xtask --package xtask --quiet
    if ($LASTEXITCODE -ne 0) {
      exit $LASTEXITCODE
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Stamp) | Out-Null
    Set-Content -Path $Stamp -Value "built $(Get-Date -Format o)"
  } finally {
    Pop-Location
  }
}

& $Target @args
exit $LASTEXITCODE
"#;

/// Installs the repo-local `clm` launcher scripts under `.build/bin`.
pub(crate) fn install(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let bin_dir = ctx.launcher_bin_dir();
    sh.create_dir(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;

    write_launcher(&bin_dir.join("clm"), &UNIX_LAUNCHER.replace("\r\n", "\n"))?;
    write_launcher(&bin_dir.join("clm.cmd"), WINDOWS_CMD_LAUNCHER)?;
    write_launcher(&bin_dir.join("clm.ps1"), WINDOWS_PS_LAUNCHER)?;
    write_launcher(&bin_dir.join("cogentlm-env.sh"), &unix_env_script(&bin_dir))?;
    write_launcher(
        &bin_dir.join("cogentlm-env.ps1"),
        &powershell_env_script(&bin_dir),
    )?;
    write_launcher(&bin_dir.join("cogentlm-env.cmd"), &cmd_env_script(&bin_dir))?;
    make_unix_launcher_executable(&bin_dir.join("clm"))?;
    make_unix_launcher_executable(&bin_dir.join("cogentlm-env.sh"))?;

    output::success("Installed clm launcher");
    output::path("Launcher directory", &bin_dir);
    output::detail("Use", "clm build core");
    print_activation_hint(&bin_dir);
    Ok(())
}

fn write_launcher(path: &Path, contents: &str) -> Result<()> {
    std::fs::write(path, contents)
        .with_context(|| format!("failed to write launcher {}", path.display()))
}

fn print_activation_hint(bin_dir: &Path) {
    if path_contains(bin_dir) {
        output::success("clm is active in this terminal session");
        return;
    }

    if cfg!(windows) {
        output::detail(
            "Activate PowerShell",
            format!(". {}", powershell_quote(&bin_dir.join("cogentlm-env.ps1"))),
        );
        output::detail(
            "Activate CMD",
            format!("call {}", cmd_quote(&bin_dir.join("cogentlm-env.cmd"))),
        );
    } else {
        output::detail(
            "Activate",
            format!("source {}", shell_quote(&bin_dir.join("cogentlm-env.sh"))),
        );
    }
}

fn path_contains(dir: &Path) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|entry| path_eq(&entry, dir))
}

fn path_eq(left: &Path, right: &Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => path_string_eq(&left, &right),
        _ => path_string_eq(left, right),
    }
}

fn path_string_eq(left: &Path, right: &Path) -> bool {
    let left = left.display().to_string();
    let right = right.display().to_string();
    if cfg!(windows) {
        left.eq_ignore_ascii_case(&right)
    } else {
        left == right
    }
}

fn unix_env_script(bin_dir: &Path) -> String {
    // Utilize the updated shell_quote that handles the drive letter conversion
    let bin_dir_quoted = shell_quote(bin_dir);

    format!(
        r#"#!/usr/bin/env sh
COGENTLM_BIN={bin_dir_quoted}
case ":${{PATH:-}}:" in
  *:"$COGENTLM_BIN":*) ;;
  *) export PATH="$COGENTLM_BIN${{PATH:+:$PATH}}" ;;
esac
"#
    )
    .replace("\r\n", "\n") // Force Unix line endings
}

fn powershell_env_script(bin_dir: &Path) -> String {
    let bin_dir = powershell_quote_str(&bin_dir.display().to_string());
    format!(
        r#"$CogentLmBin = {bin_dir}
$PathParts = @()
if ($env:Path) {{
  $PathParts = $env:Path -split [System.IO.Path]::PathSeparator
}}
if ($PathParts -notcontains $CogentLmBin) {{
  $env:Path = "$CogentLmBin$([System.IO.Path]::PathSeparator)$env:Path"
}}
"#
    )
}

fn cmd_env_script(bin_dir: &Path) -> String {
    let bin_dir = bin_dir.display().to_string().replace('%', "%%");
    format!(
        r#"@echo off
set "COGENTLM_BIN={bin_dir}"
echo ;%PATH%; | find /I ";%COGENTLM_BIN%;" >nul
if errorlevel 1 set "PATH=%COGENTLM_BIN%;%PATH%"
"#
    )
}

fn shell_quote(path: &Path) -> String {
    let mut path_str = path.display().to_string().replace('\\', "/");

    // Convert Windows drive letters (e.g., "D:/...") to Git Bash/MSYS paths (e.g., "/d/...")
    let bytes = path_str.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        path_str = format!("/{drive}{}", &path_str[2..]);
    }

    shell_quote_str(&path_str)
}

fn shell_quote_str(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn powershell_quote(path: &Path) -> String {
    powershell_quote_str(&path.display().to_string())
}

fn powershell_quote_str(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn cmd_quote(path: &Path) -> String {
    format!("\"{}\"", path.display())
}

#[cfg(unix)]
fn make_unix_launcher_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = std::fs::metadata(path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions)
        .with_context(|| format!("failed to mark {} executable", path.display()))
}

#[cfg(not(unix))]
fn make_unix_launcher_executable(_path: &Path) -> Result<()> {
    Ok(())
}
