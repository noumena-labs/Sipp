//! Cached launcher installer for the short `sipp` developer command.

use crate::output;
use crate::toolchains::cmake;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::env;
use std::path::Path;
use xshell::Shell;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/setup/launcher_tests.rs"]
mod launcher_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const UNIX_LAUNCHER: &str = r#"#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
TARGET="$ROOT/.build/xtask/debug/xtask"
STAMP="$ROOT/.build/xtask/sipp.stamp"

needs_build=0
if [ ! -x "$TARGET" ] || [ ! -f "$STAMP" ]; then
  needs_build=1
elif find "$ROOT/xtask/src" "$ROOT/xtask/Cargo.toml" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" "$ROOT/.cargo/config.toml" -newer "$STAMP" -print -quit 2>/dev/null | grep -q .; then
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
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sipp.ps1" %*
exit /b %ERRORLEVEL%
"#;

const WINDOWS_PS_LAUNCHER: &str = r#"$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptRoot "..\..")
$Target = Join-Path $Root ".build\xtask\debug\xtask.exe"
$Stamp = Join-Path $Root ".build\xtask\sipp.stamp"

$SourceRoots = @(
  (Join-Path $Root "xtask\src"),
  (Join-Path $Root "xtask\Cargo.toml"),
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

/// Installs the repo-local `sipp` launcher scripts under `.build/bin`.
pub(crate) fn install(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let bin_dir = ctx.launcher_bin_dir();
    sh.create_dir(&bin_dir)
        .with_context(|| format!("failed to create {}", bin_dir.display()))?;

    write_launcher(&bin_dir.join("sipp"), &UNIX_LAUNCHER.replace("\r\n", "\n"))?;
    write_launcher(&bin_dir.join("sipp.cmd"), WINDOWS_CMD_LAUNCHER)?;
    write_launcher(&bin_dir.join("sipp.ps1"), WINDOWS_PS_LAUNCHER)?;
    write_launcher(&bin_dir.join("sipp-env.sh"), &unix_env_script(&bin_dir)?)?;
    write_launcher(
        &bin_dir.join("sipp-env.ps1"),
        &powershell_env_script(&bin_dir)?,
    )?;
    write_launcher(&bin_dir.join("sipp-env.cmd"), &cmd_env_script(&bin_dir)?)?;
    make_unix_launcher_executable(&bin_dir.join("sipp"))?;
    make_unix_launcher_executable(&bin_dir.join("sipp-env.sh"))?;

    output::success("Installed sipp launcher");
    output::path("Launcher directory", &bin_dir);
    output::detail("Use", "sipp build core");
    print_activation_hint(&bin_dir);
    Ok(())
}

fn write_launcher(path: &Path, contents: &str) -> Result<()> {
    std::fs::write(path, contents)
        .with_context(|| format!("failed to write launcher {}", path.display()))
}

fn print_activation_hint(bin_dir: &Path) {
    if path_contains(bin_dir) {
        output::success("sipp is active in this terminal session");
        return;
    }

    if cfg!(windows) {
        output::detail(
            "Activate PowerShell",
            format!(". {}", powershell_quote(&bin_dir.join("sipp-env.ps1"))),
        );
        output::detail(
            "Activate CMD",
            format!("call {}", cmd_quote(&bin_dir.join("sipp-env.cmd"))),
        );
    } else {
        output::detail(
            "Activate",
            format!("source {}", shell_quote(&bin_dir.join("sipp-env.sh"))),
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

fn unix_env_script(bin_dir: &Path) -> Result<String> {
    let bin_dir_quoted = shell_quote(bin_dir);
    let bun_dir_quoted = shell_quote(&toolchain_sibling_dir(bin_dir, "bun"));
    let ninja_dir_quoted = shell_quote(&toolchain_sibling_dir(bin_dir, "ninja"));
    let cmake_bin_dir_quoted = shell_quote(&cmake_bin_dir(bin_dir)?);

    Ok(format!(
        r#"#!/usr/bin/env sh
SIPP_BIN={bin_dir_quoted}
SIPP_BUN={bun_dir_quoted}
SIPP_NINJA={ninja_dir_quoted}
SIPP_CMAKE_BIN={cmake_bin_dir_quoted}
case ":${{PATH:-}}:" in
  *:"$SIPP_BIN":*) ;;
  *) export PATH="$SIPP_BIN${{PATH:+:$PATH}}" ;;
esac
case ":${{PATH:-}}:" in
  *:"$SIPP_BUN":*) ;;
  *) export PATH="$SIPP_BUN${{PATH:+:$PATH}}" ;;
esac
case ":${{PATH:-}}:" in
  *:"$SIPP_NINJA":*) ;;
  *) export PATH="$SIPP_NINJA${{PATH:+:$PATH}}" ;;
esac
case ":${{PATH:-}}:" in
  *:"$SIPP_CMAKE_BIN":*) ;;
  *) export PATH="$SIPP_CMAKE_BIN${{PATH:+:$PATH}}" ;;
esac
"#
    )
    .replace("\r\n", "\n"))
}

fn powershell_env_script(bin_dir: &Path) -> Result<String> {
    let bun_dir = toolchain_sibling_dir(bin_dir, "bun");
    let ninja_dir = toolchain_sibling_dir(bin_dir, "ninja");
    let cmake_bin_dir = cmake_bin_dir(bin_dir)?;
    let bin_dir = powershell_quote_str(&bin_dir.display().to_string());
    let bun_dir = powershell_quote_str(&bun_dir.display().to_string());
    let ninja_dir = powershell_quote_str(&ninja_dir.display().to_string());
    let cmake_bin_dir = powershell_quote_str(&cmake_bin_dir.display().to_string());
    Ok(format!(
        r#"$SippBin = {bin_dir}
$SippBun = {bun_dir}
$SippNinja = {ninja_dir}
$SippCmakeBin = {cmake_bin_dir}
$PathParts = @()
if ($env:Path) {{
  $PathParts = $env:Path -split [System.IO.Path]::PathSeparator
}}
if ($PathParts -notcontains $SippBin) {{
  $env:Path = "$SippBin$([System.IO.Path]::PathSeparator)$env:Path"
}}
if ($PathParts -notcontains $SippBun) {{
  $env:Path = "$SippBun$([System.IO.Path]::PathSeparator)$env:Path"
}}
if ($PathParts -notcontains $SippNinja) {{
  $env:Path = "$SippNinja$([System.IO.Path]::PathSeparator)$env:Path"
}}
if ($PathParts -notcontains $SippCmakeBin) {{
  $env:Path = "$SippCmakeBin$([System.IO.Path]::PathSeparator)$env:Path"
}}
"#
    ))
}

fn cmd_env_script(bin_dir: &Path) -> Result<String> {
    let bun_dir = toolchain_sibling_dir(bin_dir, "bun");
    let ninja_dir = toolchain_sibling_dir(bin_dir, "ninja");
    let cmake_bin_dir = cmake_bin_dir(bin_dir)?;
    let bin_dir = bin_dir.display().to_string().replace('%', "%%");
    let bun_dir = bun_dir.display().to_string().replace('%', "%%");
    let ninja_dir = ninja_dir.display().to_string().replace('%', "%%");
    let cmake_bin_dir = cmake_bin_dir.display().to_string().replace('%', "%%");
    Ok(format!(
        r#"@echo off
set "SIPP_BIN={bin_dir}"
set "SIPP_BUN={bun_dir}"
set "SIPP_NINJA={ninja_dir}"
set "SIPP_CMAKE_BIN={cmake_bin_dir}"
echo ;%PATH%; | find /I ";%SIPP_BIN%;" >nul
if errorlevel 1 set "PATH=%SIPP_BIN%;%PATH%"
echo ;%PATH%; | find /I ";%SIPP_BUN%;" >nul
if errorlevel 1 set "PATH=%SIPP_BUN%;%PATH%"
echo ;%PATH%; | find /I ";%SIPP_NINJA%;" >nul
if errorlevel 1 set "PATH=%SIPP_NINJA%;%PATH%"
echo ;%PATH%; | find /I ";%SIPP_CMAKE_BIN%;" >nul
if errorlevel 1 set "PATH=%SIPP_CMAKE_BIN%;%PATH%"
"#
    ))
}

fn toolchain_sibling_dir(bin_dir: &Path, name: &str) -> std::path::PathBuf {
    bin_dir
        .parent()
        .map(|build_dir| build_dir.join("toolchain").join(name))
        .unwrap_or_else(|| bin_dir.join("..").join("toolchain").join(name))
}

fn cmake_bin_dir(bin_dir: &Path) -> Result<std::path::PathBuf> {
    let cmake_dir = toolchain_sibling_dir(bin_dir, "cmake");
    cmake::cmake_bin_dir(&cmake_dir)
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
