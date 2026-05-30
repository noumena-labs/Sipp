$ErrorActionPreference = "Stop"

$Root = if ($PSScriptRoot) {
  $PSScriptRoot
} else {
  Split-Path -Parent $MyInvocation.MyCommand.Path
}

$Target = Join-Path $Root ".build\xtask\debug\xtask.exe"
$Stamp = Join-Path $Root ".build\xtask\clm.stamp"
$BinDir = Join-Path $Root ".build\bin"
$EnvScript = Join-Path $BinDir "cogentlm-env.ps1"

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
      $BuildExitCode = $LASTEXITCODE
      if ($env:COGENTLM_SETUP_CHILD -eq "1") {
        exit $BuildExitCode
      }
      throw "CogentLM setup bootstrap failed with exit code $BuildExitCode"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Stamp) | Out-Null
    Set-Content -Path $Stamp -Value "built $(Get-Date -Format o)"
  } finally {
    Pop-Location
  }
}

$PathParts = @()
if ($env:Path) {
  $PathParts = $env:Path -split [System.IO.Path]::PathSeparator
}
if ($PathParts -notcontains $BinDir) {
  $env:Path = "$BinDir$([System.IO.Path]::PathSeparator)$env:Path"
}

& $Target setup @args
$SetupExitCode = $LASTEXITCODE

if ((Test-Path $EnvScript) -and ($SetupExitCode -eq 0)) {
  . $EnvScript
  Write-Host ""
  Write-Host "clm is active in this PowerShell session."
}

if ($SetupExitCode -ne 0) {
  if ($env:COGENTLM_SETUP_CHILD -eq "1") {
    exit $SetupExitCode
  }
  throw "CogentLM setup failed with exit code $SetupExitCode"
}
