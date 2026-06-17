# Troubleshooting

## sipp Is Not Found

Run setup from the repository root and keep the environment active in the same
shell:

```bash
source ./setup.sh
```

```powershell
.\setup.ps1
```

```bat
setup.cmd
```

If you cannot activate `sipp`, use `cargo xtask` with the same arguments:

```bash
cargo xtask doctor
cargo xtask test list
```

## Setup Rebuilds xtask

The setup scripts rebuild `.build/xtask/debug/xtask` when xtask source files,
workspace manifests, or Cargo configuration are newer than
`.build/xtask/sipp.stamp`. This is expected after pulling changes that affect
developer automation.

## PowerShell Blocks Script Execution

Run the script with the current-user execution policy configured by your
machine, or invoke it for the current process:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

## PATH Is Active Only In One Terminal

The launcher is installed under `.build/bin`. Setup activates that directory
for the current shell session. Open a new terminal and run setup again, or
source the generated environment script:

```bash
source .build/bin/sipp-env.sh
```

```powershell
. .build\bin\sipp-env.ps1
```

## Toolchain Or Backend Is Missing

Use:

```bash
sipp doctor
sipp toolchain status
```

Then install xtask-managed components when appropriate:

```bash
sipp toolchain install uv
sipp toolchain install all
```

CUDA is not installed by xtask. Install CUDA through NVIDIA tooling and rerun
`sipp doctor --target node --backend cuda` or the target you need.

