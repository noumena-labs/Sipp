# Setup

Run the setup script from the repository root. It builds the `xtask` binary when
needed, installs `sipp` launchers under `.build/bin`, and can bootstrap managed
toolchains and sample files for the selected workflow.

## Unix Shells

```bash
source ./setup.sh
sipp doctor
```

Running `./setup.sh` without `source` still performs setup, but it cannot modify
the current shell `PATH`. It prints the environment script to source afterward.

## Windows PowerShell

```powershell
.\setup.ps1
sipp doctor
```

The PowerShell script updates `PATH` for the current PowerShell session and
loads `.build\bin\sipp-env.ps1` when setup succeeds.

## Windows CMD

```bat
setup.cmd
sipp doctor
```

`setup.cmd` invokes the PowerShell setup script and activates `.build\bin` for
the current CMD session.

## Profiles

Use a profile when you know which development surface you need:

```bash
sipp setup --profile browser
sipp setup --profile bindings
sipp setup --profile full --yes
```

| Profile | Use for |
| --- | --- |
| `browser` | Browser package, WASM, WebGPU examples, and demos. |
| `bindings` | Native Node.js and Python binding development. |
| `full` | Full workspace development across browser and native bindings. |

Useful setup flags:

- `--yes`: accept recommended actions without prompting.
- `--no-downloads`: skip toolchain, dependency, and sample-model downloads.
- `--no-splash`: skip the interactive splash.
- `--plain`: disable bounded terminal rendering.

## Generated Files

Setup writes only repo-local generated files:

- `.build/xtask/debug/xtask` or `.build\xtask\debug\xtask.exe`
- `.build/bin/sipp`, `.build/bin/sipp.cmd`, and `.build/bin/sipp.ps1`
- `.build/bin/sipp-env.sh` and `.build/bin/sipp-env.ps1`
- xtask-managed toolchains and caches under `.build/toolchain`

