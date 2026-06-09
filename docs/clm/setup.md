# Setup

Run the setup script from the repository root. It builds the `xtask` binary when
needed, installs `clm` launchers under `.build/bin`, and can bootstrap managed
toolchains and sample files for the selected workflow.

## Unix Shells

```bash
source ./setup.sh
clm doctor
```

Running `./setup.sh` without `source` still performs setup, but it cannot modify
the current shell `PATH`. It prints the environment script to source afterward.

## Windows PowerShell

```powershell
.\setup.ps1
clm doctor
```

The PowerShell script updates `PATH` for the current PowerShell session and
loads `.build\bin\cogentlm-env.ps1` when setup succeeds.

## Windows CMD

```bat
setup.cmd
clm doctor
```

`setup.cmd` invokes the PowerShell setup script and activates `.build\bin` for
the current CMD session.

## Profiles

Use a profile when you know which development surface you need:

```bash
clm setup --profile browser
clm setup --profile bindings
clm setup --profile full --yes
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
- `.build/bin/clm`, `.build/bin/clm.cmd`, and `.build/bin/clm.ps1`
- `.build/bin/cogentlm-env.sh` and `.build/bin/cogentlm-env.ps1`
- xtask-managed toolchains and caches under `.build/toolchain`

